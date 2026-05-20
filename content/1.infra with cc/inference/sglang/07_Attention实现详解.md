---
created: 2026-05-06
---

# 第七章：Attention 实现详解

## 一、概述

SGLang 支持多种注意力后端，包括 FlashInfer、FlashAttention、FlashMLA 等。本章详细分析注意力的实现架构和各后端的特点。

## 二、注意力架构设计

### 2.1 后端注册机制

**文件位置**：`python/sglang/srt/layers/attention/attention_registry.py`

```python
# 全局后端注册表
ATTENTION_BACKENDS: Dict[str, Type[BaseAttentionBackend]] = {}


def register_attention_backend(name: str):
    """注册注意力后端的装饰器"""
    def decorator(backend_class):
        ATTENTION_BACKENDS[name] = backend_class
        return backend_class
    return decorator


def get_attention_backend(name: str) -> BaseAttentionBackend:
    """获取注意力后端实例"""
    if name not in ATTENTION_BACKENDS:
        raise ValueError(f"Unknown attention backend: {name}")
    return ATTENTION_BACKENDS[name]()
```

### 2.2 BaseAttentionBackend 接口

**文件位置**：`python/sglang/srt/layers/attention/base_attn_backend.py`

```python
class BaseAttentionBackend(abc.ABC):
    """注意力后端基类"""

    @abc.abstractmethod
    def init_forward_metadata(
        self,
        forward_batch: ForwardBatch,
        model_config: ModelConfig,
    ):
        """初始化前向传播元数据"""
        pass

    @abc.abstractmethod
    def forward_extend(
        self,
        q: torch.Tensor,           # Query [num_tokens, num_heads, head_dim]
        k: torch.Tensor,           # Key [num_tokens, num_kv_heads, head_dim]
        v: torch.Tensor,           # Value [num_tokens, num_kv_heads, head_dim]
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Prefill 阶段的注意力计算"""
        pass

    @abc.abstractmethod
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Decode 阶段的注意力计算"""
        pass

    def get_cuda_graph_runner(self):
        """获取 CUDA Graph 执行器 (可选)"""
        return None
```

### 2.3 后端选择

```python
def get_attn_backend(
    model_config: ModelConfig,
    server_args: ServerArgs,
) -> BaseAttentionBackend:
    """根据配置选择最合适的注意力后端"""

    # 用户显式指定
    if server_args.attention_backend:
        return get_attention_backend(server_args.attention_backend)

    # 根据模型和硬件自动选择
    if model_config.attention_arch == AttentionArch.MLA:
        # DeepSeek MLA 模型
        if is_flashmla_available():
            return FlashMLABackend()
        else:
            return FlashInferMLABackend()

    if is_flashinfer_available():
        return FlashInferBackend()
    else:
        return FlashAttentionBackend()
```

## 三、FlashInfer 后端

### 3.1 概述

**文件位置**：`python/sglang/srt/layers/attention/flashinfer_backend.py`

FlashInfer 是高性能的注意力库，支持多种优化。

```python
@register_attention_backend("flashinfer")
class FlashInferBackend(BaseAttentionBackend):
    """FlashInfer 注意力后端"""

    def __init__(self):
        self.forward_metadata = None

    def init_forward_metadata(self, forward_batch: ForwardBatch, model_config: ModelConfig):
        """初始化前向传播元数据"""

        if forward_batch.forward_mode.is_extend():
            # Prefill 模式
            self.forward_metadata = FlashInferPrefillMetadata(
                qo_indptr=forward_batch.qo_indptr,
                kv_indptr=forward_batch.kv_indptr,
                kv_indices=forward_batch.kv_indices,
                ...
            )
        else:
            # Decode 模式
            self.forward_metadata = FlashInferDecodeMetadata(
                qo_indptr=forward_batch.qo_indptr,
                kv_indptr=forward_batch.kv_indptr,
                ...
            )
```

### 3.2 Prefill 实现

```python
def forward_extend(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    """使用 FlashInfer 执行 Prefill 注意力"""

    metadata = self.forward_metadata

    # 写入 KV Cache
    cache_k, cache_v = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    cache_k[forward_batch.out_cache_loc] = k
    cache_v[forward_batch.out_cache_loc] = v

    # 执行注意力
    output = flashinfer.prefill_attention(
        q,
        cache_k,
        cache_v,
        qo_indptr=metadata.qo_indptr,
        kv_indptr=metadata.kv_indptr,
        kv_indices=metadata.kv_indices,
        causal=True,
        sm_scale=layer.scaling,
    )

    return output
```

### 3.3 Decode 实现

```python
def forward_decode(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    """使用 FlashInfer 执行 Decode 注意力"""

    metadata = self.forward_metadata

    # 追加新 KV
    cache_k, cache_v = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    cache_k[forward_batch.out_cache_loc] = k
    cache_v[forward_batch.out_cache_loc] = v

    # 执行 decode 注意力 (每个 query 对应完整序列的 KV)
    output = flashinfer.decode_attention(
        q,
        cache_k,
        cache_v,
        qo_indptr=metadata.qo_indptr,
        kv_indptr=metadata.kv_indptr,
        sm_scale=layer.scaling,
    )

    return output
```

## 四、FlashAttention 后端

### 4.1 概述

**文件位置**：`python/sglang/srt/layers/attention/flashattention_backend.py`

使用原生的 FlashAttention 内核。

```python
@register_attention_backend("flashattention")
class FlashAttentionBackend(BaseAttentionBackend):
    """FlashAttention 注意力后端"""

    def forward_extend(self, q, k, v, layer, forward_batch):
        """使用 FlashAttention 执行 Prefill"""

        # 写入 KV Cache
        ...

        # 准备分块参数
        block_tables = forward_batch.block_tables

        # 调用 FlashAttention
        output = flash_attn_with_kvcache(
            q=q,
            k_cache=cache_k,
            v_cache=cache_v,
            block_tables=block_tables,
            cache_seqlens=forward_batch.seq_lens,
            causal=True,
        )

        return output
```

## 五、FlashMLA 后端

### 5.1 概述

**文件位置**：`python/sglang/srt/layers/attention/flashmla_backend.py`

专为 DeepSeek-V2/V3 的 Multi-Latent Attention 优化。

```python
@register_attention_backend("flashmla")
class FlashMLABackend(BaseAttentionBackend):
    """
    FlashMLA 后端 - 专门优化 MLA 注意力。

    MLA 特点：
    1. KV 压缩到低维潜在空间
    2. 分离 RoPE 部分
    3. 大幅减少 KV Cache 大小
    """

    def forward_extend(self, q, k, v, layer, forward_batch):
        """MLA 的 Prefill 注意力"""

        # MLA 的 KV 已经是压缩形式
        # k: [num_tokens, 1, kv_lora_rank + qk_rope_head_dim]

        # 解压缩 query (如果需要)
        q_nope, q_pe = q.split([...], dim=-1)

        # 准备 KV
        kv_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        # 执行 MLA 注意力
        output = flash_mla_attention(
            q_nope, q_pe,
            kv_cache,
            ...
        )

        return output
```

### 5.2 MLA 独特之处

```python
# 标准注意力 KV 大小
# K/V: [num_tokens, num_heads, head_dim]
# 总大小: 2 * num_tokens * num_heads * head_dim

# MLA 压缩 KV 大小
# KV: [num_tokens, 1, kv_lora_rank + qk_rope_head_dim]
# 总大小: num_tokens * (kv_lora_rank + qk_rope_head_dim)

# 例如 DeepSeek-V3:
# num_heads = 128, head_dim = 192
# kv_lora_rank = 512, qk_rope_head_dim = 64
#
# 标准: 2 * 128 * 192 = 49,152 per token
# MLA: 512 + 64 = 576 per token
# 压缩比: ~85x
```

## 六、RadixAttention 层

### 6.1 层实现

**文件位置**：`python/sglang/srt/layers/attention/`/ (在模型层中使用)

```python
class RadixAttention(nn.Module):
    """
    基于基数树缓存优化的注意力层。

    集成了 KV Cache 和前缀复用。
    """

    def __init__(
        self,
        layer_id: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scaling: float,
        ...
    ):
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = scaling

        # QKV 投影
        self.qkv_proj = QKVParallelLinear(...)

        # 输出投影
        self.o_proj = RowParallelLinear(...)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        ...
    ):
        # 1. QKV 投影
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([...], dim=-1)

        # 2. 应用 RoPE
        if self.rope is not None:
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        # 3. 执行注意力 (调用后端)
        if forward_batch.forward_mode.is_extend():
            output = self.attn_backend.forward_extend(q, k, v, self, forward_batch)
        else:
            output = self.attn_backend.forward_decode(q, k, v, self, forward_batch)

        # 4. 输出投影
        output = self.o_proj(output)

        return output
```

### 6.2 RoPE 位置编码

```python
class RotaryEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""

    def __init__(self, head_dim: int, max_position: int, base: float = 10000.0):
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """应用旋转位置编码"""

        # positions: [seq_len]
        # x: [seq_len, num_heads, head_dim]

        # 计算旋转角
        freqs = torch.outer(positions.float(), self.inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()

        # 应用旋转
        x_rot = self._rotate_half(x)
        return x * cos.unsqueeze(1) + x_rot * sin.unsqueeze(1)

    def _rotate_half(self, x):
        """旋转一半维度"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
```

## 七、注意力掩码

### 7.1 因果掩码

```python
# 标准因果注意力掩码
# 每个位置只能看到自己和之前的位置

def causal_attention_mask(seq_len: int) -> torch.Tensor:
    """生成因果注意力掩码"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.bool()  # True 表示要 mask 掉

# [seq_len=4]:
# [[F, T, T, T],
#  [F, F, T, T],
#  [F, F, F, T],
#  [F, F, F, F]]
```

### 7.2 分块因果掩码

```python
# Chunked Prefill 时的掩码处理
# 每个 chunk 内保持因果关系

def chunked_causal_mask(
    chunk_starts: List[int],  # 每个 chunk 的起始位置
    chunk_lens: List[int],    # 每个 chunk 的长度
):
    """生成分块因果掩码"""
    total_len = sum(chunk_lens)
    mask = torch.ones(total_len, total_len, dtype=torch.bool)

    offset = 0
    for start, length in zip(chunk_starts, chunk_lens):
        # chunk 内的因果掩码
        chunk_mask = torch.triu(torch.ones(length, length), diagonal=1)
        mask[offset:offset+length, offset:offset+length] = chunk_mask.bool()

        # 允许看到之前 prefix
        if offset > 0:
            mask[offset:offset+length, :offset] = False

        offset += length

    return mask
```

## 八、KV Cache 管理

### 8.1 写入 KV Cache

```python
def write_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    cache_loc: torch.Tensor,
):
    """将 K/V 写入缓存"""

    # 直接索引写入
    cache_k[cache_loc] = k
    cache_v[cache_loc] = v
```

### 8.2 读取 KV Cache

```python
def read_kv_cache(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    kv_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """从缓存读取 K/V"""

    # 根据索引读取
    k = cache_k[kv_indices]
    v = cache_v[kv_indices]

    return k, v
```

## 九、性能优化

### 9.1 Flash Attention 优势

```
标准注意力 (O(n²) 内存):
for i in range(seq_len):
    for j in range(seq_len):
        attn[i, j] = Q[i] @ K[j] / sqrt(d)
        attn[i, j] = softmax(attn[i])
        out[i] += attn[i, j] * V[j]

Flash Attention (O(n) 内存):
# 分块计算，避免存储完整注意力矩阵
for block_q in range(0, seq_len, block_size):
    for block_k in range(0, seq_len, block_size):
        # 块内计算
        q_block = Q[block_q:block_q+block_size]
        k_block = K[block_k:block_k+block_size]
        ...
```

### 9.2 Paged Attention

```python
# Paged Attention 将 KV Cache 分页存储
# 支持不连续的内存布局

class PagedKVCache:
    def __init__(self, num_blocks: int, block_size: int, ...):
        # 物理存储: [num_blocks, block_size, num_heads, head_dim]
        self.k_cache = torch.zeros(...)
        self.v_cache = torch.zeros(...)

        # 逻辑到物理的映射
        self.block_tables: Dict[int, List[int]] = {}

    def get_block_table(self, seq_id: int) -> List[int]:
        """获取序列的 block 映射表"""
        return self.block_tables[seq_id]
```

## 十、后端对比

| 后端 | 适用场景 | 特点 |
|------|----------|------|
| FlashInfer | 通用 | 高性能，支持多种优化，vLLM/SGLang 默认 |
| FlashAttention | 通用 | 原生实现，兼容性好 |
| FlashMLA | DeepSeek 模型 | MLA 专用，极低 KV 内存 |
| NSA | 长序列 | 原生稀疏注意力 |
| AITER | AMD GPU | AMD AI 引擎优化 |
| TorchNative | 调试/回退 | PyTorch 原生 SDPA，无第三方依赖 |
| Triton | 自定义/实验 | Triton 实现，便于修改和实验 |
| Wave | Wave Computing | Wave 硬件后端 |
| Intel AMX | Intel CPU | AMX 指令集加速 |
| XPU | Intel GPU | Intel XPU 后端 |
| TRT-LLM MHA/MLA | NVIDIA GPU | TensorRT-LLM 注意力内核 |
| DualChunkFlashAttention | 超长序列 | 双块 FlashAttention，流式大窗口 |
| TBO (Token-Block-Offset) | 移动端 | 轻量级注意力实现 |
| HybridLinear/FLA | 线性注意力 | Gated Linear Attention 变体 |

## 十一、总结

### 11.1 架构优势

1. **可插拔后端**：通过注册机制支持多种后端
2. **统一接口**：BaseAttentionBackend 提供统一抽象
3. **自动选择**：根据模型和硬件自动选择最优后端
4. **性能优化**：Flash Attention、Paged Attention 等

### 11.2 核心数据流

```
隐藏状态 (hidden_states)
       │
       ▼
    QKV 投影 (qkv_proj)
       │
       ├── Q → RoPE → query
       ├── K → RoPE → key
       └── V → value
       │
       ▼
    注意力后端 (attn_backend)
       │
       ├── forward_extend → Prefill
       ├── forward_decode → Decode
       │
       ├── 读/写 KV Cache
       │
       ▼
    注意力输出 (attn_output)
       │
       ▼
    输出投影 (o_proj)
       │
       ▼
    最终输出 (output)
```

---

**上一章**：[模型执行](06_模型执行.md)

**下一章**：[采样与输出处理](08_采样与输出处理.md)