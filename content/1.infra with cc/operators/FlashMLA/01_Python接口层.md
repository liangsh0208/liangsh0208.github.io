---
created: 2026-05-06
---

# Python 接口层

> 源码路径：`flash_mla/__init__.py`  
> `flash_mla/flash_mla_interface.py`

---

## 1. 模块结构与导出

`flash_mla` 包通过 `flash_mla_interface.py` 向外暴露全部公开接口，然后在 `__init__.py` 中统一导出：

```python
# flash_mla/__init__.py
__version__ = "1.0.0"

from flash_mla.flash_mla_interface import (
    get_mla_metadata,
    flash_mla_with_kvcache,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_mla_sparse_fwd
)
```

底层 C++ 扩展模块以 `flash_mla.cuda` 导入，由 pybind11 绑定（`csrc/api/api.cpp` 定义了以下四个函数）：

```cpp
// csrc/api/api.cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_decode_fwd", &sparse_attn_decode_interface);
    m.def("dense_decode_fwd",  &dense_attn_decode_interface);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
    m.def("dense_prefill_fwd",  &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd",  &FMHACutlassSM100BwdRun);
}
```

---

## 2. FlashMLASchedMeta：调度元数据对象

`FlashMLASchedMeta` 是 FlashMLA 解码接口的核心状态对象，用于缓存 Tile Scheduler 的元数据，避免每次解码循环都重复计算 SM 任务分配。

```python
@dataclasses.dataclass
class FlashMLASchedMeta:
    @dataclasses.dataclass
    class Config:
        b: int                          # batch size
        s_q: int                        # 每请求的 query token 数
        h_q: int                        # query 头数（通常 128）
        page_block_size: int            # paged KV cache 每页 token 数（通常 64）
        h_k: int                        # KV 头数（通常 1，MQA 模式）
        
        causal: bool                    # 是否 causal mask
        is_fp8_kvcache: bool            # 是否使用 FP8 KV cache
        topk: Optional[int]             # 稀疏注意力 top-k 数（None 表示密集）
        
        extra_page_block_size: Optional[int]  # 额外 KV cache 的 block size
        extra_topk: Optional[int]       # 额外 KV cache 的 topk
    
    have_initialized: bool = False      # 是否已完成首次初始化
    config: Optional[Config] = None    # 运行时配置（首次调用时设置）
    
    tile_scheduler_metadata: Optional[torch.Tensor] = None
    # shape: (num_sm_parts, TileSchedulerMetaDataSize), dtype=torch.int32
    # 每个 SM 分组负责的请求范围和 block 范围
    
    num_splits: Optional[torch.Tensor] = None
    # shape: (batch_size + 1,), dtype=torch.int32
    # split-KV 方案：num_splits[i] 记录第 i 条请求被分成几片
```

### 2.1 懒初始化设计

`get_mla_metadata()` 仅创建一个**空的** `FlashMLASchedMeta` 对象：

```python
def get_mla_metadata(*args, **kwargs) -> Tuple[FlashMLASchedMeta, None]:
    """
    返回一个空的 FlashMLASchedMeta。
    实际调度元数据在首次调用 flash_mla_with_kvcache 时才真正生成。
    第二个返回值始终为 None（历史兼容原因保留该位置）。
    """
    return FlashMLASchedMeta(), None
```

真正的初始化发生在 `flash_mla_with_kvcache` 首次执行时，并记录输入张量的形状和参数作为 `Config`。后续调用会校验这些参数是否一致：

```python
if not sched_meta.have_initialized:
    sched_meta.have_initialized = True
    sched_meta.config = FlashMLASchedMeta.Config(
        q.shape[0],   # b
        q.shape[1],   # s_q
        q.shape[2],   # h_q
        k_cache.shape[1],  # page_block_size
        k_cache.shape[2],  # h_k
        causal, is_fp8_kvcache, topk,
        extra_k_page_block_size, extra_topk,
    )
else:
    # 严格校验每次调用的配置是否与首次一致
    assert sched_meta.config.b == q.shape[0], ...
    assert sched_meta.config.topk == topk, ...
    # ... 更多校验
```

**设计动机**：在大语言模型推理循环中，同一个 decode 步骤通常只有 `cache_seqlens`（每条请求的序列长度）会变化，其他形状参数保持不变。复用 `FlashMLASchedMeta` 可以节省每步重新计算 SM 分配的 GPU kernel 开销。

---

## 3. flash_mla_with_kvcache：解码统一入口

这是 FlashMLA 最核心的用户接口，同时支持**密集解码**和**稀疏解码**两种模式：

```python
def flash_mla_with_kvcache(
    q: torch.Tensor,              # [batch_size, seq_len_q, num_heads_q, head_dim]
    k_cache: torch.Tensor,        # [num_blocks, page_block_size, num_heads_k, head_dim]
    block_table: Optional[torch.Tensor],   # [batch_size, max_num_blocks_per_seq], int32
    cache_seqlens: Optional[torch.Tensor], # [batch_size], int32
    head_dim_v: int,              # Value 的维度，必须为 512
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,      # 历史兼容，必须传 None
    softmax_scale: Optional[float] = None,  # 默认 1/sqrt(head_dim)
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,  # [batch, seq_len_q, topk]，稀疏模式专用
    attn_sink: Optional[torch.Tensor] = None, # [h_q]，注意力沉淀项
    extra_k_cache: Optional[torch.Tensor] = None,  # 额外 KV cache（稀疏模式）
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,  # [b]，每条请求实际 topk 长度
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 返回 (out, softmax_lse)
    # out: [batch_size, seq_len_q, num_heads_q, head_dim_v]
    # softmax_lse: [batch_size, num_heads_q, seq_len_q], float32
```

### 3.1 内部分支逻辑

```python
if topk is not None:
    # 稀疏注意力路径
    # 约束：is_fp8_kvcache 必须为 True，causal 必须为 False
    out, lse, new_meta, new_splits = flash_mla_cuda.sparse_decode_fwd(
        q, k_cache, indices, topk_length, attn_sink,
        sched_meta.tile_scheduler_metadata, sched_meta.num_splits,
        extra_k_cache, extra_indices, extra_topk_length,
        head_dim_v, softmax_scale
    )
else:
    # 密集注意力路径
    # 约束：block_table 和 cache_seqlens 必须提供
    out, lse, new_meta, new_splits = flash_mla_cuda.dense_decode_fwd(
        q, k_cache, head_dim_v,
        cache_seqlens, block_table,
        softmax_scale, causal,
        sched_meta.tile_scheduler_metadata, sched_meta.num_splits
    )

# 回写 tile scheduler 元数据供下次复用
sched_meta.tile_scheduler_metadata = new_tile_scheduler_metadata
sched_meta.num_splits = new_num_splits
```

### 3.2 KV Cache 格式

**密集解码（BF16）**：标准 paged KV cache 格式，`k_cache` 形状为 `[num_blocks, page_block_size, h_kv, head_dim]`。

**稀疏解码（FP8）**：每个 token 的 KV Cache 为 **656 字节**，内存布局如下：

```
每个 token 的 KV Cache（V3.2 格式，head_dim=576）：

偏移 0    ─ 512 字节：512 个 float8_e4m3 值（NoPE 量化部分）
             分 4 个 tile（每 tile 128 个 float8）
偏移 512  ─ 16 字节：4 个 float32 缩放因子
             scale[i] 对应第 i 个 128-float8 tile
偏移 528  ─ 128 字节：64 个 bfloat16 值（RoPE 部分，不量化）

合计：656 字节 / token
```

这与 BF16 无量化格式相比（576×2 = 1152 字节/token），节省 **43%** 内存。

---

## 4. flash_mla_sparse_fwd：稀疏预填充接口

用于 Prefill 阶段的稀疏注意力计算，**不含 batch 维度**（需调用方手动处理多 batch）：

```python
def flash_mla_sparse_fwd(
    q: torch.Tensor,           # [s_q, h_q, d_qk], bfloat16
    kv: torch.Tensor,          # [s_kv, h_kv, d_qk], bfloat16
    indices: torch.Tensor,     # [s_q, h_kv, topk], int32
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,  # [h_q], float32
    topk_length: Optional[torch.Tensor] = None, # [s_q], int32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 返回 (output, max_logits, lse)
    # output: [s_q, h_q, d_v], bfloat16
    # max_logits: [s_q, h_q], float32（每个 query 头的最大 logit）
    # lse: [s_q, h_q], float32（log-sum-exp，用于跨层/跨 batch 合并）
```

**等价的 PyTorch 伪代码**（摘自 README）：

```python
# kv: [s_kv, d_qk]（h_kv 必须为 1）
# indices: [s_q, topk]

focused_kv = kv[indices]       # [s_q, topk, d_qk]，按 topk 索引收集 KV

# QK^T 使用 log2 底数 softmax（计算上更快）
P = (Q @ focused_kv.transpose(-1, -2)) * sm_scale * log2(e)  # [s_q, h_q, topk]
max_logits = P.max(dim=-1)
lse = log2sumexp2(P, dim=-1)   # log2-based logsumexp
S = exp2(P - lse)              # softmax
out = S @ focused_kv           # [s_q, h_q, d_v]
```

注意索引格式：无效 token 设为 `-1` 或 `>= s_kv` 的值，内核会自动屏蔽。

---

## 5. Dense MHA Prefill 接口（SM100 专用）

这三个函数实现了与 `flash_attn` 包兼容的接口，底层调用 SM100（B200）专用的 CUTLASS kernel：

```python
def flash_attn_varlen_func(
    q: torch.Tensor,          # [total_seq_q, h, d]（所有 batch 拼接后）
    k: torch.Tensor,          # [total_seq_k, h_kv, d]
    v: torch.Tensor,          # [total_seq_k, h_kv, d_v]
    cu_seqlens_qo: torch.Tensor,   # [batch_size + 1]，累积序列长度（行指针）
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    dropout_p: float = 0.0,   # 必须为 0（不支持 dropout）
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    deterministic: bool = False,  # 必须为 False
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]   # (out, lse)
```

内部通过 `torch.autograd.Function` 封装，支持 `.backward()`：

```python
class FlashAttnVarlenFunc(torch.autograd.Function):
    def forward(ctx, q, k, v, ...):
        out, lse = _flash_attn_varlen_forward(...)
        ctx.save_for_backward(q, k, v, out, lse, ...)  # 保存反向所需张量
        return out, lse
    
    def backward(ctx, do, dlse):
        del dlse  # LSE 暂不支持反向
        dq, dk, dv = _flash_attn_varlen_backward(...)
        return dq, dk, dv, None, None, None, None, None, None, None
```

---

## 6. 注意力沉淀项（Attention Sink）

`attn_sink` 是一个可选的 `[h_q]` float32 向量，用于实现"注意力沉淀"机制。当提供时，输出会乘以一个缩放因子：

```python
# 伪代码（来自 ref.py）：
if attn_sink is not None:
    output *= (1.0 / (1.0 + exp(attn_sink[head] - lse[head])))
```

物理含义：将 `attn_sink` 视为一个额外的"虚拟 token"的 log-sum-exp 值。若 `attn_sink` 足够大（趋向 +∞），当前 output 趋向 0；若 `attn_sink` 很小（趋向 -∞），则无影响。这可以用于实现 StreamingLLM 等需要"沉淀注意力"的场景。

---

## 7. 典型使用范式

### 7.1 密集 MLA 解码循环

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

# 在解码循环外初始化（只需一次）
sched_meta, _ = get_mla_metadata()

# 解码循环
for step in range(num_decode_steps):
    # ... 计算 q, 更新 kvcache, block_table, cache_seqlens ...
    
    out, lse = flash_mla_with_kvcache(
        q,               # [b, 1, 128, 576]（s_q=1 时为单步解码）
        kvcache,         # [num_blocks, 64, 1, 576]（BF16 格式）
        block_table,     # [b, max_blocks]
        cache_seqlens,   # [b]，每条请求当前序列长度
        head_dim_v=512,
        tile_scheduler_metadata=sched_meta,
        causal=False,
    )
    # out: [b, 1, 128, 512]
    # lse: [b, 128, 1]
```

### 7.2 FP8 稀疏 MLA 解码循环

```python
from tests.quant import quantize_k_cache, FP8KVCacheLayout, abs_indices2indices_in_kvcache

# 量化 KV cache（通常在 prefill 阶段完成）
fp8_kvcache = quantize_k_cache(bf16_kvcache, FP8KVCacheLayout.V32_FP8Sparse)

# 转换稀疏索引格式
indices_in_kvcache = abs_indices2indices_in_kvcache(abs_indices, block_table, block_size=64)

sched_meta, _ = get_mla_metadata()

for step in range(num_decode_steps):
    out, lse = flash_mla_with_kvcache(
        q,               # [b, 1, 128, 576]
        fp8_kvcache,     # [num_blocks, 64, 1, 656]（FP8 格式，每 token 656 字节）
        block_table=None,     # 稀疏模式下 block_table 不使用
        cache_seqlens=None,   # 稀疏模式下 cache_seqlens 不使用
        head_dim_v=512,
        tile_scheduler_metadata=sched_meta,
        is_fp8_kvcache=True,
        indices=indices_in_kvcache,  # [b, 1, topk]
    )
```

### 7.3 稀疏预填充

```python
from flash_mla import flash_mla_sparse_fwd

# 注意：无 batch 维度，q/kv 为 3D 张量
out, max_logits, lse = flash_mla_sparse_fwd(
    q=q,           # [s_q, h_q, 576]
    kv=kv,         # [s_kv, 1, 576]
    indices=indices, # [s_q, 1, topk]
    sm_scale=1.0 / math.sqrt(576),
    d_v=512,
)
```
