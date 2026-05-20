---
created: 2026-05-06
---

# Fused Attention 融合注意力

> 源码路径：`transformer_engine/common/fused_attn/`  
> `transformer_engine/pytorch/attention/`  
> `transformer_engine/common/include/transformer_engine/fused_attn.h`

---

## 1. 为什么需要 Fused Attention

标准多头注意力计算：

```python
# 朴素实现
Q, K, V = linear_qkv(x)                     # [B, S, H]
scores = Q @ K.transpose(-2, -1) / sqrt(d)   # [B, h, S, S]  ← 显存峰值
scores = softmax(scores + mask)               # [B, h, S, S]
out = scores @ V                              # [B, h, S, d]
```

**问题**：`scores` 矩阵的显存占用为 `O(B * h * S²)`，对于 S=8192、h=32 的配置约 **8GB**（FP16），严重限制了长序列训练。

**Fused Attention（Flash Attention 思路）**：将 softmax 和矩阵乘法融合为一个 kernel，分块流式处理，峰值显存降至 `O(B * h * S)`。

---

## 2. TE 的 Fused Attention 后端

TE 实现了**多个 backend**，根据输入形状和精度自动选择：

| Backend | 文件 | 适用场景 |
|---------|------|---------|
| `flash_attn.cu` | FlashAttention-2/3 | 任意序列长度，主力 backend |
| `fused_attn_f16_max512_seqlen.cu` | cuDNN 短序列 | seqlen ≤ 512，F16/BF16 |
| `fused_attn_f16_arbitrary_seqlen.cu` | cuDNN 任意长度 | F16/BF16 |
| `fused_attn_fp8.cu` | cuDNN FP8 | FP8 精度（Hopper+） |
| `context_parallel.cu` | Ring Attention | 序列并行（CP > 1） |

### 2.1 Backend 选择逻辑（`fused_attn.cpp`）

```cpp
// 优先级（简化）：
// 1. 若 seqlen <= 512 且 F16/BF16 → cuDNN max512
// 2. 若 FP8 → cuDNN FP8 backend
// 3. 其他 → FlashAttention（最通用）
NVTE_Fused_Attn_Backend get_fused_attn_backend(
    NVTEDType q_dtype, NVTEDType kv_dtype,
    NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type,
    float dropout,
    size_t num_attn_heads, size_t num_gqa_groups,
    size_t max_seqlen_q, size_t max_seqlen_kv,
    size_t head_dim_qk, size_t head_dim_v
);
```

---

## 3. FlashAttention 集成

TE 集成了 FlashAttention-2/3 算法，核心思想：

### 3.1 分块 Online Softmax

```
将 Q 按行分块（Q_i，大小 Br），K/V 按列分块（K_j/V_j，大小 Bc）

对每个 (Q_i, K_j) 块：
  S_ij = Q_i @ K_j^T / sqrt(d)    # local scores
  m_ij = max(S_ij)                  # local max
  
  # 使用 online softmax 更新全局 max 和归一化常数
  m_i = max(m_i_prev, m_ij)
  l_i = exp(m_i_prev - m_i) * l_i_prev + sum(exp(S_ij - m_ij))
  
  # 更新输出（滚动累加）
  O_i = O_i_prev * exp(m_i_prev - m_i) + exp(S_ij - m_ij) @ V_j
```

最终 `O_i = O_i / l_i`，得到正确 softmax 归一化的输出。

### 3.2 显存节省

```
标准注意力显存：O(S²)（存整个 attention matrix）
Flash Attention 显存：O(S)（每次只保存当前块）

以 S=4096, h=32, B=4, d=128 为例：
  标准：4096² * 32 * 4 * 2 bytes ≈ 4.3 GB
  Flash：4096 * 128 * 32 * 4 * 2 bytes ≈ 134 MB
```

---

## 4. FP8 Fused Attention

`fused_attn_fp8.cu` 实现了完整的 FP8 精度注意力（需要 Hopper+）：

```python
# recipe 中启用 FP8 注意力
fp8_recipe = recipe.DelayedScaling(
    fp8_dpa=True,   # FP8 Dot Product Attention（边界有 cast）
    fp8_mha=True,   # 完整 MHA 链路（无边界 cast）
)
```

### 4.1 FP8 DPA（`fp8_dpa=True, fp8_mha=False`）

```
QKV 投影（BF16 输出）
  → cast to FP8
  → FP8 Attention kernel（内部 FP8 计算）
  → cast to BF16
  → O 投影
```

### 4.2 FP8 MHA（`fp8_mha=True`）

```
QKV 投影（FP8 输出，无需 cast）
  → FP8 Attention kernel
  → FP8 输出直接送入 O 投影
```

省去了两次 cast 操作，对大 batch/长序列有约 5-10% 的额外加速。

### 4.3 内部实现

```cpp
// fused_attn_fp8.cu
// cuDNN 图 API（cudnn-frontend）调用
void fused_attn_fp8_fwd_qkvpacked(
    const NVTETensor QKV,        // [B, S, 3, h, d]，FP8
    const NVTETensor Bias,       // attention bias
    NVTETensor S,                // softmax 统计量（lse）
    NVTETensor O,                // 输出，FP8
    NVTETensorPack aux_output_tensors,
    const NVTETensor cu_seqlens, // 变长序列的累积长度
    ...
    float attn_scale,            // 1/sqrt(d)
    float dropout,
    NVTE_Mask_Type mask_type,
    ...
);
```

---

## 5. Context Parallel（Ring Attention）

`context_parallel.cu` 实现了序列并行注意力，将长序列切分到多个 GPU：

```
CP（Context Parallel）= 4，序列 S=16384

GPU 0: Q[0:4096],   K[0:4096],   V[0:4096]
GPU 1: Q[4096:8192], K[4096:8192], V[4096:8192]
...

Ring AllGather：每个 GPU 轮流将自己的 K/V 发送到所有其他 GPU
每轮计算本地 Q 与收到的 K/V 的注意力分数，并 online softmax 累加
```

**通信-计算重叠**：发送 K/V 的 P2P 通信与下一轮的注意力计算重叠执行，通信几乎完全隐藏。

```python
# 使用 Context Parallel
with te.fp8_autocast(enabled=True, ...):
    output = te_attn(
        q, k, v,
        attn_mask_type="causal",
        cp_group=context_parallel_group,  # 序列并行通信组
        cp_global_ranks=cp_ranks,
    )
```

---

## 6. KV Cache（推理优化）

`kv_cache.cu` 和 `inference.py` 实现了推理时的 KV Cache 管理：

```python
from transformer_engine.pytorch.attention import InferenceParams

inference_params = InferenceParams(
    max_batch_size=32,
    max_sequence_length=8192,
)

# 推理循环
for step in range(max_new_tokens):
    with te.fp8_autocast(enabled=True, ...):
        output = model(
            input_ids[:, step:step+1],   # 只传入当前 token
            inference_params=inference_params,
        )
    # KV Cache 在 inference_params 内部自动追加
```

KV Cache 的存储格式支持 FP8（节省显存），在每步推理时：
1. 将新的 K/V 量化为 FP8 并追加到 cache
2. 对历史 K/V（FP8）和当前 Q 计算注意力
3. 使用 paged attention 或连续 cache 两种模式

---

## 7. MultiheadAttention 模块

```python
class MultiheadAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,           # 模型宽度
        num_attention_heads: int,   # 注意力头数
        kv_channels: int = None,    # 每头维度（默认 hidden/heads）
        attention_dropout: float = 0.1,
        
        # GQA/MQA 支持
        num_gqa_groups: int = None,   # Group Query Attention 分组数
        
        # 注意力类型
        attn_mask_type: str = "causal",   # "causal", "padding", "no_mask"
        
        # 位置编码
        rotary_pos_emb_group_method: str = "consecutive",
        
        # QK Norm
        qk_norm: bool = False,       # 是否对 Q/K 做 L2 Norm
        
        # 后端控制
        fused_qkv_params: bool = True,   # QKV 是否用同一 parameter
        ...
    )
```

### 7.1 GQA（Group Query Attention）支持

```python
# 标准 MHA：每个 Q 头有独立的 K/V 头
# GQA：多个 Q 头共享一组 K/V 头
mha = te.MultiheadAttention(
    hidden_size=4096,
    num_attention_heads=32,   # Q 头数
    num_gqa_groups=8,         # K/V 头数（每 4 个 Q 共享 1 组 KV）
)
```

内部实现：K/V 矩阵维度为 `[B, S, num_gqa_groups, head_dim]`，在注意力计算前通过 expand/repeat 广播到 `num_attention_heads`。

### 7.2 QK Norm

```python
# 对 Q 和 K 施加 L2 归一化（稳定训练）
mha = te.MultiheadAttention(hidden_size=4096, num_attention_heads=32, qk_norm=True)
```

实现于 `ops/basic/l2normalization.py`，等价于 `q = q / (||q|| + eps)`。

---

## 8. RotaryPositionEmbedding（RoPE）

```python
from transformer_engine.pytorch.attention import RotaryPositionEmbedding

rotary_emb = RotaryPositionEmbedding(kv_channels)
# 生成位置编码（在推理/训练开始时调用一次）
rotary_pos_emb = rotary_emb(max_seq_len)

# 在 MultiheadAttention 前向中：
output = mha(x, rotary_pos_emb=rotary_pos_emb)
```

RoPE 的 CUDA 实现（`fused_rope/`）将旋转操作与 QKV split 融合，避免额外的显存写入。

支持两种排列方式：
- `"consecutive"`：相邻维度成对旋转（`[d0,d1,...] → [(d0,d1),(d2,d3),...]`）
- `"interleaved"`：前半/后半配对旋转（`[(d0,d_{N/2}),(d1,d_{N/2+1}),...]`）

---

## 9. DotProductAttention 详解

```python
class DotProductAttention(nn.Module):
    """纯注意力计算（不含 QKV 投影）
    
    输入：q[B,S_q,h,d], k[B,S_k,h,d], v[B,S_k,h,d]
    输出：context[B,S_q,h,d]
    """
    
    def forward(
        self,
        query_layer:   torch.Tensor,  # [B, S_q, h, d]
        key_layer:     torch.Tensor,
        value_layer:   torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[...] = None,
        core_attention_bias: Optional[torch.Tensor] = None,  # ALiBi 等
        ...
    )
```

`DotProductAttention` 内部会：
1. 根据数据类型、序列长度、是否 FP8 等条件选择 backend
2. 调用 `fused_attn.cpp` 的 C++ 接口
3. 处理变长序列（padding pack/unpack）

---

## 10. Attention Bias 类型

```python
# attn_bias_type 枚举（AttnBiasTypes）
"no_bias"          # 无额外 bias（标准注意力）
"pre_scale_bias"   # score = Q@K/sqrt(d) + bias，在 softmax 前加
"post_scale_bias"  # score = Q@K/sqrt(d)，softmax 后加
"alibi"            # ALiBi 线性位置偏置
"alibi_slopes_of_ones"  # ALiBi 斜率全为 1

# 在 TE 中用 Attention Bias 传入
with te.fp8_autocast(...):
    out = dpa(q, k, v, core_attention_bias=alibi_bias)
```
