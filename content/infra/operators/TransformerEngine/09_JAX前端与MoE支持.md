# JAX 前端与 MoE 支持

> JAX 前端源码：`transformer_engine/jax/`  
> MoE 相关：`transformer_engine/pytorch/permutation.py`  
> `transformer_engine/common/fused_router/`  
> `transformer_engine/common/permutation/`

---

## Part 1：JAX 前端

---

## 1. JAX 前端架构

```
JAX 用户代码（Flax / 原生 JAX）
         │
         ▼
jax/flax/transformer.py     ← Flax TransformerLayer
jax/flax/module.py          ← Flax 基础模块（DenseGeneral 等）
         │
         ▼
jax/*.py                    ← 纯 JAX 函数（attention、layernorm 等）
         │
         ▼
jax/cpp_extensions/         ← Python → C++ 绑定层
jax/csrc/extensions/        ← C++ XLA custom call 注册
         │
         ▼
common/ libtransformer_engine.so  ← 共用 CUDA kernel
```

JAX 的扩展机制是 **XLA Custom Call**：将 TE 的 CUDA kernel 注册为 XLA 图中的自定义算子节点，JIT 编译时 XLA 会直接调用这些节点。

---

## 2. Flax 模块层

### 2.1 `TransformerLayer`（Flax）

```python
from transformer_engine.jax.flax import TransformerLayer

class MyModel(nn.Module):
    @nn.compact
    def __call__(self, x, mask, deterministic=True):
        return TransformerLayer(
            hidden_size=1024,
            mlp_hidden_size=4096,
            num_attention_heads=16,
            
            # 归一化
            layernorm_type="rmsnorm",     # 或 "layernorm"
            
            # 激活
            mlp_activations=("silu", "linear"),  # SwiGLU
            
            # 注意力
            attn_mask_type="causal",
            
            # 分片（XLA 分布式）
            logical_axis_rules=[...],
        )(x, mask, deterministic=deterministic)
```

### 2.2 Flax 基础模块

```python
# jax/flax/module.py 中的模块
from transformer_engine.jax.flax.module import (
    DenseGeneral,        # 支持 FP8 的通用线性层
    LayerNorm,           # LayerNorm
    RMSNorm,             # RMSNorm
    MultiHeadAttention,  # 多头注意力
    RelativePositionBiases,  # T5 相对位置编码
)
```

### 2.3 激活函数支持

```python
# JAX 版 SwiGLU 配置
TransformerLayer(
    mlp_activations=("silu", "linear"),   # SwiGLU = silu(gate) * linear(x)
    # 等价 PyTorch：activation="swiglu"
)
TransformerLayer(
    mlp_activations=("gelu",),            # 标准 GELU
)
TransformerLayer(
    mlp_activations=("gelu", "linear"),   # GeGLU
)
```

---

## 3. FP8 in JAX

### 3.1 FP8 autocast

```python
from transformer_engine.jax import fp8_autocast
from transformer_engine.common import recipe

fp8_recipe = recipe.DelayedScaling()

# 使用 fp8_autocast context
with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    output = model.apply(params, x, mask)
```

### 3.2 量化函数

```python
from transformer_engine.jax import quantize, dequantize

# 显式量化
x_fp8, scale, amax = quantize(
    x,                          # 输入（BF16/FP32）
    q_dtype=jnp.float8_e4m3fn, # FP8 格式
    scale=existing_scale,       # 使用已有 scale
)

# 显式反量化
x_fp32 = dequantize(x_fp8, scale)
```

---

## 4. XLA 分布式（Sharding）

### 4.1 逻辑轴标注

JAX 使用 `logical_axis_rules` 将 Tensor 的逻辑轴映射到设备网格轴：

```python
from transformer_engine.jax.sharding import (
    MeshResource,
    global_shard_guard,
    te_einsum,
)

# 定义设备网格（TP=4, DP=2）
mesh = jax.make_mesh((2, 4), ("dp", "tp"))

axis_rules = [
    ("batch", "dp"),          # batch 维度沿 dp 切分
    ("hidden", "tp"),         # hidden 维度沿 tp 切分
    ("mlp", "tp"),            # mlp 维度沿 tp 切分
    ("heads", "tp"),          # 注意力头沿 tp 切分
]

with global_shard_guard(MeshResource(dp_resource="dp", tp_resource="tp")):
    with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        output = model.apply(params, x, mask)
```

### 4.2 自动分片

TE JAX 会根据 `logical_axis_rules` 自动在模型前向中插入正确的 XLA sharding 标注：

```python
# 内部自动处理（以 DenseGeneral 为例）：
# weight: [hidden, mlp] → shard along "tp" axis
# input:  [batch, seq, hidden] → shard batch along "dp"
# output: [batch, seq, mlp] → shard mlp along "tp"
```

---

## 5. Triton 扩展

```
jax/triton_extensions/     ← JAX 调用 Triton kernel 的绑定
```

对于 Blackwell GPU 上的 MXFP8/NVFP4，TE 也提供 Triton 实现作为 CUDA kernel 的备用：

```python
from transformer_engine.jax.triton_extensions import (
    triton_fp8_cast,
    triton_mxfp8_gemm,
)
```

---

## 6. JAX 测试与 Benchmark

```python
# jax/softmax.py 和 jax/attention.py 提供独立函数
from transformer_engine.jax import softmax, dot_product_attention

# 融合 softmax（避免 S² 显存）
out = softmax(
    logits,           # [B, h, S, S]
    mask=attn_mask,
    scale_factor=1.0/sqrt(d_head),
)

# 融合点积注意力
out = dot_product_attention(
    query,   # [B, S, h, d]
    key,
    value,
    bias=attn_bias,
    mask=causal_mask,
    dropout_rate=0.0,
    is_training=True,
    qkv_layout="bshd_bshd_bshd",
)
```

---

---

## Part 2：MoE 支持

---

## 7. MoE 的挑战

Mixture of Experts（MoE）的训练难点：

1. **Token Permutation**：不同 token 路由到不同 expert，需要重新排列
2. **不均匀 batch**：各 expert 处理的 token 数量不同，朴素实现效率低
3. **分布式 MoE**：expert 可以分布在不同 GPU 上（Expert Parallelism）
4. **FP8 兼容**：permute/unpermute 操作需要处理 FP8 Tensor

---

## 8. Token Permutation

### 8.1 标准 Permute/Unpermute

```python
from transformer_engine.pytorch.permutation import (
    moe_permute,
    moe_unpermute,
)

# Router 输出：每个 token 分配到哪些 expert
# routing_map: [num_tokens, topk] (bool)
# topk=2 表示每个 token 送到 2 个 expert

# Permute：按 expert 重排 token
permuted_input, row_id_map = moe_permute(
    input_tensor,   # [num_tokens, hidden_size]
    routing_map,    # [num_tokens, num_experts]（bool，TopK sparse）
)
# 输出 permuted_input: [num_experts * max_tokens_per_expert, hidden_size]

# Expert 计算...
expert_output = grouped_linear(permuted_input, expert_weights)

# Unpermute：恢复 token 原始顺序，合并 topk 输出
output = moe_unpermute(
    expert_output,    # [num_experts * max_tokens_per_expert, hidden_size]
    row_id_map,
    probs=router_probs,  # TopK 权重（用于加权合并）
)
```

### 8.2 带概率权重的 Permute

```python
# 带 TopK 概率的 permute（更高效的融合版本）
permuted_input, row_id_map, permuted_probs = moe_permute_with_probs(
    input_tensor,
    routing_map,
    topk_weights,   # [num_tokens, topk]
)
```

### 8.3 Padding Permute（确保对齐）

某些 FP8 GEMM 要求 M 维度对齐（如 128 的倍数）：

```python
# 带 padding 的 permute，确保每个 expert 的 token 数是 padding_size 的倍数
permuted_input, row_id_map = moe_permute_and_pad_with_probs(
    input_tensor,
    routing_map,
    topk_weights,
    padding_size=128,  # 对齐到 128
)
```

---

## 9. Fused Router

`fused_router` 模块将 Router 的 softmax → TopK → Permute 融合为单个 CUDA kernel：

```python
from transformer_engine.pytorch.router import fused_router_topk_softmax

# 融合：softmax + TopK 选择
router_probs, routing_map = fused_router_topk_softmax(
    logits,        # [num_tokens, num_experts]
    top_k=2,
    capacity_factor=1.5,   # 每个 expert 的容量上限
)

# fused_router.cu 实现：
# 1. softmax(logits) → probs
# 2. TopK(probs) → 选出每个 token 的 top-k expert
# 3. 生成 routing_map（bool matrix）
# 单 kernel 完成，避免中间 Tensor 写到 global memory
```

---

## 10. Expert Parallelism（EP）

当 expert 数量超过单 GPU 容量，需要 Expert 并行：

```python
# EP=4：4 GPU，每个 GPU 持有 num_experts/4 个 expert
# 路由时需要 AllToAll 通信：将 token 发送到正确 GPU

from transformer_engine.pytorch.permutation import moe_sort_chunks_by_index

# 准备跨卡 AllToAll 通信
sorted_input, sorted_indices = moe_sort_chunks_by_index(
    permuted_input,
    split_sizes,            # 每个目标 GPU 要发送的 token 数
    split_size=chunk_size,
)

# AllToAll：发送/接收 token
# 每个 GPU 将自己负责的 token 发送给对应 expert 所在的 GPU
dist.all_to_all(recv_buffer, sorted_input, group=ep_group)

# 在本 GPU 的 expert 上计算
local_output = local_experts(recv_buffer)

# AllToAll：将结果发回原始 GPU
dist.all_to_all(result_buffer, local_output, group=ep_group)
```

---

## 11. GroupedLinear：MoE GEMM

```python
import transformer_engine.pytorch as te

# num_gemms = expert 数量
grouped_linear = te.GroupedLinear(
    num_gemms=8,      # 8 个 expert
    in_features=4096,
    out_features=4096,
)

# 前向
output = grouped_linear(
    inp,             # [total_tokens, 4096]（排列后的连续 token）
    m_splits=[m0, m1, m2, ...],  # 每个 expert 的 token 数
)
```

内部调用 `cublaslt_grouped_gemm`：

```cpp
// 8 个 GEMM 描述符，一次 kernel 启动处理所有 expert
nvte_grouped_gemm(
    {A_0, A_1, ..., A_7},  // 各 expert 的输入
    {B_0, B_1, ..., B_7},  // 各 expert 的权重（FP8）
    {D_0, D_1, ..., D_7},  // 各 expert 的输出
    8,
    ...
);
```

性能优势（与循环调用 8 次独立 GEMM 比较）：
- 单次 kernel 启动开销（避免 8× launch overhead）
- GPU 调度更高效（全局 SM 分配优化）
- 典型加速：1.3-2× 相对于循环 GEMM

---

## 12. NVSHMEM（多节点 MoE）

对于超大规模 MoE（如数千个 expert，跨机器 EP），TE 提供了 NVSHMEM（NVIDIA Symmetric Memory）支持：

```
nvshmem_api/    ← NVSHMEM 封装
```

NVSHMEM 是一种 GPU 发起的通信模型，允许 GPU kernel 内直接通过 `nvshmem_put/get` 读写远程 GPU 的内存，无需通过 CPU。

```cpp
// 在 MoE permute kernel 内部，直接写到目标 GPU 的 expert buffer
nvshmem_float_put_nbi(
    remote_expert_buffer + offset,
    local_token_data,
    token_size,
    target_pe  // 目标 GPU 的 PE（Processing Element）编号
);
```

相比 NCCL AllToAll，NVSHMEM 的优势：
- 细粒度通信（token 级别，而非 bulk）
- 无 CPU 介入（完全 GPU 驱动）
- 更低延迟（适合小 batch/long sequence 场景）

---

## 13. Checkpoint 与 CPU Offload（MoE）

MoE 中 expert 数量多，显存压力大。TE 支持 MoE 专属优化：

```python
# Expert 参数 CPU offload（推理时按需加载）
from transformer_engine.pytorch.cpu_offload import (
    get_cpu_offload_context,
    mark_not_offload,
)

with get_cpu_offload_context(
    enabled=True,
    num_layers=num_experts,   # 按 expert 粒度 offload
    offload_activations=True,
):
    for expert_idx, expert in enumerate(experts):
        # expert 权重在使用时从 CPU 加载
        # 激活在前向后卸载到 CPU
        output_i = expert(token_i)
```
