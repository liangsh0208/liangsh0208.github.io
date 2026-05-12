# Normalization 归一化

> 源码路径：`transformer_engine/common/normalization/`  
> `transformer_engine/pytorch/module/layernorm.py`  
> `transformer_engine/pytorch/module/rmsnorm.py`  
> `transformer_engine/pytorch/ops/basic/layer_norm.py`

---

## 1. 支持的归一化类型

TE 实现了两种归一化：

| 类型 | 公式 | 参数 | 适用模型 |
|------|------|------|---------|
| LayerNorm | `y = (x - μ) / √(σ² + ε) * γ + β` | γ, β（可学习） | BERT、GPT-3 等 |
| RMSNorm | `y = x / RMS(x) * γ`，`RMS(x) = √(mean(x²) + ε)` | γ（可学习） | LLaMA、Mistral 等 |

RMSNorm 无减均值步骤，计算量更低（约省 25%），且在大模型上效果相当，因此现代 LLM 多采用 RMSNorm。

---

## 2. 为什么专门实现

PyTorch 内置的 `nn.LayerNorm` 和 `F.layer_norm` 存在的问题：

1. **无法融合 FP8 cast**：TE 的 LayerNorm 在完成归一化后，可以直接将结果量化为 FP8 写入 GEMM 输入缓冲，避免一次额外的 global memory 往返
2. **反向效率差**：PyTorch 的自动微分生成多个算子，TE 用手写 CUDA kernel 将前向和反向都融合为单次 kernel
3. **序列并行支持**：TE 的归一化支持在 sequence 维度上分布式统计（allreduce mean/variance）

---

## 3. CUDA Kernel 架构

### 3.1 线程块组织

以 LayerNorm 前向为例（`ln_fwd_kernels.cuh`）：

```
输入 shape: [B*S, H]  （B=batch, S=seq_len, H=hidden_size）

线程块组织（通过 Ktraits 模板参数化）：
  THREADS_PER_CTA  = 128 或 256（每个 CTA 的线程数）
  ROWS_PER_CTA     = 1 或 2（每个 CTA 处理的行数）
  WARPS_M          = 行方向的 warp 数
  WARPS_N          = 列方向的 warp 数
  THREADS_PER_ROW  = H / VEC_COLS_PER_LDG（每行的线程数）
```

```cpp
template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA)
void ln_fwd_tuned_kernel(ForwardKernelParams params) {
    // 计算当前线程处理的行 r 和列 c
    const index_t r = bidm * ROWS_PER_CTA + warp_m;
    const index_t c = bidn * THREADS_PER_ROW + warp_n * THREADS_PER_WARP + lane;
    
    // 每个 warp 独立统计局部 mean 和 var
    Stats stats(params, bidm, bidn, warp_m, warp_n, lane, smem_);
    ...
}
```

### 3.2 Welford 在线算法

TE 使用 Welford 算法计算均值和方差，避免两遍扫描：

```cpp
// 第一遍（加载数据）：计算 mean 和 variance
for (int it = 0; it < LDGS; it++) {
    // 向量化加载（Ivec = float4 或 bf16x8）
    Ivec in;
    in.load_from(input_ptr + c + it * THREADS_PER_ROW);
    
    // Welford 在线更新
    for (int jt = 0; jt < NUM_ELTS; jt++) {
        float val = in.data[jt];
        stats.update(val);  // 更新 count, mean, M2
    }
}

// warp 内 reduce
auto [mean, var] = stats.finalize();  // warp shuffle reduce
float rs = rsqrtf(var + epsilon);     // reciprocal sqrt
```

### 3.3 融合 FP8 Cast

最关键的优化：在计算 `y = (x - mean) * rs * gamma + beta` 的同时，直接量化为 FP8：

```cpp
// 第二遍（写入输出）
for (int it = 0; it < LDGS; it++) {
    Ovec out;
    for (int jt = 0; jt < NUM_ELTS; jt++) {
        float val = (in.data[jt] - mean) * rs;
        float y   = val * gamma[c + ...] + beta[c + ...];
        
        if constexpr (is_fp8_output) {
            // 融合量化：y * scale，然后 cast 到 FP8
            out.data[jt] = (output_t)(y * scale);
            // 原子更新 amax
            atomicMaxFloat(amax_ptr, fabsf(y));
        } else {
            out.data[jt] = (output_t)y;
        }
    }
    out.store_to(output_ptr + c + it * THREADS_PER_ROW);
}
```

这样 LayerNorm → FP8 Cast 只需要**一次** global memory 写，而不是 LayerNorm 写 BF16 → 再读 BF16 → Cast 写 FP8 的**三次**。

---

## 4. 反向传播 Kernel

### 4.1 LayerNorm 反向

```
给定：dL/dy（输出梯度），x（前向输入），γ（scale），μ（均值），σ²（方差）

需要计算：
  dL/dx = (dL/dy * γ - mean(dL/dy * γ) - x̂ * mean(dL/dy * γ * x̂)) * rs
  dL/dγ = sum(dL/dy * x̂)    （沿 batch 维度求和）
  dL/dβ = sum(dL/dy)         （沿 batch 维度求和）
  
其中 x̂ = (x - μ) * rs（归一化后的 x），rs = 1/√(σ² + ε)
```

```
源文件结构：
  ln_bwd_semi_cuda_kernel.cu  → "半精度"反向（混合 FP32/BF16）
  ln_bwd_kernels.cuh          → 具体 kernel 实现
```

TE 的反向 kernel 特点：
- γ 和 β 梯度（wgrad）通过跨行的 parallel reduction 计算
- 使用 shared memory 缓冲跨 warp 的 partial sum
- 反向输出（dL/dx）可以选择直接量化为 FP8

### 4.2 RMSNorm 反向

RMSNorm 无均值减法，反向简化：

```
dL/dx = (dL/dy * γ - x * mean(dL/dy * γ * x) / RMS²) / RMS
dL/dγ = sum(dL/dy * x / RMS)
```

省去了均值相关的计算项，速度略快于 LayerNorm 反向。

---

## 5. Kernel 选择逻辑

TE 根据 hidden_size 选择不同的 kernel 配置：

```cpp
// common/normalization/common.cpp
NormFwdTuningKey get_tuning_key(
    int hidden_size, DType itype, DType otype, bool zero_centered_gamma
) {
    // hidden_size 决定 THREADS_PER_ROW、LDGS 等参数
    // 目标：每个 thread 负载均衡，最大化 warp 利用率
    ...
}
```

典型配置（以 H=4096 为例）：

```
THREADS_PER_CTA = 256
ROWS_PER_CTA    = 1
THREADS_PER_ROW = 256
VEC_COLS_PER_LDG = 4（每次加载 float4 = 4×BF16）
LDGS             = 4096 / 256 / 4 = 4 次加载
```

---

## 6. PyTorch 模块接口

### 6.1 `te.LayerNorm` 和 `te.RMSNorm`

```python
import transformer_engine.pytorch as te

# LayerNorm（支持 zero_centered_gamma）
ln = te.LayerNorm(
    hidden_size=4096,
    eps=1e-5,
    zero_centered_gamma=False,  # 若 True，gamma 初始化为 0，实际 scale = 1 + gamma
    params_dtype=torch.bfloat16,
)

# RMSNorm
rms = te.RMSNorm(
    hidden_size=4096,
    eps=1e-6,
)

# 前向
x = torch.randn(32, 512, 4096, device="cuda", dtype=torch.bfloat16)
y = ln(x)   # shape: [32, 512, 4096]
```

### 6.2 `te.LayerNormLinear`（融合模块）

将 LayerNorm 和 Linear 合并为一个模块，用同一个 autograd Function：

```python
# 等价于：y = Linear(LayerNorm(x))
# 但 LayerNorm 的输出直接量化为 FP8 送入 Linear，无中间 BF16 存储
model = te.LayerNormLinear(
    in_features=4096,
    out_features=4096,
    eps=1e-5,
    bias=True,
    return_layernorm_output=False,  # 是否额外返回 LN 的输出（residual 连接需要）
)

# FP8 时：
# x(BF16) → LN kernel（含融合 FP8 cast）→ x_fp8 → GEMM(FP8) → y(BF16)
with te.fp8_autocast(enabled=True, ...):
    y = model(x)
```

### 6.3 zero_centered_gamma

一些大模型训练技巧中，将 γ 初始化为 0（而非 1），实际 scale = `1 + γ`：

```python
# 标准 LayerNorm：y = x_normalized * γ
# zero_centered_gamma：y = x_normalized * (1 + γ)
# 初始时 γ=0，等价于 y = x_normalized，梯度流更好
ln = te.LayerNorm(hidden_size, zero_centered_gamma=True)
# 内部：nn.Parameter 初始化为 0，前向计算为 gamma + 1
```

---

## 7. ops 系统中的 LayerNorm

在底层 ops API 中，LayerNorm 作为 `BasicOperation`：

```python
from transformer_engine.pytorch.ops.basic import LayerNorm as LayerNormOp

# 可与其他 ops 组合：
pipeline = te.ops.Sequential([
    LayerNormOp(hidden_size=4096),
    te.ops.Linear(4096, 16384),  # 等价于 LayerNormLinear
])
```

`OperationFuser` 会自动识别 `LayerNorm + Linear` 的组合，将其融合为 `LayerNormLinear` 的 fused op，从而启用融合 FP8 cast。

---

## 8. 分布式 LayerNorm（序列并行）

在序列并行（Sequence Parallelism）场景下，Tensor 的 sequence 维度被切分到多个 GPU 上：

```
GPU 0 处理 token [0, S/2)
GPU 1 处理 token [S/2, S)

LayerNorm 需要在整个 sequence 上计算 mean/var
→ 每个 GPU 先计算局部统计量，然后 AllReduce 汇总
```

TE 的 LayerNorm 原生支持这一模式，只需传入通信组：

```python
# LayerNormLinear 中的 SP 支持
output = model(x, 
    tensor_parallel_group=tp_group,
    sequence_parallel=True   # 输入 x 是序列并行分片的
)
# 内部自动处理 AllGather（在 LayerNorm 前）和 ReduceScatter（在 Linear 后）
```
