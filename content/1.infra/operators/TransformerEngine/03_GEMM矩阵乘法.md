# GEMM 矩阵乘法

> 源码路径：`transformer_engine/common/gemm/`  
> `transformer_engine/common/include/transformer_engine/gemm.h`  
> `transformer_engine/pytorch/cpp_extensions/`

---

## 1. 概述

GEMM（General Matrix Multiplication）是 Transformer 中计算量最大的算子，约占整体训练时间的 70%~80%。TE 的 GEMM 实现基于：

- **cuBLASLt**：NVIDIA 专为 Tensor Core 优化的矩阵乘法库，支持 FP8 输入
- **CUTLASS**：CUDA C++ 模板库，用于实现分组 GEMM（MoE 场景）
- **split-k 累加器**：解决 FP8 GEMM 的精度问题

---

## 2. GEMM 在 Transformer 中的位置

一个标准 Linear 层的完整训练过程（forward + backward）涉及 3 次 GEMM：

```
前向（fprop）：  Y     = X   @ W^T    [M×K] @ [K×N] → [M×N]
反向输入梯度（dgrad）：dX = dY  @ W     [M×N] @ [N×K] → [M×K]
反向权重梯度（wgrad）：dW = dY^T @ X    [N×M] @ [M×K] → [N×K]
```

FP8 训练时：
- fprop：X(FP8) @ W^T(FP8) → Y(BF16 或 FP8)
- dgrad：dY(FP8) @ W(FP8) → dX(BF16)
- wgrad：dY^T(FP8) @ X(FP8) → dW(FP32)

---

## 3. cuBLASLt GEMM 实现

### 3.1 `cublaslt_gemm.cu`

这是 TE 单个 GEMM 的核心实现。接口（C++ 层）：

```cpp
// transformer_engine/common/include/transformer_engine/gemm.h
void nvte_cublas_gemm(
    const NVTETensor A,          // 矩阵 A（可以是 FP8）
    const NVTETensor B,          // 矩阵 B（可以是 FP8）
    NVTETensor D,                // 输出矩阵
    const NVTETensor bias,       // 可选 bias
    NVTETensor pre_gelu_out,     // 可选：GELU 融合输出
    bool transa,                 // 是否转置 A
    bool transb,                 // 是否转置 B
    bool grad,                   // 是否是梯度计算
    NVTETensor workspace,        // cuBLASLt 工作空间
    bool accumulate,             // 是否累加到 D（而非覆写）
    bool use_split_accumulator,  // 是否使用分割累加器
    int math_sm_count            // 使用的 SM 数量
);
```

### 3.2 Python 接口

```python
# transformer_engine/pytorch/cpp_extensions/__init__.py
from transformer_engine_torch import (
    general_gemm,    # 通用 GEMM（支持 FP8/BF16/FP16）
    fp8_gemm,        # FP8 专用 GEMM
)
```

在 `module/linear.py` 中调用：

```python
# 前向 GEMM（fprop）
out, *_ = general_gemm(
    weight,          # 量化后的权重（Float8Tensor）
    inp,             # 量化后的输入（Float8Tensor）
    activation_dtype,
    get_workspace(),
    bias=bias if use_bias else None,
    gelu=bool(activation == "gelu"),
    use_split_accumulator=recipe.fp8_gemm_fprop.use_split_accumulator,
)
```

### 3.3 Split Accumulator（分割累加器）

FP8 GEMM 的核心数值问题：

```
普通 GEMM：result = sum(A[i] * B[i])  for i in 0..K
FP8 GEMM 中，K 次累加时若直接用 FP16 中间态，误差累积严重
```

**Split Accumulator** 解决方案（cublasLtMatmul FAST_ACCUM 选项）：

```
将 K 维切成多段（split），每段内用 FP32 累加
各段结果用 FP16 汇总
最终精度：介于全 FP32 累加和全 FP16 累加之间
```

```python
# 不同场景的 split_accumulator 策略
class DelayedScaling:
    # 所有 GEMM 不强制指定（由 use_split_accumulator 默认控制）
    
class Float8CurrentScaling:
    fp8_gemm_fprop = MMParams(use_split_accumulator=False)  # fprop 不需要
    fp8_gemm_dgrad = MMParams(use_split_accumulator=True)   # dgrad 需要
    fp8_gemm_wgrad = MMParams(use_split_accumulator=True)   # wgrad 需要

class Float8BlockScaling:
    fp8_gemm_fprop = MMParams(use_split_accumulator=True)   # 块级必须
    fp8_gemm_dgrad = MMParams(use_split_accumulator=True)
    fp8_gemm_wgrad = MMParams(use_split_accumulator=True)
```

### 3.4 FP8 GEMM 的 Scale 处理

cuBLASLt 的 FP8 GEMM 接受两个额外参数：
- `alpha_scale`（等于 `scale_A * scale_B`）：将 FP8 乘积缩放回实际值
- `amax_D`（输出 Tensor 的 amax，用于下一步量化）

```
数学表达：
  D = (A_fp8 / scale_A^{-1}) @ (B_fp8 / scale_B^{-1}) * alpha_scale
  等价于：
  D = (A_fp8 @ B_fp8) * (scale_inv_A * scale_inv_B)
```

TE 在调用 cuBLASLt 时自动处理这些 scale 参数的传递。

---

## 4. 分组 GEMM（Grouped GEMM，MoE 场景）

### 4.1 为什么需要分组 GEMM

MoE（Mixture of Experts）中每个 expert 是独立的 Linear 层，但 batch 中不同 token 路由到不同 expert：

```
N 个 expert，每个 expert 处理 m_i 个 token
朴素实现：N 次独立 GEMM 调用 → GPU 利用率低
分组 GEMM：一次内核启动处理所有 expert → 吞吐大幅提升
```

### 4.2 cuBLASLt 分组 GEMM

```cpp
// common/gemm/cublaslt_grouped_gemm.cu
void nvte_grouped_gemm(
    const std::vector<NVTETensor>& A,    // 每个 expert 的输入矩阵
    const std::vector<NVTETensor>& B,    // 每个 expert 的权重矩阵
    std::vector<NVTETensor>& D,          // 每个 expert 的输出
    int gemm_count,                      // expert 数量
    ...
);
```

cuBLASLt grouped GEMM 接受一个 GEMM 描述符数组，GPU 内核一次启动完成所有 expert 的矩阵乘法。

### 4.3 CUTLASS 分组 GEMM

```cpp
// common/gemm/cutlass_grouped_gemm.cu
// 基于 CUTLASS 的分组 GEMM，支持 FP8
```

CUTLASS 实现提供更细粒度的控制（如 epilogue 融合），用于需要定制化 epilogue（后处理）的场景，例如量化输出、加 bias 等。

---

## 5. 融合 GEMM（Fused GEMM Epilogue）

TE 利用 cuBLASLt 的 epilogue（尾处理）机制，将 GEMM 后的操作融合进同一个内核：

| 融合操作 | cuBLASLt Epilogue 参数 |
|---------|----------------------|
| 加 Bias | `CUBLAS_EPILOGUE_BIAS` |
| GELU 激活 | `CUBLAS_EPILOGUE_GELU` |
| GELU + 保存 pre-gelu 值 | `CUBLAS_EPILOGUE_GELU_AUX` |
| 量化输出为 FP8 | `CUBLAS_EPILOGUE_DEFAULT` + amax 计算 |
| 加 Bias + ReLU | `CUBLAS_EPILOGUE_RELU_BIAS` |

```python
# LayerNormMLP 前向中的融合 GEMM
# FC1: Linear + GELU，一次内核完成
out_fc1, gelu_inp, *_ = general_gemm(
    weight1,  inp,
    activation_dtype,
    workspace,
    bias=bias1,
    gelu=True,                    # 融合 GELU epilogue
    gelu_input=gelu_inp_buffer,   # 保存 pre-GELU 值（反向需要）
)

# FC2: 普通 Linear
out_fc2, *_ = general_gemm(
    weight2, out_fc1,
    activation_dtype,
    workspace,
    bias=bias2,
)
```

---

## 6. 工作空间管理（Workspace）

cuBLASLt 需要额外的工作空间，TE 使用全局工作空间池：

```python
# module/base.py
def get_workspace():
    """获取全局 cuBLASLt 工作空间（懒初始化）"""
    global _cublas_workspace
    if _cublas_workspace is None:
        _cublas_workspace = torch.empty(
            33_554_432,  # 32 MB
            dtype=torch.int8,
            device="cuda"
        )
    return _cublas_workspace
```

32 MB 对绝大多数 GEMM 够用；当需要更大工作空间时，cuBLASLt 会自动降级到较小的算法。

---

## 7. 2X Accumulator 精度模式

TE 定义了三个环境变量控制精度：

```python
# module/base.py
_2X_ACC_FPROP = bool(int(os.getenv("NVTE_FP8_FPROP_WORKSPACE_OPT", "0")))
_2X_ACC_DGRAD = bool(int(os.getenv("NVTE_FP8_DGRAD_WORKSPACE_OPT", "1")))
_2X_ACC_WGRAD = bool(int(os.getenv("NVTE_FP8_WGRAD_WORKSPACE_OPT", "1")))
```

"2X accumulator" 指使用两倍精度（FP32）的中间累加器，以牺牲少量性能换取数值稳定性。默认 dgrad 和 wgrad 开启（梯度对精度更敏感），fprop 关闭。

---

## 8. 非 TN 布局支持

标准 cuBLASLt FP8 GEMM 要求输入为 TN 布局（A 转置，B 非转置）。TE 检测并处理非 TN 情况：

```python
def is_non_tn_fp8_gemm_supported():
    """检查 cuBLASLt 是否支持非 TN 的 FP8 GEMM（需要较新版本）"""
    return tex.get_cublasLt_version() >= 120500  # cuBLASLt >= 12.5.0
```

若不支持非 TN，TE 会在 GEMM 前通过转置操作将其转换为 TN 布局。

---

## 9. 性能对比

典型 H100 80GB GPU 上，`[4096, 4096] @ [4096, 4096]` 矩阵乘法：

| 精度 | TFLOPs/s | 相对 BF16 |
|------|---------|----------|
| FP32 | ~20 | 0.25× |
| BF16 | ~80 | 1× |
| FP8 E4M3 | ~160 | 2× |
| MXFP8 (B200) | ~300+ | 3.75×+ |
| NVFP4 (B200) | ~600+ | 7.5×+ |

FP8 GEMM 的加速来源：
1. Hopper H100 的 Tensor Core 对 FP8 的原生支持（MMA 指令）
2. FP8 数据量减半，内存带宽消耗减半
3. Split accumulator 避免了完整 FP32 中间态的开销
