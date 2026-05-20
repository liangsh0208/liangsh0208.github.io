---
created: 2026-05-06
---

# 低精度 Tensor 类型

> 源码路径：`transformer_engine/pytorch/tensor/`  
> `transformer_engine/pytorch/quantized_tensor.py`

---

## 1. 设计动机

标准 PyTorch `torch.Tensor` 不携带量化元数据（scale、amax）。TE 需要在 Tensor 本身上附着这些信息，以便：
1. 在 GEMM 前将高精度值量化为 FP8/FP4
2. 在 GEMM 后将结果反量化回高精度
3. 通过 PyTorch autograd 正确传递梯度

TE 的解决方案是定义 `QuantizedTensor`（继承自 `torch.Tensor`）作为基类，再派生出各低精度类型。

---

## 2. 继承体系

```
torch.Tensor
    └── QuantizedTensor          # 量化 Tensor 基类（quantized_tensor.py）
            ├── Float8Tensor         # FP8 per-tensor scale
            ├── Float8BlockwiseTensor # FP8 块级 scale
            ├── MXFloat8Tensor       # MXFP8 (Blackwell)
            ├── NVFloat4Tensor       # NVFP4 (Blackwell)
            └── GroupedTensor        # MoE 分组 Tensor
```

每种 Tensor 对应一个 `Quantizer`（量化器）和一个 `TensorStorage`（存储容器）：

```
Quantizer        ── 负责"如何量化"（计算 scale、执行 cast）
    Float8Quantizer
    Float8CurrentScalingQuantizer
    Float8BlockQuantizer
    MXFP8Quantizer
    NVFP4Quantizer

TensorStorage    ── 负责"存什么"（实际 FP8/FP4 数据 + scale 数据）
    Float8TensorStorage
    Float8BlockwiseQTensorStorage
    MXFP8TensorStorage
    NVFP4TensorStorage
```

---

## 3. QuantizedTensor 基类

```python
class QuantizedTensor(torch.Tensor):
    """量化 Tensor 的基类
    
    持有低精度数据和反量化所需的元数据。
    支持 __torch_dispatch__ 以拦截 PyTorch 算子并做透明反量化。
    """
    
    @classmethod
    def make_like(cls, tensor, *, data, ...):
        """创建同类型的新 Tensor（用于 autograd 反向传播）"""
    
    def dequantize(self, dtype=torch.float32):
        """反量化为高精度 Tensor"""
    
    def quantize_(self, tensor):
        """原地量化（更新内部 FP8 数据和 scale）"""
```

### __torch_dispatch__ 透明反量化

TE 注册了大量 aten 算子的 dispatch，使 `Float8Tensor` 在参与普通 PyTorch 算子时自动反量化：

```python
# 例如 Float8Tensor 参与 add 运算时
x_fp8 + y_bf16
# 等价于
x_fp8.dequantize(dtype=torch.bfloat16) + y_bf16
```

这保证了 FP8 Tensor 可以透明地融入现有的 PyTorch 代码。

---

## 4. Float8Tensor（FP8 per-tensor）

### 4.1 存储结构

```python
class Float8TensorStorage:
    rowwise_data:    torch.Tensor  # shape=[M,K], dtype=float8_e4m3/e5m2
    columnwise_data: torch.Tensor  # 转置后的 FP8 数据（可选）
    rowwise_scale_inv:    torch.Tensor  # shape=[1], FP32
    columnwise_scale_inv: torch.Tensor  # shape=[1], FP32
```

同时保存行优先和列优先两份数据的原因：GEMM 计算需要不同布局：
- 前向 `Y = X * W^T`：X 按行访问，W 按列（即转置后按行）访问
- 反向 `dX = dY * W`：dY 按行，W 按行访问
- 反向 `dW = dY^T * X`：需要 dY 和 X 的转置

如果只存一份，每次 GEMM 前都要现场转置，增加带宽开销。

### 4.2 量化过程（Float8Quantizer）

```python
class Float8Quantizer(Quantizer):
    scale: torch.Tensor      # 当前 scale（标量 FP32）
    amax:  torch.Tensor      # 记录当前步 amax
    dtype: TE_DType           # E4M3 或 E5M2
    
    def make_empty(self, shape, device):
        """分配 FP8 存储空间（含 rowwise + columnwise 两份）"""
        storage = Float8TensorStorage(...)
        return Float8Tensor(storage)
    
    def update_quantized(self, src: torch.Tensor, dst: Float8Tensor):
        """执行量化：FP32/BF16 → FP8
        
        内部调用 CUDA kernel: tex.quantize(src, scale, dst, amax)
        同时更新 amax（作为副作用）
        """
```

**CUDA kernel 实现**（`common/cast/fp8/`）：

```cuda
// 核心量化操作（伪代码）
__global__ void cast_to_fp8_kernel(
    const float* input,     // 高精度输入
    float scale,            // scaling factor
    __nv_fp8_e4m3* output,  // FP8 输出
    float* amax_out         // 记录本批次 amax
) {
    float val = input[idx] * scale;
    // 原子更新 amax
    atomicMax(amax_out, fabsf(input[idx]));
    // 量化：截断到 FP8 可表示范围
    output[idx] = (__nv_fp8_e4m3)clamp(val, -FP8_MAX, FP8_MAX);
}
```

### 4.3 FSDP2 支持

Float8Tensor 专门支持 PyTorch FSDP2（全分片数据并行），通过注册特定 aten 算子：

```python
_ops_to_preserve_subclass_in_fsdp2 = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    # ... 其他 aten ops
}
```

这些算子在执行时保留 Float8Tensor 子类，而非退化为普通 torch.Tensor，使 FSDP 的分片/聚合操作能正确处理 FP8 权重。

---

## 5. Float8BlockwiseTensor（块级 FP8）

### 5.1 存储结构

```python
class Float8BlockwiseQTensorStorage:
    rowwise_data:        torch.Tensor  # FP8 数据，shape=[M, K]
    columnwise_data:     torch.Tensor  # 转置版 FP8 数据
    rowwise_scale_inv:   torch.Tensor  # 块 scale，shape=[M, K//block_size]
    columnwise_scale_inv:torch.Tensor  # 列方向块 scale
```

每个 scale 覆盖 `block_size`（默认 128）个连续元素，scale 本身为 FP32（或可选约束为 2 的幂）。

### 5.2 与 per-tensor 的差异

| 特性 | Float8Tensor | Float8BlockwiseTensor |
|------|-------------|----------------------|
| scale 数量 | 每 Tensor 1 个 | 每块 1 个 |
| scale 精度 | FP32 | FP32 或 power-of-2 |
| 额外存储开销 | 可忽略 | ~1/128 的数据量 |
| 数值精度 | 全局范围适配 | 局部范围适配 |
| GEMM 要求 | split_accum 可选 | 必须 split_accum |

---

## 6. MXFloat8Tensor（MXFP8，Blackwell）

### 6.1 OCP MX 规格

MXFP8 是 Open Compute Project 的 Microscaling 规格：

```
每 32 个连续元素 → 共享 1 个 E8M0 scale（8-bit 纯指数）
E8M0: 取值范围 2^(-127) ~ 2^127
```

### 6.2 存储结构

```python
class MXFP8TensorStorage:
    rowwise_data:         torch.Tensor  # FP8 数据
    columnwise_data:      torch.Tensor  # 转置 FP8 数据
    rowwise_scale_inv:    torch.Tensor  # E8M0 scale，shape=[M, K//32]，dtype=uint8
    columnwise_scale_inv: torch.Tensor  # dtype=uint8
```

scale 用 uint8 存储（存 E8M0 的二进制表示），实际值为 `2^(scale_bits - 127)`。

### 6.3 量化器

```python
class MXFP8Quantizer(Quantizer):
    dtype: TE_DType     # E4M3
    block_scaling_dim: int = 0  # 0=行方向，1=列方向
    
    def make_empty(self, shape, device):
        """分配 MXFP8 存储：FP8 数据 + uint8 scale"""
```

硬件加速：Blackwell GPU 的 TMA（Tensor Memory Accelerator）原生支持 MX 格式的 load/store 和 GEMM，无需软件 dequantize。

---

## 7. NVFloat4Tensor（NVFP4，Blackwell）

### 7.1 E2M1 格式

```
E2M1: 2位指数 + 1位尾数 + 1位符号 = 4-bit
可表示值: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6（共16个）
最大值: 6
```

### 7.2 双层量化存储

```python
class NVFP4TensorStorage:
    rowwise_data:          torch.Tensor  # FP4 数据（每字节存2个FP4）
    columnwise_data:       torch.Tensor
    rowwise_scale_inv:     torch.Tensor  # Level-1 局部 scale，E4M3，每16元素1个
    columnwise_scale_inv:  torch.Tensor
    rowwise_global_scale:  torch.Tensor  # Level-2 全局 scale，FP32，每Tensor 1个
    columnwise_global_scale: torch.Tensor
```

### 7.3 随机 Hadamard 变换（RHT）

在量化前对 Tensor 施加 Hadamard 变换：

```
H = Hadamard(16×16)   # 随机版：H' = H * D，D 是随机对角矩阵（±1）
x_transformed = H' @ x_block   # 每块16个元素
```

**作用**：如果原始 x 中有异常值（outliers），变换后能量扩散到所有维度，使分布更均匀，更适合 FP4 的均匀量化网格。

**量化后的逆变换**通过 scale 隐式吸收，无需显式执行 IFHT。

---

## 8. GroupedTensor（MoE 分组）

```python
class GroupedTensor(QuantizedTensor):
    """将多个 expert 的 Tensor 按组打包存储
    
    shape: 逻辑 [total_tokens, hidden_size]
    内部: 按 expert 分组存储，每个 expert 的数据连续
    """
    
    offsets: torch.Tensor  # shape=[num_experts+1]，每个 expert 的起始位置
```

GroupedTensor 配合 `GroupedLinear` 模块使用，允许用单次 cuBLASLt grouped GEMM 调用处理所有 expert，而不是对每个 expert 分别调用 GEMM。

---

## 9. Quantizer：量化器接口

```python
class Quantizer(abc.ABC):
    rowwise_usage: bool     # 是否需要行优先版本
    columnwise_usage: bool  # 是否需要列优先版本
    internal: bool          # 是否为内部中间 Tensor（影响内存分配）
    
    @abc.abstractmethod
    def make_empty(self, shape, ...) -> QuantizedTensor:
        """分配空 QuantizedTensor（给定 shape）"""
    
    @abc.abstractmethod
    def update_quantized(self, src, dst, ...) -> QuantizedTensor:
        """将 src（高精度）量化写入 dst（QuantizedTensor）"""
    
    def quantize(self, tensor, ...) -> QuantizedTensor:
        """make_empty + update_quantized 的组合"""
        dst = self.make_empty(tensor.shape, ...)
        return self.update_quantized(tensor, dst)
```

---

## 10. 使用示例

### 直接使用（低层）

```python
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor import Float8Tensor
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

# 创建量化器
scale = torch.tensor(1.0, device="cuda")
amax  = torch.tensor(0.0, device="cuda")
quantizer = Float8Quantizer(scale, amax, tex.DType.kFloat8E4M3)

# 执行量化
x = torch.randn(1024, 512, device="cuda", dtype=torch.bfloat16)
x_fp8 = quantizer.quantize(x)  # → Float8Tensor

# 反量化
x_bf16 = x_fp8.dequantize(dtype=torch.bfloat16)

print(x_fp8._data.dtype)     # torch.float8_e4m3fn
print(x_fp8._scale_inv)      # FP32 scalar
```

### 通过高层模块（常用）

```python
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# TE 模块自动处理所有量化/反量化
model = te.Linear(512, 2048)
fp8_recipe = recipe.DelayedScaling()

with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    # 内部：x 被量化为 Float8Tensor，GEMM 完成后输出自动反量化
    out = model(x)
```
