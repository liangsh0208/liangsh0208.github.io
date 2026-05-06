# 算子图系统（ops）

> 源码路径：`transformer_engine/pytorch/ops/`

---

## 1. 设计动机

`module/` 层的 `LayerNormLinear`、`LayerNormMLP` 等融合模块是**硬编码的融合**——每种组合都需要手写一个 autograd Function。这种方式缺乏灵活性：

- 如果用户想要 `RMSNorm + Linear + SwiGLU + Linear`，就需要另写一个模块
- 张量并行、通信重叠等特性需要在每个模块中重复实现

`ops/` 系统提供了一套**可组合的算子图 API**：用户将基础算子（ops）串联成流水线，系统自动识别可融合的模式并替换为高效 fused ops。

---

## 2. 核心抽象

### 2.1 `BasicOperation`：原子算子

```python
class BasicOperation(FusibleOperation):
    """不可再分的基础操作单元
    
    每个 BasicOperation 都是独立可执行的，
    也可以被 OperationFuser 融合为 FusedOperation
    """
    
    @abc.abstractmethod
    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """前向计算"""
    
    @abc.abstractmethod
    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:
        """反向计算，返回 (grad_input, [grad_params...])"""
    
    @property
    def is_fused_op(self) -> bool:
        return False
```

### 2.2 `FusedOperation`：融合算子

```python
class FusedOperation(FusibleOperation):
    """多个 BasicOperation 融合后的高效实现
    
    取代一组 BasicOperation，实现更低内存访问和更高计算密度
    """
    
    @property
    def is_fused_op(self) -> bool:
        return True
    
    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[...],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict],
    ) -> tuple[torch.Tensor, ...]:
        """接管对应 BasicOp 的前向，执行融合实现"""
```

### 2.3 `OperationContext`：前向/反向通信桥

```python
@dataclasses.dataclass
class OperationContext:
    """在前向和反向之间传递状态"""
    
    # 前向中保存的 Tensor（autograd 自动管理生命周期）
    saved_tensors: Optional[tuple[Optional[torch.Tensor], ...]] = None
    to_save: Optional[tuple[Optional[torch.Tensor], ...]] = None
    
    # 在 pipeline 的保存 Tensor 列表中的位置范围
    _saved_tensors_range: Optional[tuple[int, int]] = None
    
    requires_grad: bool = True
    
    def save_for_backward(self, *tensors):
        self.to_save = tensors
```

每个 BasicOperation 有自己的 OperationContext，但所有 Tensor 统一由 `_OperationFuserAutogradFunction` 通过 `save_for_backward` 管理，确保 autograd 能正确释放。

---

## 3. 内置 BasicOperation 列表

### 3.1 计算类

| Op | 文件 | 功能 |
|----|------|------|
| `BasicLinear` | `basic/basic_linear.py` | `y = x @ W^T`（无 bias） |
| `Bias` | `basic/bias.py` | `y = x + b` |
| `LayerNorm` | `basic/layer_norm.py` | 层归一化 |
| `RMSNorm` | `basic/rmsnorm.py` | RMS 归一化 |
| `Activation` | `basic/activation.py` | GELU/SiLU/ReLU 等 |
| `SwiGLU` | `basic/swiglu.py` | `swish(gate) * content` |
| `Dropout` | `basic/dropout.py` | Dropout |
| `Quantize` | `basic/quantize.py` | 显式量化到 FP8 |
| `L2Normalization` | `basic/l2normalization.py` | QK Norm |

### 3.2 分布式类

| Op | 文件 | 功能 |
|----|------|------|
| `AllGather` | `basic/all_gather.py` | AllGather（序列并行） |
| `AllReduce` | `basic/all_reduce.py` | AllReduce（行并行） |
| `ReduceScatter` | `basic/reduce_scatter.py` | ReduceScatter（序列并行） |
| `GroupedLinear` | `basic/grouped_linear.py` | MoE 分组 GEMM |

### 3.3 图结构类

| Op | 文件 | 功能 |
|----|------|------|
| `AddExtraInput` | `basic/add_extra_input.py` | 注入额外输入（residual 连接） |
| `MakeExtraOutput` | `basic/make_extra_output.py` | 暴露中间结果为额外输出 |
| `Identity` | `basic/identity.py` | 无操作（占位符） |
| `Reshape` | `basic/reshape.py` | 形状变换 |

---

## 4. `SequentialPipeline`：流水线容器

```python
from transformer_engine.pytorch.ops import Sequential

# 构建自定义 MLP
mlp = te.ops.Sequential([
    te.ops.LayerNorm(hidden_size=4096),
    te.ops.Linear(4096, 16384),     # 等价于 BasicLinear + Bias
    te.ops.Activation("swiglu"),
    te.ops.Linear(8192, 4096),      # SwiGLU 输出维度减半
])

# 前向
with te.fp8_autocast(enabled=True, ...):
    out = mlp(x)
```

`Sequential` 内部持有一个 `OperationFuser`，在第一次前向前自动完成融合分析。

---

## 5. `OperationFuser`：自动融合引擎

### 5.1 融合规则发现

```python
class OperationFuser:
    """将 BasicOp 流水线重写为 FusedOp 流水线"""
    
    def __init__(self, ops: list[BasicOperation], ...):
        self._ops = ops
        self._fused_ops = None  # 懒加载
    
    def _fuse_ops(self) -> list[FusibleOperation]:
        """分析 ops 序列，应用所有可用的融合规则"""
        ops = list(self._ops)
        
        # 遍历所有已注册的融合规则（按优先级）
        for fusion_function in _FUSION_FUNCTIONS:
            ops = fusion_function(ops)
        
        return ops
```

### 5.2 内置融合规则

所有融合规则定义在 `ops/fused/` 中，通过 pattern matching 识别：

**规则 1：`ForwardLinearBiasActivation`**
```
BasicLinear + Bias + Activation("gelu"/"relu")
→ 合并为单次 GEMM + epilogue 融合激活
```

**规则 2：`ForwardLinearBiasAdd`**
```
BasicLinear + Bias + AddExtraInput
→ 合并为 GEMM + bias + residual add
（利用 cuBLASLt 的 D = alpha*A*B + beta*C 能力）
```

**规则 3：`BackwardLinearAdd`**
```
反向路径上：BasicLinear.backward + AllReduce
→ 合并为单次 GEMM + 异步 AllReduce 重叠
```

**规则 4：`UserbuffersForwardLinear`**
```
AllGather + BasicLinear 或 BasicLinear + ReduceScatter
→ 合并为 UserBuffers 通信-GEMM 重叠
（需要 UserBuffers 已初始化）
```

**规则 5：`ForwardGroupedMLP`**
```
多个 (BasicLinear + Activation) 串联（MoE）
→ 合并为 grouped GEMM 调用
```

### 5.3 Autograd Function 封装

```python
class _OperationFuserAutogradFunction(torch.autograd.Function):
    """整个流水线的 autograd 函数
    
    关键设计：前向和反向可以使用不同的融合方案
    （某些融合只在前向有效，反向需要不同的处理）
    """
    
    @staticmethod
    def forward(func_ctx, input_, fuser, basic_op_kwargs, *params_and_extra_inputs):
        # 执行融合后的前向流水线
        output = fuser._forward(input_, ...)
        
        # 统一保存所有 BasicOp 的 saved_tensors
        saved_tensors = []
        for op_ctx in basic_op_ctxs:
            saved_tensors.extend(op_ctx.to_save or [])
        func_ctx.save_for_backward(*saved_tensors)
        
        return output
    
    @staticmethod  
    def backward(func_ctx, *grad_outputs):
        # 恢复各 op 的 saved_tensors
        all_saved = func_ctx.saved_tensors
        # 分配给各 op 的 OperationContext
        
        # 执行融合后的反向流水线（从后往前）
        grad_input = fuser._backward(grad_outputs, ...)
        
        return (grad_input, None, None, *grad_params)
```

---

## 6. Quantizer 传递机制

ops 系统中，相邻算子之间通过 `Quantizer` 协商数据格式：

```
Op A 的 output_quantizer ──→ Op B 的 input_quantizer
```

若 A 的输出 Quantizer 和 B 的输入 Quantizer 兼容（都是 FP8），则 A 的输出可直接以 FP8 格式传递给 B，无需先反量化为 BF16：

```python
def fuser_forward(
    self,
    basic_op_ctxs,
    input_,
    *,
    prev_op_grad_output_quantizer: Optional[Quantizer],  # 上游期望的梯度格式
    next_op_input_quantizer: Optional[Quantizer],        # 下游期望的输入格式
    ...
):
    # 如果下游需要 FP8 输入，则本 op 的输出直接量化为 FP8
    if next_op_input_quantizer is not None:
        output = quantize(output, next_op_input_quantizer)
    return output
```

这是 `fp8_mha=True` 等特性的基础：相邻 op 协商后，中间 Tensor 保持 FP8 格式，消除不必要的精度转换。

---

## 7. CPU Offloading 支持

ops 系统内置 CPU offload 钩子：

```python
from transformer_engine.pytorch.ops.basic.basic_linear import BasicLinear

# 训练时：激活值在前向后卸载到 CPU，反向前从 CPU 载入
# 需要与 cpu_offload context 配合使用
from transformer_engine.pytorch.cpu_offload import get_cpu_offload_context

with get_cpu_offload_context(enabled=True, num_layers=4):
    for layer in transformer_layers:
        output = layer(hidden_states, ...)
```

在 BasicLinear 的前向中：

```python
# 前向保存激活到 CPU（若开启 offload）
if is_cpu_offload_enabled():
    mark_activation_offload(inp)  # 异步 D2H 搬运
```

---

## 8. 完整示例：自定义 LLaMA FFN

```python
import transformer_engine.pytorch as te
from transformer_engine.pytorch import ops as te_ops

class LlamaFFN(te.ops.Sequential):
    """LLaMA-style FFN with SwiGLU"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__([
            te_ops.RMSNorm(hidden_size),
            te_ops.Linear(hidden_size, intermediate_size * 2),  # gate + up proj
            te_ops.SwiGLU(),                                      # swish(gate)*up
            te_ops.Linear(intermediate_size, hidden_size),        # down proj
        ])
        # 融合器自动识别：
        # RMSNorm + Linear → LayerNormLinear（融合 norm+GEMM）
        # Linear + SwiGLU → ForwardLinearBiasActivation（融合 GEMM+激活）

ffn = LlamaFFN(hidden_size=4096, intermediate_size=11008)

fp8_recipe = te.recipe.DelayedScaling()
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    output = ffn(hidden_states)
# 自动优化：4 个逻辑算子 → 2 次 kernel 调用（融合后）
```

---

## 9. ops vs module 的选择

| 场景 | 推荐 | 理由 |
|------|------|------|
| 标准 Transformer 架构 | `module/` 层 | 更简洁，直接用 `TransformerLayer` |
| 自定义架构（非标准融合） | `ops/` 层 | 灵活组合，自动融合 |
| 研究新的融合模式 | `ops/` + 自定义 `FusedOperation` | 可插件式注册融合规则 |
| 需要 UserBuffers 通信重叠 | `ops/` + `UserbuffersForwardLinear` | 自动处理重叠逻辑 |
