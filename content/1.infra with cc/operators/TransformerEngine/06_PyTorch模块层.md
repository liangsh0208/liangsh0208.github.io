---
created: 2026-05-06
---

# PyTorch 模块层

> 源码路径：`transformer_engine/pytorch/module/`  
> `transformer_engine/pytorch/transformer.py`  
> `transformer_engine/pytorch/attention/`

---

## 1. 模块层级

TE 的 PyTorch 模块分为三层，用户通常用最高层：

```
TransformerLayer                ← 完整 encoder/decoder block
    │
    ├── MultiheadAttention      ← QKV投影 + 注意力 + O投影
    │       └── DotProductAttention
    │
    └── LayerNormMLP            ← LN + FC1 + 激活 + FC2
            ├── LayerNorm/RMSNorm
            └── Linear（×2）

独立可用的基础模块：
    Linear
    LayerNorm / RMSNorm
    LayerNormLinear             ← 融合 LN + Linear
    GroupedLinear               ← MoE 分组 GEMM
```

所有模块继承自 `TransformerEngineBaseModule`（`module/base.py`），后者统一处理：
- FP8 元数据（scale、amax history）的注册与更新
- FP8 参数初始化（`fp8_model_init`）
- UserBuffers 通信绑定
- FSDP 兼容性

---

## 2. `TransformerEngineBaseModule` 基类

```python
class TransformerEngineBaseModule(nn.Module):
    """所有 TE 模块的基类"""
    
    def __init__(self):
        # 注册到 FP8GlobalStateManager
        # 创建 RecipeState（存 scale/amax history）
        self._fp8_metas = {}  # key: tensor 角色（如 "input_fp8"）
    
    def set_fp8_weights(self):
        """将权重参数预量化为 FP8（fp8_model_init 使用）"""
    
    def get_fp8_meta(self, tensor_key: str):
        """获取指定 Tensor 的 scale/amax 元数据"""
    
    def forward(self, *args, **kwargs):
        """子类实现，内部通过 _get_fp8_params() 获取量化器"""
```

---

## 3. `te.Linear`

最基础的模块，替代 `torch.nn.Linear`，支持 FP8 计算。

### 3.1 接口

```python
class Linear(TransformerEngineBaseModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        params_dtype: torch.dtype = torch.bfloat16,
        # 张量并行
        parallel_mode: Optional[str] = None,  # "column", "row", None
        tensor_parallel_group: Optional[dist.ProcessGroup] = None,
        sequence_parallel: bool = False,
        # 初始化
        init_method: Optional[Callable] = None,
        # 精度控制
        return_bias: bool = False,  # 若 True 返回 (output, bias) 而非 output+bias
    )
```

### 3.2 前向计算（`_Linear.forward`）

```python
# 简化后的前向流程
@staticmethod
def forward(ctx, weight, weight_workspace, inp, bias, non_tensor_args):
    # 1. 解包参数
    fp8, input_quantizer, weight_quantizer, ... = non_tensor_args
    
    # 2. 量化输入（若 FP8 模式）
    if fp8:
        inp_fp8 = input_quantizer.quantize(inp)    # BF16 → FP8
        w_fp8   = weight_quantizer.quantize(weight) # BF16 → FP8
    else:
        inp_fp8, w_fp8 = inp, weight
    
    # 3. FP8 GEMM
    out, *_ = general_gemm(
        w_fp8, inp_fp8, 
        activation_dtype,
        get_workspace(),
        bias=bias,
        use_split_accumulator=...,
    )
    
    # 4. 保存反向所需 Tensor
    ctx.save_for_backward(inp_fp8, w_fp8, ...)
    
    return out
```

### 3.3 反向计算（`_Linear.backward`）

```python
@staticmethod
def backward(ctx, grad_output):
    inp_fp8, w_fp8 = ctx.saved_tensors
    
    # dgrad: 输入梯度 = grad_output @ weight
    if not weight_requires_grad or get_dummy_wgrad:
        # 用 fp8 weight
        dgrad = general_gemm(w_fp8, grad_output_fp8, ...)
    
    # wgrad: 权重梯度 = grad_output^T @ input
    if weight.requires_grad:
        wgrad = general_gemm(grad_output_fp8, inp_fp8, ...)
    
    # bias 梯度 = sum(grad_output, dim=0)
    if bias.requires_grad:
        dbias = grad_output.sum(0)
    
    return wgrad, None, dgrad, dbias, None
```

### 3.4 FP8 权重预量化

通过 `fp8_model_init`，权重在初始化时就以 FP8 格式存储（节省显存）：

```python
with te.fp8_model_init(enabled=True):
    model = te.Linear(4096, 4096)
    # model.weight 已经是 Float8Tensor，不是 BF16 Parameter

# 或者等价的 quantized_model_init
with te.quantized_model_init(enabled=True, recipe=recipe.DelayedScaling()):
    model = te.Linear(4096, 4096)
```

---

## 4. `te.LayerNormMLP`

最常用的复合模块，对应 Transformer FFN 块（带 Pre-LN）：

```
LayerNorm → FC1 → Activation → FC2
```

### 4.1 接口

```python
class LayerNormMLP(TransformerEngineBaseModule):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,      # 通常 4 * hidden_size
        eps: float = 1e-5,
        bias: bool = True,
        
        # 激活函数
        activation: str = "gelu",   # "gelu", "relu", "reglu", "geglu", "swiglu", "qgelu"
        
        # 归一化类型
        normalization: str = "LayerNorm",  # 或 "RMSNorm"
        
        # 张量并行
        parallel_mode: Optional[str] = None,  # "column" / "row"
        tp_group: Optional[dist.ProcessGroup] = None,
        sequence_parallel: bool = False,
        
        # 返回选项
        return_layernorm_output: bool = False,  # 是否额外返回 LN 输出
        return_bias: bool = False,
    )
```

### 4.2 GLU 变体（SwiGLU、GeGLU 等）

现代 LLM（LLaMA、Mistral）使用 SwiGLU 激活：

```python
# 标准 MLP：
# FC1: [H, 4H] → act → FC2: [4H, H]

# SwiGLU MLP：
# FC1: [H, 8H]（门控和内容各 4H）
# 激活：out = swish(gate) * content
# FC2: [4H, H]

mlp = te.LayerNormMLP(
    hidden_size=4096,
    ffn_hidden_size=4096 * 8 // 2,  # 注意：TE 用 ffn_hidden_size 表示 output 维度
    activation="swiglu",
)
```

内部 SwiGLU 实现（融合 kernel）：

```python
# jit.py
@jit_fuser
def swiglu(inp: torch.Tensor) -> torch.Tensor:
    """inp: [B, S, 2*ffn_H] → [B, S, ffn_H]"""
    x, gate = inp.chunk(2, dim=-1)
    return x * torch.nn.functional.silu(gate)
```

### 4.3 FP8 执行路径（fc1 融合）

```
# FP8 时的完整 LayerNormMLP 前向（简化）：

1. LayerNorm kernel（融合 FP8 cast）
   x_ln_fp8 = ln_cast_fp8(x)            ← 一次 kernel

2. FC1 GEMM + 激活融合
   fc1_out = fp8_gemm(x_ln_fp8, w1_fp8)  ← GEMM
   act_out = gelu(fc1_out)               ← cuBLASLt epilogue 融合

3. FC2 GEMM
   out = fp8_gemm(act_out_fp8, w2_fp8)
```

---

## 5. `te.LayerNormLinear`

```python
# 等价于：output = Linear(LayerNorm(input))
# 但 LN 输出直接量化为 FP8，送入 GEMM
model = te.LayerNormLinear(
    in_features=4096,
    out_features=4096,
    normalization="RMSNorm",
    return_layernorm_output=True,  # 返回 LN 输出（用于 residual）
)

# 前向（FP8 enabled）：
# x(BF16) ──→ RMSNorm kernel ──→ x_norm_fp8 ──→ GEMM(FP8) ──→ out(BF16)
#                           └──→ x_norm(BF16) [return_layernorm_output=True 时返回]
```

---

## 6. `te.GroupedLinear`

MoE 场景下替代多个独立 Linear 的高效实现：

```python
class GroupedLinear(TransformerEngineBaseModule):
    def __init__(
        self,
        num_gemms: int,      # expert 数量
        in_features: int,
        out_features: int,
        ...
    )
    
    def forward(
        self,
        inp: torch.Tensor,            # [total_tokens, in_features]
        m_splits: List[int],          # 每个 expert 的 token 数
    ) -> torch.Tensor:
```

内部调用 `cublaslt_grouped_gemm`，将 N 个 expert 的 GEMM 合并为单次 GPU kernel 调用。

---

## 7. `te.TransformerLayer`

最顶层的模块，一个完整的 Transformer block：

```python
class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        num_gqa_groups: int = None,
        
        # 归一化
        normalization: str = "LayerNorm",  # 或 "RMSNorm"
        layernorm_epsilon: float = 1e-5,
        
        # 激活
        activation: str = "gelu",
        
        # 注意力
        attn_input_format: str = "sbhd",  # 输入格式：sbhd 或 bshd
        attn_mask_type: str = "causal",
        
        # 层类型
        layer_type: str = "encoder",  # "encoder" 或 "decoder"（decoder 多一个 cross-attn）
        
        # 张量并行
        set_parallel_mode: bool = False,
        tp_group: Optional[dist.ProcessGroup] = None,
        sequence_parallel: bool = False,
        
        # 精度
        output_layernorm: bool = False,  # 是否在末尾加 LN（Post-LN 架构）
    )
    
    def forward(
        self,
        hidden_states: torch.Tensor,       # [S, B, H] 或 [B, S, H]
        attention_mask: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,  # decoder cross-attn 用
        is_first_microbatch: Optional[bool] = None,
        rotary_pos_emb: Optional[...] = None,
        inference_params: Optional[InferenceParams] = None,
    )
```

### 7.1 Pre-LN vs Post-LN

```python
# Pre-LN（现代 LLM 标准，默认）：
# x → LN → Attn → + x → LN → MLP → + x

# Post-LN（BERT 原始）：
# x → Attn → LN → + x → MLP → LN → + x
layer = te.TransformerLayer(
    hidden_size=1024, ...,
    output_layernorm=True,  # Post-LN 架构
    apply_residual_connection_post_layernorm=True,
)
```

### 7.2 Activation Checkpointing

TE 支持 FP8 友好的 activation checkpointing：

```python
from transformer_engine.pytorch.distributed import checkpoint

# 标准方式
output = checkpoint(
    transformer_layer,
    hidden_states,
    attention_mask,
    use_reentrant=False,
)

# FP8 activation recompute：重新计算时复用 scale
# 比标准 recompute 更节省显存，且数值完全一致
with te.fp8_autocast(enabled=True, ...):
    output = checkpoint(model_block, x, ...)
```

---

## 8. 模块初始化：`fp8_model_init`

```python
# 方式一：fp8_model_init context（仅初始化参数格式）
with te.fp8_model_init(enabled=True):
    model = MyTransformerModel(...)
    # 所有 TE Linear 的权重立即以 Float8Tensor 形式存储

# 方式二：quantized_model_init（指定 recipe）
with te.quantized_model_init(
    enabled=True,
    recipe=recipe.Float8CurrentScaling()
):
    model = MyTransformerModel(...)
```

**节省显存估算**（以 LLaMA-70B 为例）：
- 标准 BF16 参数：70B × 2 bytes = 140 GB
- FP8 参数：70B × 1 byte = 70 GB
- 节省 50%，可在更少 GPU 上装载

---

## 9. ONNX 导出

```python
from transformer_engine.pytorch import export

# 将 TE 模块导出为 ONNX
torch.onnx.export(
    model,
    (dummy_input,),
    "model.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
)
```

`onnx_extensions.py` 中注册了 TE 专属算子的 ONNX symbolic，将 FP8 相关算子映射为标准 ONNX 节点（通常降精到 FP32）。

---

## 10. 模块使用对比

| 场景 | 推荐模块 | 说明 |
|------|---------|------|
| 替换 `nn.Linear` | `te.Linear` | 最小改动，FP8 自动支持 |
| FFN 块 | `te.LayerNormMLP` | 融合 LN+FC1+Act+FC2 |
| Attention + FFN | `te.TransformerLayer` | 完整 block |
| MoE Expert | `te.GroupedLinear` | 多 expert 合并 GEMM |
| 自定义架构 | `te.ops.Sequential` | 灵活组合 ops |
