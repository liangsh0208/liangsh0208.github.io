---
created: 2026-05-06
---

# Megatron-LM Transformer层实现

> 【源码定位】|【阅读建议】|【前置知识】
> - **源码定位**: `megatron/core/transformer/transformer_layer.py`, `attention.py`, `mlp.py`
> - **阅读建议**: 重点理解Attention+MLP的组合，以及spec-based架构设计
> - **前置知识**: Transformer架构、FlashAttention、CUDA kernel优化

---

## 1. 模块概述

Transformer层是核心计算单元，Megatron实现包含：
- **TransformerLayer**: 单层封装 (71KB)
- **Attention**: 多后端支持 (73KB，支持FlashAttention/MHA/GQA/MLA)
- **MLP**: 多种激活支持 (17KB)
- **MoE**: 混合专家层 (见07文档)

### 【重点】层规格模式 (ModuleSpec)

Megatron使用`ModuleSpec`解耦层定义与实现，支持运行时切换：
- Local implementation vs TransformerEngine
- Standard Attention vs FlashAttention vs MLA
- Normal MLP vs MoE

---

## 2. 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Transformer Layer Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   标准Transformer层结构:                                                    │
│                                                                             │
│   Input: [sequence_length, batch_size, hidden_size] (SBH格式)             │
│       │                                                                     │
│       ▼ 1. Input LayerNorm                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Self-Attention Block                                              │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │   │
│   │  │ LayerNorm    │→ │ Linear QKV   │→│ Attention Compute        │ │   │
│   │  │ (or RMSNorm) │  │ (TP并行)     │ │ (local/flash/cuda)       │ │   │
│   │  └──────────────┘  └──────────────┘  └──────────────────────────┘ │   │
│   │         │              (可选AllGather)                 │           │   │
│   │         │                                             │ (Attention│   │
│   │         │                                             │ Output)    │   │
│   │         │          ┌──────────────────────┐           ▼           │   │
│   │         │          │ → Linear Projection  │← [S,B,H/TP]          │   │
│   │         │          │   (RowParallel, TP)  │                        │   │
│   │         │          └──────────┬───────────┘                        │   │
│   │         │                     │ (All-Reduce)                       │   │
│   │         │                     ▼                                    │   │
│   │         │    ← Bias-Dropout-Residual-Add → [S,B,H]               │   │
│   └─────────┼──────────────────────────────────────────────────────────┘   │
│             │                                                               │
│             ▼ 2. Post-Attention LayerNorm                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ MLP/MoE Block                                                      │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│   │  │ LayerNorm    │→ │ Linear FC1   │→ │ Activation   │→ [S,B,4H/TP]│  │
│   │  └──────────────┘  │ (ColumnParallel,│  │ (GELU/SwiGLU)│            │   │
│   │                    │ TP)              │  │              │            │   │
│   │                    └────────────────┘  └──────────────┘             │   │
│   │                            │                                        │   │
│   │                    ┌───────┴────────┐                              │   │
│   │                    │ Linear FC2     │← [S,B,4H/TP]                 │   │
│   │                    │ (RowParallel)  │                               │   │
│   │                    └───────┬────────┘                              │   │
│   │                            │ (All-Reduce / Reduce-Scatter if SP)  │   │
│   │                            ▼                                        │   │
│   │    ← Bias-Dropout-Residual-Add → [S,B,H/TP] or [S/TP,B,H]         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心概念与实现

### 3.1 Transformer层基础实现

```python
# 文件: megatron/core/transformer/transformer_layer.py

class TransformerLayer(MegatronModule):
    """单层Transformer实现。
    
    设计原则:
    1. 通过ModuleSpec注入子模块，支持多种实现Variant
    2. 处理多种并行组合(TP/SP/PP/CP)
    3. 支持激活重计算(gradient checkpointing)
    """
    
    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,  # ModuleSpec注入
        layer_number: int = 1,
        hidden_dropout: float = None,
    ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        
        # 构建输入层归一化
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
            config=config,
        )
        
        # 构建自注意力 (通过spec注入，支持多后端)
        self.self_attention = build_module(
            submodules.self_attention,
            config=config,
            layer_number=layer_number,
        )
        
        # 构建Pre-MLP层归一化
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
            config=config,
        )
        
        # 构建MLP/MoE (通过spec注入)
        self.mlp = build_module(
            submodules.mlp,
            config=config,
        )
        
        # 构建残差连接 (support fused bias-dropout-add)
        self.self_attn_bda = build_module(submodules.self_attn_bda)
        self.mlp_bda = build_module(submodules.mlp_bda)
    
    def forward(self, hidden_states, attention_mask, context=None):
        """前向传播。
        
        处理流程:
        1. Input LayerNorm + Self-Attention
        2. Residual connection
        3. Pre-MLP LayerNorm + MLP/MoE
        4. Residual connection
        """
        # === Self-Attention Block ===
        residual = hidden_states
        
        # LayerNorm/RMSNorm
        input_ln_output = self.input_layernorm(hidden_states)
        
        # Self-Attention
        attention_output = self.self_attention(
            hidden_states=input_ln_output,
            attention_mask=attention_mask,
            # ... other args
        )
        
        # Residual + Dropout
        if self.config.apply_residual_connection_post_layernorm:
            residual = input_ln_output  # BERT风格
            
        hidden_states = self.self_attn_bda(
            self.training,  # 训练/推理模式
            self.config.bias_dropout_fusion,  # 融合算子
            attention_output,  # 主输出
            residual,  # 残差输入
            self.hidden_dropout,  # dropout率
        )
        
        # === MLP Block ===
        residual = hidden_states
        
        # LayerNorm
        mlp_ln_output = self.pre_mlp_layernorm(hidden_states)
        
        # MLP/MoE
        mlp_output = self.mlp(mlp_ln_output)
        
        # Residual + Dropout
        hidden_states = self.mlp_bda(
            self.training,
            self.config.bias_dropout_fusion,
            mlp_output,
            residual,
            self.hidden_dropout,
        )
        
        return hidden_states
```

### 3.2 Attention实现架构

```python
# 文件: megatron/core/transformer/attention.py

class SelfAttention(Attention):
    """自注意力实现，支持多种后端:
    
    1. Local implementation (PyTorch native)
    2. TransformerEngine FusedAttention (FlashAttention, FusedAttention)
    3. Multi-Head Attention (MHA)
    4. Grouped Query Attention (GQA)
    5. Multi-Latent Attention (MLA, DeepSeek-V2/V3)
    """
    
    def get_query_key_value_tensors(self, hidden_states):
        """获取QKV张量，处理GQA和TP切分。
        
        关键逻辑:
        - 标准MHA: QKV每组H个heads
        - GQA: Q=H heads, KV=H/G groups (节省显存和计算)
        - MLA: kompresso KV cache (DeepSeek优化)
        """
        # QKV线性映射 (ColumnParallel)
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        
        if self.config.num_query_groups is None:
            # 标准MHA
            query, key, value = mixed_qkv.chunk(3, dim=-1)
        else:
            # GQA: 多个query heads共享一组KV
            num_query_groups = self.config.num_query_groups
            # 拆分mixed_qkv为Q和KV...
            
        return query, key, value
    
    def run_attention(self, query, key, value, attention_mask):
        """执行核心attention计算。
        
        后端选择优先级:
        1. 若config.attention_backend=='flash', 使用FlashAttention
        2. 若FlashAttention可用，自动启用
        3. fallback到local implementation
        """
        if self.use_flash_attention:
            # FlashAttention: 融合算子，节省显存
            from flash_attn import flash_attn_func
            output = flash_attn_func(
                query.transpose(0, 1),  # [B,H,S,D] format
                key.transpose(0, 1),
                value.transpose(0, 1),
                causal=True,  # autoregressive mask
                softmax_scale=self.softmax_scale,
            )
        elif self.use_fused_attention:
            # TransformerEngine融合算子
            output = self.fused_attention(query, key, value, attention_mask)
        else:
            # 本地实现 (debug/兼容性)
            output = self.local_attention(query, key, value, attention_mask)
            
        return output
```

### 3.3 MLP实现

```python
# 文件: megatron/core/transformer/mlp.py

class MLP(MegatronModule):
    """MLP层实现，支持多种激活和并行策略。
    
    架构: GeGLU/GeLU/SiLU + SwiGLU variants
     - ColumnParallel(FC1): [H, 4H/TP]  (gates when using GLU)
     - Activation: GELU/SiGLU/...
     - RowParallel(FC2): [4H/TP, H] → All-Reduce
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # FC1: ColumnParallel
        # 当使用GLU时，输出2×intermediate_size (gate + up)
        ffn_hidden_size = config.ffn_hidden_size
        if config.gated_linear_unit:
            ffn_hidden_size *= 2  # 为GLU翻倍
            
        self.linear_fc1 = ColumnParallelLinear(
            config.hidden_size,
            ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,  # 保持分片
            bias=config.add_bias_linear,
        )
        
        # FC2: RowParallel
        self.linear_fc2 = RowParallelLinear(
            config.ffn_hidden_size,  # 输入分片
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            input_is_parallel=True,  # 输入是分片的
            bias=config.add_bias_linear,
        )
    
    def forward(self, hidden_states):
        # FC1: [S,B,H] → [S,B,4H/TP] (或[S,B,8H/TP] for GLU)
        intermediate, _ = self.linear_fc1(hidden_states)
        
        # 激活
        if self.config.gated_linear_unit:
            # SwiGLU: split into 2 halves
            x1, x2 = intermediate.chunk(2, dim=-1)
            # SwiGLU(x1, x2) = SiLU(x1) * x2
            intermediate = F.silu(x1) * x2
        else:
            intermediate = F.gelu(intermediate)
        
        # FC2: [S,B,4H/TP] → [S,B,H] + All-Reduce
        output, _ = self.linear_fc2(intermediate)
        
        return output
```

### 3.4 ModuleSpec注入机制

```python
# 文件: megatron/core/transformer/spec_utils.py

@dataclass
class ModuleSpec:
    """模块规格定义。
    
    实现运行时动态模块切换的工厂模式。
    
    示例使用:
    - 标准Attention vs FlashAttention
    - Normal MLP vs MoE vs Mamba
    - Local implementation vs TransformerEngine
    """
    module: type  # 实现类
    params: dict = field(default_factory=dict)  # 构造函数参数
    submodules: Any = None  # 嵌套子模块规格

class TransformerLayerSubmodules:
    """TransformerLayer的完整子模块规格。"""
    input_layernorm: ModuleSpec
    self_attention: ModuleSpec  # 可切换多种Attention实现
    pre_mlp_layernorm: ModuleSpec
    mlp: ModuleSpec  # 可切换MLP/MoE/Mamba
    # ...

# 示例: 构建GPT模型时选择实现
def get_gpt_layer_spec(use_te: bool, use_moe: bool) -> ModuleSpec:
    """获取GPT层规格。"""
    if use_te:
        # TransformerEngine backend
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=TEDotProductAttention,  # TE实现
                ),
                mlp=ModuleSpec(
                    module=TEMLP if not use_moe else MoE,  # 切换MLP/MoE
                ),
            ),
        )
    else:
        # Local implementation
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=SelfAttention,  # 本地PyTorch实现
                ),
                mlp=ModuleSpec(
                    module=MLP if not use_moe else MoELayer,
                ),
            ),
        )
```

### 3.5 与nanotron对比

| 对比项 | Megatron-LM | nanotron |
|--------|-------------|----------|
| **层定义** | ModuleSpec工厂模式 | 直接子类化 |
| **后端支持** | local/TE/flash等多后端 | 简单实现 |
| **FlashAttention** | 完善支持 (dot_product_attention.py) | 基础支持 |
| **GQA/MLA** | 原生支持 (multi_latent_attention.py) | 需手动实现 |
| **MoE集成** | 深度集成 (moe_layer.py) | 无 |
| **CUDA Graphs** | 支持 (cuda_graphs.py 125008字节) | 无 |
| **代码规模** | transformer/ 目录 ~300KB | ~5000行 |

---

## 4. 配置参数

| 参数名 | 类型 | 典型值 | 说明 |
|--------|------|--------|------|
| `attention_backend` | str | "auto" | attention后端: auto/local/flash |
| `num_query_groups` | int \| None | None | GQA组数，None=MHA |
| `multi_latent_attention` | bool | False | 启用DeepSeek MLA |
| `gated_linear_unit` | bool | True | 使用SwiGLU/GeGLU |
| `activation_func` | str | "swiglu" | 激活函数 |
| `ffn_hidden_size` | int | 4*H | FFN中间维度 |
| `recompute_granularity` | str \| None | "selective" | 激活重计算粒度 |

---

## 5. 常见问题与排查

**Q: Attention精度异常**

```python
# 诊断: 检查softmax scale
config.softmax_scale = 1.0 / math.sqrt(self.kv_channels)
# 或使用flash attention时需匹配
```

**Q: OOM在large sequence**

```python
# 方案: 启用sequence_parallel或context_parallel
config.sequence_parallel = True  # 沿序列切分
config.context_parallel_size = 4  # 长序列CP
```

---

## 6. 参考资料

- **核心文件**:
  - `megatron/core/transformer/transformer_layer.py` (71KB)
  - `megatron/core/transformer/attention.py` (73KB)
  - `megatron/core/transformer/mlp.py` (17KB)
- **交叉引用**: [03_张量并行](03_张量并行TP实现.md), [07_MoE混合专家架构](07_MoE混合专家架构.md)
- **论文**: [FlashAttention-2](https://arxiv.org/abs/2307.08691)
