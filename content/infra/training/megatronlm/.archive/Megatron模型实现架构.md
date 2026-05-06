# Megatron-LM Model 模块架构详解

> 基于 Megatron-LM 源码梳理，涵盖模型层次结构、核心组件、数据流向与分布式并行设计。

---

## 1. 目录结构总览

```
megatron/core/
├── models/                        # 各类模型实现
│   ├── gpt/gpt_model.py           # GPT 模型
│   ├── bert/bert_model.py         # BERT 模型
│   ├── T5/t5_model.py             # T5 模型
│   ├── mamba/mamba_model.py       # Mamba 模型
│   ├── multimodal/llava_model.py  # 多模态 LLaVA
│   ├── vision/clip_vit_model.py   # 视觉模型
│   └── common/                    # 共享组件
│       ├── language_module/       # 语言模型基类
│       └── embeddings/            # Embedding 组件
│
├── transformer/                   # Transformer 核心组件
│   ├── transformer_block.py       # Transformer Block (多层堆叠)
│   ├── transformer_layer.py       # 单个 Transformer Layer
│   ├── transformer_config.py      # Transformer 配置
│   ├── attention.py               # Attention 实现
│   ├── mlp.py                     # MLP 实现
│   ├── spec_utils.py              # 模块规格/构建工具
│   └── moe/                       # MoE 组件
│
└── distributed/                   # 分布式并行组件
    ├── tensor_parallel/           # 张量并行
    └── pipeline_parallel/         # 流水线并行
```

---

## 2. 类继承层次

```
MegatronModule (megatron/core/transformer/module.py)
    │
    ├── GraphableMegatronModule (支持 CUDA Graph)
    │       │
    │       ├── TransformerBlock
    │       │       └── TransformerLayer (多层)
    │       │
    │       └── BaseTransformerLayer
    │               ├── SelfAttention
    │               └── MLP
    │
    └── LanguageModule (megatron/core/models/common/language_module/)
            │
            ├── GPTModel
            ├── BertModel
            ├── T5Model
            ├── MambaModel
            └── LlavaModel
```

**关键基类说明：**

| 基类 | 文件位置 | 职责 |
|------|---------|------|
| `MegatronModule` | `transformer/module.py` | 所有模块基类，提供分布式训练基础 |
| `GraphableMegatronModule` | `transformer/module.py` | 支持 CUDA Graph 捕获 |
| `LanguageModule` | `models/common/language_module/` | 语言模型基类，处理 embedding 共享 |

---

## 3. GPT Model 核心组件

**文件位置：** `megatron/core/models/gpt/gpt_model.py`

```
GPTModel
├── embedding: LanguageModelEmbedding
│   ├── word_embeddings (Token Embedding)
│   └── position_embeddings (位置编码)
│
├── rotary_pos_emb: RotaryEmbedding / YarnRotaryEmbedding
│   └── RoPE 位置编码 (可选)
│
├── decoder: TransformerBlock
│   └── layers: List[TransformerLayer] (多层)
│       ├── input_layernorm
│       ├── self_attention
│       ├── pre_mlp_layernorm
│       └── mlp
│
├── mtp: MultiTokenPredictionBlock (可选，DeepSeek-V3 MTP)
│
└── output_layer: ColumnParallelLinear
    └── 输出 Logits 映射
```

**初始化参数：**

```python
GPTModel(
    config: TransformerConfig,           # Transformer 配置
    transformer_layer_spec: ModuleSpec,  # 层规格定义
    vocab_size: int,                     # 词表大小
    max_sequence_length: int,            # 最大序列长度
    pre_process: bool = True,            # PP: 是否包含 embedding
    post_process: bool = True,           # PP: 是否包含 output layer
    share_embeddings_and_output_weights: bool = False,  # 是否共享权重
    position_embedding_type: str = 'learned_absolute',  # 位置编码类型
    # ... 其他参数
)
```

---

## 4. Transformer Layer 结构

**文件位置：** `megatron/core/transformer/transformer_layer.py`

```
TransformerLayer (BaseTransformerLayer)
│
├── Input Layernorm
│   └── TensorParallelLayerNorm / FusedLayerNorm
│
├── Self-Attention
│   ├── LinearQKV (ColumnParallelLinear)
│   │   └── Q, K, V 投影 (融合)
│   ├── CoreAttention
│   │   ├── FlashAttention / FusedAttention
│   │   └── 支持 Multi-Head / Grouped-Query Attention
│   └── LinearProjection (RowParallelLinear)
│       └── Output 投影
│
├── Pre-MLP Layernorm
│
├── MLP
│   ├── LinearFc1 (ColumnParallelLinear)
│   │   └── 激活前投影
│   ├── Activation (SwiGLU / GELU / ReLU)
│   └── LinearFc2 (RowParallelLinear)
│       └── 输出投影
│
└── Residual Connections (Pre-Norm 架构)
```

**Forward 流程：**

```python
# TransformerLayer forward 伪代码
def forward(self, hidden_states, attention_mask, ...):
    # 1. Self-Attention
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attention(hidden_states, attention_mask, ...)
    hidden_states = residual + hidden_states  # 残差连接

    # 2. MLP
    residual = hidden_states
    hidden_states = self.pre_mlp_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states  # 残差连接

    return hidden_states
```

---

## 5. 数据流向 (Forward Pass)

```
Input IDs [batch, seq_len]
        │
        ▼
┌─────────────────────────┐
│   LanguageModelEmbedding │
│   - Token Embedding      │
│   - Position Embedding   │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   RotaryEmbedding (可选)  │
│   - 计算旋转位置编码       │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   TransformerBlock      │
│   ┌───────────────────┐ │
│   │ TransformerLayer 1│ │
│   │  - Layernorm      │ │
│   │  - Self-Attention │ │
│   │  - Layernorm      │ │
│   │  - MLP            │ │
│   └───────────────────┘ │
│   │      ...           │ │
│   ┌───────────────────┐ │
│   │ TransformerLayer N│ │
│   └───────────────────┘ │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Final Layernorm       │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Output Layer          │
│   (ColumnParallelLinear)│
│   hidden_size → vocab   │
└─────────────────────────┘
        │
        ▼
   Logits [batch, seq_len, vocab]
```

**GPTModel Forward 核心代码：**

```python
def forward(self, input_ids, position_ids, attention_mask, labels=None, ...):
    # 1. 预处理：Embedding + RoPE
    decoder_input = self.embedding(input_ids, position_ids)
    rotary_pos_emb = self.rotary_pos_emb(seq_len) if self.position_embedding_type == 'rope' else None

    # 2. Transformer Block
    hidden_states = self.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        rotary_pos_emb=rotary_pos_emb,
        ...
    )

    # 3. 后处理：Output Layer
    logits = self.output_layer(hidden_states)

    # 4. 计算损失（如果提供 labels）
    if labels is not None:
        loss = self.compute_language_model_loss(labels, logits)
        return loss

    return logits
```

---

## 6. ModuleSpec 设计模式

Megatron 使用 **ModuleSpec** 实现模块化配置，允许灵活替换组件。

**核心类：**

| 类名 | 作用 |
|------|------|
| `ModuleSpec` | 定义模块类型和初始化参数 |
| `TransformerLayerSpec` | 定义 Transformer Layer 各子模块 |
| `build_module()` | 根据 Spec 构建模块实例 |

**使用示例：**

```python
from megatron.core.transformer.spec_utils import ModuleSpec

# 定义层规格
layer_spec = TransformerLayerSubmodules(
    self_attention=ModuleSpec(
        module=SelfAttention,
        params={"num_attention_heads": 32}
    ),
    self_attn_bda=ModuleSpec(module=BiasDropoutAdd),
    pre_mlp_layernorm=ModuleSpec(module=LayerNorm),
    mlp=ModuleSpec(
        module=MLP,
        params={"hidden_size": 4096, "ffn_hidden_size": 13696}
    ),
    mlp_bda=ModuleSpec(module=BiasDropoutAdd),
)

# 使用规格构建模型
model = GPTModel(
    config=config,
    transformer_layer_spec=layer_spec,
    vocab_size=50000,
    max_sequence_length=2048,
)
```

**自定义替换组件：**

```python
# 替换为自定义 Attention
custom_layer_spec = TransformerLayerSubmodules(
    self_attention=ModuleSpec(module=MyCustomAttention),
    mlp=ModuleSpec(module=MLP),
)
```

---

## 7. 分布式并行支持

### 7.1 并行类型概览

| 并行类型 | 缩写 | 实现位置 | 关键组件 |
|---------|------|---------|---------|
| Tensor Parallel | TP | `tensor_parallel/` | ColumnParallelLinear, RowParallelLinear |
| Pipeline Parallel | PP | `pipeline_parallel/` | pre_process, post_process 标志 |
| Sequence Parallel | SP | `tensor_parallel/` | scatter/gather 操作 |
| Expert Parallel | EP | `transformer/moe/` | MoE Router |
| Data Parallel | DP | `distributed/` | DDP/FSDP |

### 7.2 Tensor Parallel (TP)

**核心思想：** 将矩阵乘法按列或行切分到多个 GPU。

```python
# ColumnParallelLinear: 按列切分权重
# Y = X @ W -> Y = [X @ W1, X @ W2, ..., X @ Wn] (每个 GPU 计算一部分)
output = ColumnParallelLinear(hidden_size, ffn_hidden_size)

# RowParallelLinear: 按行切分权重
# Y = X @ W -> Y = X1 @ W1 + X2 @ W2 + ... + Xn @ Wn (每个 GPU 计算一部分后 all-reduce)
output = RowParallelLinear(ffn_hidden_size, hidden_size)
```

**TP 下的 MLP 结构：**

```
Input [h]
    │
    ▼
ColumnParallelLinear (fc1) [h → ffn/n_tp]
    │  (每个 TP rank 计算一部分)
    ▼
Activation
    │
    ▼
RowParallelLinear (fc2) [ffn/n_tp → h]
    │  (all-reduce 聚合结果)
    ▼
Output [h]
```

### 7.3 Pipeline Parallel (PP)

**核心思想：** 将模型按层切分到不同 GPU。

```python
# GPTModel 中的 PP 处理
class GPTModel:
    def __init__(self, pre_process=True, post_process=True, ...):
        if self.pre_process:  # 第一个 PP stage
            self.embedding = LanguageModelEmbedding(...)

        self.decoder = TransformerBlock(...)

        if self.post_process:  # 最后一个 PP stage
            self.output_layer = ColumnParallelLinear(...)
```

**PP 执行流程：**

```
Stage 0: Embedding → Layer 0-7
    │  (send activations)
    ▼
Stage 1: Layer 8-15
    │  (send activations)
    ▼
Stage 2: Layer 16-23
    │  (send activations)
    ▼
Stage 3: Layer 24-31 → Output Layer
```

### 7.4 Sequence Parallel (SP)

**核心思想：** 在 Sequence 维度切分，减少激活内存。

```python
# SP 在 Attention 和 MLP 中的应用
if self.config.sequence_parallel:
    # 将序列切分到 TP ranks
    hidden_states = scatter_to_sequence_parallel_region(hidden_states)
    # 计算完成后聚合
    hidden_states = gather_from_sequence_parallel_region(hidden_states)
```

---

## 8. 关键配置类

### 8.1 TransformerConfig

**文件位置：** `megatron/core/transformer/transformer_config.py`

```python
@dataclass
class TransformerConfig:
    # 模型尺寸
    num_layers: int = 1
    hidden_size: int = 512
    num_attention_heads: int = 8
    ffn_hidden_size: int = 2048

    # 并行配置
    tensor_model_parallel_size: int = 1  # TP 大小
    pipeline_model_parallel_size: int = 1  # PP 大小
    virtual_pipeline_model_parallel_size: int = None  # VP 大小

    # 序列并行
    sequence_parallel: bool = False

    # Attention 配置
    num_query_groups: int = None  # GQA group 数
    attention_dropout: float = 0.1

    # MLP 配置
    hidden_dropout: float = 0.1
    activation_func: Callable = None  # SwiGLU/GELU

    # 位置编码
    position_embedding_type: str = 'learned_absolute'  # rope/yarn/mrope/none
    rotary_base: int = 10000

    # 精度
    bf16: bool = False
    fp16: bool = False
    fp8: str = None  # FP8 训练

    # MoE 配置
    num_moe_experts: int = None
    moe_router_load_balancing_type: str = "aux_loss"
```

### 8.2 常用模型配置示例

**GPT-3 175B 配置：**

```python
config = TransformerConfig(
    num_layers=96,
    hidden_size=12288,
    num_attention_heads=96,
    ffn_hidden_size=49152,  # 4x hidden_size
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=16,
    sequence_parallel=True,
    position_embedding_type='learned_absolute',
)
```

**LLaMA 70B 配置：**

```python
config = TransformerConfig(
    num_layers=80,
    hidden_size=8192,
    num_attention_heads=64,
    ffn_hidden_size=28672,  # 3.5x hidden_size (SwiGLU)
    num_query_groups=8,  # GQA
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=4,
    sequence_parallel=True,
    position_embedding_type='rope',
    rotary_base=500000,
)
```

---

## 9. 特殊特性支持

### 9.1 Mixture of Experts (MoE)

**文件位置：** `megatron/core/transformer/moe/`

```
MoE Layer 结构:
├── Router
│   └── gate (Linear) → Top-K 专家选择
├── Experts
│   └── List[MLP] (多个专家)
└── Token Dispatcher
    └── 将 token 分发到对应专家
```

**关键配置：**

```python
config = TransformerConfig(
    num_moe_experts=8,
    moe_router_load_balancing_type="aux_loss",
    moe_router_topk=2,
    moe_grouped_gemm=True,
)
```

### 9.2 Multi-Latent Attention (MLA)

**文件位置：** `megatron/core/transformer/multi_latent_attention.py`

DeepSeek-V2/V3 提出的低秩注意力压缩技术。

```python
# MLA 将 KV 压缩到低秩空间
# KV Cache 大幅减少
config = TransformerConfig(
    multi_latent_attention=True,
    q_lora_rank=1536,  # Q 的 LoRA rank
    kv_lora_rank=512,  # KV 的 LoRA rank
)
```

### 9.3 Multi-Token Prediction (MTP)

**文件位置：** `megatron/core/transformer/multi_token_prediction.py`

DeepSeek-V3 的多 token 预测，提升推理效率。

```python
config = TransformerConfig(
    mtp_num_layers=1,  # MTP 层数
    mtp_loss_scaling=0.3,  # MTP 损失权重
)
```

### 9.4 其他特性

| 特性 | 文件位置 | 说明 |
|------|---------|------|
| FlashAttention | `attention.py` | 高效注意力实现 |
| FP8/FP4 训练 | `fp8_utils.py` / `fp4_utils.py` | 低精度训练 |
| CUDA Graphs | `cuda_graphs.py` | 推理优化 |
| Yarn RoPE | `embeddings/yarn_rotary_embedding.py` | 长序列扩展 |

---

## 10. 模型初始化流程

```python
# 1. 创建配置
config = TransformerConfig(...)

# 2. 定义层规格
from megatron.core.transformer.transformer_layer import get_transformer_layer_spec
layer_spec = get_transformer_layer_spec(config, use_te=True)

# 3. 创建模型
model = GPTModel(
    config=config,
    transformer_layer_spec=layer_spec,
    vocab_size=tokenizer.vocab_size,
    max_sequence_length=2048,
)

# 4. 分布式初始化
from megatron.training.initialize import initialize_megatron
initialize_megatron(extra_args_provider=None, args_dict={...})

# 5. 可选：加载预训练权重
model.load_state_dict(checkpoint['model'])
```

---

## 参考资料

- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [Megatron-Core 文档](https://docs.nvidia.com/megatron-core/)
- [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)