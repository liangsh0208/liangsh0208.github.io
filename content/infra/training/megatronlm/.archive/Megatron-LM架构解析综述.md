# Megatron-LM 架构解析综述

> 基于 /Users/danchen/Documents/RL_fw/Megatron-LM 代码分析

---

## 目录

- [一、项目概述](#一项目概述)
- [二、目录结构总览](#二目录结构总览)
- [三、核心模块详解](#三核心模块详解)
- [四、核心模块实现原理](#四核心模块实现原理)
- [五、张量并行实现](#五张量并行实现)
- [六、集合通信架构](#六集合通信架构)
- [七、流水线并行通信](#七流水线并行通信)
- [八、完整训练流程通信图](#八完整训练流程通信图)
- [九、关键函数速查表](#九关键函数速查表)

---

## 一、项目概述

Megatron-LM 是 NVIDIA 开发的大规模分布式训练框架，专门用于训练超大规模 Transformer 模型。该项目集成了强化学习训练能力，支持 GPT、T5、Mamba 等多种模型架构。

**核心特性**：
- 支持数据并行、张量并行、流水线并行混合
- 原生强化学习支持（GRPO 算法）
- Flash Attention、SwiGLU 等现代优化
- 支持千亿参数规模模型训练

---

## 二、目录结构总览

```
Megatron-LM/
├── megatron/                    # 核心实现目录
│   ├── core/                    # 核心 AI 框架（分布式、模型架构）
│   │   ├── transformer/         # Transformer 完整实现
│   │   ├── models/              # 模型定义：GPT、BERT、Mamba、T5、VLM 等
│   │   ├── tensor_parallel/     # 张量并行实现
│   │   ├── pipeline_parallel/   # 流水线并行实现
│   │   ├── distributed/         # 分布式训练核心
│   │   ├── optimizer/           # 优化器实现
│   │   ├── dataset/             # 数据加载管道
│   │   ├── parallel_state.py    # 进程组管理
│   │   └── fp8_utils.py         # 低精度计算支持
│   ├── inference/               # 推理优化模块
│   ├── legacy/                  # 旧版本兼容代码
│   ├── post_training/           # 后训练处理（微调、量化等）
│   ├── rl/                      # 强化学习实现（项目特色）
│   │   ├── agent/               # RL 代理实现
│   │   ├── rl_utils.py          # GRPO 算法
│   │   ├── sequence_packing_utils.py
│   │   └── parallel_utils.py
│   └── training/                # 通用训练框架
├── examples/                    # 示例代码
├── tools/                       # 辅助工具
├── tests/                       # 测试套件
├── pretrain_*.py                # 预训练脚本
├── train_rl.py                  # RL 训练主入口
├── gpt_builders.py              # GPT 模型构建器
└── model_provider.py            # 模型实例化提供者
```

---

## 三、核心模块详解

### 3.1 核心框架层 (`megatron/core`)

| 子模块 | 功能 |
|--------|------|
| `transformer/` | Transformer 完整实现（Attention、MLP、LayerNorm） |
| `models/` | 模型定义：GPT、BERT、Mamba、T5、VLM 等 |
| `distributed/` | 分布式训练核心：DP、MP、PP、ZeRO |
| `tensor_parallel/` | 张量并行实现 |
| `pipeline_parallel/` | 流水线并行实现 |
| `optimizer/` | 优化器实现，支持内存优化 |
| `dataset/` | 高效数据加载管道 |
| `fp8_utils.py / fp4_utils.py` | 低精度计算支持 |

### 3.2 强化学习模块 (`megatron/rl`)

```
rl/
├── agent/                    # RL 代理实现（策略、奖励计算）
├── rl_utils.py              # GRPO 算法（A2C 变体）
├── sequence_packing_utils.py # 序列打包优化
├── parallel_utils.py        # RL 分布式通信优化
└── inference/               # RL 推断数据流处理
```

**核心算法**：
- **GRPO**：Group Relative Policy Optimization
- **重要性采样**：用于策略梯度估计
- **KL 散度约束**：防止策略更新过大
- **熵正则化**：鼓励探索

### 3.3 训练框架层 (`megatron/training`）

```
pretrain()
    ├── 初始化分布式环境
    ├── 配置并行策略
    ├── 构建模型
    ├── 准备数据集
    ├── 训练循环
    │   ├── forward_step()
    │   ├── backward_step()
    │   └── optimizer.step()
    └── checkpoint 管理
```

### 3.4 核心设计模式

**工厂模式**：
```
train_rl.py → model_provider() → _gpt_builder() → GPTModel
```

**策略模式**：
通过 `transformer_layer_spec` 动态切换：
- Transformer Engine 实现
- 本地实现
- 异构层配置
- MoE / 多头注意力变体

**依赖注入**：
```python
GPTModel(
    config,
    transformer_layer_spec,  # 注入层规格
    vocab_size,
    ...
)
```

---

## 四、核心模块实现原理

### 4.1 TransformerBlock 实现

**文件路径**：`megatron/core/transformer/transformer_block.py`

```python
class TransformerBlock(GraphableMegatronModule, MegatronModule):
    def __init__(self, config: TransformerConfig, spec: TransformerBlockSubmodules, ...):
        # 构建层列表
        self._build_layers()

    def _build_layers(self):
        self.layers = torch.nn.ModuleList([
            build_layer(layer_spec, i + 1)
            for i, layer_spec in enumerate(self.submodules.layer_specs)
        ])

        # 最终层归一化
        if self.has_final_layernorm_in_this_stage():
            self.final_layernorm = LayerNorm(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
```

**数据流**：
```
Input [s, b, h] → Layer 0 → Layer 1 → ... → Layer N-1 → Final LayerNorm → Output
                    ↓
              [Self-Attention + MLP]
```

### 4.2 Attention 机制实现

**文件路径**：`megatron/core/transformer/attention.py`

```python
class SelfAttention(Attention):
    def get_query_key_value_tensors(self, hidden_states, ...):
        # QKV 线性投影
        mixed_qkv, _ = apply_module(self.linear_qkv)(hidden_states)

        # Multi-Query Attention 处理
        if self.config.num_query_groups < self.world_size:
            # 当 TP 大小 > KV 组数量时，先 all_gather 再切分
            mixed_qkv = all_gather_last_dim_from_tensor_parallel_region(mixed_qkv)
            idx = get_tensor_model_parallel_rank() // (self.world_size // self.config.num_query_groups)
            mixed_qkv = mixed_qkv[:, :, idx * size : (idx + 1) * size]

        # 分离 Q, K, V
        query, key, value = torch.split(mixed_qkv, split_sizes, dim=3)
```

**Flash Attention 集成**：
```python
# 通过 Transformer Engine 的融合内核
if config.apply_rope_fusion:
    return fused_apply_rotary_pos_emb(t, freqs, interleaved=config.rotary_interleaved)
```

### 4.3 MLP 层实现

**文件路径**：`megatron/core/transformer/mlp.py`

```python
class MLP(MegatronModule):
    def forward(self, hidden_states, ...):
        # FC1: [s, b, h] → [s, b, 4h/p] (p = TP size)
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)

        # SwiGLU 激活函数
        if self.config.gated_linear_unit and self.config.bias_activation_fusion:
            intermediate_parallel = bias_swiglu_impl(
                intermediate_parallel, bias_parallel,
                self.config.activation_func_fp8_input_store
            )
        elif self.config.gated_linear_unit:
            # 基础 SwiGLU: activation(gate) * linear
            x_glu, x_linear = torch.chunk(intermediate_parallel, 2, dim=-1)
            intermediate_parallel = F.silu(x_glu) * x_linear

        # FC2: [s, b, 4h/p] → [s, b, h]
        output, output_bias = self.linear_fc2(intermediate_parallel)
```

**SwiGLU 公式**：
$$\text{SwiGLU}(x) = \text{SiLU}(xW_{gate}) \odot (xW_{linear})$$

### 4.4 RoPE 位置编码

**文件路径**：`megatron/core/transformer/rope_utils.py`

```python
def _apply_rotary_pos_emb_bshd(t, freqs, rotary_interleaved=False, mscale=1.0):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * mscale).to(t.dtype)

    # 应用旋转
    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)

def _rotate_half(x, rotary_interleaved):
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[..., ::2]   # 偶数索引
        x2 = x[..., 1::2]  # 奇数索引
        return torch.stack((-x2, x1), dim=-1).view(x.shape)
```

---

## 五、张量并行实现

### 5.1 ColumnParallelLinear（列并行）

**核心原理**：将权重矩阵按列分割，每个 GPU 保持部分列

```
原始: Y = XA，其中 A ∈ R^{input×output}

并行: A = [A_1 | A_2 | ... | A_p]
      Y_i = XA_i  (每个 GPU 计算部分输出)
```

**文件路径**：`megatron/core/tensor_parallel/layers.py:731`

```python
class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, config, ...):
        # 输出维度按 TP 大小分片
        self.output_size_per_partition = divide(output_size, world_size)

        # 权重形状: [input_size, output_size_per_partition]
        self.weight = Parameter(
            torch.empty(self.output_size_per_partition, self.input_size, ...)
        )

    def forward(self, input_, ...):
        # 1. 输入复制（或序列并行时已分片）
        if self.sequence_parallel:
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)

        # 2. 局部矩阵乘法
        output_parallel = F.linear(input_parallel, self.weight, self.bias)

        # 3. 可选：收集所有分片
        if self.gather_output:
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
```

**通信模式**：
```
Forward:  Copy → MatMul → (可选 All-Gather)
Backward: All-Reduce ← ← Split
```

### 5.2 RowParallelLinear（行并行）

**核心原理**：将权重矩阵按行分割，输入也需分片

```
原始: Y = XA + b

并行: A = [A_1; A_2; ...; A_p]^T
      X = [X_1, X_2, ..., X_p]
      Y = Σ(X_i A_i^T) + b
```

**文件路径**：`megatron/core/tensor_parallel/layers.py:1072`

```python
class RowParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, config, input_is_parallel, ...):
        # 输入维度按 TP 大小分片
        self.input_size_per_partition = divide(input_size, world_size)

        # 权重形状: [output_size, input_size_per_partition]
        self.weight = Parameter(
            torch.empty(self.output_size, self.input_size_per_partition, ...)
        )

    def forward(self, input_, ...):
        # 1. 输入分片（如果尚未分片）
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)

        # 2. 局部矩阵乘法
        output_parallel = F.linear(input_parallel, self.weight)

        # 3. All-Reduce 合并结果
        if self.sequence_parallel:
            output = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output = reduce_from_tensor_model_parallel_region(output_parallel)

        # 4. 添加偏置
        if not self.skip_bias_add:
            output = output + self.bias
```

**通信模式**：
```
Forward:  (Scatter) → MatMul → All-Reduce / Reduce-Scatter
Backward: (Gather)  ← ← ← ← ← All-Gather
```

### 5.3 Transformer 层中的通信流

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Transformer Layer (TP=4)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input [s, b, h] ──┬──→ ColumnParallel(QKV) ──→ [s, b, 3h/4]       │
│                    │        │                                      │
│                    │        └─ Forward: Identity (input replicated) │
│                    │        └─ Backward: All-Reduce (grad)          │
│                    │                                                │
│                    └──→ Attention Compute ──→ [s, b, h/4]          │
│                                               │                     │
│                    ┌─────────────────────────┘                     │
│                    │                                                │
│  Output [s, b, h] ←── RowParallel(Proj) ←─── [s, b, h/4]          │
│                         │                                           │
│                         └─ Forward: All-Reduce (output)            │
│                         └─ Backward: Identity (grad replicated)    │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  MLP: ColumnParallel(FC1) → [s, b, 4h/4] → Activation             │
│                        │                                            │
│                        └─ Forward: Identity                         │
│                        └─ Backward: All-Reduce                      │
│                                                                     │
│       RowParallel(FC2) → [s, b, h]                                 │
│                        │                                            │
│                        └─ Forward: All-Reduce                       │
│                        └─ Backward: Identity                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 六、集合通信架构

### 6.1 进程组管理

**核心文件**：`megatron/core/parallel_state.py`

#### 全局进程组变量

```python
# 主要并行组
_TENSOR_MODEL_PARALLEL_GROUP = None      # 张量并行组
_PIPELINE_MODEL_PARALLEL_GROUP = None    # 流水线并行组
_DATA_PARALLEL_GROUP = None              # 数据并行组
_MODEL_PARALLEL_GROUP = None             # 模型并行组（TP+PP）
_CONTEXT_PARALLEL_GROUP = None           # 上下文并行组

# 专家并行组
_EXPERT_MODEL_PARALLEL_GROUP = None
_EXPERT_TENSOR_PARALLEL_GROUP = None
_EXPERT_DATA_PARALLEL_GROUP = None

# 复合并行组
_TENSOR_AND_DATA_PARALLEL_GROUP = None   # TP+DP 组合
_TENSOR_AND_CONTEXT_PARALLEL_GROUP = None # TP+CP 组合
```

#### 进程组初始化

**文件路径**：`megatron/core/parallel_state.py:549`

```python
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    order: str = "tp-cp-ep-dp-pp",
    ...
) -> None:
    """
    初始化模型并行组。

    并行维度顺序（默认）: tp-cp-ep-dp-pp
    global_rank = tp_rank + cp_rank*tp_size + ep_rank*tp_size*cp_size
                + dp_rank*... + pp_rank*...
    """
```

#### 进程组生成算法

**文件路径**：`megatron/core/parallel_state.py:252`

```python
def generate_masked_orthogonal_rank_groups(
    world_size: int,
    parallel_size: List[int],  # 例如 [tp_size, pp_size, dp_size]
    mask: List[bool]           # 例如 [False, True, False] 表示获取 PP 组
) -> List[List[int]]:
    """
    生成正交并行组。

    例如: 8 GPUs, parallel_size = [2, 2, 2], mask = [False, True, False]

    DP 组: [[0, 4], [1, 5], [2, 6], [3, 7]]
    PP 组: [[0, 2], [1, 3], [4, 6], [5, 7]]
    TP 组: [[0, 1], [2, 3], [4, 5], [6, 7]]
    """
```

#### Rank 映射示意图

**8 GPUs, TP=2, PP=2, DP=2**：

```
                    TP=0    TP=1
              ┌────────┬────────┐
        PP=0  │ Rank 0 │ Rank 1 │  ← PP组: [0, 2]
              ├────────┼────────┤
        PP=1  │ Rank 2 │ Rank 3 │  ← PP组: [2, 0]
              └────────┴────────┘
                    ↑
              DP组: [0, 4], [1, 5], ...
```

### 6.2 通信原语实现

**核心文件**：`megatron/core/tensor_parallel/mappings.py`

#### 底层通信函数

```python
# All-Reduce
def _reduce(input_, group):
    """全规约操作"""
    torch.distributed.all_reduce(input_.contiguous(), group=group)
    return input_

# All-Gather（沿最后一维）
def _gather_along_last_dim(input_, group):
    """沿最后一维收集张量"""
    output = torch.empty(dim_size, dtype=input_.dtype, device='cuda')
    dist_all_gather_func(output, input_.contiguous(), group=group)
    return torch.cat(output.chunk(world_size, dim=0), dim=-1)

# Reduce-Scatter（沿最后一维）
def _reduce_scatter_along_last_dim(input_, group):
    """沿最后一维规约并分散"""
    concat_tensor = torch.cat(torch.split(input_, split_size, dim=1), dim=0)
    return _reduce_scatter_along_first_dim(concat_tensor, group)

# All-Gather（沿第一维，序列并行用）
def _gather_along_first_dim(input_, group, output_split_sizes=None):
    """沿第一维收集张量（序列维度）"""
    output = torch.empty(dim_size, dtype=input_.dtype, device='cuda')
    dist_all_gather_func(output, input_.contiguous(), group=group)
    return output
```

#### Autograd 集成通信模块

**核心设计**：将通信操作封装为 `torch.autograd.Function`，实现前向/反向自动通信。

```python
# ============ Copy + All-Reduce ============
class _CopyToModelParallelRegion(torch.autograd.Function):
    """Forward: 复制输入
       Backward: All-Reduce 梯度"""

    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return input_  # 前向：直接复制

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.group), None  # 反向：All-Reduce

# ============ All-Reduce + Copy ============
class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """Forward: All-Reduce
       Backward: 复制梯度"""

    @staticmethod
    def forward(ctx, input_, group):
        return _reduce(input_, group)  # 前向：All-Reduce

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # 反向：直接传递

# ============ Split + All-Gather ============
class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Forward: 分割输入（沿最后一维）
       Backward: All-Gather 梯度"""

    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return _split_along_last_dim(input_, group)  # 前向：Split

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output, ctx.group), None  # 反向：All-Gather

# ============ All-Gather + Split ============
class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Forward: All-Gather（沿最后一维）
       Backward: Split 梯度"""

    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return _gather_along_last_dim(input_, group)  # 前向：All-Gather

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output, ctx.group), None  # 反向：Split

# ============ 序列并行专用 ============
class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Forward: All-Gather（沿第一维，序列维度）
       Backward: Reduce-Scatter"""

    @staticmethod
    def forward(ctx, input_, group, tensor_parallel_output_grad=True, ...):
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        ctx.group = group
        return _gather_along_first_dim(input_, group)  # 前向：All-Gather 序列

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.tensor_parallel_output_grad:
            return _reduce_scatter_along_first_dim(grad_output, ctx.group), ...  # 反向：Reduce-Scatter
        else:
            return _split_along_first_dim(grad_output, ctx.group), ...

class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Forward: Reduce-Scatter（沿第一维）
       Backward: All-Gather"""

    @staticmethod
    def forward(ctx, input_, group, ...):
        return _reduce_scatter_along_first_dim(input_, group)  # 前向：Reduce-Scatter

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output, ctx.group), ...  # 反向：All-Gather
```

### 6.3 通信原语对照表

| 通信原语 | 函数 | 前向操作 | 反向操作 |
|---------|------|---------|---------|
| `copy_to_tensor_model_parallel_region` | `_CopyTo...` | Identity | All-Reduce |
| `reduce_from_tensor_model_parallel_region` | `_ReduceFrom...` | All-Reduce | Identity |
| `scatter_to_tensor_model_parallel_region` | `_ScatterTo...` | Split (last dim) | All-Gather (last dim) |
| `gather_from_tensor_model_parallel_region` | `_GatherFrom...` | All-Gather (last dim) | Split (last dim) |
| `scatter_to_sequence_parallel_region` | `_ScatterToSeq...` | Split (first dim) | All-Gather (first dim) |
| `gather_from_sequence_parallel_region` | `_GatherFromSeq...` | All-Gather (first dim) | Reduce-Scatter (first dim) |
| `reduce_scatter_to_sequence_parallel_region` | `_ReduceScatterToSeq...` | Reduce-Scatter (first dim) | All-Gather (first dim) |
| `all_gather_last_dim_from_tensor_parallel_region` | `_AllGatherFrom...` | All-Gather (last dim) | Reduce-Scatter (last dim) |
| `reduce_scatter_last_dim_to_tensor_parallel_region` | `_ReduceScatterTo...` | Reduce-Scatter (last dim) | All-Gather (last dim) |

### 6.4 序列并行通信优化

```
标准 TP:
  Forward:  Copy → Compute → All-Reduce
  Memory:   完整激活 × TP_size

序列并行 (SP):
  Forward:  Split(序列) → Compute → Reduce-Scatter
  Memory:   分片激活

通信量对比:
  TP: All-Reduce = O(batch × seq × hidden × TP_size)
  SP: Reduce-Scatter = O(batch × seq × hidden)
```

**序列并行层代码流程**：

```python
# MLP with Sequence Parallel
def forward(self, hidden_states):
    # Forward:
    # 1. 输入已在序列维度分片 [s/TP, b, h]

    # 2. FC1 (ColumnParallel): 输入保持分片
    intermediate = self.linear_fc1(hidden_states)  # No communication
    intermediate = activation(intermediate)

    # 3. FC2 (RowParallel): 输出需要合并
    if self.sequence_parallel:
        # Reduce-Scatter: 规约并分散到序列维度
        output = reduce_scatter_to_sequence_parallel_region(
            self.linear_fc2(intermediate)
        )  # Output: [s/TP, b, h]
    else:
        output = reduce_from_tensor_model_parallel_region(
            self.linear_fc2(intermediate)
        )
```

---

## 七、流水线并行通信

**核心文件**：`megatron/core/pipeline_parallel/schedules.py`

### 7.1 调度函数选择

```python
def get_forward_backward_func(pp_size, vp_size):
    """根据配置选择调度函数"""
    if pp_size > 1:
        if vp_size is not None:
            return forward_backward_pipelining_with_interleaving  # 交错流水线
        else:
            return forward_backward_pipelining_without_interleaving  # 标准 1F1B
    else:
        return forward_backward_no_pipelining  # 无流水线
```

### 7.2 1F1B 调度核心

**文件路径**：`megatron/core/pipeline_parallel/combined_1f1b.py`

```python
def combined_1f1b_schedule_for_no_pipelining(
    forward_step_func,
    backward_step_func,
    ...
):
    """
    1F1B (One Forward One Backward) 调度:
    - 阶段1: Warmup - 连续执行前向
    - 阶段2: Steady - 交错执行前向和反向
    - 阶段3: Cooldown - 连续执行反向
    """
    # Warmup 阶段: 填充流水线
    for i in range(num_warmup_microbatches):
        output = forward_step_func(...)
        send_forward(output)

    # Steady 阶段: 1F1B 交错
    for i in range(num_microbatches_remaining):
        output = forward_step_func(...)      # 1 Forward
        send_forward(output)

        grad = recv_backward()               # 1 Backward
        backward_step_func(grad)

    # Cooldown 阶段: 排空流水线
    for i in range(num_warmup_microbatches):
        grad = recv_backward()
        backward_step_func(grad)
```

### 7.3 1F1B 时序图

```
时间 →
Stage 0: [F0][F1][F2][F3][B0][F4][B1][F5][B2][B3][B4][B5]
Stage 1:     [F0][F1][F2][B0][F3][B1][F4][B2][B3][B4][B5]
Stage 2:         [F0][F1][B0][F2][B1][F3][B2][B3][B4][B5]
Stage 3:             [F0][B0][F1][B1][F2][B2][B3][B3][B4][B5]

F = Forward, B = Backward, 数字 = micro-batch ID
```

### 7.4 P2P 通信

**核心文件**：`megatron/core/pipeline_parallel/p2p_communication.py`

```python
class P2PCommunicator:
    """流水线阶段间的点对点通信"""

    def send_forward(self, tensor, Shapes):
        """发送前向激活到下一阶段"""
        if not is_pp_last_stage():
            torch.distributed.send(
                tensor,
                dst=self.get_forward_dst(),
                group=self.pp_group
            )

    def recv_forward(self, Shapes):
        """从前一阶段接收前向激活"""
        if not is_pp_first_stage():
            tensor = torch.empty(Shapes, device='cuda')
            torch.distributed.recv(
                tensor,
                src=self.get_forward_src(),
                group=self.pp_group
            )
            return tensor

    def send_backward(self, tensor, Shapes):
        """发送反向梯度到前一阶段"""
        if not is_pp_first_stage():
            torch.distributed.send(
                tensor,
                dst=self.get_backward_dst(),
                group=self.pp_group
            )

    def recv_backward(self, Shapes):
        """从后一阶段接收反向梯度"""
        if not is_pp_last_stage():
            tensor = torch.empty(Shapes, device='cuda')
            torch.distributed.recv(
                tensor,
                src=self.get_backward_src(),
                group=self.pp_group
            )
            return tensor
```

---

## 八、完整训练流程通信图

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     Training Iteration Communication Flow                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                          Forward Pass                                │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  │                                                                      │  │
│  │  Data Loader ─→ [DP Shard]                                           │  │
│  │       │                                                              │  │
│  │       ↓                                                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │ PP Stage 0 (First Stage)                                        │ │  │
│  │  │   Embedding (TP-VocabParallel) ─→ All-Gather (if needed)       │ │  │
│  │  │        ↓                                                        │ │  │
│  │  │   Transformer Layers × N:                                       │ │  │
│  │  │     Attention: ColumnParallel(QKV) → All-Gather → RowParallel   │ │  │
│  │  │     MLP: ColumnParallel(FC1) → RowParallel(FC2) → All-Reduce    │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │       │                                                              │  │
│  │       │ P2P Send (activations)                                       │  │
│  │       ↓                                                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │ PP Stage 1, 2, ... (Middle Stages)                              │ │  │
│  │  │   P2P Recv → Transformer Layers → P2P Send                      │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │       │                                                              │  │
│  │       │ P2P Send (activations)                                       │  │
│  │       ↓                                                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │ PP Stage N (Last Stage)                                         │ │  │
│  │  │   P2P Recv → Transformer Layers → Output Layer → Loss          │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                          Backward Pass                               │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  │                                                                      │  │
│  │  Loss ∂L/∂output                                                     │  │
│  │       │                                                              │  │
│  │       ↓                                                              │  │
│  │  PP Stage N: Backward through layers                                 │  │
│  │       │                                                              │  │
│  │       │ P2P Send (gradients)                                         │  │
│  │       ↓                                                              │  │
│  │  PP Stage N-1, ..., 1: P2P Recv → Backward → P2P Send               │  │
│  │       │                                                              │  │
│  │       │ P2P Send (gradients)                                         │  │
│  │       ↓                                                              │  │
│  │  PP Stage 0: Backward through embedding                              │  │
│  │                                                                      │  │
│  │  Each TP Layer:                                                      │  │
│  │    RowParallel: All-Gather (grad w.r.t input)                        │  │
│  │    ColumnParallel: All-Reduce (grad w.r.t input)                     │  │
│  │                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                       Gradient Synchronization                       │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  │                                                                      │  │
│  │  Data Parallel Gradients:                                            │  │
│  │    ┌─────────────────────────────────────────────────────────────┐   │  │
│  │    │ All-Reduce (DP Group) or                                    │   │  │
│  │    │ Reduce-Scatter (Distributed Optimizer + ZeRO)               │   │  │
│  │    └─────────────────────────────────────────────────────────────┘   │  │
│  │                                                                      │  │
│  │  TP Gradients: (已在 forward/backward 中处理)                        │  │
│  │    一对 ColumnParallel+RowParallel 自动处理梯度同步                   │  │
│  │                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         Optimizer Step                               │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  │                                                                      │  │
│  │  Standard Optimizer:                                                 │  │
│  │    Each rank independently: param -= lr × grad                       │  │
│  │                                                                      │  │
│  │  Distributed Optimizer (ZeRO):                                       │  │
│  │    ┌─────────────────────────────────────────────────────────────┐   │  │
│  │    │ Each rank maintains a shard of optimizer states.            │   │  │
│  │    │                                                             │   │  │
│  │    │ Updates:                                                    │   │  │
│  │    │   1. All-Gather to get full gradients                       │   │  │
│  │    │   2. Update local parameter shard                            │   │  │
│  │    │   3. All-Gather to broadcast updated parameters              │   │  │
│  │    └─────────────────────────────────────────────────────────────┘   │  │
│  │                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 九、关键函数速查表

### 9.1 通信原语函数

| 文件位置 | 函数名 | 用途 |
|---------|--------|-----|
| `parallel_state.py:215` | `create_group()` | 创建进程组 |
| `parallel_state.py:549` | `initialize_model_parallel()` | 初始化所有并行组 |
| `parallel_state.py:252` | `generate_masked_orthogonal_rank_groups()` | 生成正交并行组 |

### 9.2 张量并行函数

| 文件位置 | 函数名 | 用途 |
|---------|--------|-----|
| `mappings.py:469` | `copy_to_tensor_model_parallel_region()` | 复制输入，反向 All-Reduce |
| `mappings.py:475` | `reduce_from_tensor_model_parallel_region()` | 前向 All-Reduce |
| `mappings.py:481` | `scatter_to_tensor_model_parallel_region()` | 前向 Split，反向 All-Gather |
| `mappings.py:487` | `gather_from_tensor_model_parallel_region()` | 前向 All-Gather，反向 Split |
| `mappings.py:493` | `scatter_to_sequence_parallel_region()` | 序列并行：前向 Split(first dim) |
| `mappings.py:499` | `gather_from_sequence_parallel_region()` | 序列并行：前向 All-Gather |
| `mappings.py:513` | `reduce_scatter_to_sequence_parallel_region()` | 序列并行：前向 Reduce-Scatter |

### 9.3 流水线并行函数

| 文件位置 | 函数名 | 用途 |
|---------|--------|-----|
| `schedules.py:45` | `get_forward_backward_func()` | 获取调度函数 |
| `schedules.py:168` | `custom_backward()` | 直接调用 C++ autograd |
| `combined_1f1b.py` | `combined_1f1b_schedule_for_no_pipelining()` | 1F1B 调度实现 |
| `p2p_communication.py` | `P2PCommunicator` | 流水线阶段通信 |

### 9.4 Transformer 组件函数

| 文件位置 | 类/函数名 | 用途 |
|---------|--------|-----|
| `transformer_block.py` | `TransformerBlock` | Transformer 层容器 |
| `attention.py` | `SelfAttention` | 自注意力实现 |
| `mlp.py` | `MLP` | MLP 层实现 |
| `rope_utils.py` | `apply_rotary_pos_emb()` | RoPE 位置编码 |
| `layers.py:731` | `ColumnParallelLinear` | 列并行线性层 |
| `layers.py:1072` | `RowParallelLinear` | 行并行线性层 |

---

## 总结

Megatron-LM 的分布式架构通过以下设计实现了高效的大规模训练：

1. **层次化进程组管理**：正交并行维度，支持灵活组合
2. **通信与计算融合**：通过 Autograd Function 隐式集成通信
3. **通信量优化**：序列并行减少激活显存，Reduce-Scatter 替代 All-Reduce
4. **流水线气泡消除**：1F1B 调度最大化 GPU 利用率
5. **模块化设计**：Column/Row Parallel Linear 作为基本构建块

---

> 文档生成时间：2026-03-04
>
> 分析代码路径：/Users/danchen/Documents/RL_fw/Megatron-LM