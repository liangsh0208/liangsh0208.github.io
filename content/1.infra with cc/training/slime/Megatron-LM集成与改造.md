---
created: 2026-05-06
---

# Slime 对 Megatron-LM 的集成与改造

## 一、集成方式概述

Slime 通过以下方式调用 Megatron-LM：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Slime 框架                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     slime/backends/megatron_utils/                    │ │
│  │                                                                       │ │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │ │
│  │   │ initialize  │    │ model.py    │    │ loss.py     │              │ │
│  │   │             │    │             │    │             │              │ │
│  │   │ • 分布式初始化│    │ • 模型构建   │    │ • 损失计算   │              │ │
│  │   │ • 随机种子   │    │ • 训练循环   │    │ • 优势估计   │              │ │
│  │   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘              │ │
│  │          │                  │                  │                      │ │
│  │          └──────────────────┼──────────────────┘                      │ │
│  │                             │                                         │ │
│  │                             ▼                                         │ │
│  │   ┌─────────────────────────────────────────────────────────────┐    │ │
│  │   │                     Megatron-LM API 调用                     │    │ │
│  │   │                                                              │    │ │
│  │   │  from megatron.core import mpu, tensor_parallel             │    │ │
│  │   │  from megatron.core.models.gpt import GPTModel              │    │ │
│  │   │  from megatron.core.optimizer import get_megatron_optimizer │    │ │
│  │   │  from megatron.training.checkpointing import ...            │    │ │
│  │   └─────────────────────────────────────────────────────────────┘    │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、核心集成模块

### 2.1 分布式初始化 (`initialize.py`)

Slime 直接调用 Megatron 的分布式初始化 API：

```python
from megatron.core import mpu, tensor_parallel
from megatron.training.global_vars import set_args

def init(args):
    # 1. 设置全局参数
    set_args(args)

    # 2. 初始化模型并行组 (TP/PP/EP/DP/CP)
    mpu.initialize_model_parallel(
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.virtual_pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        ...
    )

    # 3. 设置随机种子
    tensor_parallel.model_parallel_cuda_manual_seed(seed, ...)

    # 4. 构建 tokenizer (Megatron 内置)
    _build_tokenizer(args)
```

**改造点：**
- 添加了 `deterministic_mode` 支持
- 添加了 `custom_megatron_init_path` 自定义初始化钩子

---

### 2.2 模型构建 (`model_provider.py`)

Slime 封装了 Megatron 的 `GPTModel` 创建过程：

```python
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.transformer_config import TransformerConfig

def model_provider(pre_process=True, post_process=True, vp_stage=None) -> GPTModel:
    # 1. 从参数构建 TransformerConfig
    config = core_transformer_config_from_args(args)

    # 2. 选择 layer spec (Transformer Engine 或 Local)
    if args.num_experts:
        transformer_layer_spec = get_gpt_decoder_block_spec(config, ...)
    else:
        if use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(...)
        else:
            transformer_layer_spec = get_gpt_layer_local_spec(...)

    # 3. 构建 GPT 模型
    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        ...
    )

    return model
```

**改造点：**

| 改造项 | 说明 |
|--------|------|
| **Bridge 模式** | 支持 `megatron.bridge.AutoBridge` 从 HF 自动转换 |
| **Critic 输出层** | 为 Critic 模型替换 `LinearForLastLayer` (output_size=1) |
| **参数冻结** | 添加 `freeze_model_params()` 支持选择性训练 |
| **MTP 支持** | 支持 Multi-Token Prediction 训练 |
| **VP Stage** | 支持 Virtual Pipeline 并行阶段参数 |

---

### 2.3 训练循环 (`model.py`)

Slime 重写了训练循环以适配 RL 训练：

```python
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer import get_megatron_optimizer

def train(rollout_id, model, optimizer, opt_param_scheduler, data_iterator, num_microbatches):
    # 1. 配置 DDP 和梯度同步
    config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
    config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]

    # 2. 获取 Megatron 的 forward_backward 函数
    forward_backward_func = get_forward_backward_func()

    # 3. 执行训练步骤
    for step_id in range(num_steps_per_rollout):
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches[step_id],
            forward_only=False,
        )

        # 4. 优化器更新
        optimizer.step()
        opt_param_scheduler.step(increment=args.global_batch_size)
```

**改造点：**

| 改造项 | 说明 |
|--------|------|
| **Rollout-based 训练** | 以 rollout_id 为单位组织训练循环 |
| **自定义损失函数** | 替换为 RL 损失 (PPO/GRPO/REINFORCE) |
| **动态 batch size** | 支持基于 token 数量的动态 batch |
| **前向钩子控制** | 添加 `enable/disable_forward_pre_hook` |
| **手动 GC** | 添加手动垃圾回收控制 |

---

### 2.4 检查点管理 (`checkpoint.py`)

Slime 扩展了 Megatron 的检查点功能：

```python
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint

def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, ...):
    load_path = args.load

    # 检查是否为 Megatron 格式
    if _is_megatron_checkpoint(load_path):
        # 使用 Megatron 原生加载
        return _load_checkpoint_megatron(...)
    else:
        # 使用 HF 格式加载
        return _load_checkpoint_hf(...)

def _load_checkpoint_hf(ddp_model, optimizer, args, load_path):
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint)
    bridge.load_hf_weights(ddp_model)
```

**改造点：**

| 改造项 | 说明 |
|--------|------|
| **双格式支持** | 同时支持 Megatron 和 HuggingFace 检查点 |
| **Bridge 集成** | 使用 `megatron.bridge` 进行格式转换 |
| **性能优化** | Patch `EnumerableShardingSpec.__post_init__` 跳过慢速验证 |
| **ROCm 兼容** | 添加 `ROCmFileSystemWriterAsync` 补丁 |

---

## 三、关键改造详解

### 3.1 HuggingFace 权重转换 (`megatron_to_hf/`)

Slime 实现了完整的 Megatron → HuggingFace 权重转换：

```
megatron_to_hf/
├── __init__.py          # 转换分发器
├── qwen3_next.py        # Qwen3-Next 转换
├── qwen3moe.py          # Qwen3-MoE 转换
├── glm4.py              # GLM4 转换
├── glm4moe.py           # GLM4-MoE 转换
├── deepseekv3.py        # DeepSeek V3 转换
├── llama.py             # Llama 转换
├── mimo.py              # MiMo 转换
└── processors/          # 后处理器
    ├── padding_remover.py
    ├── quantizer_compressed_tensors.py
    └── quantizer_fp8.py
```

**转换流程：**

```
Megatron 权重 (分布式张量)
        │
        ▼
┌─────────────────────┐
│ 1. remove_padding   │  移除 TP/EP padding
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 2. _convert_to_hf   │  名称映射 & 维度转换
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 3. quantize_params  │  可选量化 (int4/fp8)
└──────────┬──────────┘
           │
           ▼
HuggingFace 权重 (标准张量)
```

**示例转换逻辑 (Qwen3-MoE)：**

```python
def convert_qwen3moe_to_hf(args, name, param):
    # MLP down proj
    if "experts.down_proj" in name:
        name = name.replace("experts.down_proj", "experts.down_proj")
        param = param.T  # 转置

    # MLP gate proj
    elif "experts.gate_proj" in name:
        name = name.replace("experts.gate_proj", "experts.gate_proj")
        param = param.T

    # MLP up proj
    elif "experts.up_proj" in name:
        name = name.replace("experts.up_proj", "experts.up_proj")
        param = param.T

    # Router
    elif "router" in name:
        name = name.replace("router", "gate")

    return [(name, param)]
```

---

### 3.2 权重同步机制 (`update_weight/`)

Slime 实现了高效的训练→推理引擎权重同步：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         权重同步流程                                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│ Megatron 训练模型   │
│ (分布式 TP/EP/PP)   │
└──────────┬──────────┘
           │
           │ 1. AllGather TP 分片
           ▼
┌─────────────────────┐
│ 完整参数 (TP rank 0)│
└──────────┬──────────┘
           │
           │ 2. AllGather EP 分片 (仅 MoE)
           ▼
┌─────────────────────┐
│ 完整参数 (PP rank 0)│
└──────────┬──────────┘
           │
           │ 3. Megatron → HF 名称转换
           ▼
┌─────────────────────┐
│ HF 格式参数         │
└──────────┬──────────┘
           │
           │ 4. NCCL Broadcast 到 SGLang
           ▼
┌─────────────────────┐
│ SGLang 推理引擎     │
└─────────────────────┘
```

**核心实现：**

```python
class UpdateWeightFromDistributed:
    def update_weights(self):
        self.weight_version += 1

        # 1. 暂停 SGLang 生成
        ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])

        # 2. 同步非专家参数 (TP)
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." not in name:
                param = all_gather_param(name, param)  # AllGather TP
                converted = convert_to_hf(name, param)
                # Broadcast 到 SGLang

        # 3. 同步专家参数 (EP)
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." in name:
                param = all_gather_param(name, param)  # AllGather TP
                # AllGather EP
                converted = convert_to_hf(name, param)
                # Broadcast 到 SGLang

        # 4. 恢复 SGLang 生成
        ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
```

---

### 3.3 数据加载改造 (`data.py`)

Slime 为 RL 训练重写了数据加载逻辑：

**Context Parallelism 支持：**

```python
def get_batch(data_iterator, keys, ...):
    tokens = data_iterator.get_next(keys)["tokens"]

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    # THD 格式 (推荐)
    if qkv_format == "thd":
        # 切片 tokens 到 CP 分片
        tokens = [slice_with_cp(t, ...) for t in tokens]
        tokens = torch.cat(tokens)

        # 构建 cu_seqlens
        cu_seqlens = torch.tensor(cu_seqlens_list).cuda() * cp_size

        # 创建 PackedSeqParams
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            qkv_format="thd",
        )

    return {"tokens": tokens, "packed_seq_params": packed_seq_params, ...}
```

**动态 Batch Size：**

```python
def get_data_iterator(args, model, rollout_data):
    if args.use_dynamic_batch_size:
        # 基于 token 数量计算最优 micro-batch 数量
        num_microbatches = [
            get_minimum_num_micro_batch_size(samples, args.max_tokens_per_gpu)
            for samples in rollout_data["total_lengths"]
        ]

        # 跨 DP rank 同步
        dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=dp_group)

        # 均衡序列长度
        micro_batch_indices = get_seqlen_balanced_partitions(samples, num_mbs)

    return data_iterator, num_microbatches
```

---

### 3.4 损失函数改造 (`loss.py`)

Slime 实现了完整的 RL 损失计算：

```python
def compute_advantages_and_returns(args, rollout_data):
    if args.advantage_estimator == "grpo":
        # GRPO: Group Relative Policy Optimization
        advantages = get_grpo_returns(rewards, ...)
    elif args.advantage_estimator == "gae":
        # GAE: Generalized Advantage Estimation
        advantages, returns = get_advantages_and_returns_batch(values, rewards, ...)
    elif args.advantage_estimator == "reinforce":
        # REINFORCE with baseline
        advantages = get_reinforce_plus_plus_baseline_advantages(...)

def compute_policy_loss(args, batch, num_microbatches):
    # PPO Clip Loss
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip_high)

    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    # KL Loss (可选)
    if args.use_kl_loss:
        kl_loss = compute_approx_kl(log_probs, ref_log_probs)
        policy_loss += args.kl_loss_coef * kl_loss

    # Entropy Bonus (可选)
    if args.entropy_coef > 0:
        entropy = calculate_log_probs_and_entropy(...)[1]
        policy_loss -= args.entropy_coef * entropy

    return policy_loss
```

---

## 四、API 调用映射表

| 功能 | Megatron-LM 原生 API | Slime 封装/改造 |
|------|---------------------|-----------------|
| 分布式初始化 | `mpu.initialize_model_parallel()` | `initialize.init()` |
| 模型构建 | `GPTModel()` | `model_provider.get_model_provider_func()` |
| 优化器 | `get_megatron_optimizer()` | `model.setup_model_and_optimizer()` |
| 检查点加载 | `load_checkpoint()` | `checkpoint._load_checkpoint_megatron()` / `_load_checkpoint_hf()` |
| 检查点保存 | `save_checkpoint()` | `checkpoint.save_checkpoint()` |
| 前向后向 | `get_forward_backward_func()` | `model.forward_only()` / `train()` |
| 数据并行 | `DDP` | `model.train_one_step()` |
| TP 张量操作 | `tensor_parallel.scatter/gather()` | `update_weight.all_gather_param()` |
| 模型并行组 | `mpu.get_*_parallel_group()` | 直接使用 |

---

## 五、改造总结

### 5.1 核心改造清单

| 类别 | 改造项 | 目的 |
|------|--------|------|
| **模型支持** | HuggingFace 检查点加载 | 支持主流预训练模型 |
| **模型支持** | Megatron→HF 权重转换 | 与 SGLang 引擎兼容 |
| **训练模式** | Rollout-based 循环 | 适配 RL 训练范式 |
| **训练模式** | PPO/GRPO 损失函数 | 强化学习策略优化 |
| **训练模式** | 动态 Batch Size | 优化 GPU 利用率 |
| **分布式** | Context Parallelism | 支持超长序列训练 |
| **分布式** | EP 权重同步 | 支持 MoE 模型 |
| **内存** | 权重 CPU 备份 | 支持训练/推理内存共享 |
| **扩展性** | 自定义钩子 | 支持用户自定义扩展 |

### 5.2 设计原则

1. **最小侵入**：尽量使用 Megatron 公开 API，避免修改源码
2. **格式兼容**：完整支持 HuggingFace 格式的输入输出
3. **高效同步**：优化训练→推理引擎的权重传输
4. **模块化**：每个模型类型独立转换器，便于扩展

### 5.3 依赖关系

```
slime/backends/megatron_utils/
├── initialize.py    ──依赖──▶ megatron.core.mpu
├── model_provider.py ──依赖──▶ megatron.core.models.gpt
├── model.py         ──依赖──▶ megatron.core.optimizer
│                         megatron.core.pipeline_parallel
├── checkpoint.py    ──依赖──▶ megatron.training.checkpointing
│                         megatron.bridge
├── loss.py          ──依赖──▶ slime.utils.ppo_utils (自研)
├── data.py          ──依赖──▶ megatron.core.packed_seq_params
└── update_weight/   ──依赖──▶ slime.backends.megatron_to_hf (自研)
```