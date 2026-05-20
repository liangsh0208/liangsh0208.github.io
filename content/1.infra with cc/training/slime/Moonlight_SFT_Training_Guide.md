---
created: 2026-05-06
---

# Moonlight SFT 操作流程与性能优化分析

> 本文档详细分析 Moonlight 模型在 Slime 框架中进行 SFT (Supervised Fine-Tuning) 训练的操作流程、核心实现以及性能优化策略

---

## 目录

1. [SFT 概述与特性](#1-sft-概述与特性)
2. [操作流程详解](#2-操作流程详解)
3. [核心实现分析](#3-核心实现分析)
4. [性能优化策略](#4-性能优化策略)
5. [Moonlight 模型配置](#5-moonlight-模型配置)
6. [最佳实践](#6-最佳实践)

---

## 1. SFT 概述与特性

### 1.1 Slime SFT 与传统 SFT 的区别

| 特性 | 传统 SFT 框架 | Slime SFT |
|------|--------------|-----------|
| 数据流 | 需要 rollout 生成 | 直接使用标注数据 |
| 损失计算 | 仅 response 部分 | 支持 multi-turn loss mask |
| 分布式训练 | DDP/FSDP | Megatron-LM (TP/PP/CP/EP) |
| 动态 batch | 通常固定 | 支持动态 batch 均衡 |
| 视为 RL 的特例 | 无 | 是 (无 reward 的 rollout) |

### 1.2 SFT 在 Slime 中的定位

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Slime 训练模式对比                                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    SFT      │     │   GRPO/PPO  │     │  OPD/蒸馏   │
│             │     │             │     │             │
│ - 无生成    │     │ - 需要生成  │     │ - 需要生成  │
│ - 固定数据  │     │ - 动态数据  │     │ - 教师模型  │
│ - sft_loss  │     │ - ppo_loss  │     │ - kl_loss   │
│ - 无 reward │     │ - 需要 RM   │     │ + sft_loss  │
└─────────────┘     └─────────────┘     └─────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                   共享训练基础设施:
                   - Megatron-LM 后端
                   - 分布式并行策略
                   - Checkpoint 管理
                   - 动态 batch 支持
```

---

## 2. 操作流程详解

### 2.1 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Slime SFT 训练架构                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                            Ray Cluster                                    │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    Placement Group                                   │  │
│  │                                                                     │  │
│  │   ┌──────────────────────────────────────────────────────────────┐  │  │
│  │   │              Training Actors (8 GPUs)                        │  │  │
│  │   │                                                              │  │  │
│  │   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │  │  │
│  │   │  │Actor 0  │ │Actor 1  │ │Actor 2  │ │Actor 3  │ TP=4      │  │  │
│  │   │  │(rank 0) │ │(rank 1) │ │(rank 2) │ │(rank 3) │            │  │  │
│  │   │  └─────────┘ └─────────┘ └─────────┘ └─────────┘            │  │  │
│  │   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │  │  │
│  │   │  │Actor 4  │ │Actor 5  │ │Actor 6  │ │Actor 7  │            │  │  │
│  │   │  │(rank 4) │ │(rank 5) │ │(rank 6) │ │(rank 7) │            │  │  │
│  │   │  └─────────┘ └─────────┘ └─────────┘ └─────────┘            │  │  │
│  │   │                                                              │  │  │
│  │   │  MoE: EP=8 (专家并行)                                        │  │  │
│  │   └──────────────────────────────────────────────────────────────┘  │  │
│  │                                                                     │  │
│  │   SFT 模式: 无需 Rollout 引擎 (debug-train-only)                   │  │
│  │                                                                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 启动流程

```bash
# Step 1: 环境准备
export PYTHONBUFFERED=16
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export NCCL_NVLS_ENABLE=1  # 如果有 NVLink

# Step 2: 启动 Ray 集群
ray start --head \
    --node-ip-address ${MASTER_ADDR} \
    --num-gpus 8 \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

# Step 3: 提交训练任务
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{
        "env_vars": {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1"
        }
    }' \
    -- python3 train_async.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${SFT_ARGS[@]} \
    ...
```

### 2.3 训练主流程

```python
# train_async.py 核心流程

def train(args):
    # ========== 阶段 1: 初始化 ==========
    configure_logger()
    pgs = create_placement_groups(args)      # GPU资源分配
    init_tracking(args)                      # WandB/TensorBoard

    # ========== 阶段 2: 创建模型 ==========
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # ========== 阶段 3: 异步训练循环 ==========
    # 预取第一批数据
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # 1. 获取当前数据
        rollout_data_curr_ref = ray.get(rollout_data_next_future)

        # 2. 预取下一批数据 (异步训练关键)
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        # 3. 训练
        ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

        # 4. 保存检查点
        if should_run_periodic_action(rollout_id, args.save_interval, ...):
            actor_model.save_model(rollout_id)

        # 5. 评估 (可选)
        if should_run_periodic_action(rollout_id, args.eval_interval, ...):
            ray.get(rollout_manager.eval.remote(rollout_id))

    # ========== 阶段 4: 清理 ==========
    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)
```

### 2.4 SFT 数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SFT 数据流                                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐
│ 训练数据集   │  openhermes2_5.parquet
│ (JSONL/     │
│  Parquet)   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Dataset (utils/data.py)                                  │
│                                                                             │
│  1. read_file() - 读取文件                                                  │
│  2. _build_messages() - 构建对话格式                                        │
│  3. (可选) apply_chat_template - 应用模板                                   │
│  4. filter_long_prompt() - 过滤超长样本                                     │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       │ Sample(prompt=messages, label=..., metadata=...)
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               SFT Rollout (rollout/sft_rollout.py)                          │
│                                                                             │
│  generate_rollout(args, rollout_id, data_buffer):                          │
│                                                                             │
│  for sample in samples:                                                     │
│      messages = sample.prompt                                               │
│      tools = sample.metadata.get("tools")                                  │
│                                                                             │
│      # 核心: 生成 token_ids 和 loss_mask                                   │
│      token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(                  │
│          messages, tools=tools                                              │
│      )                                                                      │
│                                                                             │
│      response_length = get_response_lengths([loss_mask])[0]                │
│                                                                             │
│      sample.tokens = token_ids                                              │
│      sample.response_length = response_length                               │
│      sample.reward = 0  # SFT 无 reward                                    │
│      sample.loss_mask = loss_mask[-response_length:]                       │
│                                                                             │
│  return samples                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       │ Sample(tokens=[...], loss_mask=[...], response_length=N)
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    训练数据处理                                              │
│                                                                             │
│  1. process_rollout_data() - DP分片                                        │
│  2. get_data_iterator() - 构建迭代器 (动态batch均衡)                        │
│  3. get_batch() - 构建训练批次                                              │
│     - PackedSeqParams for FlashAttention                                   │
│     - CP切分 (如果启用)                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SFT Loss 计算 (loss.py)                                   │
│                                                                             │
│  sft_loss_function(args, batch, logits, sum_of_sample_mean):               │
│                                                                             │
│      # 1. 计算 log_probs                                                   │
│      log_probs = get_log_probs_and_entropy(                                │
│          logits, tokens, response_lengths...                                │
│      )                                                                      │
│                                                                             │
│      # 2. 计算 NLL loss                                                    │
│      loss = -sum_of_sample_mean(log_probs)                                 │
│                                                                             │
│      # 3. 按元素平均或按样本平均                                            │
│      # calculate_per_token_loss: True → 按token平均                        │
│      # calculate_per_token_loss: False → 按样本平均                        │
│                                                                             │
│      return loss, {"loss": loss}                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心实现分析

### 3.1 多轮对话 Loss Mask 生成

**文件**: `utils/mask_utils.py`

```python
class MultiTurnLossMaskGenerator:
    """
    多轮对话 loss mask 生成器

    核心功能:
    - 为 assistant 的回复生成 loss_mask=1
    - 为 user/system 消息生成 loss_mask=0
    - 支持工具调用 (tools) 场景
    """

    def __init__(self, tokenizer, tokenizer_type="qwen"):
        self.tokenizer = tokenizer
        # 计算系统消息长度和生成token长度
        self.system_message_length, self.gen_token_length = self.get_system_message_length()
        self.tokenizer_type = tokenizer_type

    def get_system_message_length(self) -> tuple[int, int]:
        """
        计算 chat template 中的系统消息长度

        通过测试字符串比较计算:
        - system_message_length: 系统消息模板的token长度
        - gen_token_length: 生成开始标记的token长度
        """
        test_string = "FOR TESTING ONLY"
        test_messages = [
            {"role": "user", "content": test_string},
            {"role": "user", "content": test_string},
        ]

        # 获取原始token
        raw_token_ids = self.tokenizer(test_string, add_special_tokens=False)["input_ids"]

        # 获取带模板的token
        chat_template_token = self.tokenizer.apply_chat_template(
            test_messages, add_special_tokens=False, tokenize=False
        )
        chat_template_token_ids = self.tokenizer(chat_template_token, add_special_tokens=False)["input_ids"]

        # 找到两个 test_string 在模板中的位置
        idx_1, idx_2 = self.find_all_sublist_indices(chat_template_token_ids, raw_token_ids)

        # 计算系统消息长度
        end_interval = len(chat_template_token_ids) - len(raw_token_ids) - idx_2
        system_message_length = idx_1 - ((idx_2 - idx_1) - end_interval - len(raw_token_ids))

        # 计算生成token长度 (如 <|assistant|> 标记)
        gen_token_length = len(
            self.tokenizer.apply_chat_template(
                test_messages, add_generation_prompt=True, tokenize=True
            )
        ) - len(chat_template_token_ids)

        return system_message_length, gen_token_length

    def gen_multi_turn_loss_mask_qwen(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        """
        为 Qwen 模型生成多轮对话 loss mask

        示例输入:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well!"},
        ]

        示例输出:
        token_ids = [151644, 8948, ..., 151645, ...]  # 完整对话token
        loss_mask = [0, 0, ..., 0, 1, 1, 1, ..., 0, 0, ..., 1, 1, ...]  # 只有assistant部分为1
        """
        all_loss_masks = []
        all_token_ids = []

        for i, message in enumerate(messages):
            # 生成当前消息的token
            if i == 0:
                message_ids = self.tokenizer.apply_chat_template(
                    [message], tokenize=True, tools=tools, return_dict=False
                )
            else:
                message_ids = self.tokenizer.apply_chat_template(
                    [message], tokenize=True, return_dict=False
                )

            # 去除系统消息前缀 (非首条消息)
            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length:]

            # 生成 loss mask
            if message["role"] == "assistant":
                # assistant 消息: 前面gen_token_length个token为0，其余为1
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                # user/system 消息: 全部为0
                loss_mask = [0] * len(message_ids)

            # 支持单条消息级别的 loss mask 控制
            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks
```

### 3.2 SFT Loss 函数

**文件**: `backends/megatron_utils/loss.py`

```python
def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable,
) -> tuple[torch.Tensor, dict]:
    """
    SFT 损失函数

    计算: loss = -mean(log_prob(t)) for t in response_tokens

    Args:
        args: 配置参数
        batch: 包含 tokens, loss_masks, response_lengths 等
        logits: 模型输出 [1, T, V]
        sum_of_sample_mean: 归约函数

    Returns:
        loss: 标量损失值
        metrics: {"loss": loss_value}
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    # 1. 计算 log_probs
    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
    )

    # 2. 拼接所有样本的 log_probs
    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)

    # 3. 计算负对数似然
    loss = -sum_of_sample_mean(log_probs)

    # 4. 确保梯度正确传播 (空序列情况)
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return loss, {"loss": loss.clone().detach()}
```

### 3.3 Loss 类型分发

```python
def loss_function(args, batch, num_microbatches, logits):
    """
    损失函数分发器
    """
    # 计算归约函数
    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,  # 关键参数
        args.qkv_format,
    )

    # 分发到具体损失函数
    match args.loss_type:
        case "policy_loss":
            func = policy_loss_function
        case "value_loss":
            func = value_loss_function
        case "sft_loss":
            func = sft_loss_function
        case "custom_loss":
            func = load_function(args.custom_loss_function_path)

    # 可选: 梯度检查点重计算
    if args.recompute_loss_function:
        loss, log = checkpoint(func, args, batch, logits, sum_of_sample_mean)
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # 损失缩放 (与Megatron集成)
    if not args.calculate_per_token_loss:
        # 按样本平均: loss = loss * num_microbatches / global_batch_size * dp_size
        loss = loss * num_microbatches / args.global_batch_size * dp_size
    else:
        # 按token平均: loss = loss * cp_size
        loss = loss * mpu.get_context_parallel_world_size()

    return loss, normalizer, logging_dict
```

### 3.4 数据迭代器与动态 Batch

```python
def get_data_iterator(args, model, rollout_data):
    """
    构建数据迭代器

    支持两种模式:
    1. 固定 micro_batch_size
    2. 动态 batch (基于 max_tokens_per_gpu)
    """
    if not args.use_dynamic_batch_size:
        # 固定模式
        num_microbatches = [num_local_gbs // args.micro_batch_size]
        data_iterator = DataIterator(rollout_data, args.micro_batch_size)
    else:
        # 动态模式: 计算每个step的micro-batch数量
        for i in range(num_steps_per_rollout):
            samples = rollout_data["total_lengths"][start:end]
            num_mb = get_minimum_num_micro_batch_size(
                samples, args.max_tokens_per_gpu * cp_size
            )
            num_microbatches.append(num_mb)

        # AllReduce 取最大值 (DP同步)
        dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX)

        # 序列长度均衡
        partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)
        micro_batch_indices = [...]

        data_iterator = DataIterator(rollout_data, None, micro_batch_indices)

    return data_iterator, num_microbatches
```

---

## 4. 性能优化策略

### 4.1 并行策略配置

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Moonlight SFT 并行策略                                    │
└─────────────────────────────────────────────────────────────────────────────┘

Moonlight-16B-A3B 模型特点:
- 27层 Transformer
- 64个 MoE 专家, TopK=6
- MLA (Multi-Latent Attention)
- 2个共享专家

推荐配置:
┌────────────────────────────────────────────────────────────────────────────┐
│ --tensor-model-parallel-size 4     # 张量并行                              │
│ --sequence-parallel                # 序列并行 (配合TP)                     │
│ --expert-model-parallel-size 8     # 专家并行 (64专家 / 8 = 8专家/张量组)  │
│ --expert-tensor-parallel-size 1    # 专家内部TP                             │
│ --pipeline-model-parallel-size 1   # 流水线并行                             │
│ --context-parallel-size 1          # 上下文并行                             │
└────────────────────────────────────────────────────────────────────────────┘

GPU分配示意 (8 GPU):
┌────────────────────────────────────────────────────────────────────────────┐
│  GPU 0-3: TP Group 0                                                      │
│    - 模型参数分片 (1/4)                                                    │
│    - 专家: 0-7 (EP维度)                                                    │
│                                                                            │
│  GPU 4-7: TP Group 1                                                      │
│    - 模型参数分片 (1/4)                                                    │
│    - 专家: 8-15 (EP维度)                                                   │
│                                                                            │
│  实际上 EP=8, 每个 GPU 负责 64/8 = 8 个专家                               │
└────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 内存优化

```bash
# ========== 激活重计算 ==========
--recompute-granularity full          # 完整层重计算
--recompute-method uniform            # 均匀分布
--recompute-num-layers 1              # 重计算1层

# 效果: 以计算换内存，减少约 30-50% 激活内存

# ========== 动态 Batch ==========
--use-dynamic-batch-size
--max-tokens-per-gpu 8192             # 每 GPU 最大 token 数

# 效果: 根据 max_tokens 动态调整 micro-batch 数量
#       平衡 GPU 内存利用率

# ========== MLA 优化 ==========
--multi-latent-attention              # 使用 MLA
--kv-lora-rank 512                    # KV 压缩维度
--qk-head-dim 128                     # QK 头维度
--v-head-dim 128                      # V 头维度

# 效果: KV Cache 压缩，减少显存占用
```

### 4.3 计算优化

```bash
# ========== 精度控制 ==========
--accumulate-allreduce-grads-in-fp32  # 梯度累加用 FP32
--attention-softmax-in-fp32           # Attention Softmax 用 FP32

# 效果: 保持数值稳定性

# ========== DeepEP 优化 (MoE) ==========
--moe-enable-deepep                   # 启用 DeepEP
--moe-token-dispatcher-type flex      # 灵活的 token 分发

# 效果: 优化 MoE 专家间通信

# ========== 分组 GEMM ==========
--moe-grouped-gemm                    # MoE 分组 GEMM

# 效果: 批量矩阵乘法，提升 MoE 效率
```

### 4.4 异步训练流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         异步训练时序图                                       │
└─────────────────────────────────────────────────────────────────────────────┘

时间 ──────────────────────────────────────────────────────────────────────▶

Rollout Manager:
├─────────────────┐                 ┌─────────────────┐
│ Generate Data 0 │ ...             │ Generate Data N │
└────────┬────────┘                 └─────────────────┘
         │
         ▼ prefetch
┌─────────┐
│ Data 1  │─────────────────────────────────────────────┐
└─────────┘                                             │
                                                        ▼
Training Actor:                                   ┌─────────────────┐
┌─────────────────┐     ┌─────────────────┐      │ Train Step N    │
│ Train Step 0    │     │ Train Step 1    │ ...  └─────────────────┘
└─────────────────┘     └─────────────────┘

重叠计算:
- 当 Step 0 训练时, Step 1 的数据在后台预取
- 减少数据加载等待时间

关键代码 (train_async.py):
┌────────────────────────────────────────────────────────────────────────────┐
│ # 预取第一批                                                               │
│ rollout_data_next_future = rollout_manager.generate.remote(start_id)      │
│                                                                            │
│ for rollout_id in range(num_rollout):                                     │
│     # 同步获取当前数据                                                     │
│     rollout_data_curr = ray.get(rollout_data_next_future)                 │
│                                                                            │
│     # 异步预取下一批数据                                                   │
│     if rollout_id + 1 < num_rollout:                                      │
│         rollout_data_next_future = rollout_manager.generate.remote(id+1)  │
│                                                                            │
│     # 训练 (同时数据在后台加载)                                            │
│     train_step(rollout_data_curr)                                         │
└────────────────────────────────────────────────────────────────────────────┘
```

### 4.5 检查点优化

```bash
# ========== 异步保存 ==========
--async-save                          # 异步保存检查点

# 效果: 保存检查点不阻塞训练

# ========== 仅训练模式 ==========
--debug-train-only                    # 跳过 rollout 生成

# SFT 专用: 不需要推理引擎，直接用标注数据
```

### 4.6 性能指标

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Moonlight-16B-A3B SFT 性能参考                           │
└─────────────────────────────────────────────────────────────────────────────┘

硬件配置: 8x H800 (80GB)

配置:
- TP=4, EP=8
- max_tokens_per_gpu=8192
- dynamic_batch_size

预期性能:
┌────────────────────────────┬──────────────────────────────────────────────┐
│ 指标                       │ 参考值                                       │
├────────────────────────────┼──────────────────────────────────────────────┤
│ 吞吐量                     │ ~5000 tokens/sec/GPU                         │
│ MFU (Model FLOPS Util)     │ ~45-55%                                      │
│ 内存利用率                 │ 85-90% (激活重计算后)                        │
│ SeqLen 4K 时 batch_size    │ ~8-16 samples/micro-batch                   │
└────────────────────────────┴──────────────────────────────────────────────┘

瓶颈分析:
1. MoE 通信: EP 间的 all-to-all 通信是主要瓶颈
2. MLA: KV Cache 压缩减少内存瓶颈
3. 动态 Batch: 减少填充浪费，提高计算效率
```

---

## 5. Moonlight 模型配置

### 5.1 模型架构参数

```bash
# moonlight.sh 核心配置
MODEL_ARGS=(
    # ========== 基础架构 ==========
    --num-layers 27
    --hidden-size 2048
    --ffn-hidden-size 11264           # Dense 层 FFN
    --num-attention-heads 16
    --kv-channels 128
    --vocab-size 163840

    # ========== Attention ==========
    --multi-latent-attention          # MLA 架构
    --kv-lora-rank 512                # KV 压缩维度
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --qk-layernorm

    # ========== RoPE ==========
    --position-embedding-type rope
    --rotary-base 50000
    --rotary-scaling-factor 1

    # ========== MoE 配置 ==========
    --num-experts 64
    --moe-layer-freq "[0,1,1,...,1]"  # 第一层Dense，其余MoE
    --moe-ffn-hidden-size 1408        # 每个专家的 FFN 维度
    --moe-router-topk 6               # 每次激活6个专家
    --moe-shared-expert-intermediate-size 2816  # 2个共享专家

    # ========== Router 配置 ==========
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-topk-scaling-factor 2.446
    --moe-token-dispatcher-type alltoall
)
```

### 5.2 MoE 结构分析

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Moonlight MoE 结构                                       │
└─────────────────────────────────────────────────────────────────────────────┘

Layer 0: Dense Layer (FFN)
├─────────────────────────────────────────────────────────────────┐
│ Input → Attention → FFN (11264) → Output                       │
└─────────────────────────────────────────────────────────────────┘

Layer 1-26: MoE Layer
├─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Shared Expert (2个)                                            │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Expert_0 (1408) + Expert_1 (1408) = 2816 总维度           ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Router (Sigmoid + TopK=6)                                     │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ 64 Experts, 每次选择 Top-6 激活                            ││
│  │ Expert FFN 维度: 1408                                      ││
│  │ 总激活参数: 6 × 1408 = 8448                                ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Output = Σ(router_score_i × Expert_i(x)) + Shared_Expert(x)   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

参数量计算:
- Dense Layer FFN: 2048 × 11264 × 2 ≈ 46M
- MoE Layer: 64 × 2048 × 1408 × 2 ≈ 368M (专家)
           + 2 × 2048 × 1408 × 2 ≈ 11.5M (共享专家)
           ≈ 380M per MoE Layer

Total: 46M + 26 × 380M ≈ 9.9B (激活)
       总参数 ≈ 16B
       稀疏激活 ≈ 3.3B (A3B = Active 3B)
```

---

## 6. 最佳实践

### 6.1 配置模板

```bash
#!/bin/bash
# run-moonlight-sft.sh

# ===== 模型配置 =====
source "${SCRIPT_DIR}/models/moonlight.sh"

# ===== 检查点配置 =====
CKPT_ARGS=(
   --hf-checkpoint /path/to/Moonlight-16B-A3B
   --load /path/to/checkpoint/
   --save /path/to/checkpoint/
   --save-interval 1000
)

# ===== SFT 专用配置 =====
SFT_ARGS=(
   # 核心配置
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only                    # 跳过推理引擎

   # 数据配置
   --prompt-data /path/to/data.parquet
   --input-key messages
   --rollout-shuffle
   --num-epoch 3
   --rollout-batch-size 128
   --global-batch-size 128

   # Loss mask 类型
   --loss-mask-type qwen                 # 或 qwen3, distill_qwen
)

# ===== 性能配置 =====
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

# ===== 优化器配置 =====
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
)

# ===== MoE 优化 =====
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --moe-enable-deepep
   --moe-token-dispatcher-type flex
)
```

### 6.2 性能调优指南

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         性能调优决策树                                       │
└─────────────────────────────────────────────────────────────────────────────┘

1. 内存不足 (OOM)
   │
   ├─ 启用激活重计算
   │  --recompute-granularity full
   │  --recompute-num-layers 1-4
   │
   ├─ 减小 max-tokens-per-gpu
   │  --max-tokens-per-gpu 4096 (从 8192 降低)
   │
   ├─ 增加张量并行
   │  --tensor-model-parallel-size 8 (从 4 增加)
   │
   └─ 启用专家并行
      --expert-model-parallel-size 增大

2. 训练速度慢
   │
   ├─ 检查 GPU 利用率
   │  nvidia-smi dmon -s u
   │  │
   │  ├─ 利用率低 → 增大 max-tokens-per-gpu
   │  └─ 利用率高 → 考虑其他瓶颈
   │
   ├─ 检查通信开销
   │  │
   │  ├─ MoE 通信多 →
   │  │  --moe-enable-deepep
   │  │  --moe-token-dispatcher-type flex
   │  │
   │  └─ TP 通信多 →
   │     减小 TP, 增加其他并行
   │
   └─ 使用异步训练
      python train_async.py

3. 收敛问题
   │
   ├─ 学习率调整
   │  --lr 5e-6 ~ 2e-5
   │  --lr-warmup-fraction 0.05 ~ 0.1
   │
   ├─ 梯度裁剪
   │  --clip-grad 1.0
   │
   └─ 精度问题
      --accumulate-allreduce-grads-in-fp32
      --attention-softmax-in-fp32
```

### 6.3 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| OOM during forward | 激活内存过大 | 启用 recompute |
| MoE 通信瓶颈 | EP=allreduce 效率低 | 使用 DeepEP + flex dispatcher |
| 利用率不均衡 | 动态 batch 未启用 | `--use-dynamic-batch-size` |
| Loss NaN | 梯度爆炸 | 降低 LR, 启用梯度裁剪 |
| 训练卡住 | NCCL 通信死锁 | 检查 NCCL 配置, 设置超时 |
| 收敛慢 | 学习率不当 | 调整 LR 和 warmup |

### 6.4 监控与调试

```bash
# GPU 利用率监控
watch -n 1 nvidia-smi

# 详细GPU使用
nvidia-smi dmon -s pucvmet -i 0,1,2,3,4,5,6,7

# NCCL 日志
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# PyTorch 分析
export TORCH_PROFILER=1
# 在代码中添加 torch.profiler

# Ray Dashboard
http://localhost:8265
```

---

## 附录: 关键文件速查

| 功能 | 文件路径 | 核心函数/类 |
|------|----------|-------------|
| SFT Rollout | `rollout/sft_rollout.py` | `generate_rollout()` |
| Loss Mask | `utils/mask_utils.py` | `MultiTurnLossMaskGenerator` |
| SFT Loss | `backends/megatron_utils/loss.py` | `sft_loss_function()` |
| 异步训练 | `train_async.py` | `train()` |
| Placement | `ray/placement_group.py` | `create_placement_groups()` |
| Actor Group | `ray/actor_group.py` | `RayTrainGroup` |
| Model Provider | `backends/megatron_utils/model_provider.py` | `get_model_provider_func()` |
| Moonlight 配置 | `scripts/models/moonlight.sh` | `MODEL_ARGS` |