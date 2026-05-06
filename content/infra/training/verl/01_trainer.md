# Trainer 模块

**路径**：`verl/trainer/`

训练模块是整个框架的算法编排层，运行在单个 Driver 进程上，通过 Ray RPC 调用各 Worker 完成分布式计算。

---

## 1. 入口文件

| 文件 | 用途 |
|------|------|
| `main_ppo.py` | PPO 系列算法主入口，负责解析 Hydra 配置、初始化模型/tokenizer/数据集，然后调用 `RayPPOTrainer` |
| `main_ppo_sync.py` | 严格同步版主入口，适用于调试 |
| `main_eval.py` | 纯评估入口，不做训练更新 |
| `main_generation_server.py` | 启动生成服务（离线批量推理）|
| `sft_trainer.py` | 单机 SFT（监督微调）trainer |
| `sft_trainer_ray.py` | Ray 分布式 SFT trainer |

`main_ppo.py` 的典型初始化流程：
```python
# 1. 读取 Hydra config（YAML）
config = OmegaConf.load(...)

# 2. 初始化 tokenizer / processor
tokenizer = AutoTokenizer.from_pretrained(config.model.path)

# 3. 创建数据集（RL Dataset）
train_dataset = create_rl_dataset(config.data.train_files, ...)

# 4. 配置 role_worker_mapping（哪个 Role 用哪个 Worker 类）
role_worker_mapping = {
    Role.ActorRollout: ActorRolloutWorker,
    Role.Critic: CriticWorker,
    ...
}

# 5. 创建并运行 Trainer
trainer = RayPPOTrainer(config, tokenizer, role_worker_mapping, ...)
trainer.init_workers()
trainer.fit()
```

---

## 2. `RayPPOTrainer`

**文件**：`verl/trainer/ppo/ray_trainer.py`

### 2.1 初始化

```python
class RayPPOTrainer:
    def __init__(
        self,
        config,                       # Hydra OmegaConf 配置
        tokenizer,
        role_worker_mapping,          # {Role: WorkerClass}
        resource_pool_manager,        # Ray 资源池管理器
        ray_worker_group_cls,         # RayWorkerGroup
        train_dataset, val_dataset,
        collate_fn, train_sampler,
    ):
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine  # 默认 True
        self.use_reference_policy = need_reference_policy(config)
        self.use_critic = need_critic(config)
        self.use_rm = need_reward_model(config)
        self.ref_in_actor = lora_rank > 0  # LoRA 模式下 ref 和 actor 共享权重
        ...
```

关键布尔标志（由 `utils.py` 中的辅助函数根据 `config.algorithm.adv_estimator` 推断）：
- `use_critic`：GAE 需要 Critic；GRPO/REINFORCE 不需要
- `use_reference_policy`：`use_kl_in_reward=True` 时需要
- `use_rm`：有外部奖励模型时需要

### 2.2 `init_workers()`

按 `role_worker_mapping` 为每个 Role 创建 `RayWorkerGroup`，并绑定到对应的 ResourcePool：

```python
def init_workers(self):
    # 1. 为每个 ResourcePool 收集该 pool 上的所有 Role
    for resource_pool, class_dict in self.resource_pool_to_cls.items():
        # 2. 将多个 Role 打包到 ColocatedWorker（允许 Actor+Rollout 同卡）
        worker_dict_cls = create_colocated_worker_cls(class_dict)
        wg = RayWorkerGroup(resource_pool, worker_dict_cls)
        spawn_wg = wg.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)

    # 3. 保存各 WorkerGroup 引用
    self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
    self.critic_wg = all_wg[str(Role.Critic)]
    ...
```

### 2.3 `fit()` —— 训练主循环

**完整数据流**（见 `ray_trainer.py:1290`）：

```python
def fit(self):
    for epoch in range(total_epochs):
        for batch_dict in self.train_dataloader:
            batch = DataProto.from_single_dict(batch_dict)

            # ① Rollout：生成 token 序列
            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

            # ② Reward：计算奖励分数
            batch_reward = self._compute_reward_colocate(batch)
            reward_tensor, reward_extra_infos_dict = extract_reward(batch)

            # ③ Old log_prob：重计算旧策略 log_prob（PPO 需要）
            old_log_prob = self._compute_old_log_prob(batch)

            # ④ Ref log_prob：参考策略 log_prob（KL 惩罚需要）
            if self.use_reference_policy:
                ref_log_prob = self._compute_ref_log_prob(batch)

            # ⑤ Values：价值估计（GAE 需要）
            if self.use_critic:
                values = self._compute_values(batch)

            # ⑥ Advantage：在 Driver 本地计算优势（轻量操作）
            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl)
            batch = compute_advantage(batch, adv_estimator, gamma, lam)

            # ⑦ Update Critic
            if self.use_critic:
                critic_output = self._update_critic(batch)

            # ⑧ Update Actor
            actor_output = self._update_actor(batch)

            # ⑨ 权重同步：将新训练的权重同步给 Rollout 引擎
            self.checkpoint_manager.update_weights(self.global_steps)
```

### 2.4 KL 惩罚

```python
def apply_kl_penalty(data: DataProto, kl_ctrl, kl_penalty="kl"):
    kld = core_algos.kl_penalty(old_log_probs, ref_log_prob, kl_penalty)
    beta = kl_ctrl.value
    token_level_rewards = token_level_scores - beta * kld
    kl_ctrl.update(current_kl, n_steps=batch_size)
```

两种 KL 控制器（`FixedKLController` / `AdaptiveKLController`）：
- Fixed：`beta` 固定不变
- Adaptive：根据当前 KL 与目标 KL 的比值，按比例调整 `beta`

---

## 3. `core_algos.py` —— 算法核心

### 3.1 Advantage Estimator 注册机制

使用装饰器注册，支持扩展：
```python
ADV_ESTIMATOR_REGISTRY: dict[str, Callable] = {}

@register_adv_est(AdvantageEstimator.GAE)
def compute_gae_advantage_return(...):
    ...

@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(...):
    ...
```

同理，Policy Loss 也有注册机制：
```python
POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}

@register_policy_loss("ppo_clip")
def ppo_clip_loss(old_log_prob, log_prob, advantages, response_mask, ...):
    ...
```

### 3.2 GAE（Generalized Advantage Estimation）

```python
def compute_gae_advantage_return(
    token_level_rewards,   # (bs, response_len)
    values,                # (bs, response_len)
    response_mask,         # (bs, response_len)  -- EOS 之后为 0
    gamma,                 # 折扣因子
    lam,                   # GAE lambda
):
    # 反向遍历时间步
    for t in reversed(range(gen_len)):
        delta = r_t + gamma * V_{t+1} - V_t
        A_t = delta + gamma * lam * A_{t+1}
    returns = advantages + values
    advantages = masked_whiten(advantages, response_mask)  # 标准化
```

### 3.3 GRPO（Group Relative Policy Optimization）

```python
def compute_grpo_outcome_advantage(
    token_level_rewards,   # (bs, response_len)，只在 EOS 位置有值
    response_mask,
    index,                 # uid 分组标识 (bs,)
    norm_adv_by_std_in_grpo=True,
):
    scores = token_level_rewards.sum(dim=-1)  # 每条回答的总分

    # 按 uid 分组，计算组内均值和标准差
    for idx in groups:
        mean = mean(scores_in_group)
        std  = std(scores_in_group)

    # 标准化：A_i = (r_i - mean_group) / (std_group + eps)
    advantages = normalized_scores.unsqueeze(-1) * response_mask
```

**Dr.GRPO 变体**：设置 `norm_adv_by_std_in_grpo=False`，只减均值不除以标准差，避免方差为 0 时的梯度爆炸。

### 3.4 其他 Advantage Estimators

| 名称 | 核心思想 |
|------|---------|
| `reinforce_plus_plus` | 使用 token-level reward 减均值作为优势，无 Critic |
| `reinforce_plus_plus_baseline` | 类似 GRPO，但使用批内全局均值作为 baseline |
| `rloo` | Leave-One-Out 估计，组内其他样本平均作为 baseline |
| `remax` | 以贪心解码的分数作为 baseline |
| `opo` | On-Policy Optimization，直接用 rollout log_prob 作为 old_log_prob |
| `gdpo` | Group reward-Decoupled normalization，对每个奖励维度分别归一化再加权 |
| `optimal_token_baseline` | 基于 token 级别最优 baseline，利用 `sum_pi_squared` |

### 3.5 Policy Loss（`agg_loss`）

```python
def agg_loss(loss_mat, loss_mask, loss_agg_mode, loss_scale_factor):
    """支持多种 loss 聚合模式：
    - token_mean：对所有有效 token 取平均
    - seq_mean_token_sum：先对每条序列的 token 求和，再取序列均值
    - seq_mean：序列级别均值
    """
```

PPO clip loss 核心：
```python
ratio = exp(log_prob - old_log_prob)         # importance ratio
clipped = clip(ratio, 1-eps, 1+eps) * adv
loss = -min(ratio * adv, clipped)            # PPO clip
```

---

## 4. `reward.py` —— Reward 提取

```python
def extract_reward(batch: DataProto) -> tuple[torch.Tensor, dict]:
    """从 batch 中提取 token_level_scores 和额外奖励信息"""
    reward_tensor = batch.batch["token_level_scores"]
    reward_extra_infos_dict = {
        k: v for k, v in batch.non_tensor_batch.items()
        if k.startswith("reward_")
    }
    return reward_tensor, reward_extra_infos_dict
```

---

## 5. 配置结构 (`trainer/config/`)

Hydra YAML 配置层级：

```
config/
├── actor/        # ActorConfig：optimizer, ppo_epochs, loss_agg_mode, use_remove_padding...
├── critic/       # CriticConfig：ppo_infer_max_token_len_per_gpu...
├── rollout/      # RolloutConfig：n（采样数）, temperature, tensor_model_parallel_size...
├── algorithm/    # AlgoConfig：adv_estimator, gamma, lam, kl_ctrl, use_kl_in_reward...
├── data/         # 数据集路径、max_prompt_length、max_response_length...
├── model/        # 模型路径、trust_remote_code...
├── reward/       # 奖励模型配置
├── ref/          # 参考策略配置
└── engine/       # 引擎选择（fsdp/megatron）、并行度配置
```

---

## 6. 知识蒸馏（`trainer/distillation/`）

支持将教师模型的 logits 作为软标签训练学生模型：
- `fsdp/`：FSDP 后端蒸馏
- `megatron/`：Megatron 后端蒸馏
- 损失函数包括 KL 散度（前向 / 反向）、SFT 交叉熵等
