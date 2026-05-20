---
created: 2026-05-09
---

# ms-swift RLHF 训练器：对齐算法实现

ms-swift 内置了丰富的偏好对齐训练算法，从经典的 DPO/PPO 到最新的 GRPO 家族（DAPO/GSPO/SAPO/CISPO 等）。所有 RLHF Trainer 继承自 `trl` 基类并通过 `SwiftMixin` + `RLHFTrainerMixin` 注入框架能力。

---

## 1. RLHF 训练器继承体系

```
HuggingFace trl
        │
        ├── DPOTrainer (trl) ──┬── RLHFTrainerMixin ──┬── SwiftMixin ──▶ DPOTrainer (swift)
        │                      │                        │
        ├── GRPOTrainer (trl) ─┤                        ├──▶ GRPOTrainer (swift)
        │                      │                        │
        ├── PPOTrainer (trl) ──┤                        ├──▶ PPOTrainer (swift)
        │                      │                        │
        ├── KTOTrainer (trl) ─┤                        ├──▶ KTOTrainer (swift)
        │                      │                        │
        ├── ORPOTrainer (trl) ─┤                        ├──▶ ORPOTrainer (swift)
        │                      │                        │
        ├── CPOTrainer (trl) ──┤                        ├──▶ CPOTrainer (swift)
        │                      │                        │
        └── RewardTrainer (trl)─┘                        └──▶ RewardTrainer (swift)

swift 特有:
- RolloutTrainerMixin ──▶ GRPOTrainer / PPOTrainer (rollout 生成加速)
- GKDTrainer (trl 无基类，独立实现)
```

---

## 2. 公共 Mixin 层

### 2.1 RLHFTrainerMixin

**文件**: `swift/rlhf_trainers/rlhf_mixin.py`

```python
class RLHFTrainerMixin:
    """RLHF 训练器公共能力"""
    
    def get_per_token_logps(self, logits, labels, label_pad_token_id=-100):
        """计算 per-token log probabilities"""
        log_probs = F.log_softmax(logits, dim=-1)
        # gather 对应 label 位置的 logprob
        per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
        # mask padding
        loss_mask = (labels != label_pad_token_id)
        return per_token_logps, loss_mask
    
    def concatenated_forward(self, model, batch):
        """将 chosen + rejected 拼接为单个 batch 前向传播"""
        ...
```

### 2.2 RolloutTrainerMixin

**文件**: `swift/rlhf_trainers/rollout_mixin.py` (~81KB)

专门服务于需要**从策略模型生成样本**的算法（GRPO、PPO）：

```python
class RolloutTrainerMixin:
    def prepare_rollout(self):
        """初始化 rollout 引擎"""
        if self.args.use_vllm:
            # vLLM 异步生成
            self.vllm_client = VllmClient(...)
        else:
            # TransformersEngine 生成
            self.engine = TransformersEngine(self.model, template=self.template)
    
    def generate_completions(self, prompts):
        """生成 completions 用于 rollout"""
        if self.args.use_vllm:
            return self.vllm_client.generate(prompts)
        else:
            return self.engine.infer(prompts)
```

---

## 3. DPO — Direct Preference Optimization

**文件**: `swift/rlhf_trainers/dpo_trainer.py` (~24KB)

### 3.1 多损失类型支持

```python
class DPOTrainer(RLHFTrainerMixin, SwiftMixin, DataLoaderMixin, HFDPOTrainer):
    def __init__(self, model, ref_model, ...):
        args = kwargs['args']
        self.label_smoothing = args.label_smoothing
        
        # 支持多种损失类型 (可组合使用)
        self.loss_type = args.loss_type  # sigmoid / ipo / hinge / bco_pair / nca_pair / apo_zero / apo_down / simpo / sppo_hard
        self.loss_weights = args.loss_weights  # 多 loss 加权
        
        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.f_divergence_type = getattr(args, 'f_divergence_type', 'reverse_kl')
```

### 3.2 核心损失计算

```python
def concatenated_forward(self, model, batch, is_ref_model=False):
    # 1. 拼接 chosen + rejected 序列
    # 2. 前向传播
    outputs = model(**batch, use_cache=False)
    all_logits = outputs.logits
    
    # 3. 计算 per-token logprobs
    per_token_logps, loss_mask = self.get_per_token_logps(all_logits, labels)
    
    # 4. 分离 chosen_logps 和 rejected_logps
    chosen_logps = per_token_logps[:chosen_len].sum()
    rejected_logps = per_token_logps[chosen_len:].sum()
    
    return chosen_logps, rejected_logps

def compute_loss(self, model, inputs, ...):
    policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(model, inputs)
    ref_chosen_logps, ref_rejected_logps = self.concatenated_forward(self.ref_model, inputs, is_ref_model=True)
    
    # 计算 log ratios
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = policy_logratios - ref_logratios
    
    # 根据 loss_type 计算最终损失
    if 'sigmoid' in loss_types:
        loss += -F.logsigmoid(self.args.beta * logits)
    if 'ipo' in loss_types:
        loss += (logits - 1 / (2 * self.args.beta)) ** 2
    if 'hinge' in loss_types:
        loss += torch.relu(1 - self.args.beta * logits)
    ...
    
    return loss
```

### 3.3 特殊变体

| 变体 | 实现 | 说明 |
|-----|------|------|
| IPO | `loss_type='ipo'` | Identity Preference Optimization |
| SimPO | `loss_type='simpo'` | Simple Preference Optimization |
| BCO Pair | `loss_type='bco_pair'` | Binary Classifier Optimization |
| APO Zero/Down | `loss_type='apo_zero'` | Advantage Preference Optimization |
| NCA Pair | `loss_type='nca_pair'` | Noise-contrastive Alignment |
| f-divergence | `f_divergence_type` | 前向/反向 KL、JS 散度等 |

---

## 4. GRPO — Group Relative Policy Optimization

**文件**: `swift/rlhf_trainers/grpo_trainer.py` (~135KB，框架最大 RLHF 文件)

GRPO 是 DeepSeek-R1 论文中验证有效的强化学习算法，ms-swift 对其做了深度扩展。

### 4.1 初始化流程

```python
class GRPOTrainer(RolloutTrainerMixin, SwiftMixin, HFGRPOTrainer):
    def __init__(self, model, ref_model, reward_model, reward_funcs, ...):
        # 1. 算法参数
        self._prepare_algorithm_params()  # num_generations, beta, etc.
        
        # 2. 父类初始化
        super().__init__(model, ref_model, ...)
        
        # 3. CHORD 数据集准备
        self._prepare_chord_dataset()
        
        # 4. Rollout 准备 (vLLM / Transformers)
        self.prepare_rollout()
        
        # 5. Reward 准备
        self._prepare_rewards(reward_funcs, reward_model)
        
        # 6. Liger Kernel 加速
        self._prepare_liger_loss()
        
        # 7. 评估条件检查
        assert len(self.eval_dataset) >= total_eval_batch_size
```

### 4.2 输入准备与生成

```python
def _prepare_inputs(self, generation_batch):
    """
    训练时:
    - 接收 generation batch (per_device_train_batch_size * steps_per_generation)
    - 生成 completions 一次，缓存后分多个 accumulation step 使用
    - 每 steps_per_generation * num_iterations 才重新生成
    
    评估时:
    - 不缓存，每个 batch 独立生成
    """
    if self.training and self._buffered_inputs is not None:
        return self._get_next_buffered_batch()
    
    # 生成 completions
    if self.args.use_vllm:
        completions = self.vllm_client.generate(generation_batch)
    else:
        completions = self.engine.infer(generation_batch)
    
    # 计算 advantage (Group Relative)
    rewards = self._compute_rewards(completions)
    advantages = self._compute_group_relative_advantages(rewards)
    
    # 组装训练 batch
    inputs = {
        'input_ids': ..., 
        'labels': ...,
        'advantages': advantages,
        'old_logps': old_logprobs,
    }
    self._buffered_inputs = inputs
    return inputs
```

### 4.3 GRPO Loss

```python
def compute_loss(self, model, inputs, ...):
    # 1. 前向得到当前策略的 logprobs
    outputs = model(**inputs)
    policy_logps = self.get_per_token_logps(outputs.logits, inputs['labels'])
    
    # 2. 参考模型的 logprobs (不计算梯度)
    with torch.no_grad():
        ref_outputs = self.ref_model(**inputs)
        ref_logps = self.get_per_token_logps(ref_outputs.logits, inputs['labels'])
    
    # 3. 比率裁剪 (PPO-like clipping)
    ratio = torch.exp(policy_logps - inputs['old_logps'])
    clipped_ratio = torch.clamp(ratio, 1 - self.args.epsilon, 1 + self.args.epsilon)
    
    # 4. GRPO 损失
    advantages = inputs['advantages']
    loss1 = -advantages * ratio
    loss2 = -advantages * clipped_ratio
    policy_loss = torch.max(loss1, loss2).mean()
    
    # 5. KL 散度惩罚
    kl_div = (policy_logps - ref_logps).mean()
    loss = policy_loss + self.args.beta * kl_div
    
    return loss
```

### 4.4 GRPO 家族扩展算法

ms-swift 在标准 GRPO 基础上支持丰富的变体，通过 `algorithm` 参数切换：

| 算法 | 参数 | 说明 |
|-----|------|------|
| GRPO | `algorithm='grpo'` | 标准 Group Relative Policy Optimization |
| DAPO | `algorithm='dapo'` | Decoupled Advantage Preference Optimization |
| GSPO | `algorithm='gspo'` | Group-based Self-Play Optimization |
| SAPO | `algorithm='sapo'` | Self-Adaptive Preference Optimization |
| CISPO | `algorithm='cispo'` | Contrastive Iterative Self-Play Optimization |
| RLOO | `algorithm='rloo'` | REINFORCE Leave-One-Out |
| Reinforce++ | `algorithm='reinforce++'` | REINFORCE with baseline |
| CHORD | `algorithm='chord'` | Collaborative Hindsight Optimization |

```python
def _prepare_algorithm_params(self):
    algorithm = self.args.algorithm
    if algorithm == 'grpo':
        self._setup_grpo()
    elif algorithm == 'dapo':
        self._setup_dapo()  # 解耦优势估计
    elif algorithm == 'rloo':
        self._setup_rloo()  # leave-one-out 基线
    elif algorithm == 'chord':
        self._setup_chord()  # 协作 hindsight
    ...
```

### 4.5 vLLM 集成

**文件**: `swift/rlhf_trainers/vllm_client.py` (~19KB)

```python
class VllmClient:
    """连接 vLLM 引擎进行异步生成"""
    def __init__(self, model_name, tensor_parallel_size=1, ...):
        self.llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, ...)
    
    def generate(self, prompts, sampling_params):
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.text for output in outputs]
```

**GRPOVllmEngine**: `swift/infer_engine/grpo_vllm_engine.py` 专门为 GRPO rollout 优化的 vLLM 封装。

---

## 5. PPO — Proximal Policy Optimization

**文件**: `swift/rlhf_trainers/ppo_trainer.py` (~4.2KB)

PPO 的 ms-swift 实现相对轻量，主要包装 `trl.PPOTrainer`：

```python
class PPOTrainer(SwiftMixin, TrlPPOTrainer):
    def __init__(self, model, ref_model, reward_model, ...):
        super().__init__(model, ref_model, ...)
        # value model 在 SwiftRLHF pipeline 中准备
        self.value_model = kwargs.pop('value_model', None)
```

PPO 需要 4 个模型：
1. **Policy model** (可训练)
2. **Value model** (可训练)
3. **Reference model** (冻结)
4. **Reward model** (冻结)

---

## 6. KTO / ORPO / CPO / GKD / Reward Model

### 6.1 KTO — Kahneman-Tversky Optimization

**文件**: `swift/rlhf_trainers/kto_trainer.py`

基于前景理论（Prospect Theory），不需要成对的偏好数据，只需要知道每个 completion 是"好"还是"坏"。

```python
class KTOTrainer(SwiftMixin, HFKTOTrainer):
    # loss 同时考虑 KL 约束和期望效用差异
```

### 6.2 ORPO — Odds Ratio Preference Optimization

**文件**: `swift/rlhf_trainers/orpo_trainer.py`

将 SFT 和偏好对齐合并为单一目标，无需参考模型：

```python
class ORPOTrainer(SwiftMixin, HFORPOTrainer):
    # loss = SFT_loss + lambda * OR_loss
```

### 6.3 CPO — Contrastive Preference Optimization

**文件**: `swift/rlhf_trainers/cpo_trainer.py`

对比偏好优化，类似 DPO 但更简洁：

```python
class CPOTrainer(SwiftMixin, HFCPOTrainer):
    # 简化版偏好对比损失
```

### 6.4 GKD — Generalized Knowledge Distillation

**文件**: `swift/rlhf_trainers/gkd_trainer.py` (~61KB)

使用教师模型指导策略模型训练，支持多种对齐方式：

```python
class GKDTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer):
    def __init__(self, model, teacher_model, ...):
        self.teacher_model = teacher_model  # 冻结的教师模型
        # 支持多种 distillation 损失: forward KL, reverse KL, JS, etc.
```

### 6.5 Reward Model 训练

**文件**: `swift/rlhf_trainers/reward_trainer.py`

```python
class RewardTrainer(SwiftMixin, HFRewardTrainer):
    # 训练 reward model，用于后续 PPO/GRPO
```

---

## 7. 奖励模型系统

### 7.1 ORM — Outcome Reward Model

**文件**: `swift/rewards/orm.py` (~18KB)

```python
class OutcomeRewardModel:
    """结果奖励模型"""
    def __call__(self, prompts, completions):
        # 对 completion 的结果进行打分
        # 支持规则奖励和模型奖励
        return rewards  # float list
```

### 7.2 PRM — Process Reward Model

**文件**: `swift/rewards/prm.py` (~5.2KB)

```python
class ProcessRewardModel:
    """过程奖励模型"""
    def __call__(self, prompts, completions, steps):
        # 对推理过程中的每一步打分
        return step_rewards
```

### 7.3 Reward Plugin

**文件**: `swift/rewards/rm_plugin.py`

允许用户注册自定义奖励函数：

```python
rm_plugins = {}

def register_reward_plugin(name, func):
    rm_plugins[name] = func

# 使用示例
@register_reward_plugin('my_reward')
def my_reward_func(prompts, completions):
    return [len(c) for c in completions]  # 按长度奖励
```

### 7.4 内置奖励函数

```python
# swift/rewards/orms.py
orms = {
    'accuracy': accuracy_reward,      # 答案正确性
    'format': format_reward,           # 格式合规性
    'length': length_reward,          # 长度控制
    'code': code_reward,              # 代码可执行性
    'math': math_reward,              # 数学答案正确性
}
```

---

## 8. Multi-turn RL 与 Rollout 环境

### 8.1 Multi-turn Rollout

**文件**: `swift/rollout/multi_turn.py` (~37KB)

```python
class MultiTurnRollout:
    """多轮对话的 rollout 管理"""
    def rollout(self, env, policy, max_turns=5):
        # 1. 发送第一轮 query
        # 2. 接收 response
        # 3. 检查是否调用 tool
        # 4. 若有 tool_call，执行 tool 并下一轮
        # 5. 累计 multi-turn reward
        return conversation_history, total_reward
```

### 8.2 Gym-style 环境

**文件**: `swift/rollout/gym_env.py`

```python
class TextGenEnv(gym.Env):
    """文本生成的 Gym 环境封装"""
    def step(self, action):
        # action: 生成的 token
        # return: next_observation, reward, done, info
```

---

## 9. SwiftRLHF 管道协调

**文件**: `swift/pipelines/train/rlhf.py`

```python
class SwiftRLHF(SwiftSft):
    def run(self):
        # 1. 准备所有模型 (policy/ref/value/teacher/reward)
        # 2. 准备数据集 (可能被 SwiftSft._prepare_dataset 处理)
        # 3. 获取 Trainer
        trainer_cls = TrainerFactory.get_trainer_cls(self.args)
        trainer = trainer_cls(
            model=self.model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            args=training_args,
            template=self.template,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        # 4. 训练
        trainer.train()
```

---

## 10. 关键代码路径索引

| 功能 | 关键文件/函数 |
|-----|-------------|
| DPO Trainer | `swift/rlhf_trainers/dpo_trainer.py::DPOTrainer` |
| GRPO Trainer | `swift/rlhf_trainers/grpo_trainer.py::GRPOTrainer` |
| PPO Trainer | `swift/rlhf_trainers/ppo_trainer.py::PPOTrainer` |
| KTO Trainer | `swift/rlhf_trainers/kto_trainer.py::KTOTrainer` |
| ORPO Trainer | `swift/rlhf_trainers/orpo_trainer.py::ORPOTrainer` |
| CPO Trainer | `swift/rlhf_trainers/cpo_trainer.py::CPOTrainer` |
| GKD Trainer | `swift/rlhf_trainers/gkd_trainer.py::GKDTrainer` |
| Reward Trainer | `swift/rlhf_trainers/reward_trainer.py::RewardTrainer` |
| RLHF 公共 Mixin | `swift/rlhf_trainers/rlhf_mixin.py::RLHFTrainerMixin` |
| Rollout Mixin | `swift/rlhf_trainers/rollout_mixin.py::RolloutTrainerMixin` |
| GRPO 参数 | `swift/rlhf_trainers/arguments.py::GRPOConfig` |
| DPO 参数 | `swift/rlhf_trainers/arguments.py::DPOConfig` |
| vLLM Client | `swift/rlhf_trainers/vllm_client.py::VllmClient` |
| ORM 奖励 | `swift/rewards/orm.py` |
| PRM 奖励 | `swift/rewards/prm.py` |
| Reward Plugin | `swift/rewards/rm_plugin.py` |
| 多轮 Rollout | `swift/rollout/multi_turn.py` |
| Gym 环境 | `swift/rollout/gym_env.py` |
| RLHF Pipeline | `swift/pipelines/train/rlhf.py::SwiftRLHF` |
| Trainer 工厂 | `swift/trainers/trainer_factory.py::TrainerFactory` |
