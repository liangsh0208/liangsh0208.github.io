---
created: 2026-05-06
---

# RewardTrainer 奖励建模

> **【源码定位】**
> - 训练器: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/trainer/reward_trainer.py`
> - 配置: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/trainer/reward_config.py`

---

## 1. RewardTrainer 概述

`RewardTrainer` 用于训练奖励模型 (Reward Model)，这是传统 RLHF 流程中的关键组件。

### 用途

- 训练显式奖励模型用于 RLHF
- 为 PPO 等传统 RL 方法提供奖励信号
- 评估生成质量

### 与其他训练器的区别

| 特性 | RewardTrainer | GRPO/DPO |
|------|---------------|----------|
| 输出 | 标量奖励值 | 直接优化策略 |
| 需要 | 偏好评分数据 | 偏好对比数据 |
| 用于 | PPO 等传统 RL | 端到端训练 |
| 复杂度 | 需额外模型 | 端到端，简化 |

---

## 2. RewardConfig 配置

**源码位置**: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/trainer/reward_config.py`

```python
@dataclass
class RewardConfig(_BaseConfig):
    """
    奖励模型训练配置类
    """
    max_length: int | None = field(default=None)
    
    # 用于 Bradley-Terry 模型训练
    label_smoothing: float = field(default=0.0)
    
    # 用于回归任务
    regression: bool = field(default=False)
    
    # 损失函数类型
    loss_type: str = field(default="pairwise")  # pairwise/ranking/regression
```

### 关键参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_length` | int | None | 最大序列长度 |
| `label_smoothing` | float | 0.0 | 标签平滑系数 |
| `regression` | bool | False | 回归模式 (直接预测分数) |
| `loss_type` | str | "pairwise" | pairwise/ranking/regression |

---

## 3. Bradley-Terry 模型

### 原理

学习一个奖励函数 `r(x, y)`，满足：

```
P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))
```

即偏好回答的奖励高于非偏好回答。

### 损失函数

```python
# Pairwise Loss (默认)
L = -log σ(r_chosen - r_rejected)

# With Label Smoothing
L = -[ (1-α) * log σ(r_chosen - r_rejected) + α * log σ(r_rejected - r_chosen) ]
```

---

## 4. 训练模式

### Pairwise 模式

标准的偏好对比训练：

```python
RewardConfig(
    loss_type="pairwise",
    label_smoothing=0.0,
)

# 数据格式: {"prompt": "...", "chosen": "...", "rejected": "..."}
```

### Ranking 模式

多段排序训练：

```python
RewardConfig(
    loss_type="ranking",
)

# 数据格式: {"prompt": "...", "completions": [...], "ranks": [...]}
```

### Regression 模式

直接回归评分：

```python
RewardConfig(
    regression=True,
    loss_type="regression",
)

# 数据格式: {"text": "...", "score": float}
```

---

## 5. 实战示例

### 基础奖励模型训练

```python
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset

# 偏好数据
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# 从 SFT 模型初始化奖励模型
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# 配置
training_args = RewardConfig(
    output_dir="./output/reward-model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    max_length=512,
    label_smoothing=0.1,  # 标签平滑
)

# 训练
trainer = RewardTrainer(
    model=model_name,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model()
```

### 回归模式训练

```python
from trl import RewardTrainer, RewardConfig

# 直接评分数据
# {"text": "生成文本", "score": 4.5}

training_args = RewardConfig(
    output_dir="./output/regression-reward",
    regression=True,
    loss_type="regression",
    learning_rate=1e-5,
)

trainer = RewardTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=training_args,
    train_dataset=dataset,
)
```

### 使用训练好的奖励模型

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 加载奖励模型
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "./output/reward-model"
)
tokenizer = AutoTokenizer.from_pretrained("./output/reward-model")

# 计算奖励
text = "这是一个测试文本"
inputs = tokenizer(text, return_tensors="pt")
reward = reward_model(**inputs).logits.item()
print(f"Reward: {reward}")
```

---

## 6. 完整参数表

### RewardTrainer 特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_length` | int | None | 最大序列长度 |
| `label_smoothing` | float | 0.0 | 标签平滑系数 |
| `regression` | bool | False | 回归模式 |
| `loss_type` | str | "pairwise" | 损失类型 |

### 数据格式

#### Pairwise 模式

```python
{
    "prompt": "用户输入...",
    "chosen": "好的回答...",
    "rejected": "不好的回答..."
}
```

#### Ranking 模式

```python
{
    "prompt": "用户输入...",
    "completions": ["回答1", "回答2", "回答3"],
    "ranks": [1, 3, 2]  # 排名，越小越好
}
```

#### Regression 模式

```python
{
    "text": "需要评分的文本...",
    "score": 4.5  # 分数
}
```

---

## 7. 与其他训练器的关系

```
RewardTrainer 训练奖励模型
        ↓
   奖励模型用于 PPO 等传统 RL
        ↓
PPOTrainer 使用 RewardTrainer 的输出
```

在现代 RLHF 中，DPO 和 GRPO 已经可以直接从偏好数据学习，**无需显式训练奖励模型**。但在某些场景下（如需要细粒度奖励控制），RewardTrainer 仍有其价值。
