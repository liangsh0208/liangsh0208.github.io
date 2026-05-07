# DPOTrainer 直接偏好优化

> **【源码定位】**
> - 训练器: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/trainer/dpo_trainer.py` (~84KB)
> - 配置: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/trainer/dpo_config.py`
> - CLI脚本: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/scripts/dpo.py`

---

## 1. DPOTrainer 概述

`DPOTrainer` 实现直接偏好优化(Direct Preference Optimization)，从偏好数据 `{prompt, chosen, rejected}` 直接学习策略，无需显式训练奖励模型。

### 核心优势

- 无需单独的奖励模型训练
- 从偏好数据直接优化
- 支持多种 loss 变体
- 支持多 loss 组合 (MPO)

---

## 2. DPO 算法原理

### 核心公式

```
L_DPO = -log σ(β * log(π_θ(y_w|x) / π_ref(y_w|x)) 
                - β * log(π_θ(y_l|x) / π_ref(y_l|x)))
```

其中：
- `π_θ`: 当前策略模型
- `π_ref`: 参考模型 (通常为 SFT 模型)
- `y_w`: 偏好回答 (chosen)
- `y_l`: 非偏好回答 (rejected)
- `β`: KL 散度系数，控制偏离参考模型的程度

### 算法流程

1. 对同一个 prompt，分别计算 chosen 和 rejected 的 logprobs
2. 计算相对于参考模型的 log-ratio
3. 优化 Bradley-Terry 偏好模型

---

## 3. DPOConfig 配置

**源码位置**: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/trainer/dpo_config.py`

```python
@dataclass
class DPOConfig(_BaseConfig):
    """
    DPO 训练配置类
    """
    beta: float = field(default=0.1)  # KL 散度系数
    loss_type: list[str] = field(default_factory=lambda: ["sigmoid"])
    loss_weights: list[float] = field(default_factory=lambda: [1.0])
    precompute_ref_log_probs: bool = field(default=False)
    sync_ref_model: bool = field(default=False)
    ref_model_mixup_alpha: float = field(default=None)
    ref_model_sync_steps: int = field(default=64)
```

### 关键参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `beta` | float | 0.1 | KL 散度系数 |
| `loss_type` | list[str] | ["sigmoid"] | Loss 类型列表 |
| `loss_weights` | list[float] | [1.0] | 各 loss 权重 |
| `precompute_ref_log_probs` | bool | False | 预计算参考模型 logprobs |
| `sync_ref_model` | bool | False | 同步参考模型 |

---

## 4. 支持的 Loss 类型

### 单 Loss 模式

```python
# sigmoid (默认，标准 DPO)
loss_type=["sigmoid"]

# hinge (偏好边缘最大化)
loss_type=["hinge"]

# ipo (身份偏好优化)
loss_type=["ipo"]

# nca_pair (噪声对比估计)
loss_type=["nca_pair"]

# robust (鲁棒 DPO)
loss_type=["robust"]

# bco_pair (偏好比率优化)
loss_type=["bco_pair"]

# sppo_hard (自博弈偏好优化)
loss_type=["sppo_hard"]

# aot (优势配对训练)
loss_type=["aot"]

# discopop (分离 DPO)
loss_type=["discopop"]

# sft (监督微调 loss)
loss_type=["sft"]
```

### 多 Loss 组合 (MPO)

```python
# 组合 DPO 和 SFT loss
DPOConfig(
    loss_type=["sigmoid", "sft"],
    loss_weights=[0.8, 0.2]
)
```

---

## 5. Reference Model 处理

### 预计算参考 Logprobs

当显存受限时，可以预计算参考模型的 logprobs：

```python
DPOConfig(
    precompute_ref_log_probs=True,  # 预计算并缓存
)
```

### 参考模型同步

用于 TR-DPO (Trust Region DPO) 等算法：

```python
from trl.trainer.callbacks import SyncRefModelCallback

DPOConfig(
    sync_ref_model=True,
    ref_model_mixup_alpha=0.5,  # 混合系数
    ref_model_sync_steps=64,    # 同步步数
)
```

---

## 6. 数据整理器

```python
@dataclass
class DataCollatorForPreference(DataCollatorMixin):
    """
    同时处理 chosen 和 rejected 序列
    返回格式: 前半部分是 chosen，后半部分是 rejected
    支持 precomputed reference logprobs
    """
    # 将 (prompt, chosen) 和 (prompt, rejected) 分别编码
    # batch 组织为 [chosen_batch, rejected_batch]
```

### 数据格式

```python
{
    "prompt": "用户输入...",
    "chosen": "好的回答...",
    "rejected": "不好的回答..."
}
```

---

## 7. 实战示例

### 基础 DPO 训练

```python
from trl import DPOTrainer, DPOConfig, ModelConfig
from datasets import load_dataset

# 偏好数据格式: {"prompt": "...", "chosen": "...", "rejected": "..."}
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# 从 SFT 模型开始
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# 配置
training_args = DPOConfig(
    output_dir="./output/qwen-dpo",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-6,
    beta=0.1,  # KL 散度系数
    loss_type=["sigmoid"],
    # 可选: 预计算参考模型 logprobs 节省显存
    precompute_ref_log_probs=False,
)

# 训练
trainer = DPOTrainer(
    model=model_name,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

### MPO 多 Loss 训练

```python
from trl import DPOConfig

# 组合 DPO + SFT
training_args = DPOConfig(
    output_dir="./output/qwen-mpo",
    beta=0.1,
    loss_type=["sigmoid", "sft"],
    loss_weights=[0.8, 0.2],  # 80% DPO + 20% SFT
    learning_rate=1e-6,
)
```

### 使用 CLI

```bash
trl dpo --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --output_dir qwen-dpo
```

---

## 8. 完整参数表

### DPO 特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `beta` | float | 0.1 | KL 系数 |
| `loss_type` | list | ["sigmoid"] | Loss 类型列表 |
| `loss_weights` | list | [1.0] | 各 loss 权重 |
| `precompute_ref_log_probs` | bool | False | 预计算参考模型 logprobs |
| `sync_ref_model` | bool | False | 同步参考模型 |
| `ref_model_mixup_alpha` | float | None | 混合系数 |
| `ref_model_sync_steps` | int | 64 | 同步步数 |
| `ld_alpha` | float | None | LD-DPO 参数 |

### 支持的 Loss 类型

| Loss 类型 | 论文 | 说明 |
|-----------|------|------|
| sigmoid | DPO | 标准 DPO |
| hinge | - | 偏好边缘最大化 |
| ipo | IPO | 身份偏好优化 |
| exo_pair | EXO | 显式交叉熵优化 |
| nca_pair | NCA | 噪声对比对齐 |
| robust | Robust DPO | 鲁棒 DPO |
| bco_pair | BCO | 偏好比率优化 |
| sppo_hard | SPPO | 自博弈偏好优化 |
| aot | AOT | 优势配对训练 |
| discopop | DiscoPOP | 分离 DPO |
| sft | - | 监督微调 loss |
