---
created: 2026-05-06
---

# GRPOTrainer 组相对策略优化

> **【源码定位】**
> - 训练器: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/trainer/grpo_trainer.py` (~151KB, 最大训练器)
> - 配置: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/trainer/grpo_config.py` (~57KB)
> - CLI脚本: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/scripts/grpo.py`

---

## 1. GRPOTrainer 概述

`GRPOTrainer` 是 TRL 中**最重要和实现最复杂的训练器**，是 DeepSeek-R1 的核心算法实现。

### 为什么是 DeepSeek-R1 的核心

GRPO (Group Relative Policy Optimization) 相比 PPO：
- 无需单独训练价值模型 (Critic-Free)
- 使用组内奖励归一化替代优势估计
- 内存效率更高，实现更简单
- 更适合大规模 LLM 训练

---

## 2. GRPO 算法核心

### 算法流程

```
1. 对每个 prompt 生成 G 个 completions
2. 计算每个 completion 的奖励
3. 计算组内优势: A_i = (r_i - mean(r)) / std(r)
4. 策略更新 (带裁剪):
   L_GRPO = -min(
       (π_θ / π_old) * A,
       clip(π_θ / π_old, 1-ε, 1+ε) * A
   ) + β * KL(π_θ || π_ref)
```

### 关键创新

| 特性 | PPO | GRPO |
|------|-----|------|
| 价值模型 | 需要训练 | 不需要 |
| 优势估计 | GAE | 组内归一化 |
| 内存占用 | 高 | 低 |
| 实现复杂度 | 高 | 中等 |

---

## 3. GRPOConfig 配置

**源码位置**: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/trainer/grpo_config.py`

### 核心参数

```python
@dataclass
class GRPOConfig(_BaseConfig):
    """
    GRPO 训练配置类
    """
    # 生成参数
    num_generations: int = field(default=8)           # G: 每组生成数量
    max_completion_length: int = field(default=256)   # 最大生成长度
    temperature: float = field(default=1.0)           # 采样温度
    top_p: float = field(default=1.0)                 # Top-p 采样
    top_k: int = field(default=50)                     # Top-k 采样
    repetition_penalty: float = field(default=1.0)    # 重复惩罚
    
    # 训练参数
    num_iterations: int = field(default=1)            # μ: 每批次更新次数
    epsilon: float = field(default=0.2)               # 裁剪系数下界
    epsilon_high: float | None = field(default=None)    # 裁剪系数上界 (可选)
    
    # KL 约束
    beta: float = field(default=0.0)                  # KL 系数 (0=无参考模型)
    
    # Loss 类型
    loss_type: str = field(default="dapo")            # 多种变体支持
    
    # 奖励缩放
    scale_rewards: str = field(default="group")       # group/batch/none
    
    # vLLM 配置
    use_vllm: bool = field(default=False)
    vllm_mode: str = field(default="colocate")        # colocate/server
    vllm_gpu_memory_utilization: float = field(default=0.3)
    vllm_tensor_parallel_size: int = field(default=1)
```

### 关键参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_generations` | int | 8 | 每组生成数量 G |
| `max_completion_length` | int | 256 | 最大生成长度 |
| `temperature` | float | 1.0 | 采样温度 |
| `num_iterations` | int | 1 | 每批次更新次数 μ |
| `epsilon` | float | 0.2 | 裁剪系数下界 |
| `epsilon_high` | float | None | 裁剪系数上界 |
| `beta` | float | 0.0 | KL 系数 (0=无参考模型) |
| `loss_type` | str | "dapo" | loss 类型 |
| `scale_rewards` | str | "group" | 奖励缩放方式 |
| `use_vllm` | bool | False | 使用 vLLM 生成 |

---

## 4. 8种 Loss 变体支持

GRPOTrainer 支持多种 loss 变体，对应不同论文:

| Loss 类型 | 论文 | 特点 |
|-----------|------|------|
| `grpo` | DeepSeekMath | 原始实现，按序列长度归一化 |
| `dr_grpo` | Dr. GRPO | 按 max_completion_length 常数归一化，消除长度偏差 |
| `dapo` | DAPO | 按全局 batch 有效 token 数归一化 (默认) |
| `bnpo` | - | 按本地 batch 有效 token 数归一化 |
| `cispo` | MiniMax-M1 | 裁剪重要性采样权重而非 advantage-scaled 权重 |
| `sapo` | SAPO | 软裁剪，温度控制的平滑门控 |
| `luspo` | LUSPO | 序列级 loss，消除长度偏差 |
| `vespo` | VESPO | Gamma 加权函数，序列级软策略优化 |

### 配置示例

```python
from trl import GRPOConfig

# DeepSeek-R1 原始 GRPO
grpo_config = GRPOConfig(loss_type="grpo")

# DAPO (推荐默认)
dapo_config = GRPOConfig(
    loss_type="dapo",
    mask_truncated_completions=True,
    epsilon=0.2,
    epsilon_high=0.28,
)

# Dr. GRPO (消除长度偏差)
dr_grpo_config = GRPOConfig(
    loss_type="dr_grpo",
    scale_rewards=False,  # 禁用缩放
)

# SAPO (软裁剪)
sapo_config = GRPOConfig(
    loss_type="sapo",
    epsilon=0.28,
)
```

---

## 5. 奖励函数系统

### 内置奖励函数

**位置**: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/rewards/`

```python
# 1. accuracy_reward - 数学答案验证
from trl.rewards import accuracy_reward
# 使用 math_verify 验证数学答案
# 支持 LaTeX 解析和标准答案匹配

# 2. reasoning_accuracy_reward - 推理答案验证
from trl.rewards import reasoning_accuracy_reward
# 用于推理模型 (如 DeepSeek-R1 格式)
# 可配置推理内容分隔符 (如 "</think>")

# 3. think_format_reward - 格式奖励
from trl.rewards import think_format_reward
# 奖励格式正确的推理标记
# 检查 <think>...</think> 格式
```

### 自定义奖励函数

```python
def custom_reward(prompts, completions, **kwargs) -> list[float | None]:
    """
    自定义奖励函数签名
    
    Args:
        prompts: 输入 prompts 列表
        completions: 生成的 completions 列表
        **kwargs: 可选参数包括:
            - trainer_state: 训练器状态
            - log_extra: 额外日志信息
            - log_metric: 日志指标
    
    Returns:
        每个 completion 的奖励列表，None 表示跳过该样本
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # 自定义奖励计算逻辑
        if "正确答案" in completion:
            reward = 1.0
        elif "部分正确" in completion:
            reward = 0.5
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards

# 使用自定义奖励函数
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B",
    reward_funcs=[accuracy_reward, custom_reward],
    args=GRPOConfig(...),
    train_dataset=dataset,
)
```

---

## 6. vLLM 集成加速

### Colocate 模式

vLLM 与训练共用 GPU，自动管理显存:

```python
from trl import GRPOTrainer, GRPOConfig

training_args = GRPOConfig(
    use_vllm=True,
    vllm_mode="colocate",  # 共用 GPU
    vllm_gpu_memory_utilization=0.3,  # vLLM 占用 30% 显存
    vllm_tensor_parallel_size=1,
)
```

### Server 模式

连接到独立的 vLLM 服务:

```bash
# 1. 启动 vLLM 服务
trl vllm-serve --model Qwen/Qwen2.5-0.5B-Instruct

# 2. 训练配置
```

```python
training_args = GRPOConfig(
    use_vllm=True,
    vllm_mode="server",
    vllm_server_url="http://localhost:8000",
)
```

---

## 7. 实战示例

### DeepSeek-R1 风格训练

```python
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward, think_format_reward
from datasets import load_dataset

# 加载数据 (需要包含 prompt 和 solution 字段)
dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

# DeepSeek-R1 训练配置
training_args = GRPOConfig(
    output_dir="./output/r1-style",
    num_train_epochs=1,
    
    # 生成参数
    num_generations=8,           # G: 每组 8 个 completions
    max_completion_length=2048,  # 长思维链
    temperature=0.9,
    top_p=0.95,
    
    # 训练参数
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_iterations=2,            # μ: 每批次更新 2 次
    epsilon=0.2,               # 裁剪下界
    epsilon_high=0.28,         # 裁剪上界 (DAPO)
    
    # KL 约束
    beta=0.001,                # 很小的 KL 约束
    
    # Loss 类型
    loss_type="dapo",          # DAPO 推荐
    scale_rewards="group",     # 组内标准化
    
    # vLLM 加速
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.3,
)

# 训练
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=[accuracy_reward, think_format_reward],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model()
```

### 极简 GRPO (GPU 受限)

```python
from trl import GRPOTrainer, GRPOConfig

training_args = GRPOConfig(
    output_dir="./output/grpo-minimal",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=4,      # 减少以节省显存
    max_completion_length=512,
    beta=0,                 # 无参考模型
    use_vllm=False,         # 使用 transformers 生成
)
```

### 使用 CLI

```bash
trl grpo --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name trl-lib/DeepMath-103K \
    --reward_funcs accuracy_reward \
    --output_dir qwen-grpo
```

---

## 8. 完整参数表

### GRPO 特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_generations` | int | 8 | 每组生成数量 G |
| `max_completion_length` | int | 256 | 最大生成长度 |
| `temperature` | float | 1.0 | 采样温度 |
| `top_p` | float | 1.0 | Top-p 采样 |
| `top_k` | int | 50 | Top-k 采样 |
| `num_iterations` | int | 1 | 每批次更新次数 μ |
| `epsilon` | float | 0.2 | 裁剪系数下界 |
| `epsilon_high` | float | None | 裁剪系数上界 |
| `beta` | float | 0.0 | KL 系数 |
| `loss_type` | str | "dapo" | loss 类型 |
| `scale_rewards` | str | "group" | group/batch/none |
| `use_vllm` | bool | False | 使用 vLLM |
| `vllm_mode` | str | "colocate" | colocate/server |
| `vllm_gpu_memory_utilization` | float | 0.3 | vLLM 显存占用 |
| `mask_truncated_completions` | bool | False | 掩码截断 completions |

---

## 9. 关键论文引用

| 方法 | 论文 | 链接 |
|------|------|------|
| GRPO | DeepSeekMath | [2402.03300](https://huggingface.co/papers/2402.03300) |
| DeepSeek-R1 | Incentivizing Reasoning | [2501.12948](https://huggingface.co/papers/2501.12948) |
| DAPO | Foundational RL Post-Training | [2503.14476](https://huggingface.co/papers/2503.14476) |
| Dr. GRPO | Surviving KL Divergence | [2503.20783](https://huggingface.co/papers/2503.20783) |
| CISPO | MiniMax-M1 | - |
| SAPO | Soft Advantage Policy | - |
| LUSPO | Length Unbiased SPo | - |
| VESPO | Value-Exploration SPo | - |
