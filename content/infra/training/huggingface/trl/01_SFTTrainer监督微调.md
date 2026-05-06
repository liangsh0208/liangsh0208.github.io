# SFTTrainer 监督微调

> **【源码定位】**
> - 训练器: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/trainer/sft_trainer.py` (~77KB)
> - 配置: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/trainer/sft_config.py`
> - CLI脚本: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/scripts/sft.py`

---

## 1. SFTTrainer 概述

`SFTTrainer` 是 TRL 中最基础的训练器，用于监督微调(Supervised Fine-Tuning)。它继承自 `_BaseTrainer`，为指令跟随训练提供了丰富的功能支持。

### 核心功能

- **completion_only_loss**: 仅计算 assistant 回复部分的 loss
- **packing**: 将多个样本打包成固定长度块，减少 padding
- **padding-free**: 与 FlashAttention 配合使用，展平序列
- **原生支持 VLM** (Vision Language Models)

---

## 2. SFTConfig 配置

**源码位置**: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/trainer/sft_config.py`

```python
@dataclass
class SFTConfig(_BaseConfig):
    """
    SFT 训练配置类
    """
    max_length: int | None = field(default=1024)
    packing: bool = field(default=False)  # 序列打包
    packing_strategy: str = field(default="bfd")  # bfd/bfd_split/wrapped
    completion_only_loss: bool | None = field(default=None)
    assistant_only_loss: bool = field(default=False)
    loss_type: str = field(default="nll")  # "nll" 或 "dft"
```

### 关键参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_length` | int | 1024 | 最大序列长度 |
| `packing` | bool | False | 启用序列打包 |
| `packing_strategy` | str | "bfd" | 打包策略: bfd/bfd_split/wrapped |
| `completion_only_loss` | bool | None | 只计算回复部分 loss |
| `assistant_only_loss` | bool | False | 只计算 assistant 部分 loss |
| `loss_type` | str | "nll" | "nll" 或 "dft" |

---

## 3. Packing 序列打包

### 什么是 Packing

将多个短样本打包成固定长度的序列块，减少 padding waste，提高训练效率。

### 打包算法

TRL 使用 **最佳适应递减 (BFD - Best Fit Decreasing)** 算法：

1. 按长度降序排序所有样本
2. 对每个样本，找到能容纳它的最满块
3. 如果没有合适块，创建新块

### 数据整理器实现

```python
@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    """
    支持 completion_mask 和 assistant_masks
    支持 padding-free 模式 (生成 position_ids 替代 attention_mask)
    """
    pad_token_id: int
    max_length: int | None = None
    completion_only_loss: bool = True
    padding_free: bool = False  # FlashAttention 优化模式
```

---

## 4. Padding-Free 模式

### 原理

与 FlashAttention 配合使用，展平 batch 中的所有序列：

```
# 传统方式 (有 padding)
[[tok1, tok2, tok3, PAD, PAD],
 [tok4, tok5, PAD, PAD, PAD]]

# Padding-Free 方式
[tok1, tok2, tok3, tok4, tok5]
+ position_ids 标识序列边界
```

### 启用方法

```python
from trl import SFTConfig

config = SFTConfig(
    padding_free=True,  # 启用 padding-free
    # 需要 FlashAttention 支持
)
```

---

## 5. VLM 视觉语言模型支持

SFTTrainer 原生支持多模态训练：

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# VLM 数据集包含图像字段
dataset = load_dataset("...")  # 包含 "image" 和 "messages" 字段

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    args=SFTConfig(
        output_dir="./vlm-sft",
        max_length=2048,
    ),
    train_dataset=dataset,
)
```

### 图像处理流程

1. `prepare_multimodal_messages()` - 将图像插入到消息中的 image 占位符位置
2. Processor 自动处理图像转换为 vision embeddings
3. 训练时同时优化 vision encoder 和 language model (可配置冻结)

---

## 6. 实战示例

### 基础 SFT 训练

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 加载数据
dataset = load_dataset("trl-lib/Capybara", split="train")

# 配置
training_args = SFTConfig(
    output_dir="./output/qwen-sft",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    packing=True,  # 启用序列打包
    max_length=2048,
)

# 创建训练器
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    args=training_args,
    train_dataset=dataset,
    # 可选: peft_config 传入 PEFT 配置
)

# 训练
trainer.train()
```

### 带 LoRA 的 SFT

```python
from trl import SFTTrainer, SFTConfig, ModelConfig
from peft import get_peft_config

model_config = ModelConfig(
    model_name_or_path="meta-llama/Llama-2-7b",
    use_peft=True,
    lora_r=32,
    lora_alpha=16,
    lora_target_modules=["q_proj", "v_proj"],
    load_in_4bit=True,  # QLoRA
)

training_args = SFTConfig(
    output_dir="./output/llama-lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
)

trainer = SFTTrainer(
    args=training_args,
    train_dataset=dataset,
    peft_config=get_peft_config(model_config),
)
```

### 使用 CLI

```bash
trl sft --model_name_or_path Qwen/Qwen2.5-0.5B \
    --dataset_name trl-lib/Capybara \
    --output_dir qwen-sft
```

---

## 7. 完整参数表

### SFT 特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_length` | int | 1024 | 最大序列长度 |
| `packing` | bool | False | 序列打包 |
| `completion_only_loss` | bool | None | 只计算回复部分 loss |
| `assistant_only_loss` | bool | False | 只计算 assistant 部分 loss |
| `loss_type` | str | "nll" | 或 "dft" |

### 通用参数 (继承自 _BaseConfig)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_dir` | str | 必填 | 输出目录 |
| `num_train_epochs` | float | 3.0 | 训练轮数 |
| `per_device_train_batch_size` | int | 8 | 每设备 batch 大小 |
| `gradient_accumulation_steps` | int | 1 | 梯度累积步数 |
| `learning_rate` | float | 5e-5 | 学习率 |
| `bf16` | bool | True | 使用 bfloat16 |
| `gradient_checkpointing` | bool | True | 梯度检查点 |
| `logging_steps` | int | 10 | 日志记录间隔 |
| `use_liger_kernel` | bool | False | Liger Kernel 加速 |
| `torch_empty_cache_steps` | int | None | 清空缓存步数 |
