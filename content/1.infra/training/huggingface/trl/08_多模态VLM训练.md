# 多模态 VLM 训练

> **【源码定位】**
> - 数据处理: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/data_utils.py`
> - 数据整理器: 各训练器的 data collator
> - 聊天模板: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/chat_template_utils.py`

---

## 1. VLM 支持概述

TRL 原生支持视觉语言模型 (Vision Language Models) 的训练，包括：

- **SFT**: 监督微调 VLM
- **DPO**: 基于图像的偏好优化
- **GRPO**: 视觉推理强化学习

### 支持模型

| 模型系列 | SFT | DPO | GRPO | 示例模型 |
|----------|-----|-----|------|----------|
| Qwen2-VL | ✓ | ✓ | ✓ | Qwen/Qwen2.5-VL-3B-Instruct |
| LLaVA | ✓ | ✓ | ✓ | llava-hf/llava-1.5-7b-hf |
| Phi-Vision | ✓ | ✓ | ✓ | microsoft/Phi-3.5-vision-instruct |
| InternVL | ✓ | ✓ | ✓ | OpenGVLab/InternVL2-4B |

---

## 2. 数据格式

### 基础 VLM 格式

```python
# 单轮对话
{
    "image": "path/to/image.jpg",  # 或 PIL.Image
    "messages": [
        {"role": "user", "content": [
            {"type": "image"},  # 图像占位符
            {"type": "text", "text": "描述这张图片"}
        ]},
        {"role": "assistant", "content": "这张图片展示了..."}
    ]
}
```

### 偏好数据格式 (DPO)

```python
{
    "image": "path/to/image.jpg",
    "prompt": [
        {"type": "image"},
        {"type": "text", "text": "描述这张图片中的场景"}
    ],
    "chosen": "这是一个美丽的海滩场景...",  # 更好的描述
    "rejected": "这是一张图片..."  # 较差的描述
}
```

### GRPO 推理数据

```python
{
    "image": "path/to/math_problem.jpg",
    "prompt": "解决图片中的数学问题",
    "solution": "42"  # 正确答案
}
```

---

## 3. 图像处理流程

### 数据预处理

**源码位置**: `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/data_utils.py`

```python
def prepare_multimodal_messages(messages, images=None) -> list[dict]:
    """
    将图像插入到消息中的 image 占位符位置
    
    Args:
        messages: 包含 image 占位符的消息列表
        images: 图像列表或单个图像
    
    Returns:
        处理后的消息，image 占位符被替换为实际图像
    """
    ...

def apply_chat_template(
    example,
    processing_class,  # Processor (VLM 专用)
    tools=None
) -> dict:
    """
    应用 chat template，同时处理图像
    返回包含 pixel_values 的字典
    """
    ...
```

### 处理器 (Processor) 工作流程

```
输入:
  - 消息列表 (包含 image 占位符)
  - 图像数据

处理:
  1. 检测 image 占位符 (如 <image>)
  2. 使用 image processor 转换为 pixel_values
  3. 文本部分使用 tokenizer 编码
  4. 合并为模型输入

输出:
  {
      "input_ids": [...],
      "attention_mask": [...],
      "pixel_values": [...],  # 图像特征
      "image_grid_thw": [...]  # 图像位置 (部分模型)
  }
```

---

## 4. SFT VLM 训练

### 配置示例

```python
from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info  # Qwen2-VL 专用

training_args = SFTConfig(
    output_dir="./output/vlm-sft",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # VLM 显存开销大
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    max_length=2048,
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    args=training_args,
    train_dataset=vlm_dataset,
    # 图像处理自动完成
)

trainer.train()
```

### 冻结 Vision Encoder

```python
from transformers import Qwen2VLForConditionalGeneration

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct"
)

# 仅训练 language model
for param in model.visual.parameters():
    param.requires_grad = False
```

---

## 5. DPO VLM 训练

### 配置示例

```python
from trl import DPOTrainer, DPOConfig

training_args = DPOConfig(
    output_dir="./output/vlm-dpo",
    beta=0.1,
    loss_type=["sigmoid"],
    per_device_train_batch_size=1,  # 同时处理 chosen + rejected
    gradient_accumulation_steps=4,
)

trainer = DPOTrainer(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    args=training_args,
    train_dataset=vlm_preference_dataset,
)

trainer.train()
```

---

## 6. GRPO VLM 训练

### 视觉推理场景

```python
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward
import re

def vision_math_reward(prompts, completions, images, solution, **kwargs):
    """
    针对图像数学题的奖励函数
    """
    rewards = []
    for completion, sol in zip(completions, solution):
        content = completion[0]["content"] if completion else ""
        
        # 提取答案 (假设格式: "答案是 <answer>")
        match = re.search(r'答案是?\s*(\d+)', content)
        if match:
            pred = match.group(1)
            rewards.append(1.0 if pred == str(sol) else 0.0)
        else:
            rewards.append(0.0)
    
    return rewards

training_args = GRPOConfig(
    output_dir="./output/vlm-grpo",
    num_generations=4,  # VLM 显存敏感
    max_completion_length=1024,
    use_vllm=False,  # VLM 暂不支持 vLLM
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    reward_funcs=[vision_math_reward],
    args=training_args,
    train_dataset=vision_math_dataset,
)
```

---

## 7. 数据整理器处理

### VLM 数据整理器特点

```python
@dataclass
class VLMCollator:
    """
    VLM 专用数据整理器
    处理 pixel_values 和 input_ids 的对齐
    """
    
    def __call__(self, features):
        # 1. 收集所有图像
        images = [f["image"] for f in features]
        
        # 2. 使用 processor 处理
        batch = processor(
            text=[f["text"] for f in features],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        
        # 3. 获取 labels (SFT) 或 preference masks (DPO)
        if "labels" in features[0]:
            batch["labels"] = ...
        
        return batch
```

---

## 8. 显存优化

### VLM 训练显存占用

| 模型 | FP16/BF16 | 4-bit QLoRA | 8-bit |
|------|-----------|-------------|-------|
| Qwen2.5-VL-3B | ~12GB | ~6GB | ~8GB |
| Qwen2.5-VL-7B | ~24GB | ~10GB | ~16GB |
| LLaVA-1.5-7B | ~20GB | ~9GB | ~14GB |
| InternVL2-4B | ~10GB | ~5GB | ~7GB |

### 优化策略

```python
from trl import SFTConfig, ModelConfig

# 1. 使用 QLoRA
model_config = ModelConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    use_peft=True,
    lora_r=64,
    lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

# 2. 冻结 vision encoder
training_args = SFTConfig(
    freeze_vision_encoder=True,  # 仅训练 LLM 部分
)

# 3. 梯度检查点
training_args = SFTConfig(
    gradient_checkpointing=True,
)

# 4. 减小序列长度
training_args = SFTConfig(
    max_length=1024,  # 而非 2048
)
```

---

## 9. 完整训练示例

### VLM SFT + LoRA

```python
from trl import SFTTrainer, SFTConfig, ModelConfig
from peft import get_peft_config

def format_vlm_dataset(example):
    """格式化 VLM 数据集"""
    return {
        "image": example["image"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example["question"]}
                ]
            },
            {
                "role": "assistant",
                "content": example["answer"]
            }
        ]
    }

# 配置
model_config = ModelConfig(
    model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    use_peft=True,
    lora_r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    freeze_vision_encoder=True,  # 冻结视觉编码器
)

training_args = SFTConfig(
    output_dir="./output/vlm-sft-lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,  # LoRA 可使用更高学习率
    max_length=2048,
    gradient_checkpointing=True,
)

# 训练
trainer = SFTTrainer(
    model=model_config.model_name_or_path,
    args=training_args,
    train_dataset=dataset.map(format_vlm_dataset),
    peft_config=get_peft_config(model_config),
)
trainer.train()
trainer.save_model()
```

---

## 10. 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `image token not found` | 图像占位符格式错误 | 检查模型指定的 image token |
| `pixel_values mismatch` | 图像尺寸不一致 | 使用 processor 统一处理 |
| `OOM during vision encode` | 高清图像 | 降低图像分辨率 |
| `No grad for vision model` | 梯度检查点配置 | 检查 freeze_vision_encoder |
| `Invalid shape for attention` | 图像位置编码 | 确保 image_grid_thw 正确传递 |
