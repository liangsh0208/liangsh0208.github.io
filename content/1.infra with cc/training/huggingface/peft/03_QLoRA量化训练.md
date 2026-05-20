---
created: 2026-05-06
---

# 03_QLoRA量化训练

## 文档元信息

| 属性 | 内容 |
|------|------|
| 源码定位 | `/Users/danchen/Documents/1.RL_fw/huggingface/peft/src/peft/utils/integrations.py` |
| 依赖库 | `bitsandbytes`, `transformers` |
| 论文参考 | QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023) |

---

## 1. 4-bit 量化基础

### 1.1 什么是 QLoRA

QLoRA 在 LoRA 基础上结合 4-bit 量化，实现：
- **显存效率**：将 70B 模型量化到 ~35GB 显存
- **训练可行**：单卡消费级 GPU (如 24GB RTX 4090) 微调大型模型
- **性能接近**：相比全参数微调性能损失 <1%

### 1.2 Normal Float 4 (NF4)

NF4 是一种信息量最优的 4-bit 数据类型：

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",     # 4-bit Normal Float
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # 嵌套量化
)
```

### 1.3 双重量化（Double Quantization）

对量化常数进行二次量化，进一步节省显存：

```
原始权重 (FP16/32) → 量化到 4-bit → 量化常数 (32-bit) → 二次量化到 8-bit
```

---

## 2. QLoRA 配置

### 2.1 BitsAndBytesConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `load_in_4bit` | bool | False | 启用 4-bit 量化 |
| `bnb_4bit_quant_type` | str | "fp4" | 量化类型: "nf4" 或 "fp4" |
| `bnb_4bit_compute_dtype` | dtype | torch.float32 | 计算精度 |
| `bnb_4bit_use_double_quant` | bool | False | 启用双重量化 |
| `load_in_8bit` | bool | False | 启用 8-bit 量化 |

### 2.2 完整配置示例

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. 配置 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 2. 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)

# 3. 准备模型用于训练
model = prepare_model_for_kbit_training(model)

# 4. 配置 LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    use_gradient_checkpointing="unsloth",
)

# 5. 应用 LoRA
model = get_peft_model(model, lora_config)
```

---

## 3. prepare_model_for_kbit_training

**文件**: `utils/integrations.py`

```python
def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_func=None):
    """
    准备量化模型用于训练：
    1. 冻结基础模型参数
    2. 启用梯度检查点
    3. 处理输入/输出嵌入归一化
    4. 转换可训练参数为 32-bit
    """
    # 1. 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. 启用梯度检查点
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_func)
        model.enable_input_require_grads()
    
    # 3. 确保输入嵌入可训练（用于梯度检查点）
    if hasattr(model, "get_input_embeddings"):
        model.get_input_embeddings().register_forward_hook(
            lambda module, input, output: output.requires_grad_(True)
        )
    
    return model
```

---

## 4. 分页优化器 (Paged Optimizers)

分页优化器使用 CPU 内存作为 GPU 显存的扩展，避免 OOM：

```python
from peft.utils.integrations import get_optimizer_for_params

# 使用分页 AdamW
optimizer = get_optimizer_for_params(
    model.parameters(),
    "paged_adamw_8bit",  # 或 "paged_adamw_32bit"
    lr=1e-4,
)

# 或使用 bitsandbytes 直接
from bitsandbytes.optim import PagedAdamW8bit

optimizer = PagedAdamW8bit(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
)
```

---

## 5. 梯度检查点

```python
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules="all-linear",
    use_gradient_checkpointing="unsloth",  # 扩展梯度检查点
    # 或使用标准版
    # use_gradient_checkpointing=True,
)
```

---

## 6. QLoRA vs LoRA

| 特性 | LoRA | QLoRA |
|------|------|-------|
| **基础模型精度** | FP16/BF16 | 4-bit NF4 |
| **显存需求** | 较高 | 极低（约 1/4） |
| **训练速度** | 正常 | 稍慢（量化/反量化开销） |
| **适用模型** | 中小模型 | 大型模型 (7B-70B+) |
| **单卡可行性** | 取决于模型大小 | 7B-13B 可用 24GB 显卡 |

---

## 7. 优化器组合建议

```python
# 1. 标准 QLoRA
# 学习率：1e-4 ~ 2e-4
# 需要 prepare_model_for_kbit_training

# 2. 分页优化器（避免 OOM）
from bitsandbytes.optim import PagedAdamW8bit

optimizer = PagedAdamW8bit(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
)

# 3. 8-bit 优化器（非分页）
from bitsandbytes.optim import Adam8bit

optimizer = Adam8bit(model.parameters(), lr=1e-4)
```

---

## 8. 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| OOM (显存不足) | 量化后仍有参数更新 | 使用分页优化器 |
| 训练不稳定 | NF4 量化精度 | 使用双重量化 + FP16 计算 |
| 速度很慢 | 量化/反量化开销 | 使用 `torch.compile` 优化 |
| 加载失败 | bitsandbytes 版本 | 更新 `pip install -U bitsandbytes` |

---

## 相关文档

- [02_LoRA实现详解.md](02_LoRA实现详解.md) - LoRA 核心实现
- [08_实战配置指南.md](08_实战配置指南.md) - 完整训练脚本

---

*文档生成日期: 2026-04-20*
