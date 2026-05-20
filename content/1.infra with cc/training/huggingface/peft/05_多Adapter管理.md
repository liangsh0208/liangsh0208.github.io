---
created: 2026-05-06
---

# 05_多Adapter管理

## 文档元信息

| 属性 | 内容 |
|------|------|
| 源码定位 | `/Users/danchen/Documents/1.RL_fw/huggingface/peft/src/peft/peft_model.py` |
| 核心方法 | `load_adapter`, `set_adapter`, `add_weighted_adapter` |
| 应用场景 | 多任务、多领域、模型组合 |

---

## 1. 概述

PEFT 支持在同一个基础模型上加载和管理**多个 Adapter**，实现：
- 多任务切换（翻译、总结、问答等）
- 多领域适配（法律、医疗、金融等）
- Adapter 组合与插值

---

## 2. load_adapter - 加载新 Adapter

### 2.1 基本用法

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# 1. 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# 2. 装载第一个 adapter（默认名称 "default"）
model = PeftModel.from_pretrained(base_model, "user/llama2-lora-translation")

# 3. 装载第二个 adapter（指定名称）
model.load_adapter("user/llama2-lora-summarization", adapter_name="summarize")

# 4. 装载第三个 adapter
model.load_adapter("user/llama2-lora-qa", adapter_name="qa")
```

### 2.2 方法签名

```python
def load_adapter(
    self,
    model_id: str,                    # Adapter 路径或 HuggingFace ID
    adapter_name: str = "default",       # Adapter 名称
    is_trainable: bool = False,         # 是否可训练
    **kwargs
) -> None:
    """
    从指定路径加载 Adapter 并注册到模型
    """
    # 1. 加载 adapter 配置和权重
    peft_config = PeftConfig.from_pretrained(model_id)
    adapters_weights = load_peft_weights(model_id)
    
    # 2. 注入 adapter
    self._register_loaded_adapter(adapter_name, peft_config, adapters_weights)
```

---

## 3. set_adapter - 切换 Adapter

### 3.1 基本用法

```python
# 切换到翻译 adapter
model.set_adapter("default")
translation = model.generate(**inputs)

# 切换到总结 adapter
model.set_adapter("summarize")
summary = model.generate(**inputs)

# 切换到 QA adapter
model.set_adapter("qa")
answer = model.generate(**inputs)
```

### 3.2 获取可用适配器

```python
# 查看所有已加载的 adapter
print(model.peft_config.keys())
# dict_keys(['default', 'summarize', 'qa'])
```

---

## 4. add_weighted_adapter - 加权组合

### 4.1 基本用法

将多个 Adapter 按权重组合：

```python
# 创建组合 adapter（翻译风格 + 专业领域）
model.add_weighted_adapter(
    adapters=["translation", "legal"],
    weights=[0.7, 0.3],
    adapter_name="translation_legal",
    combination_type="cat",  # 组合方式
)
```

### 4.2 组合类型

| 类型 | 说明 |
|------|------|
| `"linear"` | 线性加权平均 |
| `"slerp"` | 球面线性插值 |
| `"cat"` | 拼接（支持秩相加） |

### 4.3 方法签名

```python
def add_weighted_adapter(
    self,
    adapters: List[str],                    # 源 adapter 名称列表
    weights: List[float],                   # 对应权重
    adapter_name: str,                       # 新 adapter 名称
    combination_type: str = "linear",        # 组合类型
    **kwargs
) -> None:
    """
    按权重组合多个现有 adapter 创建新 adapter
    """
```

---

## 5. 完整示例：多任务模型

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# 1. 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)

# 2. 装载第一个 adapter（翻译任务）
model = PeftModel.from_pretrained(base_model, "user/lora-translation")

# 3. 加载其他 adapters
model.load_adapter("user/lora-summarization", adapter_name="summarize")
model.load_adapter("user/lora-coding", adapter_name="coding")
model.load_adapter("user/lora-chat", adapter_name="chat")

# 4. 定义任务切换函数
def generate_with_task(model, task_adapter, prompt):
    """使用指定 adapter 生成"""
    model.set_adapter(task_adapter)
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0])

# 5. 不同任务使用不同 adapter
translation_result = generate_with_task(model, "default", "Translate to French: Hello")
summary_result = generate_with_task(model, "summarize", "Summarize this article: ...")
code_result = generate_with_task(model, "coding", "def fibonacci(n):")

# 6. 组合 adapter（法律 + 翻译）
model.add_weighted_adapter(
    adapters=["default", "legal"],
    weights=[0.5, 0.5],
    adapter_name="legal_translation",
    combination_type="linear"
)
model.set_adapter("legal_translation")
```

---

## 6. API 参考

### 6.1 核心方法

| 方法 | 用途 |
|------|------|
| `load_adapter()` | 加载新 adapter |
| `set_adapter()` | 切换当前激活的 adapter |
| `add_weighted_adapter()` | 加权组合多个 adapter |
| `unload_adapter()` | 卸载指定 adapter |
| `delete_adapter()` | 删除指定 adapter |

### 6.2 属性

| 属性 | 说明 |
|------|------|
| `peft_config` | 所有 adapter 的配置字典 |
| `active_adapters` | 当前激活的 adapter 列表 |

---

## 7. 常见问题

### 7.1 切换不生效

```python
# 错误：直接赋值会覆盖属性
model.active_adapter = "summarize"  # ❌

# 正确：使用 set_adapter 方法
model.set_adapter("summarize")  # ✅
```

### 7.2 显存管理

```python
# 高显存模式：同时加载多个 adapter
model.load_adapter("adapter1", adapter_name="task1")
model.load_adapter("adapter2", adapter_name="task2")

# 低显存模式：按需加载/卸载
model.load_adapter("task1", adapter_name="temp")
# ... 使用 task1 ...
model.delete_adapter("temp")
model.load_adapter("task2", adapter_name="temp")
```

### 7.3 训练多个 Adapter

```python
# 可以同时训练多个 adapter（如果兼容）
model.set_adapter(["default", "summarize"])

# 检查兼容性
assert model.peft_type == PeftType.LORA  # 必须是相同类型
```

---

## 相关文档

- [01_PeftModel包装器.md](01_PeftModel包装器.md) - PeftModel 基类
- [06_Adapter合并与导出.md](06_Adapter合并与导出.md) - 合并与导出

---

*文档生成日期: 2026-04-20*
