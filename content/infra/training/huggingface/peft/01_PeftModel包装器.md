# 01_PeftModel包装器

## 文档元信息

| 属性 | 内容 |
|------|------|
| 源码定位 | `/Users/danchen/Documents/1.RL_fw/huggingface/peft/src/peft/peft_model.py` |
| 依赖模块 | `mapping.py`, `tuners/tuners_utils.py` |
| 核心类 | `PeftModel`, `PeftModelForCausalLM`, `AutoPeftModel` |

---

## 1. get_peft_model 入口函数

**文件**: `mapping_func.py`

`get_peft_model()` 是所有 PEFT 方法的统一入口：

```python
from peft import get_peft_model, LoraConfig

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"]
)

# 包装基础模型，注入 Adapter
model = get_peft_model(base_model, peft_config)
```

### 1.1 内部逻辑

```python
def get_peft_model(model, peft_config, adapter_name="default", mixed=False, **kwargs):
    """
    1. 根据 peft_config.peft_type 获取对应的 tuner 类
    2. 创建 PeftModel 实例
    3. 注入 adapter 层
    4. 冻结基础模型参数
    """
    # 确定 PEFT 模型类
    peft_model_class = MODEL_TYPE_TO_PEFT_MODEL_MAPPING.get(
        peft_config.task_type, PeftModel
    )
    
    # 创建实例
    return peft_model_class(model, peft_config, adapter_name=adapter_name, **kwargs)
```

---

## 2. PeftModel 基类

**文件**: `peft_model.py`

```python
class PeftModel(PushToHubMixin, torch.nn.Module):
    """
    所有 PEFT 方法的统一包装器。
    通过 PEFT_TYPE_TO_TUNER_MAPPING 路由到具体的 tuner 实现。
    """
    
    def __init__(self, model, peft_config, adapter_name="default", ...):
        super().__init__()
        
        # 1. 存储基础模型
        self.base_model = model
        
        # 2. 根据 peft_type 获取对应的 tuner 类
        cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]
        
        # 3. 创建 tuner 实例，注入 adapter 层
        self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)
        
        # 4. 处理梯度检查点、数据类型转换等
        ...
```

### 2.1 核心属性

| 属性 | 说明 |
|------|------|
| `self.base_model` | 被包装的基础模型（注入 adapter 后） |
| `self.peft_config` | PEFT 配置字典 |
| `self.active_adapter` | 当前激活的 adapter 名称 |

---

## 3. forward 包装

### 3.1 基类 forward

```python
def forward(self, *args, **kwargs):
    """
    调用基础模型的 forward，adapter 层会自动生效
    因为 adapter 已在初始化时注入到模型结构中
    """
    return self.base_model(*args, **kwargs)
```

### 3.2 任务特定的 forward

针对不同 `task_type`，PEFT 提供专门的模型类：

```python
MODEL_TYPE_TO_PEFT_MODEL_MAPPING = {
    "SEQ_CLS": PeftModelForSequenceClassification,
    "SEQ_2_SEQ_LM": PeftModelForSeq2SeqLM,
    "CAUSAL_LM": PeftModelForCausalLM,
    "TOKEN_CLS": PeftModelForTokenClassification,
    "QUESTION_ANS": PeftModelForQuestionAnswering,
    "FEATURE_EXTRACTION": PeftModelForFeatureExtraction,
}
```

### 3.3 PeftModelForCausalLM 示例

```python
class PeftModelForCausalLM(PeftModel):
    def forward(self, input_ids=None, ...):
        # 处理 prompt learning 的虚拟 token
        if self.peft_config.is_prompt_learning:
            batch_size = _get_batch_size(input_ids, inputs_embeds)
            # 将虚拟 token 拼接到输入前
            ...
        
        # 调用基础模型 forward
        return self.base_model(...)
```

---

## 4. save_pretrained 方法

```python
def save_pretrained(self, save_directory, ...):
    """
    只保存 adapter 参数，不保存基础模型
    显著减小存储需求 (通常从 GB 级降到 MB 级)
    """
    # 1. 创建保存目录
    os.makedirs(save_directory, exist_ok=True)
    
    # 2. 保存配置
    self.peft_config.save_pretrained(save_directory)
    
    # 3. 只保存 adapter 权重
    output_state_dict = get_peft_model_state_dict(
        self, state_dict=kwargs.get("state_dict", None)
    )
    
    # 4. 保存权重文件
    save_filename = os.path.join(save_directory, "adapter_model.safetensors")
    safe_save_file(output_state_dict, save_filename)
```

---

## 5. AutoPeftModel 自动加载

```python
from peft import AutoPeftModelForCausalLM

# 自动从 PEFT 适配器配置推断基础模型
model = AutoPeftModelForCausalLM.from_pretrained(
    "user/my-lora-adapter",
    device_map="auto"
)
# 上述代码会自动加载配置中的 base_model_name_or_path 指定的基础模型
```

---

## 6. 打印可训练参数

```python
model.print_trainable_parameters()

# 输出示例:
# trainable params: 9,437,184 || all params: 2,994,922,752 || trainable%: 0.3151
```

---

## 相关文档

- [00_整体架构与设计理念.md](./00_整体架构与设计理念.md) - 整体架构概览
- [02_LoRA实现详解.md](./02_LoRA实现详解.md) - Tuner 与 Adapter 层实现
- [06_Adapter合并与导出.md](./06_Adapter合并与导出.md) - 合并与导出详情

---

*文档生成日期: 2026-04-20*
