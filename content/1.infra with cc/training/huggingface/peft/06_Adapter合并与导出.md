---
created: 2026-05-06
---

# 06_Adapter合并与导出

## 文档元信息

| 属性 | 内容 |
|------|------|
| 源码定位 | `/Users/danchen/Documents/1.RL_fw/huggingface/peft/src/peft/peft_model.py` |
| 核心方法 | `merge_and_unload`, `save_pretrained` |
| 辅助文件 | `utils/merge_utils.py` |

---

## 1. merge_and_unload - 合并并卸载

将 Adapter 参数合并回基础模型，并返回普通模型（不再依赖 PEFT）：

```python
# 合并后不再是 PEFT 模型，性能更好但无法切换 adapter
merged_model = model.merge_and_unload()

# 保存合并后的完整模型
merged_model.save_pretrained("./merged_model")
```

### 1.1 合并原理

```
合并前: W' = W + (alpha/r) * B * A
合并后: W_new = W' (永久修改)
```

### 1.2 方法签名

```python
def merge_and_unload(
    self,
    progressbar: bool = False,     # 显示进度条
    safe_merge: bool = False,        # 安全合并（检查冲突）
    adapter_names: Optional[List[str]] = None  # 指定要合并的 adapter
) -> torch.nn.Module:
    """
    1. 将 adapter 权重合并到基础模型
    2. 返回合并后的普通 PyTorch 模型
    3. 不再支持 adapter 切换
    """
    # 遍历所有 LoraLayer
    for module in self.model.modules():
        if isinstance(module, LoraLayer):
            # 执行合并: W = W + (alpha/r) * B @ A
            module.merge(adapter_names)
    
    # 返回去包装的基础模型
    return self.model
```

### 1.3 安全合并

```python
# 安全合并会检查权重冲突
model.merge_and_unload(safe_merge=True)

# 如果检测到异常值或冲突，会发出警告或报错
```

---

## 2. save_pretrained - 保存 Adapter

### 2.1 基本用法

```python
# 只保存 adapter 参数（约 20MB）
model.save_pretrained("./qwen_lora_adapter")

# 目录内容:
# ./qwen_lora_adapter/
#   ├── adapter_config.json      # 配置
#   └── adapter_model.safetensors  # 权重
```

### 2.2 方法实现

```python
def save_pretrained(
    self,
    save_directory: str,                    # 保存目录
    safe_serialization: bool = True,         # 使用 safetensors
    selected_adapters: Optional[List[str]] = None,  # 指定保存的 adapter
    **kwargs
) -> None:
    """
    保存 adapter 配置和权重，不保存基础模型
    """
    # 1. 创建保存目录
    os.makedirs(save_directory, exist_ok=True)
    
    # 2. 保存配置
    self.peft_config[adapter_name].save_pretrained(save_directory)
    
    # 3. 只提取 adapter 权重
    output_state_dict = get_peft_model_state_dict(
        self,
        state_dict=kwargs.get("state_dict", None),
        adapter_name=adapter_name
    )
    
    # 4. 保存权重
    if safe_serialization:
        save_filename = os.path.join(save_directory, "adapter_model.safetensors")
        safe_save_file(output_state_dict, save_filename)
    else:
        save_filename = os.path.join(save_directory, "adapter_model.bin")
        torch.save(output_state_dict, save_filename)
```

### 2.3 保存特定 Adapter

```python
# 保存多个 adapter 中的特定一个
model.save_pretrained(
    "./my_adapter",
    selected_adapters=["summarize"]  # 只保存 "summarize" adapter
)
```

---

## 3. 多 Adapter 高级合并

### 3.1 使用 TIES 算法

```python
from peft import load_peft_weights
from peft.utils.merge_utils import ties_merge

# 加载多个 adapter 权重
adapter1 = load_peft_weights("adapter1_path")
adapter2 = load_peft_weights("adapter2_path")
adapter3 = load_peft_weights("adapter3_path")

# 使用 TIES 合并 (Trim, Elect Sign & Merge)
merged = ties_merge([adapter1, adapter2, adapter3], density=0.6)

# 加载合并后的权重
model.load_state_dict(merged, strict=False)
```

### 3.2 使用 DARE 算法

```python
from peft.utils.merge_utils import dare_linear

# 使用 DARE (Drop And REscale) 合并
task_arithmetic(adapter1, adapter2, merge_func=dare_linear)
```

### 3.3 合并算法对比

| 算法 | 特点 | 适用场景 |
|------|------|----------|
| **Linear** | 简单加权平均 | 相似任务 |
| **TIES** | 修剪不重要参数后合并 | 多任务组合 |
| **DARE** | 随机丢弃后重缩放 | 大量模型合并 |
| **Slerp** | 球面插值 | 两个模型平滑过渡 |

---

## 4. 完整工作流示例

### 4.1 训练并保存

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B")

# 2. 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
)

# 3. 应用 LoRA
model = get_peft_model(model, lora_config)

# 4. 训练...
trainer.train()

# 5. 保存 adapter（约 20MB）
model.save_pretrained("./my_lora_adapter")

# 目录内容:
# ./my_lora_adapter/
#   ├── adapter_config.json
#   └── adapter_model.safetensors
```

### 4.2 加载并合并

```python
from peft import PeftModel

# 1. 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B")

# 2. 加载 adapter
model = PeftModel.from_pretrained(base_model, "./my_lora_adapter")

# 3. 可选：合并 adapter（用于部署）
merged_model = model.merge_and_unload()

# 4. 保存合并后的模型（完整模型，较大）
merged_model.save_pretrained("./merged_qwen_model")

# 或保留分离（支持切换，适合开发）
# model.save_pretrained("./adapter_only")
```

### 4.3 模型部署选择

| 场景 | 推荐方案 | 文件大小 |
|------|----------|----------|
| **云端部署** | 分离加载（base + adapter） | Base: 6GB + Adapter: 20MB |
| **边缘部署** | 合并后单文件 | 6GB |
| **多租户** | 基础模型共享 + 各用户 adapter | Base: 6GB 共享 + N * 20MB |

---

## 5. 关键 API 汇总

| 方法 | 用途 | 返回值 |
|------|------|--------|
| `save_pretrained()` | 保存 adapter 配置和权重 | None |
| `merge_and_unload()` | 合并并返回普通模型 | `nn.Module` |
| `merge_adapter()` | 仅合并，保持 PEFT 结构 | None |
| `unload_adapter()` | 卸载 adapter 恢复基础模型 | None |

---

## 6. 常见问题

### 6.1 合并失败

```python
# 错误：量化模型不能直接合并
model = AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True)
model = PeftModel.from_pretrained(model, "adapter")
merged = model.merge_and_unload()  # ❌ 可能失败

# 解决：先反量化或加载 FP16 模型
model = model.to(torch.float16)
merged = model.merge_and_unload()
```

### 6.2 权重不匹配

```python
# 确保基础模型版本正确
# adapter_config.json 中的 base_model_name_or_path 必须匹配
```

---

## 相关文档

- [01_PeftModel包装器.md](01_PeftModel包装器.md) - PeftModel 基类
- [05_多Adapter管理.md](05_多Adapter管理.md) - 多 Adapter 管理

---

*文档生成日期: 2026-04-20*
