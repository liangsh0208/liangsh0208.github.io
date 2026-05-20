---
created: 2026-05-09
---

# ms-swift PEFT/Tuners：参数高效微调方法实现

ms-swift 实现了多种轻量微调方法，同时通过桥接层复用 HuggingFace `peft` 库的成熟实现。本文档解析 `SwiftModel` 统一包装器、各类 tuner 的实现细节、以及训练管道中的集成方式。

---

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      SwiftModel (nn.Module)                 │
│  ─────────────────────────────────────────────────────────  │
│  base_model: PreTrainedModel                               │
│  adapters: Dict[str, SwiftOutput]                         │
│  active_adapters: Set[str]                                 │
│                                                             │
│  + activate_adapter(name)                                  │
│  + deactivate_adapter(name)                                │
│  + merge_and_unload()                                      │
│  + state_dict(adapter_name=...)                            │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   LoRA (native)      Adapter (native)     PEFT LoRA
   swift/tuners/       swift/tuners/         swift/tuners/
   lora.py             adapter.py            peft.py
```

---

## 2. SwiftModel — 统一包装器

**文件**: `swift/tuners/base.py` (~44KB)

`SwiftModel` 是 ms-swift 所有 tuner 的**统一基座**，解决了一个核心问题：当模型上同时存在多个 adapter（如 LoRA + Adapter）时，如何管理它们的激活/切换/合并/保存。

### 2.1 初始化

```python
class SwiftModel(nn.Module):
    EXTRA_STATE_DIR = 'extra_states'

    def __init__(self, model, config, extra_state_keys=None, inference_mode=False):
        super().__init__()
        self.adapters = {}
        self.active_adapters = set()
        
        # 解嵌套 SwiftModel (支持多层包装)
        if isinstance(model, SwiftModel):
            self.adapters = model.adapters
            self.active_adapters = model.active_adapters
            model = model.base_model
        
        self.base_model = model
        
        # 注册 adapter
        if isinstance(config, SwiftConfig):
            self.adapters[DEFAULT_ADAPTER] = self._prepare_model(model, config, DEFAULT_ADAPTER)
        elif isinstance(config, dict):
            for adapter_name, _config in config.items():
                self.adapters[adapter_name] = self._prepare_model(model, _config, adapter_name)
        
        # 设置 forward 函数签名与 base_model 一致
        self.forward = MethodType(self._create_forward(), self)
        
        # 激活所有新注册的 adapter
        for adapter_name in new_adapters:
            self.activate_adapter(adapter_name)
```

### 2.2 Adapter 激活与切换

```python
def activate_adapter(self, adapter_name):
    """激活指定 adapter，将其参数设为可训练"""
    output = self.adapters[adapter_name]
    output.activate_callback(self.base_model, adapter_name)
    self.active_adapters.add(adapter_name)

def deactivate_adapter(self, adapter_name):
    """冻结指定 adapter，使其不参与训练"""
    output = self.adapters[adapter_name]
    output.deactivate_callback(self.base_model, adapter_name)
    self.active_adapters.discard(adapter_name)
```

**应用场景**:
- 模型融合：同时加载多个 LoRA，按需切换
- A/B 测试：同一基模型上对比不同 adapter 效果
- RLHF：policy 模型和 ref 模型共享基座，通过 adapter 切换

### 2.3 状态字典管理

```python
def state_dict(self, adapter_name=None, peft_format=False, **kwargs):
    """
    获取 adapter 状态字典
    
    Args:
        adapter_name: 指定 adapter，None 则获取全部
        peft_format: 是否添加 'base_model.model.' 前缀（兼容 peft 格式）
    """
```

### 2.4 合并与卸载

```python
def merge_and_unload(self, adapter_name=None):
    """将 LoRA 权重合并到基模型，卸载 adapter"""
    
def unload(self):
    """卸载所有 adapter，恢复原始模型"""
```

---

## 3. LoRA 实现

### 3.1 LoRAConfig

**文件**: `swift/tuners/lora.py`

```python
@dataclass
class LoRAConfig(LoraConfig, SwiftConfig):
    use_qa_lora: bool = False              # QA-LoRA (量化感知)
    use_merged_linear: bool = False       # 使用合并的 Linear 层
    enable_lora: List[bool] = None       # 合并线性层中哪些部分启用
    lora_dtype: Optional[str] = None    # LoRA 专用 dtype (fp32/fp16/bf16)
    lorap_lr_ratio: float = 2.0**4        # LoRA+ 的 B 矩阵学习率倍数
    lorap_emb_lr: float = 1e-6          # LoRA+ embedding 学习率
```

### 3.2 LoRA 准备流程

```python
class LoRA(SwiftAdapter):
    @staticmethod
    def prepare_model(model, config, adapter_name):
        # 1. 构建 LoraModel (复用 peft.LoraModel)
        LoraModel(model, config, adapter_name)
        
        # 2. 注册回调函数
        def state_dict_callback(state_dict, adapter_name, cfg=None):
            return lora_state_dict(state_dict, adapter_name, cfg.bias)
        
        def mark_trainable_callback(model, cfg=None):
            mark_lora_as_trainable(model, adapter_name, cfg.bias)
        
        def optimizer_group_callback(model, **defaults):
            # LoRA+: 为 lora_B 设置更高的学习率
            if config.lorap_lr_ratio is None:
                return None, None
            # 按模块分组，lora_B 的学习率 = base_lr * lorap_lr_ratio
            ...
```

### 3.3 LoRA 层扩展

**文件**: `swift/tuners/lora_layers.py` (~28KB)

自定义 LoRA 层实现，支持：
- 量化感知 LoRA (与 GPTQ/AWQ 量化模型兼容)
- 合并线性层（MergedLinear）
- Embedding LoRA

```python
class LoraLayer:
    def __init__(self, ...):
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = lora_alpha / r
```

### 3.4 LoRA+ 优化

LoRA+ 的核心思想：为 LoRA 的 B 矩阵设置比 A 矩阵更高的学习率（通常 `lora_B_lr = 16 * lora_A_lr`）。

```python
def optimizer_group_callback(model, **defaults):
    if config.lorap_lr_ratio is None:
        return None, None
    
    # 收集 lora_A 和 lora_B 的参数
    lora_A_params = []
    lora_B_params = []
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            lora_A_params.append(param)
        elif 'lora_B' in name:
            lora_B_params.append(param)
    
    return [
        {'params': lora_A_params, 'lr': defaults['lr']},
        {'params': lora_B_params, 'lr': defaults['lr'] * config.lorap_lr_ratio},
    ], defaults
```

---

## 4. 其他原生 Tuners

### 4.1 Adapter — Bottleneck 适配器

**文件**: `swift/tuners/adapter.py`

```python
class AdapterModule(nn.Module):
    def __init__(self, in_features, bottleneck_dim):
        self.down_proj = nn.Linear(in_features, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, in_features)
    
    def forward(self, x):
        return x + self.up_proj(F.relu(self.down_proj(x)))
```

在目标层后插入 bottleneck adapter，仅训练 adapter 参数。

### 4.2 Prompt Tuning / Prefix Tuning

**文件**: `swift/tuners/prompt.py`

```python
class PromptModule(nn.Module):
    """Prompt Tuning: 在输入前添加可训练的 soft prompt tokens"""
    def __init__(self, num_tokens, token_dim):
        self.prompt_embeddings = nn.Embedding(num_tokens, token_dim)
```

### 4.3 ReFT — Representation Fine-Tuning

**文件**: `swift/tuners/reft.py`

ReFT 不在权重空间做低秩更新，而是在**隐藏状态层**施加可训练的变换：

```python
class ReFTModule(nn.Module):
    def __init__(self, hidden_dim, reft_rank):
        self.reft_transform = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, hidden_states):
        return hidden_states + self.reft_transform(hidden_states)
```

### 4.4 LLaMA-Pro

**文件**: `swift/tuners/llamapro.py`

LLaMA-Pro 通过在相邻 Transformer 层之间**插入新层**并仅训练新层来增强模型能力：

```python
def expand_layers(model, expansion_ratio=2):
    """在现有层之间插入新层"""
    original_layers = model.model.layers
    new_layers = []
    for i in range(len(original_layers) - 1):
        new_layers.append(original_layers[i])  # 原层冻结
        # 插入新层（从相邻层平均初始化）
        new_layer = create_expanded_layer(original_layers[i], original_layers[i+1])
        new_layers.append(new_layer)  # 新层可训练
    return new_layers
```

### 4.5 NEFTune

**文件**: `swift/tuners/neftune.py`

在训练时向 embedding 添加噪声，提升指令遵循的鲁棒性：

```python
class NEFTune:
    def __init__(self, noise_alpha=5):
        self.noise_alpha = noise_alpha
    
    def forward(self, input_embeds):
        noise = torch.randn_like(input_embeds) * self.noise_alpha / (input_embeds.shape[1] ** 0.5)
        return input_embeds + noise
```

### 4.6 其他方法

| Tuner | 文件 | 说明 |
|-------|------|------|
| LongLoRA | `swift/tuners/longlora/` | 长上下文 LoRA，扩展位置编码 |
| ResTuning | `swift/tuners/restuning.py` | 残差调优 |
| SCETuning | `swift/tuners/scetuning/` | 选择性单元调优 |
| Side | `swift/tuners/side.py` | Side network 调优 |

---

## 5. PEFT 桥接层

**文件**: `swift/tuners/peft.py` (~15KB)

复用 HuggingFace `peft` 库的 tuner 实现：

```python
from peft import (
    LoraConfig as PeftLoraConfig,
    AdaLoraConfig,
    BOFTConfig,
    LoHaConfig,
    LoKrConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    VeraConfig,
    ...
)

class PeftAdapter(SwiftAdapter):
    @staticmethod
    def prepare_model(model, config, adapter_name):
        # 使用 peft.get_peft_model() 包装模型
        peft_model = get_peft_model(model, config, adapter_name)
        return peft_model
```

通过桥接层支持的 tuner 类型：

| 方法 | 来源 |
|-----|------|
| AdaLoRA | peft |
| BOFT | peft |
| DoRA | peft |
| LoHa | peft |
| LoKr | peft |
| OFT | peft |
| Vera | peft |
| Prefix Tuning | peft |
| Prompt Tuning | peft |
| IA3 | peft (via tuner_plugin) |

---

## 6. Tuner 注册与映射

### 6.1 全局映射

**文件**: `swift/tuners/mapping.py`

```python
class SwiftTuners(Enum):
    LORA = 'lora'
    ADAPTER = 'adapter'
    PROMPT = 'prompt'
    REFT = 'reft'
    PART = 'part'          # LLaMA-Pro
    SIDE = 'side'
    ...

SWIFT_MAPPING = {
    SwiftTuners.LORA: LoRA,
    SwiftTuners.ADAPTER: Adapter,
    SwiftTuners.PROMPT: Prompt,
    SwiftTuners.REFT: ReFT,
    ...
}
```

### 6.2 TunerMixin 管道集成

**文件**: `swift/pipelines/train/tuner.py` (~19KB)

```python
class TunerMixin:
    def prepare_model(self, args, model, task_type=None):
        if args.tuner_type == 'full':
            return self._prepare_full_model(args, model)
        
        # 构建 tuner 配置
        tuner = tuners_map[args.tuner_type]
        config = self._build_tuner_config(args, model)
        
        # 调用 Swift.prepare_model
        model = Swift.prepare_model(model, config)
        
        return model
    
    def _build_tuner_config(self, args, model):
        target_modules = get_target_modules(args, model)
        modules_to_save = get_modules_to_save(args, model)
        
        if args.tuner_type == 'lora':
            return LoRAConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                modules_to_save=modules_to_save,
                lora_dropout=args.lora_dropout,
                use_rslora=args.use_rslora,
                lorap_lr_ratio=args.lorap_lr_ratio,
                ...
            )
```

### 6.3 target_modules 解析

```python
def get_target_modules(args, model):
    target_modules = args.target_modules.copy()
    if 'all-linear' in target_modules:
        target_modules.remove('all-linear')
        target_modules += find_all_linears(model)
    if 'all-embedding' in target_modules:
        target_modules.remove('all-embedding')
        target_modules += find_embedding(model)
    return target_modules
```

---

## 7. RLHF 场景下的 LoRA 插件

**文件**: `swift/tuner_plugin/`

RLHF 训练中，policy 模型需要加载 LoRA，而 ref/reward 模型不需要更新。`tuner_plugin` 提供了针对 RLHF 的 LoRA 管理：

```python
# swift/tuner_plugin/lora_llm.py
class LoRALLMPlugin:
    def prepare_model(self, args, model):
        # 为 policy 模型准备 LoRA
        # 为 ref 模型冻结所有 LoRA
        ...

# swift/tuner_plugin/ia3.py
class IA3Plugin:
    # IA3 的 RLHF 适配
```

---

## 8. 关键代码路径索引

| 功能 | 关键文件/函数 |
|-----|-------------|
| SwiftModel 包装器 | `swift/tuners/base.py::SwiftModel` |
| LoRA 配置 | `swift/tuners/lora.py::LoRAConfig` |
| LoRA 层实现 | `swift/tuners/lora_layers.py` |
| Adapter | `swift/tuners/adapter.py::AdapterModule` |
| Prompt Tuning | `swift/tuners/prompt.py::PromptModule` |
| ReFT | `swift/tuners/reft.py::ReFTModule` |
| LLaMA-Pro | `swift/tuners/llamapro.py` |
| NEFTune | `swift/tuners/neftune.py` |
| LongLoRA | `swift/tuners/longlora/` |
| PEFT 桥接 | `swift/tuners/peft.py` |
| Tuner 映射 | `swift/tuners/mapping.py::SWIFT_MAPPING` |
| Tuner 管道集成 | `swift/pipelines/train/tuner.py::TunerMixin` |
| RLHF LoRA 插件 | `swift/tuner_plugin/lora_llm.py` |
| IA3 插件 | `swift/tuner_plugin/ia3.py` |
| 目标模块解析 | `swift/pipelines/train/tuner.py::get_target_modules()` |
| 多模态目标正则 | `swift/utils/model_utils.py::get_multimodal_target_regex()` |
