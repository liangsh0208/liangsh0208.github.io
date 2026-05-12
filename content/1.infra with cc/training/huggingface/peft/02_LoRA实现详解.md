# 02_LoRA实现详解

## 文档元信息

| 属性 | 内容 |
|------|------|
| 源码定位 | `/Users/danchen/Documents/1.RL_fw/huggingface/peft/src/peft/tuners/lora/` |
| 核心文件 | `layer.py`, `model.py`, `config.py` |
| 论文参考 | LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021) |

---

## 1. LoRA 核心公式

$$
W' = W + \frac{\alpha}{r} \cdot B \cdot A
$$

其中：
- $W$：原始权重矩阵（冻结）
- $A$：低秩分解矩阵，形状 $(r \times in\_features)$
- $B$：低秩分解矩阵，形状 $(out\_features \times r)$
- $r$：低秩维度
- $\alpha$：缩放系数
- $\frac{\alpha}{r}$：缩放因子

---

## 2. LoraConfig 配置

**文件**: `tuners/lora/config.py`

```python
@dataclass
class LoraConfig(PeftConfig):
    """LoRA 具体配置"""
    
    r: int = 8                        # 秩 / 低秩维度
    lora_alpha: int = 8               # 缩放系数
    target_modules: Optional[list] = None  # 目标模块名
    lora_dropout: float = 0.0         # Dropout 率
    bias: str = "none"                # 偏置训练策略
    use_rslora: bool = False           # 是否使用 Rank-Stabilized LoRA
    use_dora: bool = False             # 是否使用 DoRA
    init_lora_weights: bool = True     # 初始化方法
    modules_to_save: Optional[list] = None   # 额外训练的模块
```

### 2.1 关键参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `r` | int | 8 | 低秩维度，越大表示更多可训练参数 |
| `lora_alpha` | int | 8 | 缩放系数，控制 LoRA 更新的幅度 |
| `target_modules` | list | None | 要替换的模块名列表 |
| `lora_dropout` | float | 0.0 | LoRA 层的 dropout 率 |
| `bias` | str | "none" | 是否训练偏置 |
| `use_rslora` | bool | False | Rank-Stabilized LoRA |
| `use_dora` | bool | False | Weight-Decomposed LoRA |

---

## 3. LoraLayer 实现

**文件**: `tuners/lora/layer.py`

```python
class LoraLayer(BaseTunerLayer):
    """
    LoRA 层的核心实现。
    公式: W' = W + (alpha/r) * B * A
    """
    
    def __init__(self, base_layer, **kwargs):
        super().__init__()
        self.base_layer = base_layer  # 原始层 (如 nn.Linear)
        
        # 各 adapter 的配置
        self.r = {}          # 秩
        self.lora_alpha = {} # alpha 值
        self.scaling = {}    # 缩放因子 (alpha/r)
        
        # LoRA 参数
        self.lora_A = nn.ModuleDict()  # 形状: (r, in_features)
        self.lora_B = nn.ModuleDict()  # 形状: (out_features, r)
        
        # DoRA 支持
        self.lora_magnitude_vector = nn.ModuleDict()
```

### 3.1 层初始化

```python
def update_layer(self, adapter_name, r, lora_alpha, config, **kwargs):
    """创建 LoRA 参数并初始化"""
    
    # 1. 获取输入/输出维度
    in_features = self.base_layer.in_features
    out_features = self.base_layer.out_features
    
    # 2. 创建 A 和 B 参数
    self.lora_A[adapter_name] = nn.Linear(in_features, r, bias=False)
    self.lora_B[adapter_name] = nn.Linear(r, out_features, bias=False)
    
    # 3. 初始化
    self.reset_lora_parameters(adapter_name, init_lora_weights)
    
    # 4. 计算缩放因子
    self.scaling[adapter_name] = lora_alpha / r
    
    # 5. RS-LoRA: 使用 sqrt(r) 除法
    if config.use_rslora:
        self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
```

### 3.2 初始化方法

| 初始化 | 状态 |
|--------|------|
| `True` (默认) | A: Kaiming 均匀, B: 零初始化 |
| `"gaussian"` | A: 正态分布, B: 零初始化 |
| `"pissa"` | 主奇异值分解初始化 |
| `"olora"` | OLoRA 正交初始化 |
| `"eva"` | 数据驱动的 SVD 初始化 |
| `"loftq"` | 量化感知初始化 |

```python
def reset_lora_parameters(self, adapter_name, init_lora_weights):
    """
    标准初始化: A 用 Kaiming, B 用零
    确保训练初期: W' = W + 0 = W，不改变基础模型输出
    """
    if init_lora_weights is True:
        # A: Kaiming 均匀初始化
        nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
        # B: 零初始化
        nn.init.zeros_(self.lora_B[adapter_name].weight)
```

### 3.3 Forward 计算

```python
def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    # 1. 计算基础模型输出
    result = self.base_layer(x, *args, **kwargs)
    
    # 2. 获取当前激活的 adapter
    adapter = self.active_adapters[0] if self.active_adapters else None
    
    if adapter is None or adapter not in self.lora_A:
        return result
    
    # 3. 计算 LoRA 分支: B * A * x
    lora_output = self.lora_B[adapter](self.lora_A[adapter](x))
    
    # 4. 缩放并加到基础输出
    result += lora_output * self.scaling[adapter]
    
    return result
```

---

## 4. LoraModel (Tuner 实现)

**文件**: `tuners/lora/model.py`

```python
class LoraModel(BaseTuner):
    """
    LoRA Tuner 实现，负责:
    1. 解析目标模块
    2. 注入 LoraLayer
    3. 管理多个 adapter
    """
    
    prefix: str = "lora_"
    
    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)
        # 基类会自动调用 inject_adapter
```

### 4.1 模块注入流程

```python
def _create_and_replace(self, config, adapter_name, target, target_name, parent):
    """将目标模块替换为带 LoRA 的包装模块"""
    
    # 1. 创建 LoraLayer 包装
    new_module = self._create_new_module(config, adapter_name, target)
    
    # 2. 替换原模块
    setattr(parent, target_name, new_module)
```

---

## 5. DoRA (可选增强)

**Weight-Decomposed LoRA** 将幅度和方向解耦：

```python
if config.use_dora:
    # 计算权重幅度
    self.lora_magnitude_vector[adapter_name] = nn.Parameter(
        torch.norm(base_layer.weight, dim=1)
    )
```

---

## 6. 优化器组合建议

```python
# 1. 标准 LoRA
# 学习率：1e-4 ~ 2e-4
# batch size：可以较大，因为只更新少量参数

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    use_rslora=True,
)

# 2. DoRA (对低 rank 更优)
# rank 可以用更小（如 4-8），因为 DoRA 学习幅度和方向分离
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    use_dora=True,
)

# 3. RS-LoRA (大 rank)
# rank 可以用更大（如 256-512）
# scaling = alpha / sqrt(r) 而非 alpha / r
peft_config = LoraConfig(
    r=256,
    lora_alpha=32,
    use_rslora=True,
)
```

---

## 相关文档

- [01_PeftModel包装器.md](01_PeftModel包装器.md) - PeftModel 基类
- [03_QLoRA量化训练.md](03_QLoRA量化训练.md) - 量化训练
- [07_模型层映射表.md](07_模型层映射表.md) - target_modules 映射

---

*文档生成日期: 2026-04-20*
