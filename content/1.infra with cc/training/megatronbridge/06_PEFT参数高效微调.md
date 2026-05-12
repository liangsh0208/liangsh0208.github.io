# PEFT 参数高效微调

> 本文档详细分析 PEFT 参数高效微调系统的设计与实现，包括 LoRA/DoRA 实现、权重合并和张量并行兼容。

---

## 目录

1. [概述](#概述)
2. [核心：PEFT 基类](#核心peft-基类)
3. [LoRA 实现](#lora-实现)
4. [VLMLoRA](#vlmlora)
5. [LoRA 权重合并](#lora-权重合并)
6. [适配器层类型](#适配器层类型)
7. [ModuleMatcher](#modulematcher)
8. [使用示例](#使用示例)
9. [检查点过滤](#检查点过滤)
10. [相关模块](#相关模块)
11. [总结](#总结)

---

## 概述

Megatron-Bridge 提供了完整的 PEFT（Parameter-Efficient Fine-Tuning）支持，主要包括 LoRA（Low-Rank Adaptation）和 DoRA（Weight-Decomposed Low-Rank Adaptation）实现。PEFT 允许仅训练少量参数即可高效微调大型语言模型。

**源码位置**: `src/megatron/bridge/peft/`

## 核心：PEFT 基类

```python
@dataclass
class PEFT(ABC):
    """参数高效微调抽象基类。定义 PEFT 方法的接口。"""
    
    params_to_save: set[str] = field(default_factory=set, init=False, repr=False)
    """运行时状态，不应序列化到检查点"""
    
    @abstractmethod
    def transform(self, module: nn.Module, name: Optional[str] = None, 
                  prefix: Optional[str] = None) -> nn.Module:
        """
        转换单个模块。
        
        Args:
            module: 要转换的模块
            name: 模块名称
            prefix: 模块名称前缀
        
        Returns:
            转换后的模块
        """
        raise NotImplementedError("The transform method should be implemented by subclasses.")
```

### __call__ 方法

```python
def __call__(self, model: ModelType, training: bool = True) -> ModelType:
    """
    将 PEFT 方法应用到整个模型。
    
    流程：
    1. 冻结模型参数
    2. 遍历模型结构，应用 transform
    3. 启用梯度检查点（如需要）
    4. 设置训练模式
    """
    # 1. 冻结参数
    self.freeze_model(model, training=training)
    
    # 2. 应用变换
    self._walk_model(model, self.transform)
    
    # 3. 启用 recompute 输入梯度
    if training:
        maybe_enable_recompute_inputs_grad(model)
    
    # 4. 设置训练模式
    if isinstance(model, list):
        for model_chunk in model:
            model_chunk.train(mode=training)
    else:
        model.train(mode=training)
    
    return model
```

### 辅助方法

```python
def enable_adapter_layers(self, model: ModelType) -> None:
    """启用所有适配器层"""
    def enable(module: nn.Module) -> nn.Module:
        method = getattr(module, "enable_adapter_layers", None)
        if callable(method):
            method()
        return module
    self._walk_model(model, enable)

def disable_adapter_layers(self, model: ModelType) -> None:
    """禁用所有适配器层"""
    def disable(module: nn.Module) -> nn.Module:
        method = getattr(module, "disable_adapter_layers", None)
        if callable(method):
            method()
        return module
    self._walk_model(model, disable)

@contextmanager
def disable_adapter(self, model: ModelType):
    """上下文管理器：临时禁用适配器"""
    try:
        self.disable_adapter_layers(model)
        yield
    finally:
        self.enable_adapter_layers(model)

def freeze_model(self, model: ModelType, training: bool = True) -> None:
    """冻结模型参数"""
    def freeze_parameters(module):
        for param in module.parameters(recurse=False):
            param.requires_grad = False
        return module
    self._walk_model(model, freeze_parameters)

def set_params_to_save(self, model: ModelType) -> None:
    """设置需要保存的参数（仅适配器参数）"""
    models_to_process = model if isinstance(model, list) else [model]
    self.params_to_save = set()
    for model_chunk in models_to_process:
        for name, param in model_chunk.named_parameters():
            if param.requires_grad:
                self.params_to_save.add(name)

def adapter_key_filter(self, key) -> bool:
    """检查点过滤器：仅保存适配器参数"""
    if isinstance(key, tuple):
        return key[1].requires_grad
    return key in self.params_to_save or ".adapter." in key or key.endswith(".adapters")
```

## LoRA 实现

```python
@dataclass
class LoRA(PEFT, ModuleMatcher):
    """
    LoRA（Low-Rank Adaptation）实现。
    
    LoRA 使用低秩投影来适应预训练模型的权重到新任务。
    """
    
    target_modules: List[str] = field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    """目标模块名称列表，支持通配符"""
    
    dim: int = 32
    """低秩投影维度"""
    
    alpha: int = 32
    """缩放因子"""
    
    dropout: float = 0.0
    """Dropout 率"""
    
    dropout_position: Literal["pre", "post"] = "pre"
    """Dropout 位置：'pre' 或 'post'"""
    
    lora_A_init_method: str = "xavier"
    """LoRA A 矩阵初始化方法"""
    
    lora_B_init_method: str = "zero"
    """LoRA B 矩阵初始化方法"""
    
    a2a_experimental: bool = False
    """实验性 All-to-All 通信策略"""
    
    lora_dtype: torch.dtype = None
    """LoRA 权重数据类型"""
    
    normalize_moe_lora: bool = False
    """MoE 专家层的 LoRA rank 归一化"""
```

### transform 方法

```python
def transform(self, module: nn.Module, name: Optional[str] = None, 
              prefix: Optional[str] = None) -> nn.Module:
    """将 LoRA 应用到模块"""
    # 跳过已转换的模块
    adapter_types = (LinearAdapter, LoRALinear, LoRATopKRouter, TELinearAdapter)
    if isinstance(module, adapter_types):
        return module
    
    # 检查是否匹配目标模块
    if (ans := self.match(module, name, prefix)) is not None:
        (match, full_name) = ans
        
        # 处理 nn.Linear 或 TE Linear
        if isinstance(module, nn.Linear) or (module.__class__ == te.Linear):
            # 选择合适的适配器类
            if hasattr(module.weight.data, "_local_tensor") or HAVE_BNB:
                lora_cls = patch_linear_module
            elif module.__class__ == te.Linear:
                lora_cls = TELinearAdapter
            else:
                lora_cls = LinearAdapter
            
            return lora_cls(
                module,
                dim=self.dim,
                alpha=self.alpha,
                dropout=self.dropout,
                lora_A_init_method=self.lora_A_init_method,
                lora_dtype=self.lora_dtype,
            )
        
        # 处理 Megatron Core 线性层
        is_expert = is_expert_linear(full_name)
        attrs = get_adapter_attributes_from_linear(module, is_expert=is_expert)
        dim = self._get_effective_dim(module, is_expert)
        
        # 创建并行适配器
        adapter = ParallelLinearAdapter(
            attrs.in_features,
            attrs.out_features,
            dim,
            base_linear_name=full_name,
            activation="identity",
            norm_type=None,
            column_init_method=self.lora_A_init_method,
            row_init_method=self.lora_B_init_method,
            gather_output=False,
            input_is_parallel=attrs.input_is_parallel,
            dropout=self.dropout,
            dropout_position=self.dropout_position,
            model_parallel_config=getattr(module, "config", None),
            alpha=self.alpha,
            is_expert=is_expert,
            a2a_experimental=self.a2a_experimental,
            disable_tensor_parallel_comm=attrs.disable_tensor_parallel_comm,
            disable_sequence_parallel_comm=attrs.disable_sequence_parallel_comm,
            base_linear_is_parallel=attrs.base_linear_is_parallel,
        )
        
        # 返回包装后的模块
        if isinstance(module, TopKRouter):
            return LoRATopKRouter(module, adapter)
        if enable_op_fuser:
            return TEFusedLoRALinear(module, adapter)
        else:
            return LoRALinear(module, adapter)
    
    return module
```

### MoE LoRA 归一化

```python
def _get_effective_dim(self, module: nn.Module, is_expert: bool) -> int:
    """计算有效的 LoRA rank"""
    if not self.normalize_moe_lora or not is_expert:
        return self.dim
    
    topk = getattr(getattr(module, "config", None), "moe_router_topk", None)
    if topk is None or topk <= 0:
        raise ValueError("normalize_moe_lora requires moe_router_topk")
    
    if self.dim % topk != 0:
        raise ValueError(f"LoRA dim={self.dim} must be divisible by moe_router_topk={topk}")
    
    return self.dim // topk
```

## VLMLoRA

```python
@dataclass
class VLMLoRA(LoRA):
    """
    视觉-语言模型的 LoRA 实现。
    允许分别冻结视觉模型、视觉投影和语言模型。
    """
    
    freeze_vision_model: bool = True
    freeze_vision_projection: bool = True
    freeze_language_model: bool = True
    
    def freeze_model(self, model: nn.Module, training: bool = True) -> None:
        """冻结指定的模型组件"""
        modules_to_freeze = []
        
        model = unwrap_model(model)[0]
        if hasattr(model, "llava_model"):
            model = model.llava_model
        
        if self.freeze_vision_model and model.vision_model is not None:
            modules_to_freeze.append(model.vision_model)
        if self.freeze_vision_projection and model.vision_projection is not None:
            modules_to_freeze.append(model.vision_projection)
        if self.freeze_language_model and model.language_model is not None:
            modules_to_freeze.append(model.language_model)
        
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
        
        if training:
            if isinstance(model, list):
                for model_chunk in model:
                    model_chunk.train(mode=True)
            else:
                model.train(mode=True)
```

## LoRA 权重合并

```python
class LoRAMerge(PEFT):
    """LoRA 权重合并到基础模型"""
    
    def merge(
        self,
        base_weight: torch.Tensor,
        linear_out: torch.Tensor,
        linear_in: torch.Tensor,
        alpha: int,
        dim: int,
    ) -> torch.Tensor:
        """
        合并 LoRA 权重到基础权重。
        处理张量并行分片。
        
        ColumnParallelLinear:
            - base_weight: (out_features/TP, in_features)
            - linear_in: (dim/TP, in_features) ← 需要收集
            - linear_out: (out_features/TP, dim)
        
        RowParallelLinear:
            - base_weight: (out_features, in_features/TP)
            - linear_in: (dim, in_features/TP)
            - linear_out: (out_features/TP, dim) ← 需要收集
        """
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        
        if tp_size == 1:
            # 无张量并行，简单合并
            lora_weight = alpha / dim * (linear_out @ linear_in)
            return base_weight + lora_weight
        
        tp_group = parallel_state.get_tensor_model_parallel_group()
        
        # ColumnParallel: 收集 linear_in
        if linear_in.shape[0] * tp_size == dim and linear_out.shape[1] == dim:
            linear_in_list = [torch.empty_like(linear_in) for _ in range(tp_size)]
            dist.all_gather(linear_in_list, linear_in, group=tp_group)
            linear_in_full = torch.cat(linear_in_list, dim=0)
            lora_weight = alpha / dim * (linear_out @ linear_in_full)
        
        # RowParallel: 收集 linear_out
        elif linear_out.shape[0] * tp_size == base_weight.shape[0]:
            linear_out_list = [torch.empty_like(linear_out) for _ in range(tp_size)]
            dist.all_gather(linear_out_list, linear_out, group=tp_group)
            linear_out_full = torch.cat(linear_out_list, dim=0)
            lora_weight = alpha / dim * (linear_out_full @ linear_in)
        
        else:
            lora_weight = alpha / dim * (linear_out @ linear_in)
        
        return base_weight + lora_weight
```

## 适配器层类型

### LinearAdapter

```python
class LinearAdapter(nn.Module):
    """标准 Linear 层的 LoRA 适配器"""
    
    def __init__(
        self,
        to_wrap: nn.Linear,
        dim: int,
        alpha: int,
        dropout: float = 0.0,
        lora_A_init_method: str = "xavier",
        lora_dtype: torch.dtype = None,
    ):
        super().__init__()
        self.to_wrap = to_wrap
        self.dim = dim
        self.alpha = alpha
        
        in_features = to_wrap.in_features
        out_features = to_wrap.out_features
        
        # LoRA A: (dim, in_features)
        self.linear_in = nn.Linear(in_features, dim, bias=False)
        # LoRA B: (out_features, dim)
        self.linear_out = nn.Linear(dim, out_features, bias=False)
        
        # 初始化
        if lora_A_init_method == "xavier":
            nn.init.xavier_uniform_(self.linear_in.weight)
        elif lora_A_init_method == "kaiming":
            nn.init.kaiming_uniform_(self.linear_in.weight)
        
        if lora_dtype is not None:
            self.linear_in = self.linear_in.to(lora_dtype)
            self.linear_out = self.linear_out.to(lora_dtype)
```

### TELinearAdapter

```python
class TELinearAdapter(nn.Module):
    """Transformer Engine Linear 层的 LoRA 适配器"""
    
    def __init__(self, to_wrap: te.Linear, dim: int, alpha: int, ...):
        super().__init__()
        self.to_wrap = to_wrap
        # ... 类似 LinearAdapter
```

### LoRALinear

```python
class LoRALinear(nn.Module):
    """Megatron Core 线性层的 LoRA 包装器"""
    
    def __init__(self, to_wrap: nn.Module, adapter: ParallelLinearAdapter):
        super().__init__()
        self.to_wrap = to_wrap
        self.adapter = adapter
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基础输出
        base_output = self.to_wrap(x)
        # LoRA 输出
        lora_output = self.adapter(x)
        return base_output + (self.alpha / self.dim) * lora_output
```

### LoRATopKRouter

```python
class LoRATopKRouter(nn.Module):
    """MoE TopK Router 的 LoRA 包装器"""
    
    def __init__(self, router: TopKRouter, adapter: ParallelLinearAdapter):
        super().__init__()
        self.router = router
        self.adapter = adapter
```

## ModuleMatcher

```python
@dataclass
class ModuleMatcher:
    """模块匹配器，用于确定哪些模块应应用 PEFT"""
    
    target_modules: List[str] = field(default_factory=list)
    exclude_modules: List[str] = field(default_factory=list)
    
    def match(
        self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None
    ) -> Optional[Tuple[bool, str]]:
        """
        检查模块是否匹配目标。
        
        Returns:
            (True, full_name) 如果匹配
            None 如果不匹配
        """
        if not name:
            return None
        
        full_name = f"{prefix}.{name}" if prefix else name
        
        # 检查排除列表
        for pattern in self.exclude_modules:
            if fnmatch.fnmatch(full_name, pattern):
                return None
        
        # 检查目标列表
        for pattern in self.target_modules:
            if fnmatch.fnmatch(full_name, pattern):
                return (True, full_name)
        
        return None
```

## 使用示例

### 基本用法

```python
from megatron.bridge.peft import LoRA
from megatron.bridge.models.conversion.auto_bridge import AutoBridge

# 加载模型
bridge = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-8B")

# 创建 LoRA 配置
lora = LoRA(
    target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
    dim=32,
    alpha=64,
    dropout=0.1,
)

# 创建 Provider 并注册 hook
provider = bridge.to_megatron_provider()
provider.register_pre_wrap_hook(lambda chunks: lora(chunks, training=True))

# 创建模型
model = provider.provide_distributed_model(ddp_config=ddp_config)
```

### 自定义目标模块

```python
# 仅对特定层应用 LoRA
lora = LoRA(
    target_modules=[
        "*.layers.0.*.linear_qkv",
        "*.layers.1.*.linear_qkv",
        "*.layers.2.*.linear_qkv",
    ],
    dim=16,
    alpha=32,
)
```

### 保存 LoRA 适配器

```python
# 训练完成后保存
bridge = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-8B")
lora = LoRA(target_modules=["linear_qkv", "linear_proj"], dim=32)

# 设置需要保存的参数
lora.set_params_to_save(model)

# 保存为 HF PEFT 格式
bridge.save_hf_adapter(
    model,
    "./my-lora-adapter",
    peft_config=lora,
    base_model_name_or_path="meta-llama/Meta-Llama-3-8B",
)

# 使用 HuggingFace PEFT 加载
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = PeftModel.from_pretrained(base, "./my-lora-adapter")
```

### VLMLoRA 用于多模态

```python
from megatron.bridge.peft import VLMLoRA

vlm_lora = VLMLoRA(
    target_modules=["language_model.*.linear_qkv", "language_model.*.linear_proj"],
    dim=32,
    alpha=64,
    freeze_vision_model=True,      # 冻结视觉编码器
    freeze_vision_projection=False,# 微调投影层
    freeze_language_model=True,    # 冻结语言模型（仅训练 LoRA）
)
```

## 检查点过滤

在保存检查点时，PEFT 提供过滤器以仅保存适配器权重：

```python
from megatron.bridge.training.checkpointing import apply_peft_adapter_filter_to_state_dict

# 创建状态字典过滤器
def state_dict_filter(state_dict):
    return {k: v for k, v in state_dict.items() if lora.adapter_key_filter(k)}

# 或使用内置函数
filtered_state_dict = apply_peft_adapter_filter_to_state_dict(state_dict, lora)
```

## 相关模块

| 模块 | 关系说明 |
|------|---------|
| [ModelProvider](03_ModelProvider模式.md) | 通过 `register_pre_wrap_hook()` 将 LoRA 注入模型 |
| [检查点系统](08_检查点系统.md) | `apply_peft_adapter_filter_to_state_dict()` 过滤适配器权重 |
| [配置系统](05_配置系统.md) | ConfigContainer 的 `peft` 字段接受 PEFT 实例 |
| [AutoBridge](02_AutoBridge自动桥接.md) | `save_hf_adapter()` 导出 LoRA 为 HF PEFT 格式 |
| [训练配方](09_训练配方Recipes.md) | `_peft_common()` 提供 PEFT 配置模板 |

---

## 总结

Megatron-Bridge 的 PEFT 系统通过：

1. **抽象基类**: `PEFT` 定义统一接口
2. **模块匹配**: `ModuleMatcher` 支持通配符模式
3. **多种适配器**: 支持 `nn.Linear`、TE Linear、MoE Router
4. **张量并行**: 自动处理 TP 分片的权重合并
5. **检查点优化**: 仅保存适配器参数

提供了灵活、高效的参数高效微调解决方案。