---
created: 2026-05-06
---

# ModelProvider 模式

> 本文档详细分析 ModelProvider 模式，讲解分布式模型的实例化、Hook 系统和并行策略配置。

---

## 目录

1. [概述](#概述)
2. [核心类：ModelProviderMixin](#核心类modelprovidermixin)
3. [GPTModelProvider](#gptmodelprovider)
4. [并行配置](#并行配置)
5. [Hook 系统](#hook-系统)
6. [权重加载](#权重加载)
7. [分布式模型创建](#分布式模型创建)
8. [相关模块](#相关模块)
9. [总结](#总结)

---

## 概述

ModelProvider 是 Megatron-Bridge 中的核心设计模式，用于**标准化模型实例化流程**。它解决了分布式训练中模型创建的复杂性问题，提供统一的接口来处理：

- 模型并行（TP/PP/EP/CP）
- 混合精度训练
- 分布式数据并行（DDP/FSDP）
- 虚拟流水线并行（VPP）

**源码位置**: `src/megatron/bridge/models/model_provider.py`

## 核心类：ModelProviderMixin

```python
class ModelProviderMixin(abc.ABC, Generic[ModelT]):
    """
    实现 ModelProvider 模式的 Mixin 类。
    
    提供统一的 provide_distributed_model() 方法处理分布式训练设置，
    以及类似 HuggingFace 的 from_hf_pretrained() / save_hf_pretrained() 接口。
    
    子类必须实现 provide() 方法定义模型架构。
    """
```

## 设计理念

### 解决的问题

1. **生态碎片化**: 不同框架有不同的模型创建方式
2. **分布式复杂性**: TP/PP/EP 等并行方式配置复杂
3. **Hook 管理**: 模型创建前后需要执行多种操作

### 解决方案

```python
# 统一的接口
provider = GPTModelProvider(...)
model = provider.provide_distributed_model(ddp_config=ddp_config)

# HF 风格的 API
provider = GPTModelProvider.from_hf_pretrained("gpt2")
provider.save_hf_pretrained("./output")
```

## 抽象方法：provide

```python
@abc.abstractmethod
def provide(
    self, 
    pre_process: bool | None = None, 
    post_process: bool | None = None, 
    vp_stage: int | None = None
) -> ModelT:
    """
    子类必须实现此方法，返回具体的 Megatron 模型实例。
    
    Args:
        pre_process: 是否包含嵌入层（流水线并行首阶段）
        post_process: 是否包含输出层（流水线并行末阶段）
        vp_stage: 虚拟流水线阶段索引
    
    Returns:
        Megatron 模型实例（如 GPTModel）
    """
```

### GPTModelProvider 实现示例

```python
class GPTModelProvider(ModelProviderMixin, PreTrainedModelProvider):
    def provide(self, pre_process=True, post_process=True, vp_stage=None) -> GPTModel:
        """创建 GPTModel 实例"""
        return GPTModel(
            config=self.transformer_config,
            transformer_layer_spec=self.transformer_layer_spec,
            vocab_size=self.padded_vocab_size,
            max_sequence_length=self.seq_length,
            pre_process=pre_process,
            post_process=post_process,
            post_layer_norm=self.post_layer_norm,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
        )
```

## 核心方法：provide_distributed_model

```python
def provide_distributed_model(
    self,
    ddp_config: DistributedDataParallelConfig | None = None,
    model_type=ModelType.encoder_or_decoder,
    overlap_param_gather_with_optimizer_step: bool = False,
    fp16: bool | None = None,
    bf16: bool | None = None,
    use_megatron_fsdp: bool = False,
    use_torch_fsdp2: bool = False,
    wrap_with_ddp: bool = True,
    data_parallel_random_init: bool = False,
    use_cpu_initialization: None | bool = False,
    init_model_with_meta_device: bool | None = None,
    pre_wrap_hook: Callable | list[Callable] | None = None,
    post_wrap_hook: Callable | None = None,
    mixed_precision_wrapper: Callable | None = Float16Module,
    pg_collection: ProcessGroupCollection | None = None,
) -> list[ModelT]:
    """
    实例化并包装模型用于分布式训练。
    
    这是创建可用于 Megatron 生态系统的模型的主要入口点。
    """
```

### 执行流程

```python
def provide_distributed_model(self, ...):
    # 1. 初始化分布式环境（如需要）
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    
    # 2. 初始化模型并行状态
    if pg_collection is None:
        if not parallel_state.is_initialized():
            self.initialize_model_parallel(seed=0)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    
    # 3. 组合 pre_wrap_hooks
    final_pre_wrap_hook = pre_wrap_hook or self.pre_wrap_hook
    
    # 4. 调用 get_model 创建模型
    model = get_model(
        self,
        ddp_config=ddp_config,
        model_type=model_type,
        ...
    )
    
    # 5. 应用 post_wrap_hook
    if final_post_wrap_hook:
        model = final_post_wrap_hook(model)
    
    return model
```

## Hook 系统

### pre_wrap_hook

模型包装**前**执行的钩子：

```python
@property
def pre_wrap_hook(self) -> Callable | None:
    """
    返回组合的 pre-wrap 钩子。
    
    钩子在模型被 DDP 包装前执行，用于：
    - 冻结层
    - 添加 PEFT 适配器
    - 修改模型结构
    """

def register_pre_wrap_hook(self, hook: Callable, prepend: bool = False) -> None:
    """
    注册 pre-wrap 钩子。
    
    Args:
        hook: 接受模型列表，返回修改后的模型列表
        prepend: 是否插入到执行链开头
    """
```

**示例：添加 LoRA**

```python
from megatron.bridge.peft import LoRA

lora = LoRA(target_modules=["linear_qkv", "linear_proj"], dim=32)
provider = bridge.to_megatron_provider()

def apply_lora(model_chunks):
    return lora(model_chunks, training=True)

provider.register_pre_wrap_hook(apply_lora)
model = provider.provide_distributed_model(ddp_config=ddp_config)
```

### post_wrap_hook

模型包装**后**执行的钩子：

```python
@property
def post_wrap_hook(self) -> Callable | None:
    """
    返回组合的 post-wrap 钩子。
    
    钩子在模型被 DDP 包装后执行，用于：
    - 日志记录
    - 添加自定义属性
    - 模型分析
    """

def register_post_wrap_hook(self, hook: Callable, prepend: bool = False) -> None:
    """注册 post-wrap 钩子"""
```

## get_model 函数

核心模型创建函数：

```python
def get_model(
    model_provider: ModelProviderMixin,
    ddp_config: DistributedDataParallelConfig,
    model_type=ModelType.encoder_or_decoder,
    ...,
    pg_collection: ProcessGroupCollection,
) -> list[MegatronModule]:
    """
    创建并配置模型用于分布式训练。
    
    处理完整的模型创建管线：
    - 模型实例化（含流水线并行配置）
    - GPU 内存分配
    - 混合精度包装
    - Float8 张量校正
    - DDP 包装
    """
```

### 创建流程

```python
def get_model(model_provider, ddp_config, ...):
    # 1. 应用精度设置
    if fp16:
        model_provider.fp16 = fp16
    if bf16:
        model_provider.bf16 = bf16
    
    # 2. 创建模型（meta device 或 GPU）
    if init_model_with_meta_device:
        with torch.device("meta"):
            model = _create_model(model_provider, model_type, pg_collection)
    else:
        model = _create_model(model_provider, model_type, pg_collection)
    
    # 3. 应用 pre_wrap_hook
    if pre_wrap_hook:
        model = pre_wrap_hook(model)
    
    # 4. 设置 TP 属性
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
    
    # 5. GPU 分配
    if not use_torch_fsdp2 and not model_config.use_cpu_initialization:
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())
    
    # 6. 混合精度包装
    if (model_config.fp16 or model_config.bf16) and mixed_precision_wrapper:
        model = [mixed_precision_wrapper(model_config, m) for m in model]
    
    # 7. FP8 校正
    if correct_amax_history_if_needed:
        correct_amax_history_if_needed(model)
    
    # 8. DDP 包装
    if wrap_with_ddp:
        model = _ddp_wrap(model, ...)
    
    return model
```

## _create_model 函数

处理流水线并行的模型创建：

```python
def _create_model(
    model_provider: ModelProviderMixin,
    model_type: ModelType,
    pg_collection: ProcessGroupCollection,
) -> list[MegatronModule]:
    """
    创建模型实例，处理虚拟流水线并行 (VPP)。
    """
    vp_size = getattr(model_provider, "virtual_pipeline_model_parallel_size", None)
    pp_group = pg_collection.pp
    
    if (pp_group.size() > 1) and (vp_size is not None):
        # 虚拟流水线并行：创建多个模型块
        model = []
        for i in range(vp_size):
            pre_process = is_vp_first_stage(vp_stage=i, vp_size=vp_size) and is_pp_first_stage(pp_group)
            post_process = is_vp_last_stage(vp_stage=i, vp_size=vp_size) and is_pp_last_stage(pp_group)
            this_model = model_provider.provide(
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=i,
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        # 标准流水线并行
        pre_process = is_pp_first_stage(pp_group)
        post_process = is_pp_last_stage(pp_group)
        model = model_provider.provide(
            pre_process=pre_process,
            post_process=post_process,
        )
        model.model_type = model_type
    
    return [model] if not isinstance(model, list) else model
```

## 模型并行初始化

```python
def initialize_model_parallel(
    self, 
    seed: int | None = None, 
    seed_kwargs: dict | None = None, 
    **model_parallel_kwargs
) -> None:
    """
    初始化模型并行和随机种子。
    
    根据提供者实例的属性设置：
    - tensor_model_parallel_size
    - pipeline_model_parallel_size
    - virtual_pipeline_model_parallel_size
    - context_parallel_size
    - expert_model_parallel_size
    - expert_tensor_parallel_size
    """
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=getattr(self, "tensor_model_parallel_size", 1),
        pipeline_model_parallel_size=getattr(self, "pipeline_model_parallel_size", 1),
        virtual_pipeline_model_parallel_size=getattr(self, "virtual_pipeline_model_parallel_size", None),
        context_parallel_size=getattr(self, "context_parallel_size", 1) or 1,
        expert_model_parallel_size=getattr(self, "expert_model_parallel_size", 1) or 1,
        expert_tensor_parallel_size=getattr(self, "expert_tensor_parallel_size", None),
        **model_parallel_kwargs,
    )
    if seed is not None:
        model_parallel_cuda_manual_seed(seed, **(seed_kwargs or {}))
```

## HuggingFace 兼容接口

### from_hf_pretrained

```python
@classmethod
def from_hf_pretrained(
    cls,
    pretrained_model_name_or_path: str | Path,
    trust_remote_code: bool = False,
    mode: InstantiationMode | None = None,
    config_name: str | None = None,
    **kwargs,
):
    """
    从 HuggingFace Hub 或本地目录加载模型配置。
    
    Example:
        >>> provider = GPTModelProvider.from_hf_pretrained("gpt2")
        >>> provider.seq_length = 1024
        >>> model = provider.provide_distributed_model(ddp_config=ddp_config)
    """
```

### save_hf_pretrained

```python
def save_hf_pretrained(
    self,
    save_directory: str | Path,
    config_format: str | None = None,
    config_name: str | None = None,
    **kwargs,
):
    """
    保存模型配置到目录。
    """
```

## 属性

### meta_model

```python
@property
def meta_model(self) -> list[ModelT]:
    """
    返回在 meta device 上实例化的模型，用于检查架构而不分配 GPU 内存。
    """
    return self(wrap_with_ddp=False, init_model_with_meta_device=True)
```

## 类型定义

### GetModelKwargs

```python
class GetModelKwargs(TypedDict, total=False):
    """provide_distributed_model 方法的类型提示"""
    ddp_config: DistributedDataParallelConfig | None
    model_type: ModelType
    overlap_param_gather_with_optimizer_step: bool
    fp16: bool | None
    bf16: bool | None
    use_megatron_fsdp: bool
    use_torch_fsdp2: bool
    wrap_with_ddp: bool
    data_parallel_random_init: bool
    use_cpu_initialization: bool | None
    init_model_with_meta_device: bool | None
    pre_wrap_hook: Callable | list[Callable] | None
    post_wrap_hook: Callable | None
    mixed_precision_wrapper: Callable | None
```

### ModelParallelKwargs

```python
class ModelParallelKwargs(TypedDict, total=False):
    """模型并行覆盖参数"""
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    num_layers_in_first_pipeline_stage: int | None
    num_layers_in_last_pipeline_stage: int | None
    context_parallel_size: int
    expert_model_parallel_size: int
    expert_tensor_parallel_size: int
    sequence_parallel: bool
    virtual_pipeline_model_parallel_size: int | None
    hierarchical_context_parallel_sizes: list[int] | None
    pipeline_dtype: torch.dtype
```

## 使用示例

### 基本使用

```python
from megatron.bridge.models import GPTModelProvider
from megatron.core.distributed import DistributedDataParallelConfig

# 创建 Provider
provider = GPTModelProvider(
    num_layers=24,
    hidden_size=1024,
    num_attention_heads=16,
    seq_length=2048,
)

# 配置 DDP
ddp_config = DistributedDataParallelConfig(
    grad_reduce_in_fp32=True,
    overlap_grad_reduce=True,
)

# 创建分布式模型
model = provider.provide_distributed_model(
    ddp_config=ddp_config,
    wrap_with_ddp=True,
)
```

### 通过 AutoBridge 使用

```python
from megatron.bridge.models.conversion.auto_bridge import AutoBridge

# 从 HF 模型创建
bridge = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-8B")

# 获取 Provider
provider = bridge.to_megatron_provider()

# 配置并行设置
provider.tensor_model_parallel_size = 4
provider.pipeline_model_parallel_size = 2

# 创建模型
model = provider.provide_distributed_model(ddp_config=ddp_config)
```

### 使用 PEFT

```python
from megatron.bridge.peft import LoRA

# 创建 LoRA 配置
lora = LoRA(
    target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
    dim=32,
    alpha=64,
)

# 注册 pre-wrap hook
provider.register_pre_wrap_hook(lambda chunks: lora(chunks, training=True))

#创建带 LoRA 的模型
model = provider.provide_distributed_model(ddp_config=ddp_config)
```

## 设计优势

### 1. 统一接口

所有模型提供者都实现相同的接口，便于扩展：

```python
class MyCustomProvider(ModelProviderMixin):
    def provide(self, pre_process=True, post_process=True, vp_stage=None):
        return MyCustomModel(...)
```

### 2. 关注点分离

- **provide()**: 定义模型架构
- **provide_distributed_model()**: 处理分布式设置
- **Hook 系统**: 允许灵活扩展

### 3. 延迟初始化

仅在需要时分配资源：

```python
# 仅检查架构，不分配 GPU
meta_model = provider.meta_model
print(meta_model[0])
```

### 4. 配置继承

Provider 属性自动传播到模型创建过程：

```python
provider.tensor_model_parallel_size = 4  # 自动用于 parallel_state 初始化
provider.bf16 = True  # 自动应用混合精度
```

## 相关模块

| 模块 | 关系说明 |
|------|---------|
| [AutoBridge](02_AutoBridge自动桥接.md) | `to_megatron_provider()` 方法返回 ModelProvider 实例 |
| [Bridge模式](01_Bridge模式与模型转换.md) | Bridge 定义模型转换规则，ModelProvider 负责实例化 |
| [配置系统](05_配置系统.md) | Provider 包含并行配置（TP/PP/EP/CP），由 ConfigContainer 管理 |
| [训练框架](04_训练框架.md) | 训练循环使用 Provider 创建分布式模型 |
| [检查点系统](08_检查点系统.md) | Provider 的权重加载与检查点系统协作 |
| [PEFT](06_PEFT参数高效微调.md) | 通过 `register_pre_wrap_hook()` 将 LoRA 注入模型 |

---

## 总结

ModelProvider 模式通过：

1. **抽象化**: 隐藏分布式训练的复杂性
2. **标准化**: 统一的模型创建接口
3. **可扩展**: Hook 系统支持灵活扩展
4. **类型安全**: 泛型类型确保类型推导

使得用户可以专注于模型架构设计，而无需关心底层的分布式训练细节。