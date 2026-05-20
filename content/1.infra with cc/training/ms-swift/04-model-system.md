---
created: 2026-05-09
---

# ms-swift 模型系统：注册、加载与架构映射

模型系统是 ms-swift 承上启下的核心层：向下对接 HuggingFace `transformers` 生态，向上为 Template、Tuner、Trainer 提供统一且带元信息的模型实例。它不重新实现 Transformer 架构，而是通过一套 **元数据驱动的注册-加载-补丁机制** 来适配 1000+ 模型。

---

## 1. 核心设计理念

```
┌─────────────────────────────────────────────────────────────────┐
│                        model_system                              │
│                                                                  │
│   model_type ──▶ ModelMeta ──▶ ModelLoader ──▶ PreTrainedModel   │
│       │                                                          │
│       └──▶ template, arch, mcore_model_type, loader...         │
│                                                                  │
│   MODEL_MAPPING: Dict[str, ModelMeta]  (全局注册表)              │
│   MODEL_ARCH_MAPPING: Dict[str, ModelKeys] (架构路径映射)        │
└─────────────────────────────────────────────────────────────────┘
```

- **`model_type`**: 模型家族唯一标识符，如 `'qwen3'`, `'llama'`, `'qwen2_vl'`
- **`ModelMeta`**: 每个 `model_type` 对应的完整元数据（模板、架构、下载 ID 等）
- **`MODEL_MAPPING`**: 全局注册表，运行时通过 `model_type` 查询模型信息
- **`ModelKeys` / `MultiModelKeys`**: 定义模型内部组件路径字符串（用于 freeze 控制和 LoRA target 匹配）

---

## 2. 数据结构与注册表

### 2.1 ModelMeta — 模型元数据

**文件**: `swift/model/model_meta.py` (160 lines)

```python
@dataclass
class ModelMeta:
    model_type: Optional[str]                  # 唯一标识
    model_groups: List[ModelGroup]             # 支持的模型 ID 列表
    loader: Optional[Type[BaseModelLoader]] = None  # 自定义加载器

    template: Optional[str] = None             # 默认模板名称
    model_arch: Optional[str] = None           # 架构类型标识
    mcore_model_type: Optional[str] = None     # Megatron-core 模型类型
    architectures: List[str] = field(default_factory=list)  # HF config architectures
    torch_dtype: Optional[torch.dtype] = None

    is_multimodal: bool = False
    is_reward: bool = False
    task_type: Optional[str] = None
```

**关键行为**:
- `loader` 为空时，自动使用默认 `ModelLoader`
- `__post_init__` 中根据 `model_type` 是否为 `MLLMModelType`/`RMModelType` 自动设置 `is_multimodal`/`is_reward`

### 2.2 Model — 模型实例描述

```python
@dataclass
class Model:
    ms_model_id: Optional[str] = None   # ModelScope 模型 ID
    hf_model_id: Optional[str] = None   # HuggingFace 模型 ID
    model_path: Optional[str] = None    # 本地路径
```

`ModelGroup` 将多个相关模型组织在一起，可以覆盖不同参数规模（7B/14B/72B）的同一架构模型。

### 2.3 ModelInfo — 运行时模型信息

**文件**: `swift/model/model_meta.py` (126-160 lines)

```python
@dataclass
class ModelInfo:
    model_type: str
    model_dir: str
    torch_dtype: torch.dtype
    max_model_len: int
    quant_method: Literal['gptq', 'awq', 'bnb', 'aqlm', 'hqq', None]
    quant_bits: int
    is_moe_model: bool = False
    is_multimodal: bool = False
    config: Optional[PretrainedConfig] = None
```

`ModelInfo` 在**加载时动态推断**得出，包含量化方法、MoE 标志等运行时才能确定的信息。

### 2.4 注册机制

**文件**: `swift/model/register.py` (31-42 lines)

```python
def register_model(model_meta: ModelMeta, *, exist_ok: bool = False) -> None:
    model_type = model_meta.model_type
    if not exist_ok and model_type in MODEL_MAPPING:
        raise ValueError(f'The `{model_type}` has already been registered.')
    if model_meta.model_arch:
        model_meta.model_arch = get_model_arch(model_meta.model_arch)
    MODEL_MAPPING[model_type] = model_meta
```

所有模型模型的注册发生在各模型文件**模块导入时**：

```python
# swift/model/models/qwen.py (约98KB, 框架最大模型文件)
register_model(ModelMeta(
    LLMModelType.qwen,
    [ModelGroup([Model('qwen/Qwen-7B-Chat'), ...])],
    loader=QwenLoader,    # 自定义加载器
    template='qwen',
))
register_model(ModelMeta(
    MLLMModelType.qwen2_vl,
    [ModelGroup([Model('qwen/Qwen2-VL-7B-Instruct'), ...])],
    template='qwen2_vl',
    model_arch=MLLMModelArch.qwen2_vl,
    is_multimodal=True,
))
```

### 2.5 常量定义

**文件**: `swift/model/constant.py`

所有支持的 `model_type` 字符串以类属性的形式定义：

```python
class LLMModelType:
    qwen = 'qwen'
    qwen2 = 'qwen2'
    qwen3 = 'qwen3'
    llama = 'llama'
    deepseek = 'deepseek'
    deepseek_v2 = 'deepseek_v2'
    deepseek_v3 = 'deepseek_v3'
    internlm = 'internlm'
    internlm2 = 'internlm2'
    internlm3 = 'internlm3'
    chatglm4 = 'chatglm4'
    glm4 = 'glm4'
    baichuan = 'baichuan'
    mistral = 'mistral'
    gemma = 'gemma'
    ...

class MLLMModelType:
    qwen_vl = 'qwen_vl'
    qwen2_vl = 'qwen2_vl'
    qwen3_vl = 'qwen3_vl'
    internvl_chat = 'internvl_chat'
    minicpmv = 'minicpmv'
    ...

class RMModelType:
    internlm2_reward = 'internlm2_reward'
    qwen2_reward = 'qwen2_reward'
    ...
```

---

## 3. 模型架构路径映射 (ModelKeys)

**文件**: `swift/model/model_arch.py` (200+ lines)

### 3.1 数据结构

```python
@dataclass
class ModelKeys:
    arch_name: str = None
    embedding: str = None         # 词嵌入层路径
    module_list: str = None       # Transformer 层列表路径
    lm_head: str = None            # 输出头路径
    q_proj: str = None
    k_proj: str = None
    v_proj: str = None
    o_proj: str = None
    attention: str = None
    mlp: str = None
    down_proj: str = None
    qkv_proj: str = None           # 融合 QKV（如 ChatGLM）

@dataclass
class MultiModelKeys(ModelKeys):
    language_model: Union[str, List[str]] = field(default_factory=list)  # LLM 部分
    aligner: Union[str, List[str]] = field(default_factory=list)         # 对齐层
    vision_tower: Union[str, List[str]] = field(default_factory=list)    # ViT
    generator: Union[str, List[str]] = field(default_factory=list)
```

### 3.2 注册示例

```python
# LLaMA 架构
register_model_arch(ModelKeys(
    LLMModelArch.llama,
    module_list='model.layers',
    mlp='model.layers.{}.mlp',
    down_proj='model.layers.{}.mlp.down_proj',
    attention='model.layers.{}.self_attn',
    o_proj='model.layers.{}.self_attn.o_proj',
    q_proj='model.layers.{}.self_attn.q_proj',
    k_proj='model.layers.{}.self_attn.k_proj',
    v_proj='model.layers.{}.self_attn.v_proj',
    embedding='model.embed_tokens',
    lm_head='lm_head',
))

# InternLM2 架构（融合 QKV）
register_model_arch(ModelKeys(
    LLMModelArch.internlm2,
    module_list='model.layers',
    mlp='model.layers.{}.feed_forward',
    down_proj='model.layers.{}.feed_forward.w2',
    attention='model.layers.{}.attention',
    o_proj='model.layers.{}.attention.wo',
    qkv_proj='model.layers.{}.attention.wqkv',
    embedding='model.tok_embeddings',
    lm_head='output',
))

# Qwen2-VL 多模态架构
register_model_arch(MultiModelKeys(
    MLLMModelArch.qwen2_vl,
    module_list='model.layers',
    ...,
    language_model=['model'],           # LLM 组件
    vision_tower=['visual'],            # ViT 组件
    aligner=['visual.merger'],         # 对齐层
))
```

### 3.3 用途

| 功能 | 使用方式 |
|-----|---------|
| **LoRA target 选择** | `target_modules=['q_proj', 'v_proj']` → 按路径字符串匹配 |
| **freeze 控制** | `freeze_vit=True` → 匹配 `vision_tower` 路径下的参数 |
| **freeze_llm** | 匹配 `language_model` 路径下的参数不更新 |
| **freeze_aligner** | 匹配 `aligner` 路径下的参数不更新 |
| **LLaMA-Pro** | 通过 `module_list` 定位可扩展层 |

---

## 4. 模型加载核心流程

### 4.1 入口函数

**文件**: `swift/model/register.py` (160+ lines)

`ModelLoader` 是默认的模型加载器，关键方法：

```python
class ModelLoader(BaseModelLoader):
    def __init__(self, model_info, model_meta, *, load_model=False, attn_impl=None, ...):
        # 收集加载参数
    
    def load(self) -> Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]:
        # 实际加载流程
```

### 4.2 加载流程图

```
args.get_model_processor()
    │
    ▼
get_model_info_meta(model_id_or_path)
    ├── 从 MODEL_MAPPING 匹配 model_type（通过 model_id 或 architecture）
    ├── 推断 torch_dtype、quantization、task_type
    └── 返回 ModelInfo + ModelMeta
    │
    ▼
ModelLoader(model_info, model_meta).load()
    ├── 1. 下载/定位模型文件 (safe_snapshot_download)
    ├── 2. 加载 tokenizer/processor
    ├── 3. 加载 AutoConfig
    ├── 4. 处理 attn_impl (flash_attention_2/sdpa/eager)
    ├── 5. 处理 rope_scaling
    ├── 6. [可选] 使用 Unsloth 加载 (load_by_unsloth)
    ├── 7. [可选] AWQ 兼容性补丁 (_patch_awq_compat)
    ├── 8. 调用 AutoModelForCausalLM.from_pretrained()
    │       └── patch_automodel() 注入元信息
    ├── 9. 附加属性: model.model_info, model.model_meta, model.model_dir
    ├── 10. [可选] dummy model 处理 (return_dummy_model)
    └── 11. 序列分类任务: 动态替换 SequenceClassification head
```

### 4.3 关键补丁机制

**文件**: `swift/model/patcher.py` (23KB, ~540 lines)

| 补丁函数 | 作用 |
|---------|------|
| `patch_automodel()` | 在 `AutoModel.from_pretrained` 后注入 `model_info` / `model_meta` 属性 |
| `patch_automodel_for_sequence_classification()` | 将 CausalLM 动态转换为 SeqCls 模型 |
| `patch_attach_align_device_hook_on_blocks()` | 修复 device_map 下 hook 的挂载顺序 |
| `patch_get_dynamic_module()` | 修复动态模块加载的缓存问题 |
| `patch_mp_ddp()` | 修复多进程 DDP 的同步问题 |
| `patch_tp_plan()` | 兼容 transformers 5.0+ 的 tensor parallel plan |
| `get_lm_head_model()` | 获取用于生成任务的模型头 |

### 4.4 Unsloth 集成

**文件**: `swift/model/register.py` (45-95 lines)

当 `use_unsloth=True` 时，通过 `load_by_unsloth()` 函数加载模型：

```python
def load_by_unsloth(args):
    assert is_unsloth_available()
    # 分发不同的 Unsloth 模型类
    if model_meta.is_multimodal:
        from unsloth import FastVisionModel as UnslothModel
    elif model_info.is_moe_model:
        from unsloth import FastModel as UnslothModel
    else:
        from unsloth import FastLanguageModel as UnslothModel
    
    model, processor = UnslothModel.from_pretrained(...)
```

Unsloth 通过 Kernel 融合和梯度检查点优化，可实现 **2-5x 训练加速**。

---

## 5. 模型家族定制加载器

对于需要特殊处理的模型家族，ms-swift 提供自定义 `ModelLoader` 子类：

| 模型文件 | 模型家族 | 特殊处理 |
|---------|---------|---------|
| `qwen.py` (~98KB) | Qwen 全系列 | `use_flash_attn` 标志、causal mask 修复、特殊 token 处理 |
| `llama.py` | Llama 全系列 | 重置 `pretraining_tp` |
| `internlm.py` | InternLM/InternVL | InternLM2 MLP patch、InternVL resize 逻辑 |
| `deepseek.py` | DeepSeek V2/V3/VL | MLA (Multi-head Latent Attention) 特殊处理 |
| `glm.py` | ChatGLM/GLM4 | GLM 特有的 tokenization 和架构映射 |
| `minicpm.py` | MiniCPM | MiniCPM-V 多模态加载 |
| `llava.py` | LLaVA 系列 | HF vs non-HF 版本兼容 |
| `microsoft.py` | Phi/Florence | Phi3-vision 和 Florence2 |

### 5.1 QwenLoader 示例

```python
class QwenLoader(ModelLoader):
    # 处理 Qwen 系列特有的配置转换
    # 如 use_flash_attn 在不同版本间的兼容性
    # 处理 Qwen2.5-Omni 的多模态组件初始化
    # 处理 Qwen3 的特殊 token (enable_thinking 相关)
```

### 5.2 DeepSeekLoader 示例

```python
# DeepSeek-V2/V3 使用 MLA (Multi-head Latent Attention)
# 需要在加载时对 attention 层做特殊处理
# DeepSeek-VL2 有额外的多模态组件需要初始化
```

---

## 6. 多模态模型架构映射

多模态模型通过 `MultiModelKeys` 标识其内部组件：

```
┌─────────────────────────────────────────────────────────────┐
│                     MLLM Architecture                        │
│                                                              │
│   ┌──────────┐     ┌──────────┐     ┌──────────────────┐   │
│   │ ViT/     │────▶│ Aligner  │────▶│   LLM (decoder)  │   │
│   │ Vision   │     │ (MLP/Q-  │     │   (causal LM)    │   │
│   │ Tower    │     │  Former) │     │                  │   │
│   └──────────┘     └──────────┘     └──────────────────┘   │
│                                                              │
│   freeze_vit      freeze_aligner    freeze_llm            │
│   (可选)          (可选)             (可选)                 │
└─────────────────────────────────────────────────────────────┘
```

通过 `ModelKeys` 路径匹配，可以精确控制训练时哪些组件更新：

```bash
swift sft \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --freeze_vit true \\
    --freeze_aligner true \\
    --target_modules all-linear
```

此时只有 LLM 部分的 linear 层会插入 LoRA 并被训练。

---

## 7. 关键代码路径索引

| 功能 | 文件路径 |
|-----|---------|
| 全局注册表 | `swift/model/model_meta.py::MODEL_MAPPING` |
| 注册函数 | `swift/model/register.py::register_model()` |
| 默认加载器 | `swift/model/register.py::ModelLoader` |
| 模型信息推断 | `swift/model/model_meta.py::get_model_info_meta()` |
| 架构映射注册 | `swift/model/model_arch.py::register_model_arch()` |
| 架构路径定义 | `swift/model/model_arch.py::ModelKeys` / `MultiModelKeys` |
| 模型补丁中心 | `swift/model/patcher.py` |
| 模型类型常量 | `swift/model/constant.py` |
| Qwen 加载器 | `swift/model/models/qwen.py` |
| Llama 加载器 | `swift/model/models/llama.py` |
| DeepSeek 加载器 | `swift/model/models/deepseek.py` |
| Unsloth 加载 | `swift/model/register.py::load_by_unsloth()` |
| 获取模型列表 | `swift/model/register.py::get_model_list()` |
