---
created: 2026-05-09
---

# ms-swift 数据系统：模板与数据管道

数据系统是 ms-swift 最复杂也最关键的模块之一。它负责将原始对话数据（文本/图像/视频/音频）转换为模型训练所需的 token 序列和张量格式。其核心设计是将**数据格式化逻辑**（Template）与**模型架构**完全解耦，使同一套 Template 可服务于任意模型。

---

## 1. 整体架构

```
原始数据                    预处理后数据                 tokenized数据                训练batch
   │                           │                          │                         │
   ▼                           ▼                          ▼                         ▼
┌──────────┐    ┌─────────────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
│ Dataset  │───▶│ RowPreprocessor         │──▶│ Template.encode()│──▶│ Template.data_      │
│ (jsonl/  │    │ (字段映射/过滤/预处理)  │   │ (对话→token IDs)  │   │   collator()         │
│  Hugging│    │                         │   │                  │   │ (padding/masking/SP)  │
│  Face)  │    │ EncodePreprocessor      │   │                  │   │                      │
└──────────┘    └─────────────────────────┘   └──────────────────┘   └──────────────────────┘
```

---

## 2. 数据集加载层

### 2.1 DatasetLoader — 统一加载器

**文件**: `swift/dataset/loader.py` (~150 lines)

```python
class DatasetLoader(BaseDatasetLoader):
    def __init__(self, num_proc=1, streaming=False, hub_token=None, ...):
        # 统一的加载配置

    def _load_dataset_path(self, dataset_path, dataset_meta):
        # 本地文件加载 (jsonl/json/csv/txt)
        ext = os.path.splitext(dataset_path)[1]
        dataset = hf_load_dataset(file_type, data_files=dataset_path, streaming=self.streaming)
        # 列重命名 + 预处理 + 过滤无用列
        return dataset

    def _load_repo_dataset(self, dataset_id, subset, use_hf=None):
        # Hub 加载 (HuggingFace / ModelScope)
        hub = get_hub(use_hf)
        dataset = hub.load_dataset(dataset_id, subset, split, streaming=..., revision=...)
        return dataset
```

**支持的数据源**:
- 本地文件：`jsonl`, `json`, `csv`, `txt`
- HuggingFace Hub：`dataset_name/subset`
- ModelScope Hub：国内镜像加速
- 流式模式：`streaming=True` 不加载到内存

### 2.2 注册表机制

**文件**: `swift/dataset/register.py`

```python
DATASET_MAPPING: Dict[str, DatasetMeta] = {}

def register_dataset(dataset_meta: DatasetMeta):
    DATASET_MAPPING[dataset_meta.dataset_id] = dataset_meta

@dataclass
class DatasetMeta:
    dataset_id: str
    subsets: List[str]
    preprocess_func: Callable  # 预处理函数
```

ms-swift 内置 150+ 数据集的预处理函数，用户只需指定数据集 ID 即可一键加载。

### 2.3 高阶加载 API

**文件**: `swift/dataset/loader.py`

```python
def load_dataset(dataset, *, split_dataset_ratio=0.0, shuffle=True, **kwargs):
    """
    高阶数据集加载函数
    
    Args:
        dataset: 数据集标识，支持多种形式：
            - 'dataset_name#500'  # 取前500条
            - 'dataset_name#0.1'  # 取10%
            - 'dataset_name/subset#1000'
            - 本地路径
        split_dataset_ratio: 从训练集划分验证集的比例
        shuffle: 是否打乱
    """
```

### 2.4 预处理链

**文件**: `swift/dataset/preprocessor/core.py`

```python
class RowPreprocessor:
    @staticmethod
    def safe_rename_columns(dataset, columns_mapping):
        # 安全的列重命名

    @staticmethod
    def remove_useless_columns(dataset):
        # 移除无关列，保留标准列 (instruction, input, output, conversations, ...)

class EncodePreprocessor:
    def __init__(self, template: Template, strict=False):
        self.template = template
        self.strict = strict

    def __call__(self, dataset):
        # 对每行数据调用 template.encode()
        return dataset.map(self._encode_row, ...)
```

---

## 3. Template — 数据格式化中枢

**文件**: `swift/template/base.py` (~103KB)

Template 是整个 ms-swift 框架的**核心数据格式化类**，它将原始对话数据（系统提示、用户输入、助手回复、工具调用、多模态附件）转换为模型训练/推理所需的 token 序列。

### 3.1 初始化参数

```python
class Template(ProcessorMixin):
    def __init__(self, processor, template_meta, 
                 default_system=None, max_length=None,
                 truncation_strategy='raise',  # 'raise'/'left'/'right'/'split'
                 padding_free=False,
                 sequence_parallel_size=1,
                 loss_scale='default',
                 template_backend='swift',  # 'swift' 或 'jinja'
                 ...
    ):
```

### 3.2 核心方法：encode()

```python
def encode(self, inputs: TemplateInputs) -> Dict[str, Any]:
    """
    将对话输入编码为模型训练所需的 token 序列
    
    Args:
        inputs: 包含 messages（对话消息列表）和可选的多模态数据
    
    Returns:
        {
            'input_ids': torch.LongTensor,           # token IDs
            'labels': torch.LongTensor,               # 训练标签 (-100 为忽略)
            'attention_mask': torch.LongTensor,       # 注意力掩码
            'pixel_values': torch.Tensor,            # [可选] 图像特征
            'image_grid_thw': torch.Tensor,          # [可选] 图像网格大小
            'loss_scale': torch.Tensor,              # [可选] 逐 token 损失权重
            ...
        }
    """
```

**编码流程**:

```
Template.encode(inputs)
    │
    ├── 1. 解析对话消息
    │      将 message['role'] (system/user/assistant/tool) 分离
    │
    ├── 2. 系统提示处理
    │      组合 system message + default_system
    │
    ├── 3. 多模态数据处理
    │      加载图片/视频/音频 → 转换为 pixel_values / feature
    │
    ├── 4. 应用 Chat Template（jinja 或 swift backend）
    │      将对话格式化为模型特定的字符串/模板
    │
    ├── 5. Tokenization
    │      字符串 → token IDs
    │
    ├── 6. 构造 Labels
    │      对话部分的 token label = token ID
    │      非对话部分 (system, user) = -100 (忽略)
    │
    ├── 7. Truncation
    │      按 truncation_strategy 截断超长序列
    │
    └── 8. 返回编码结果字典
```

### 3.3 多模态处理

Template 原生支持**文本、图像、视频、音频、边界框** 五种模态：

```python
special_tokens = ['<image>', '<video>', '<audio>', '<bbox>', '<ref-object>']
special_keys = ['images', 'videos', 'audios', 'objects']

# 图像处理流程
images = inputs.images
template.image_placeholder = ['<image>']  # 模型特定的图像占位符
# <image> 占位符在 text 中被替换后，对应的 pixel_values 被插入到特征序列中

# 多模态模型示例 (Qwen2-VL)
# 输入: "描述这张图片 <image>"
# encode 后:
#   input_ids: [token_ids_for_text..., token_ids_for_image..., token_ids_for_text...]
#   pixel_values: [processed_image_features]
#   image_grid_thw: [(temporal, height, width)]
```

### 3.4 训练模式切换

```python
template.set_mode('train')    # 训练模式: labels 正常设置
template.set_mode('rlhf')     # RLHF 模式: 生成 chosen/rejected pair
template.set_mode('kto')      # KTO 模式: 生成 KL pair
template.set_mode('transformers')  # 推理模式
```

---

## 4. Data Collator 矩阵

**文件**: `swift/template/base.py` (~500+ lines)

Template 根据 `task_type` 和 `mode` 自动选择对应的 data collator：

```python
def data_collator(self, features: List[Dict], padding_to=None) -> Dict[str, torch.Tensor]:
    if self.task_type == 'causal_lm':
        return self._data_collator(features, padding_to)
    elif self.task_type == 'rlhf':
        return self._rlhf_data_collator(features)
    elif self.task_type == 'kto':
        return self._kto_data_collator(features)
    elif self.task_type == 'seq_cls':
        return self._seq_cls_data_collator(features)
    elif self.task_type == 'embedding':
        return self._embedding_data_collator(features)
    elif self.task_type == 'reranker':
        return self._reranker_data_collator(features)
```

### 4.1 Causal LM Collator

```python
def _data_collator(self, features, padding_to=None):
    # 1. 提取各字段
    input_ids = [f['input_ids'] for f in features]
    labels = [f['labels'] for f in features]
    
    # 2. Padding 到 batch 内最长序列 (或 padding_to)
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    # 3. 构造 attention_mask
    attention_mask = (padded_input_ids != pad_token_id).long()
    
    # 4. [可选] sequence parallel 处理
    if self.sequence_parallel_size > 1:
        # 调整批次维度以适配 SP
        ...
    
    # 5. [可选] padding-free 处理
    if self.padding_free:
        # 移除 padding token，拼接为连续序列
        ...
    
    return {'input_ids': padded_input_ids, 'labels': padded_labels, 'attention_mask': attention_mask, ...}
```

### 4.2 RLHF Collator

```python
def _rlhf_data_collator(self, features):
    # features 中每个元素包含 'chosen' 和 'rejected' 两部分
    # 组装为拼接的 chosen_ids + rejected_ids
    # labels 同样拼接，用于后续分别计算 logprobs
```

### 4.3 KTO Collator

```python
def _kto_data_collator(self, features):
    # KTO 需要同时包含 desired (KL参考) 和 undesired 的 completion
    # 组装为 pair 形式
```

### 4.4 其他 Collator

| Collator | 用途 | 特殊处理 |
|---------|------|---------|
| `_seq_cls_data_collator` | 序列分类 | labels 为分类整数 |
| `_embedding_data_collator` | Embedding | 构造 (query, positive, negative) triplet |
| `_reranker_data_collator` | 重排序 | query + document pair |

---

## 5. 高级数据特性

### 5.1 LazyLLMDataset — 延迟编码

**文件**: `swift/dataset/lazy_dataset.py`

```python
class LazyLLMDataset:
    def __init__(self, dataset, encode_func, strict=False, random_state=None):
        self.dataset = dataset
        self.encode_func = encode_func  # Template.encode
    
    def __getitem__(self, idx):
        # 训练时才调用 encode，节省内存
        return self.encode_func(self.dataset[idx])
```

Lazy 模式下，原始数据在训练循环中**按需编码**，而非预处理阶段一次性完成。

### 5.2 Packing — Padding-Free 训练

**文件**: `swift/dataset/packing.py`

```python
class PackingDataset:
    """将多条短序列拼接为一条长序列，消除 padding"""
    def __init__(self, template, dataset, packing_length=...):
        # 1. 预编码所有样本获取长度
        # 2. 使用 bin-packing 算法将样本分组成不超过 packing_length 的批次
        # 3. 每组内样本用 EOS token 分隔，拼接为单条序列
```

**Packing 效果**: 训练速度提升 **100%+**（因为消除了无效的 padding 计算）。

### 5.3 Sequence Parallel 数据适配

**文件**: `swift/sequence_parallel/ulysses.py`

当 `sequence_parallel_size > 1` 时：

```python
# Ulysses 序列并行: 将序列维度切分到多个 GPU
# input_ids: [B, L] -> [B, L//SP] per GPU

sequence_parallel.prepare_inputs(inputs)
    ├── 调整 input_ids 的分布
    ├── 调整 attention_mask
    └── 调整 position_ids
```

### 5.4 Streaming 模式

支持流式数据集（不加载到内存）：

```python
swift sft --dataset 'dataset_name' --streaming true
# 使用 IterableDataset，数据在网络流中按需读取
```

---

## 6. Template 注册系统

### 6.1 模板注册

**文件**: `swift/template/register.py`

```python
TEMPLATES: Dict[str, Type[Template]] = {}

def register_template(template_type: str, template_cls: Type[Template], *, exist_ok=False):
    TEMPLATES[template_type] = template_cls
```

### 6.2 模板查找

```python
def get_template(template_type: str) -> Type[Template]:
    if template_type in TEMPLATES:
        return TEMPLATES[template_type]
    raise ValueError(f'Unknown template: {template_type}')
```

### 6.3 常用模板列表

| 模板名 | 对应模型 | 说明 |
|-------|---------|------|
| `default` | 通用 | 默认对话模板 |
| `qwen` | Qwen 系列 | Qwen 系统提示格式 |
| `qwen2_vl` | Qwen2-VL | 含图像占位符 `<image>` |
| `llama` | Llama 系列 | Llama chat template |
| `chatglm` | ChatGLM | GLM 特殊分隔符 |
| `internlm` | InternLM | InternLM 格式 |
| `deepseek` | DeepSeek | DeepSeek 对话格式 |

---

## 7. 关键代码路径索引

| 功能 | 关键文件/函数 |
|-----|-------------|
| 数据集加载 | `swift/dataset/loader.py::load_dataset()` |
| Hub 加载 | `swift/dataset/loader.py::DatasetLoader._load_repo_dataset()` |
| 本地文件加载 | `swift/dataset/loader.py::DatasetLoader._load_dataset_path()` |
| 预处理 | `swift/dataset/preprocessor/core.py::RowPreprocessor` |
| 编码预处理 | `swift/dataset/preprocessor/core.py::EncodePreprocessor` |
| Template 基类 | `swift/template/base.py::Template` |
| Template.encode | `swift/template/base.py::Template.encode()` |
| Data Collator | `swift/template/base.py::Template.data_collator()` |
| 模板注册 | `swift/template/register.py::register_template()` |
| Lazy 数据集 | `swift/dataset/lazy_dataset.py::LazyLLMDataset` |
| Packing | `swift/dataset/packing.py::PackingDataset` |
| Stream 数据集 | `swift/dataset/loader.py::DatasetLoader` (streaming 模式) |
| 多模态 Vision 工具 | `swift/template/vision_utils.py` |
