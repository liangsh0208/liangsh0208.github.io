---
created: 2026-05-06
---

# Features 类型系统

> **【文档定位】** 数据集字段类型定义与复杂结构支持
>
> **【版本信息】** 基于 HuggingFace datasets v4.8.5.dev0 (`src/datasets/features/features.py` ~2300行)
>
> **【前置知识】** Python 类型系统、PyArrow 类型、数据序列化

---

## 模块概述

`Features` 定义数据集各列的数据类型，支持复杂嵌套结构。它是对 Arrow Schema 的高级封装，不仅支持基本数值类型，还支持图像、音频、视频等多模态数据类型，以及嵌套结构如 List、Sequence、ClassLabel 等。

### 核心特性

| 特性 | 说明 |
|------|------|
| **丰富的内置类型** | Value、ClassLabel、Image、Audio、Video 等 |
| **嵌套结构支持** | List、Sequence、Array2D/3D/4D/5D、Translation |
| **自动编解码** | 自动处理 Python 对象与 Arrow 格式的转换 |
| **类型推导** | load_dataset 时可自动从数据推断类型 |
| **Schema 兼容** | 底层使用 PyArrow Schema 存储 |

---

## 架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Features 类型系统架构                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                     Features (dict 子类)                              │  │
│  │                                                                     │  │
│  │  class Features(dict):                                               │  │
│  │      # 定义数据集的字段结构                                          │  │
│  │                                                                     │  │
│  │      def encode_example(example) → dict:                           │  │
│  │          """Python 对象 → Arrow 兼容格式"""                         │  │
│  │                                                                     │  │
│  │      def decode_example(example) → Python 对象:                    │  │
│  │          """Arrow 格式 → Python 对象（如 PIL Image）"""             │  │
│  │                                                                     │  │
│  │      @property                                                      │  │
│  │      def type → pa.StructType:                                     │  │
│  │          """转换为 PyArrow Schema"""                                │  │
│  │                                                                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      基础类型 (Scalar)                               │  │
│  │                                                                     │  │
│  │   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐       │  │
│  │   │ Value()       │   │ ClassLabel()  │   │ Translation() │       │  │
│  │   │ - int8-64     │   │ - num_classes │   │ - languages   │       │  │
│  │   │ - uint8-64    │   │ - names       │   │ - nested      │       │  │
│  │   │ - float16-64  │   │ - id2label    │   │               │       │  │
│  │   │ - string      │   │ - label2id    │   │               │       │  │
│  │   │ - bool        │   │               │   │               │       │  │
│  │   │ - binary      │   │               │   │               │       │  │
│  │   └───────────────┘   └───────────────┘   └───────────────┘       │  │
│  │                                                                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      容器类型 (Container)                            │  │
│  │                                                                     │  │
│  │   ┌───────────────┐   ┌───────────────┐   ┌───────────────────┐     │  │
│  │   │ List()        │   │ Sequence()    │   │ TranslationVariable│     │  │
│  │   │ - feature     │   │ - feature     │   │ - languages       │     │  │
│  │   │ - nested      │   │ - length      │   │ - variable len    │     │  │
│  │   └───────────────┘   └───────────────┘   └───────────────────┘     │  │
│  │                                                                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      多维数组类型 (Array)                              │  │
│  │                                                                     │  │
│  │   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐       │  │
│  │   │ Array2D()     │   │ Array3D()     │   │ Array4D()     │       │  │
│  │   │ Array5D()     │   │ - shape       │   │ - dtype       │       │  │
│  │   └───────────────┘   └───────────────┘   └───────────────┘       │  │
│  │                                                                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      多模态类型 (Multimodal)                          │  │
│  │                                                                     │  │
│  │   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐       │  │
│  │   │ Image()       │   │ Audio()       │   │ Video()       │       │  │
│  │   │ - decode      │   │ - decode      │   │ - decode      │       │  │
│  │   │ - mode (RGB)  │   │ - sampling_rate│  │ - fps         │       │  │
│  │   │               │   │ - mono        │   │               │       │  │
│  │   └───────────────┘   └───────────────┘   └───────────────┘       │  │
│  │                                                                     │  │
│  │   ┌───────────────┐   ┌───────────────┐                           │  │
│  │   │ Pdf()         │   │ Json()        │                           │  │
│  │   │ - decode      │   │ - nested      │                           │  │
│  │   └───────────────┘   └───────────────┘                           │  │
│  │                                                                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 类型详解

### 基础类型 (Value)

```python
from datasets import Features, Value

# 数值类型
integer = Value('int32')      # int8, int16, int32, int64
unsigned = Value('uint32')    # uint8, uint16, uint32, uint64
float_num = Value('float32') # float16, float32, float64
decimal = Value('float64')

# 字符串和布尔
string = Value('string')
binary = Value('binary')
boolean = Value('bool')

# 时间类型
timestamp = Value('timestamp[s]')  # 秒级时间戳
date = Value('date32')           # 日期

features = Features({
    'id': Value('int64'),
    'text': Value('string'),
    'score': Value('float32'),
    'is_valid': Value('bool'),
})
```

### 分类标签 (ClassLabel)

```python
from datasets import Features, ClassLabel

# 方式1: 通过类别数量
labels = ClassLabel(num_classes=10)

# 方式2: 通过类别名称
labels = ClassLabel(names=['cat', 'dog', 'bird', 'fish'])

# 方式3: 通过 names_file (每行一个类别名)
labels = ClassLabel(names_file='labels.txt')

features = Features({
    'image_path': Value('string'),
    'label': ClassLabel(names=['positive', 'negative', 'neutral'])
})

# 使用
# 自动转换: int <-> str
example = {'image_path': 'a.jpg', 'label': 'positive'}  # str 输入
dataset.features['label'].int2str(0)  # 'positive'
dataset.features['label'].str2int('positive')  # 0
```

### 列表和序列

```python
from datasets import Features, Value, List, Sequence

# List: 变长列表
tags = List(Value('string'))
# Example: {'tags': ['AI', 'ML', 'NLP']}

# Sequence: 类似 List，但 length 字段存储真实长度
sequence = Sequence(Value('int32'))
# 与 List 基本相同，在 NLP 任务中常用

# 嵌套列表
nested = List(List(Value('float32')))
# Example: {'matrix': [[1.0, 2.0], [3.0, 4.0]]}

features = Features({
    'sentence': Value('string'),
    'tokens': Sequence(Value('string')),     # ['Hello', 'world']
    'token_ids': Sequence(Value('int32')),    # [101, 7592, 2088]
    'attention_mask': List(Value('int32')),   # [1, 1, 1, 0, 0]
})
```

### 多维数组

```python
from datasets import Features, Array2D, Array3D, Array4D, Array5D

# 2D 数组 (如图片特征)
image_features = Array2D(shape=(224, 224), dtype='float32')

# 3D 数组 (如视频帧特征)
video_features = Array3D(shape=(16, 224, 224), dtype='float32')

# 4D 数组 (如批量图片)
batch_images = Array4D(shape=(32, 224, 224, 3), dtype='uint8')

# 5D 数组 (如批量视频)
batch_videos = Array5D(shape=(8, 16, 224, 224, 3), dtype='float32')

features = Features({
    'embedding': Array2D(shape=(768,), dtype='float32'),
    'image_feat': Array3D(shape=(7, 7, 2048), dtype='float32'),
})
```

### 图像类型 (Image)

```python
from datasets import Features, Image, ClassLabel

# 基础图像（自动解码为 PIL Image）
features = Features({
    'image': Image(),
    'label': ClassLabel(names=['cat', 'dog'])
})

# 禁用自动解码（返回图片路径/bytes）
features = Features({
    'image': Image(decode=False),  # 不进行 decode，节省内存
    'label': ClassLabel(names=['cat', 'dog'])
})

# 指定颜色模式
features = Features({
    'image_rgb': Image(mode='RGB'),      # 彩色图
    'image_gray': Image(mode='L'),      # 灰度图
})

# 使用示例
# from PIL import Image as PILImage
# example = {
#     'image': PILImage.open('cat.jpg'),  # PIL Image 对象
#     'label': 0
# }
```

### 音频类型 (Audio)

```python
from datasets import Features, Audio, ClassLabel

# 基础音频（自动解码为 dict）
features = Features({
    'audio': Audio(),  # 返回 {'path': ..., 'array': ..., 'sampling_rate': ...}
    'transcription': Value('string')
})

# 指定采样率（自动重采样）
features = Features({
    'audio': Audio(sampling_rate=16000),  # 自动重采样到 16kHz
    'text': Value('string')
})

# 转换为单声道
features = Features({
    'audio': Audio(mono=True, sampling_rate=16000),
    'label': ClassLabel(names=['speech', 'music', 'noise'])
})

# 解码后的数据结构
# {
#     'path': '/path/to/audio.wav',
#     'array': np.array([...]),  # 音频波形数组
#     'sampling_rate': 16000
# }
```

### 嵌套结构

```python
from datasets import Features, Value, Sequence, ClassLabel

# 复杂嵌套结构
features = Features({
    'id': Value('int64'),
    'document': {
        'title': Value('string'),
        'content': Value('string'),
        'metadata': {
            'author': Value('string'),
            'date': Value('string'),
            'tags': Sequence(Value('string'))
        }
    },
    'annotations': Sequence({
        'start': Value('int32'),
        'end': Value('int32'),
        'label': ClassLabel(names=['PERSON', 'ORG', 'LOC']),
        'text': Value('string')
    })
})

# 数据示例
example = {
    'id': 1,
    'document': {
        'title': 'Sample',
        'content': 'Hello world',
        'metadata': {
            'author': 'Alice',
            'date': '2024-01-01',
            'tags': ['AI', 'NLP']
        }
    },
    'annotations': [
        {'start': 0, 'end': 5, 'label': 0, 'text': 'Hello'},
        {'start': 6, 'end': 11, 'label': 1, 'text': 'world'}
    ]
}
```

---

## 代码示例

### 完整 Features 定义

```python
from datasets import Features, Value, ClassLabel, Sequence, Image, Audio, Array2D

# 多模态数据集特征
multimodal_features = Features({
    # 基础信息
    'id': Value('string'),
    'timestamp': Value('timestamp[s]'),

    # 图像模态
    'image': Image(mode='RGB'),
    'image_features': Array2D(shape=(2048,), dtype='float32'),

    # 音频模态
    'audio': Audio(sampling_rate=16000),
    'duration': Value('float32'),

    # 文本模态
    'text': Value('string'),
    'tokens': Sequence(Value('string')),
    'input_ids': Sequence(Value('int32')),

    # 标签
    'category': ClassLabel(names=['nature', 'urban', 'people']),
    'tags': Sequence(Value('string')),

    # 元数据
    'metadata': {
        'source': Value('string'),
        'quality_score': Value('float32'),
        'is_annotated': Value('bool')
    }
})
```

### 类型转换

```python
from datasets import Features, Value

# 创建数据集时指定特征
from datasets import Dataset

data = [
    {'name': 'Alice', 'age': 30, 'score': 95.5},
    {'name': 'Bob', 'age': 25, 'score': 88.0}
]

features = Features({
    'name': Value('string'),
    'age': Value('int32'),
    'score': Value('float32')
})

dataset = Dataset.from_list(data, features=features)

# 查看实际类型
print(dataset.features)
# {'name': Value(dtype='string', id=None),
#  'age': Value(dtype='int32', id=None),
#  'score': Value(dtype='float32', id=None)}

# 类型转换
dataset = dataset.cast_column('age', Value('int64'))
dataset = dataset.cast(Features({
    'name': Value('string'),
    'age': Value('int64'),
    'score': Value('float64')
}))
```

### 自定义编码/解码

```python
from datasets import Features

# encode_example: Python → Arrow
decoded = {'label': 'cat'}  # Python 字符串
encoded = features.encode_example(decoded)
# --> {'label': 0}  (转为 int)

# decode_example: Arrow → Python
arrow_data = {'label': 0}
decoded = features.decode_example(arrow_data)
# --> {'label': 'cat'}  (转回 str)
```

---

## 配置参数表

### Value 支持的 dtype

| dtype | 说明 | 示例值 |
|-------|------|--------|
| `int8` ~ `int64` | 有符号整数 | `Value('int32')` |
| `uint8` ~ `uint64` | 无符号整数 | `Value('uint8')` |
| `float16` ~ `float64` | 浮点数 | `Value('float32')` |
| `string` | UTF-8 字符串 | `Value('string')` |
| `large_string` | 大字符串 | `Value('large_string')` |
| `binary` | 二进制数据 | `Value('binary')` |
| `large_binary` | 大二进制数据 | `Value('large_binary')` |
| `bool` | 布尔值 | `Value('bool')` |
| `timestamp[s/ms/us/ns]` | 时间戳 | `Value('timestamp[us]')` |
| `date32` / `date64` | 日期 | `Value('date32')` |

### ClassLabel 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `num_classes` | int | 类别数量 |
| `names` | list[str] | 类别名称列表 |
| `names_file` | str | 类别名称文件路径 |
| `id` | str | 标识符 |

### Array2D/3D/4D/5D 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `shape` | tuple | 数组形状 |
| `dtype` | str | 数据类型 |
| `id` | str | 标识符 |
| `nullable` | bool | 是否可为 null |

### Image 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `decode` | bool | True | 是否自动解码为 PIL Image |
| `mode` | str | None | 颜色模式 (RGB, L, etc.) |
| `id` | str | None | 标识符 |

### Audio 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sampling_rate` | int | None | 采样率 (自动重采样) |
| `mono` | bool | False | 是否转为单声道 |
| `decode` | bool | True | 是否自动解码 |
| `id` | str | None | 标识符 |

---

## 常见问题排查

### 类型相关问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| `TypeError` 转换失败 | Arrow 类型推断与实际不符 | 显式传入 `features=Features({...})` |
| `None` 值导致转换失败 | Arrow 默认 nullable 与数据冲突 | 设置 `disable_nullable=True`；确保数据填充完整 |
| 嵌套结构变深后无法处理 | Feature 结构复杂度过高 | 扁平化结构；或分多步 `map` 处理 |
| 保存后读取特征丢失 | 未导出 dataset_info.json | 使用 `save_to_disk()` 而非直接操作 Arrow 文件 |

### 图像/音频相关问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| Image/Audio 加载慢 | 每行都 decode 二进制数据 | 使用 `features.Image(decode=False)` 禁用自动解码 |
| 图像模式不匹配 | 数据集图片模式不一致 | 统一使用 `Image(mode='RGB')` 转换 |
| 音频采样率不一致 | 不同文件采样率不同 | 使用 `Audio(sampling_rate=16000)` 统一重采样 |

### ClassLabel 相关问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| `KeyError: label` | 标签字符串不在 names 列表中 | 确保所有标签都在 `names` 中，或使用 `num_classes` |
| id2label 映射错误 | names 列表顺序问题 | 检查 names 列表顺序与实际 id 是否一致 |

---

*文档生成于: 2026/04/20*
