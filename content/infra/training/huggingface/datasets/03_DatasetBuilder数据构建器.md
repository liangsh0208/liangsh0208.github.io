# DatasetBuilder 数据构建器

> **【文档定位】** 自定义数据集加载与构建流程
>
> **【版本信息】** 基于 HuggingFace datasets v4.8.5.dev0 (`src/datasets/builder.py` ~2000行)
>
> **【前置知识】** Python 抽象基类、生成器、数据下载

---

## 模块概述

`DatasetBuilder` 是数据集构建器基类，定义了数据加载和准备的标准流程。开发者可通过继承该基类创建自定义数据集加载脚本，支持从各种数据源（文件、API、数据库等）构建标准化的 Arrow 格式数据集。

### 核心特性

| 特性 | 说明 |
|------|------|
| **统一构建流程** | `download_and_prepare` -> `_split_generators` -> `_generate_examples` -> `as_dataset` |
| **多格式支持** | 内置 CSV/JSON/Parquet/Text/Images 等解析器 |
| **自动缓存** | 通过 fingerprint 避免重复下载和生成 |
| **多配置支持** | 支持多语言、多子集的 `BUILDER_CONFIG` |
| **增量更新** | 支持版本控制和数据验证 |

### Builder 类型

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| `GeneratorBasedBuilder` | 基于生成器（最常用） | 需要从文件/API生成示例 |
| `ArrowBasedBuilder` | 直接读取 Arrow 文件 | 已有 Arrow/Parquet 文件 |
| `FolderBasedBuilder` | 基于文件夹结构 | 图像/音频文件夹分类 |

---

## 架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DatasetBuilder 构建流程                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    DatasetBuilder 基类                               │  │
│  │                                                                      │  │
│  │  class DatasetBuilder:                                               │  │
│  │      BUILDER_CONFIGS = []           # 预定义配置列表                   │  │
│  │      BUILDER_CONFIG_CLASS = BuilderConfig                            │  │
│  │      DEFAULT_CONFIG_NAME = None    # 默认配置名                      │  │
│  │      VERSION = None                # 数据集版本                       │  │
│  │                                                                      │  │
│  │      # 必须实现的抽象方法                                             │  │
│  │      _info() → DatasetInfo           # 数据集元信息                 │  │
│  │      _split_generators() → list[SplitGenerator]  # 分片生成器        │  │
│  │      _generate_examples() → Generator     # 示例生成器              │  │
│  │                                                                      │  │
│  └───────────────────────────────────┬──────────────────────────────────┘  │
│                                      │                                    │
│                                      ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                   构建流程 (download_and_prepare)                   │  │
│  │                                                                      │  │
│  │   1. 检查本地缓存 (通过 fingerprint)                                  │  │
│  │         ↓ 不存在或失效                                                │  │
│  │   2. 创建 DownloadManager 管理下载                                  │  │
│  │         ↓                                                            │  │
│  │   3. 调用 _split_generators() 获取 SplitGenerator 列表              │  │
│  │         ↓                                                            │  │
│  │   4. 对每个 Split，调用 _generate_examples() 生成 (key, example) 对   │  │
│  │         ↓                                                            │  │
│  │   5. 使用 ArrowWriter 写入 Arrow/Parquet 缓存文件                     │  │
│  │         ↓                                                            │  │
│  │   6. 保存 dataset_info.json 元信息文件                               │  │
│  │                                                                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                      │                                    │
│                                      ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                        as_dataset()                                 │  │
│  │                                                                      │  │
│  │   读取 Arrow/Parquet 文件 → MemoryMappedTable → Dataset 对象          │  │
│  │                                                                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 抽象基类详解

### DatasetBuilder 核心实现

```python
class DatasetBuilder:
    """
    抽象基类，所有数据集构建器的基类。
    子类必须实现: info(), _split_generators(), _generate_examples()
    """
    # 构建器配置
    BUILDER_CONFIGS = []        # 预定义配置列表
    BUILDER_CONFIG_CLASS = BuilderConfig  # 配置类
    DEFAULT_CONFIG_NAME = None  # 默认配置名
    VERSION = None              # 数据集版本

    def __init__(self, cache_dir=None, config_name=None, ...):
        self.config = self._create_builder_config(config_name)
        self.info = self._info()
        self._cache_dir = self._build_cache_dir()

    def download_and_prepare(self, output_dir=None, ...):
        """
        下载数据源并写入 Arrow/Parquet 文件。
        流程: 检查本地缓存 -> 下载文件 -> 生成示例 -> 写入 Arrow
        """
        # 检查是否需要重新生成（通过 fingerprint）
        # 如已存在且验证通过，直接返回
        # 否则：
        #   1. 创建 DownloadManager
        #   2. 调用 _split_generators() 获取 Splits
        #   3. 对每个 Split，调用 _generate_examples() 生成示例
        #   4. 使用 ArrowWriter 写入缓存文件

    def as_dataset(self, split=None, ...):
        """将准备好的数据加载为 Dataset 对象"""
        # 读取 Arrow 文件，创建 MemoryMappedTable/InMemoryTable
        # 包装为 Dataset 对象返回

    def _info(self):
        """返回 DatasetInfo，描述数据集结构和特征"""
        raise NotImplementedError

    def _split_generators(self, dl_manager):
        """返回 SplitGenerator 列表，描述各 split 如何生成"""
        raise NotImplementedError

    def _generate_examples(self, **gen_kwargs):
        """
        生成 (key, example) 对的生成器
        key: 唯一标识符
        example: dict 格式的样本数据
        """
        raise NotImplementedError
```

### GeneratorBasedBuilder

```python
class GeneratorBasedBuilder(DatasetBuilder):
    """
    基于生成器的构建器，最常用的 Builder 类型
    """
    def _generate_examples(self, filepath, split):
        """
        必须实现的生成器方法

        参数:
            filepath: 通过 dl_manager 下载的文件路径
            split: 当前分片名称

        Yields:
            (key, example) 键值对
        """
        # 读取文件
        with open(filepath) as f:
            for i, line in enumerate(f):
                yield i, {'text': line.strip()}
```

---

## 代码示例

### 完整自定义数据集示例

```python
# my_dataset.py
from datasets import DatasetBuilder, GeneratorBasedBuilder, DatasetInfo, Features, Value

class MyDataset(GeneratorBasedBuilder):
    """自定义数据集示例"""

    VERSION = "1.0.0"

    def _info(self):
        return DatasetInfo(
            description="My custom dataset",
            features=Features({
                'text': Value('string'),
                'label': Value('int32')
            }),
            homepage="https://example.com",
            citation="@article{mydata2024}"
        )

    def _split_generators(self, dl_manager):
        """定义数据集分片"""
        # 下载数据
        urls = {
            'train': 'https://example.com/train.txt',
            'test': 'https://example.com/test.txt'
        }
        downloaded_files = dl_manager.download(urls)

        return [
            SplitGenerator(
                name='train',
                gen_kwargs={'filepath': downloaded_files['train']}
            ),
            SplitGenerator(
                name='test',
                gen_kwargs={'filepath': downloaded_files['test']}
            )
        ]

    def _generate_examples(self, filepath):
        """生成示例"""
        with open(filepath, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                text, label = line.strip().split('\t')
                yield idx, {
                    'text': text,
                    'label': int(label)
                }


# 使用
from datasets import load_dataset
dataset = load_dataset('path/to/my_dataset.py')
```

### 带配置的数据集

```python
from datasets import DatasetBuilder, BuilderConfig, GeneratorBasedBuilder

class MultiLanguageDataset(GeneratorBasedBuilder):
    """多语言数据集，支持配置选择"""

    BUILDER_CONFIGS = [
        BuilderConfig(name="en", version="1.0.0", description="English"),
        BuilderConfig(name="zh", version="1.0.0", description="Chinese"),
        BuilderConfig(name="fr", version="1.0.0", description="French"),
    ]

    DEFAULT_CONFIG_NAME = "en"

    def _info(self):
        return DatasetInfo(
            features=Features({
                'text': Value('string'),
                'language': Value('string')
            })
        )

    def _split_generators(self, dl_manager):
        lang = self.config.name
        url = f"https://example.com/{lang}.txt"
        path = dl_manager.download(url)
        return [SplitGenerator(name='train', gen_kwargs={'filepath': path, 'lang': lang})]

    def _generate_examples(self, filepath, lang):
        with open(filepath) as f:
            for i, line in enumerate(f):
                yield i, {'text': line.strip(), 'language': lang}


# 使用特定配置
dataset_en = load_dataset('my_multilang.py', name='en')
dataset_zh = load_dataset('my_multilang.py', name='zh')
```

### 图像数据集 Builder

```python
from datasets import DatasetBuilder, Features, Image, ClassLabel, Value
from pathlib import Path

class ImageFolderDataset(DatasetBuilder):
    """从图片文件夹构建数据集"""

    def _info(self):
        return DatasetInfo(
            features=Features({
                'image': Image(),
                'label': ClassLabel(names=['cat', 'dog']),
                'filename': Value('string')
            })
        )

    def _split_generators(self, dl_manager):
        data_dir = Path(dl_manager.manual_dir)
        return [
            SplitGenerator(
                name='train',
                gen_kwargs={'data_dir': data_dir / 'train'}
            ),
            SplitGenerator(
                name='test',
                gen_kwargs={'data_dir': data_dir / 'test'}
            )
        ]

    def _generate_examples(self, data_dir):
        label_names = ['cat', 'dog']
        for label_idx, label_name in enumerate(label_names):
            label_dir = data_dir / label_name
            for image_path in label_dir.glob('*.jpg'):
                yield str(image_path), {
                    'image': str(image_path),
                    'label': label_idx,
                    'filename': image_path.name
                }


# 使用 (需要本地数据目录)
# load_dataset('my_image_dataset.py', data_dir='path/to/images')
```

### 本地文件数据集

```python
import csv
from datasets import GeneratorBasedBuilder, Features, Value

class CSVDataset(GeneratorBasedBuilder):
    """加载本地 CSV 文件"""

    def _info(self):
        return DatasetInfo(
            features=Features({
                'name': Value('string'),
                'age': Value('int32'),
                'score': Value('float32')
            })
        )

    def _split_generators(self, dl_manager):
        # 使用 data_files 参数传入
        return [SplitGenerator(
            name='train',
            gen_kwargs={'filepath': dl_manager.manual_dir}
        )]

    def _generate_examples(self, filepath):
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                yield i, {
                    'name': row['name'],
                    'age': int(row['age']),
                    'score': float(row['score'])
                }


# 使用
from datasets import load_dataset
dataset = load_dataset(
    'my_csv.py',
    data_files='path/to/data.csv'
)
```

---

## 配置参数表

### DatasetBuilder 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `BUILDER_CONFIGS` | list[BuilderConfig] | 可选配置列表 |
| `BUILDER_CONFIG_CLASS` | class | 配置类（默认 BuilderConfig） |
| `DEFAULT_CONFIG_NAME` | str/None | 默认配置名 |
| `VERSION` | str/None | 数据集版本 |

### download_and_prepare 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_dir` | str | None | 输出目录 |
| `download_config` | DownloadConfig | None | 下载配置 |
| `download_mode` | DownloadMode | None | 下载模式 |
| `ignore_verifications` | bool | False | 忽略验证 |
| `try_from_hf_gcs` | bool | True | 尝试从 HF GCS 获取 |
| `use_auth_token` | str/bool | None | 认证 token |

### SplitGenerator 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | str/NamedSplit | 分片名称 (train/test/validation) |
| `gen_kwargs` | dict | 传递给 _generate_examples 的参数 |

### BuilderConfig 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | str | 配置名称 |
| `version` | str | 版本号 |
| `data_dir` | str | 数据目录 |
| `data_files` | dict/list | 数据文件 |
| `description` | str | 配置描述 |

---

## 常见问题排查

### 自定义数据集加载失败

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| `ModuleNotFoundError` | 脚本路径错误 | 确保 `.py` 文件路径正确，或使用绝对路径 |
| `_split_generators` 未实现 | 抽象方法未定义 | 确保继承 `GeneratorBasedBuilder` 并实现所有抽象方法 |
| 数据下载失败 | URL 无效或网络问题 | 检查 URL；设置 `HF_DATASETS_OFFLINE` 为 1 测试本地 |
| 缓存未更新 | 相同 fingerprint | 删除缓存目录或修改版本号 |

### 数据生成问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 特征类型不匹配 | _info() 中 features 定义与实际数据不符 | 检查 Features 定义与实际生成的 example 是否一致 |
| KeyError | example 字典缺少字段 | 确保 _generate_examples 返回的 dict 包含所有 feature 字段 |
| 生成器无限循环 | 没有正确 yield 终止 | 确保生成器在数据读取完毕后退出 |

### 多配置问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 配置名找不到 | BUILDER_CONFIG 未定义 | 定义 BUILDER_CONFIGS 列表或使用 `DEFAULT_CONFIG_NAME` |
| 使用错误配置 | 未指定 name 参数 | 使用 `load_dataset(..., name='config_name')` |

### 安装到 Hub

```bash
# 本地测试
load_dataset('path/to/dataset_script.py')

# 上传到 HuggingFace Hub
huggingface-cli login
huggingface-cli repo create my-dataset --type dataset
# 推送数据集文件
# 然后通过 load_dataset("username/my-dataset") 加载
```

---

*文档生成于: 2026/04/20*
