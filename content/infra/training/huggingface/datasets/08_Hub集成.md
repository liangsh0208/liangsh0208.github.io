# Hub 集成

> **【文档定位】** HuggingFace Hub 数据集加载与发布
>
> **【版本信息】** 基于 HuggingFace datasets v4.8.5.dev0 (`src/datasets/load.py` ~2200行)
>
> **【前置知识】** HuggingFace Hub、huggingface_hub 库

---

## 模块概述

HuggingFace Hub 是开源数据集社区平台，datasets 提供深度集成功能，支持一键加载 Hub 上的 10万+ 数据集，以及将自己的数据集发布到 Hub 供社区使用。

### 核心功能

| 功能 | 说明 |
|------|------|
| **load_dataset()** | 从 Hub 加载数据集（自动缓存、支持流式） |
| **push_to_hub()** | 将本地数据集发布到 Hub |
| **load_from_disk()** | 从本地磁盘加载已下载/保存的数据集 |
| **save_to_disk()** | 将数据集保存到本地磁盘 |
| **Streaming** | 无需下载完整数据集即可流式访问 |
| **版本控制** | 通过 Git commit hash 固定数据集版本 |

---

## 架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Hub 集成架构                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      HuggingFace Hub                               │  │
│  │                                                                      │  │
│  │   Available Datasets                                                  │  │
│  │   ├── username/dataset-1 (GLUE)                                     │  │
│  │   ├── username/dataset-2 (SQuAD)                                    │  │
│  │   ├── username/dataset-3 (ImageNet)  ... 100k+                      │  │
│  │                                                                      │  │
│  │   Repo Structure:                                                    │  │
│  │   ├── dataset_info.json   (Features, splits, size)                 │  │
│  │   ├── *.arrow / *.parquet (Data shards)                            │  │
│  │   └── README.md           (Dataset card)                             │  │
│  │                                                                      │  │
│  └───────────────────────────────────┬────────────────────────────────┘  │
│                                        │                                  │
│                                        ▼                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      User / Local                                  │  │
│  │                                                                      │  │
│  │   ┌───────────────────┐         ┌───────────────────┐             │  │
│  │   │  load_dataset()   │         │  push_to_hub()    │             │  │
│  │   │                   │         │                   │             │  │
│  │   │  Download & Cache │         │  Upload & Share   │             │  │
│  │   │  - metadata       │         │  - Arrow files    │             │  │
│  │   │  - data files     │         │  - dataset_info   │             │  │
│  │   │  - streaming      │         │  - README         │             │  │
│  │   └───────────────────┘         └───────────────────┘             │  │
│  │                                                                      │  │
│  │   ┌───────────────────┐         ┌───────────────────┐             │  │
│  │   │  save_to_disk()   │         │  load_from_disk() │             │  │
│  │   │                   │         │                   │             │  │
│  │   │  Local Storage    │◄───────►│  Local Loading    │             │  │
│  │   │  - Arrow format   │         │  - Memory mapped  │             │  │
│  │   │  - Metadata       │         │  - Features       │             │  │
│  │   └───────────────────┘         └───────────────────┘             │  │
│  │                                                                      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 从 Hub 加载数据集

### load_dataset() 基础用法

```python
from datasets import load_dataset

# 基础加载
dataset = load_dataset("rajpurkar/squad")
# Returns: DatasetDict({'train': ..., 'validation': ...})

# 加载特定 split
train = load_dataset("rajpurkar/squad", split="train")
validation = load_dataset("rajpurkar/squad", split="validation")

# 加载所有 split
all_splits = load_dataset("rajpurkar/squad")
train = all_splits['train']
val = all_splits['validation']
```

### 流式加载

```python
# 流式模式（不下载完整数据集）
dataset = load_dataset(
    "rajpurkar/squad",
    split="train",
    streaming=True
)

# 迭代使用
for example in dataset:
    print(example)
    break
```

### 分片选择语法

```python
from datasets import load_dataset

# 加载前 1000 条
dataset = load_dataset("rajpurkar/squad", split="train[:1000]")

# 加载后 10%
dataset = load_dataset("rajpurkar/squad", split="train[-10%:]")

# 加载中间段
dataset = load_dataset("rajpurkar/squad", split="train[50%:60%]")

# 组合多个分片
dataset = load_dataset("rajpurkar/squad", split="train[:1000]+validation[:100]")

# 数据集交错
dataset = load_dataset("rajpurkar/squad", split="train[:50%]|validation[:50%]")
```

### 版本控制

```python
from datasets import load_dataset

# 加载特定版本（通过 git tag）
dataset = load_dataset("rajpurkar/squad", revision="v1.0.0")

# 加载特定 commit
dataset = load_dataset("rajpurkar/squad", revision="abc123d")

# 加载 main 分支最新
dataset = load_dataset("rajpurkar/squad", revision="main")
```

### 认证与私有数据集

```python
from datasets import load_dataset

# 方式1: 使用 token 参数
dataset = load_dataset(
    "username/private-dataset",
    token="hf_xxxxxxxxxxxxxx"
)

# 方式2: 使用环境变量
import os
os.environ['HF_TOKEN'] = "hf_xxxxxxxxxxxxxx"
dataset = load_dataset("username/private-dataset")

# 方式3: 先登录
from huggingface_hub import login
login(token="hf_xxxxxxxxxxxxxx")
dataset = load_dataset("username/private-dataset")
```

---

## 发布到 Hub

### push_to_hub() 基础用法

```python
from datasets import load_dataset
from huggingface_hub import login

# 先登录
login()

# 加载本地或远程数据集
dataset = load_dataset("local_dataset")

# 发布到 Hub
dataset.push_to_hub("username/my-dataset")

# 发布特定 split
dataset['train'].push_to_hub("username/my-dataset", split="train")
dataset['test'].push_to_hub("username/my-dataset", split="test")
```

### 完整发布流程

```python
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
from huggingface_hub import login
import os

# 1. 登录
login()

# 2. 准备数据
data = [
    {"text": "Hello world", "label": 0},
    {"text": "How are you", "label": 1},
]

features = Features({
    "text": Value("string"),
    "label": ClassLabel(names=["greeting", "question"])
})

# 3. 创建数据集
dataset = Dataset.from_list(data, features=features)

# 4. 创建 DatasetDict
dataset_dict = DatasetDict({
    "train": dataset,
    "validation": dataset.select(range(1)),
    "test": dataset.select(range(1))
})

# 5. 生成数据集卡片 (README.md)
readme_content = """---
tags:
- text-classification
- custom-dataset
license: apache-2.0
---

# My Custom Dataset

## Description
This is a sample dataset for demonstration.

## Features
- text: string
- label: classification label
"""

# 6. 发布
dataset_dict.push_to_hub(
    "username/my-custom-dataset",
    private=False,           # 是否私有
    token=None,              # 可选 token
    branch="main",           # 分支
    create_pr=False,         # 是否创建 PR
    embed_external_files=True  # 嵌入外部文件
)

# 7. 单独上传 README
dataset_dict.push_to_hub(
    "username/my-custom-dataset",
    data_dir=".",
    readme_path="path/to/README.md"
)
```

### 私有数据集

```python
from datasets import load_dataset
from huggingface_hub import login

login()

# 发布为私有数据集
dataset.push_to_hub(
    "username/private-dataset",
    private=True,  # 仅自己可见
    token=None
)

# 加载私有数据集（需要 token）
private_dataset = load_dataset(
    "username/private-dataset",
    token=True  # 使用已登录的 token
)
```

---

## 本地数据管理

### save_to_disk() 与 load_from_disk()

```python
from datasets import load_dataset, load_from_disk

# 从 Hub 加载
dataset = load_dataset("rajpurkar/squad", split="train")

# 保存到本地（Arrow 格式，支持内存映射）
dataset.save_to_disk("/path/to/local/squad")

# 从本地加载（内存映射）
local_dataset = load_from_disk("/path/to/local/squad")

# 访问数据和从 Hub 加载完全相同
print(local_dataset[0])
```

### 导出为其他格式

```python
import json
import csv

# 导出为 JSON
with open("output.json", "w") as f:
    for example in dataset:
        f.write(json.dumps(example) + "\n")

# 导出为 JSON Lines
with open("output.jsonl", "w") as f:
    for example in dataset:
        f.write(json.dumps(example) + "\n")

# 导出为 CSV
dataset.to_csv("output.csv")

# 导出为 Pandas DataFrame
df = dataset.to_pandas()
df.to_parquet("output.parquet")

# 导出为 Parquet（推荐）
dataset.to_parquet("output.parquet")
```

---

## 代码示例

### 完整工作流：创建、处理、发布

```python
from datasets import Dataset, DatasetDict, Features, Value, Image, ClassLabel
from PIL import Image as PILImage
from huggingface_hub import login
import os

# 1. 登录
login()

# 2. 创建自定义数据集
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
labels = ["cat", "dog"]

def gen_data():
    for img_path, label in zip(image_paths, labels):
        yield {
            "image": PILImage.open(img_path),
            "label": label,
            "filename": os.path.basename(img_path)
        }

features = Features({
    "image": Image(),
    "label": ClassLabel(names=["cat", "dog"]),
    "filename": Value("string")
})

dataset = Dataset.from_generator(gen_data, features=features)

# 3. 处理数据
dataset = dataset.map(lambda x: {"label": x["label"].upper()})

# 4. 拆分为 train/test
dataset = dataset.shuffle(seed=42)
train_test = dataset.train_test_split(test_size=0.2)
dataset_dict = DatasetDict({
    "train": train_test["train"],
    "test": train_test["test"]
})

# 5. 本地保存
dataset_dict.save_to_disk("./my_processed_dataset")

# 6. 发布到 Hub
dataset_dict.push_to_hub(
    "username/my-image-dataset",
    private=False
)

# 7. 验证
loaded = load_dataset("username/my-image-dataset")
print(loaded)
```

### 批量下载与离线使用

```python
from datasets import load_dataset, load_from_disk
import os

# 设置离线索引目录
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 在线时下载
dataset = load_dataset("rajpurkar/squad")
dataset.save_to_disk("/cache/squad")

# 离线时使用
offline_dataset = load_from_disk("/cache/squad")
```

### 数据集验证

```python
from datasets import load_dataset_builder

# 查看数据集信息（不下载数据）
builder = load_dataset_builder("rajpurkar/squad")
print(builder.info.description)
print(builder.info.features)
print(builder.info.splits)

# 输出:
# Stanford Question Answering Dataset (SQuAD)
# Features:
#       id: string
#       title: string
#       context: string
#       question: string
#       answers:
#               text: list<item: string>
#               answer_start: list<item: int32>
# Splits:
#       train: 87599 examples
#       validation: 10570 examples
```

---

## 配置参数表

### load_dataset() 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | str | - | 数据集路径（Hub repo_id 或本地脚本） |
| `name` | str | None | 配置名（如 "en", "fr"） |
| `data_dir` | str | None | 数据文件所在目录 |
| `data_files` | dict/list | None | 显式指定数据文件 |
| `split` | str | None | 分片选择（支持切片语法） |
| `cache_dir` | str | None | 缓存目录 |
| `features` | Features | None | 强制指定特征类型 |
| `download_config` | DownloadConfig | None | 下载配置 |
| `download_mode` | str | None | "reuse_cache_if_exists" / "force_redownload" |
| `verification_mode` | str | None | 数据校验模式 |
| `keep_in_memory` | bool | None | 是否强制驻留内存 |
| `revision` | str | None | Git commit hash（版本控制） |
| `token` | str | None | HuggingFace Token |
| `streaming` | bool | False | 流式模式 |
| `num_proc` | int | None | 下载预处理进程数 |
| `trust_remote_code` | bool | False | 信任远程代码 |

### push_to_hub() 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `repo_id` | str | - | Hub 仓库 ID（如 "username/repo"） |
| `split` | str | None | 分片名称 |
| `private` | bool | False | 是否私有 |
| `token` | str | None | Token |
| `branch` | str | "main" | Git 分支 |
| `create_pr` | bool | False | 是否创建 PR |
| `max_shard_size` | str | "500MB" | 最大分片大小 |
| `embed_external_files` | bool | True | 嵌入外部文件 |

---

## 常见问题排查

### Hub 加载问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| `GatedRepoError` | 数据集需要授权 | 使用 `token` 参数传入 HuggingFace token |
| `RepositoryNotFound` | 数据集不存在或名称错误 | 检查 repo_id 拼写；确认数据集存在 |
| 数据集版本不一致 | 不同环境默认获取最新 commit | 显式指定 `revision="abc123"` 固定版本 |
| 自定义数据集脚本加载失败 | 远程代码执行未启用 | 设置 `trust_remote_code=True` |
| 网络超时或缓慢 | 国际带宽或本地网络问题 | 使用 `HF_ENDPOINT` 指定镜像；或先下载再本地加载 |

### 发布问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| `RepositoryNotFound` (发布时) | 账号不存在 repo | 先通过 Web 或 API 创建仓库 |
| 权限错误 | 无写入权限 | 检查 token 权限；确认是 repository owner |
| 上传失败 | 网络中断或文件过大 | 使用 `max_shard_size` 参数分片上传 |
| 发布后不显示 | 缓存未刷新 | 刷新 Hub 页面；使用新的 revision 访问 |

### 本地加载问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| `FileNotFoundError` | 路径错误或文件不存在 | 检查路径；使用绝对路径 |
| 加载后 features 改变 | 本地 Arrow 文件与 info 不匹配 | 使用 `load_from_disk`，不要使用 `pa.ipc` 直接读取 |
| 内存不足 | 数据集太大 | 使用 `load_from_disk` 获得内存映射支持 |

---

*文档生成于: 2026/04/20*
