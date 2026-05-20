---
created: 2026-05-06
---

# Tokenizer核心API

> **【源码定位】**
> - Tokenizer核心: `tokenizers/src/tokenizer/mod.rs`
> - Python绑定: `bindings/python/src/tokenizer.rs`
> - 预置实现: `bindings/python/py_src/tokenizers/implementations/`

---

## 1. Tokenizer类概述

`Tokenizer`是面向用户的统一接口（Facade模式），内部封装`TokenizerImpl`实现所有分词功能。

### 1.1 类层次结构

```python
Tokenizer (Python Facade)
    └── TokenizerImpl<M, N, PT, PP, D> (Rust核心)
            ├── normalizer: Option<N>
            ├── pre_tokenizer: Option<PT>
            ├── model: M                    # 必需
            ├── post_processor: Option<PP>
            ├── decoder: Option<D>
            └── added_vocabulary: AddedVocabulary
```

---

## 2. 初始化方法

### 2.1 from_pretrained - 从Hub加载

```python
from tokenizers import Tokenizer

# 从HuggingFace Hub加载预训练分词器
tokenizer = Tokenizer.from_pretrained("bert-base-cased")
tokenizer = Tokenizer.from_pretrained("gpt2")
tokenizer = Tokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
```

**底层实现要点**：
- 需要启用`http` feature
- 自动下载`tokenizer.json`配置
- 解析并实例化所有Pipeline组件

### 2.2 from_file - 从本地加载

```python
# 加载本地保存的分词器
tokenizer = Tokenizer.from_file("path/to/tokenizer.json")
```

### 2.3 程序化构建

```python
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

# 1. 初始化Tokenizer核心（必须指定Model）
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]", max_input_chars_per_word=100))

# 2. 配置各阶段组件
tokenizer.normalizer = normalizers.Sequence([...])
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
tokenizer.post_processor = processors.BertProcessing(...)
tokenizer.decoder = decoders.WordPiece(prefix="##")
```

---

## 3. 核心编码方法

### 3.1 encode - 单条编码

```python
encoding = tokenizer.encode(
    sequence: str | PreTokenizedInput,
    pair: Optional[str] = None,
    add_special_tokens: bool = True,
    # 返回Encoding对象
)

# 示例
output = tokenizer.encode("Hello, world!", add_special_tokens=True)
print(output.tokens)   # ['[CLS]', 'hello', ',', 'world', '!', '[SEP]']
print(output.ids)      # [101, 7592, 117, 2088, 106, 102]
```

**底层Rust方法体系**：

| 方法名 | 约束 | 职责 |
|--------|------|------|
| `encode()` | N + M | 完整编码（带字节偏移） |
| `encode_char_offsets()` | N + M | 字符级偏移编码 |
| `encode_fast()` | M | 跳过偏移计算（高性能） |

### 3.2 encode_batch - 批量编码

```python
encodings = tokenizer.encode_batch(
    inputs: List[str],
    add_special_tokens: bool = True,
    # 返回List[Encoding]
)

# 示例
texts = ["First sentence.", "Second sentence."]
encodings = tokenizer.encode_batch(texts, add_special_tokens=True)
```

**性能优势**：
- 使用`Rayon`在Rust侧并行处理
- 绕过Python GIL限制
- 批量内存分配优化

**底层约束**：需要 `N + M + PT + PP + D + Send`

### 3.3 encode_char_offsets - 字符偏移

```python
# 获取字符级偏移（而非字节偏移）
encoding = tokenizer.encode_char_offsets("Hello world")
print(encoding.offsets)  # [(0, 5), (6, 11)]
```

---

## 4. 解码方法

### 4.1 decode - 单条解码

```python
text = tokenizer.decode(
    ids: List[int],
    skip_special_tokens: bool = True,
    # 返回str
)

# 示例
token_ids = [101, 7592, 117, 2088, 106, 102]
text = tokenizer.decode(token_ids, skip_special_tokens=True)
print(text)  # "hello, world!"
```

### 4.2 decode_batch - 批量解码

```python
texts = tokenizer.decode_batch(
    sequences: List[List[int]],
    skip_special_tokens: bool = True,
)
```

**底层约束**：需要 `M + Send`

### 4.3 流式解码（用于生成场景）

```python
# 流式解码保持状态，正确处理token边界
decode_stream = tokenizer.decode_stream(skip_special_tokens=False)

# 模拟LLM逐个生成token
token_ids = [713, 16, 41, 1246]  # "This is an example"
for token_id in token_ids:
    chunk = decode_stream.step(token_id)
    if chunk:
        print(chunk, end="", flush=True)  # 输出：This is an example
```

---

## 5. 模型与组件管理

### 5.1 获取模型信息

```python
# 词汇表操作
vocab = tokenizer.get_vocab()           # Dict[str, int]
vocab_size = tokenizer.get_vocab_size()

# ID与token转换
token_id = tokenizer.token_to_id("hello")  # int or None
token = tokenizer.id_to_token(7592)        # "hello"

# 特殊token管理
tokenizer.add_special_tokens(["[SPECIAL]"])
tokenizer.add_tokens(["custom_token"])
```

### 5.2 组件访问

```python
# 访问各Pipeline组件（Python绑定层）
normalizer = tokenizer.normalizer
pre_tokenizer = tokenizer.pre_tokenizer
model = tokenizer.model
post_processor = tokenizer.post_processor
decoder = tokenizer.decoder
```

---

## 6. 配置参数

### 6.1 截断(Truncation)配置

```python
from tokenizers import Tokenizer, processors

# 方法1：通过enable_truncation
tokenizer.enable_truncation(
    max_length=512,
    stride=128,                    # 滑动窗口步长
    strategy="longest_first",      # longest_first/only_first/only_second
    direction="right",             # left/right
)

# 方法2：通过post_processor
# 某些Processor内置truncation/padding逻辑
```

### 6.2 填充(Padding)配置

```python
# 启用填充
tokenizer.enable_padding(
    direction="right",             # left/right
    pad_id=0,
    pad_type_id=0,
    pad_token="[PAD]",
    length=512,                    # None表示batch内最长
    # strategy: "batch_longest" or "fixed"
)
```

---

## 7. 序列化

### 7.1 保存分词器

```python
# 保存为JSON格式
tokenizer.save("tokenizer.json")

# 保存为Pretty格式（调试用）
tokenizer.save("tokenizer.json", pretty=True)
```

### 7.2 加载分词器

```python
# 从文件加载
tokenizer = Tokenizer.from_file("tokenizer.json")

# 从JSON字符串加载
tokenizer = Tokenizer.from_str(json_string)
```

### 7.3 JSON结构

```json
{
    "version": "1.0",
    "truncation": { ... },
    "padding": { ... },
    "added_tokens": [ ... ],
    "normalizer": { "type": "Sequence", "normalizers": [...] },
    "pre_tokenizer": { "type": "BertPreTokenizer" },
    "post_processor": { "type": "BertProcessing", ... },
    "decoder": { "type": "WordPiece", "prefix": "##" },
    "model": { "type": "BPE", ... }
}
```

---

## 8. 实战示例

### 8.1 完整分词流程（BERT风格）

```python
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

# 1. 初始化Tokenizer核心
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]", max_input_chars_per_word=100))

# 2. 配置Normalizer
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),                    # Unicode分解
    normalizers.Lowercase(),              # 小写化
    normalizers.StripAccents(),          # 去除重音
])

# 3. 配置PreTokenizer
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

# 4. 配置PostProcessor
tokenizer.post_processor = processors.BertProcessing(
    ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ("[CLS]", tokenizer.token_to_id("[CLS]")),
)

# 5. 配置Decoder
tokenizer.decoder = decoders.WordPiece(prefix="##")

# 6. 训练
special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=30000, special_tokens=special_tokens)
tokenizer.train(files=["wiki.txt"], trainer=trainer)

# 7. 保存/加载
tokenizer.save("bert_tokenizer.json")
tokenizer = Tokenizer.from_file("bert_tokenizer.json")

# 8. 使用
encoding = tokenizer.encode("Hello, world!", add_special_tokens=True)
print(encoding.tokens)   # ['[CLS]', 'hello', ',', 'world', '!', '[SEP]']
print(encoding.ids)      # [101, 7592, 117, 2088, 106, 102]
print(encoding.offsets)  # [(0,0), (0,5), (5,6), (7,12), (12,13), (0,0)]
```

### 8.2 从HuggingFace Hub加载

```python
# 直接使用预训练分词器
tokenizer = Tokenizer.from_pretrained("bert-base-cased")

# 或使用transformers集成
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # 底层使用tokenizers库
```

---

## 9. 注意事项

### 9.1 通用使用模式

1. **Model是必需的**：初始化`Tokenizer`时必须指定Model
2. **Pipeline顺序**：Raw -> Normalizer -> PreTokenizer -> Model -> PostProcessor -> Decoder
3. **encode和decode对称**：确保配置正确以保持一致性

### 9.2 常见问题

| 问题 | 解决 |
|------|------|
| 特殊token被拆分 | 使用`add_special_tokens()`而非`add_tokens()` |
| 与transformers结果不一致 | 检查normalizer/pre_tokenizer配置对齐 |
| 批量处理慢 | 使用`encode_batch`而非Python循环 |
