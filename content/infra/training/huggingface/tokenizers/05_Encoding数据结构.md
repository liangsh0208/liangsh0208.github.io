# Encoding数据结构

> **【源码定位】**
> - Encoding结构: `tokenizers/src/tokenizer/encoding.rs`
> - Offset定义: `tokenizers/src/tokenizer/` (tokenizer模块)
> - 对齐追踪逻辑: EncodingImpl实现

---

## 1. Encoding概述

`Encoding`是分词的输出数据结构，包含完整的分词结果及与原始文本的对齐信息。

### 1.1 设计目的

- **完整信息**：包含token ID、字符串、偏移量、注意力掩码等
- **双向映射**：支持字符<->token<->word的相互转换
- **序列处理**：支持pair输入、overflow处理、填充对齐

---

## 2. 核心结构

**文件**: `tokenizers/src/tokenizer/encoding.rs` (lines 10-562)

```rust
#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct Encoding {
    ids: Vec<u32>,                    // Token IDs
    type_ids: Vec<u32>,              // Segment IDs (BERT的0/1)
    tokens: Vec<String>,             // Token字符串
    words: Vec<Option<u32>>,         // Word ID映射（PreToken产生）
    offsets: Vec<Offsets>,           // 字符偏移区间（相对原始文本）
    special_tokens_mask: Vec<u32>,   // 特殊标记掩码(1=特殊token)
    attention_mask: Vec<u32>,        // Attention掩码(1=有效token)
    overflowing: Vec<Encoding>,      // 截断产生的溢出片段
    sequence_ranges: AHashMap<usize, Range<usize>>,  // 多序列范围
}

// 偏移量类型
type Offsets = (usize, usize);  // (start, end) 半开区间
```

### 2.1 字段详解

| 字段 | 类型 | 说明 |
|------|------|------|
| `ids` | `Vec<u32>` | Token在词汇表中的ID |
| `type_ids` | `Vec<u32>` | Segment ID（pair的第一句为0，第二句为1） |
| `tokens` | `Vec<String>` | Token字符串表示 |
| `words` | `Vec<Option<u32>>` | Word索引（None表示特殊token） |
| `offsets` | `Vec<Offsets>` | Token在原始文本中的(char_start, char_end) |
| `special_tokens_mask` | `Vec<u32>` | 1表示特殊token（如[CLS]），0表示普通token |
| `attention_mask` | `Vec<u32>` | 1表示有效token，0表示padding |
| `overflowing` | `Vec<Encoding>` | 长文本截断产生的额外片段 |
| `sequence_ranges` | `HashMap` | 多序列时各序列的索引范围 |

---

## 3. 偏移量追踪

### 3.1 核心概念

```
原始文本: "Hello, world!"
          0123456789012

tokens:   [CLS]    hello    ,     world    !     [SEP]
ids:      [101]    7592    117    2088    106     102
offsets:  [(0,0)  (0,5)   (5,6)  (7,12)  (12,13) (0,0)]
           ↑                ↑
      特殊token为(0,0)  字符位置半开区间
```

### 3.2 Offset计算流程

```
Pipeline各阶段维护偏移映射：

Raw Text -> Normalizer -> PreTokenizer -> Model -> PostProcessor
   ↓            ↓              ↓            ↓            ↓
 Original  保留偏移      按词分割      维护偏移     调整（添加特殊token）
```

### 3.3 字节偏移vs字符偏移

```rust
// 两种编码模式
pub enum OffsetType {
    Byte,    // 字节级偏移（UTF-8编码）
    Char,    // 字符级偏移（Unicode标量值）
}

// Python API区分
tokenizer.encode()            # 字节偏移
tokenizer.encode_char_offsets()  # 字符偏移
```

---

## 4. 双向映射方法

### 4.1 Word级映射

```rust
/// token索引范围 -> word -> char区间
pub fn word_to_tokens(&self, word: u32, sequence_id: usize) 
    -> Option<(usize, usize)>;

/// word -> char区间
pub fn word_to_chars(&self, word: u32, sequence_id: usize) 
    -> Option<Offsets>;
```

```python
# Python使用示例
encoding = tokenizer.encode("Hello world")

# word_to_tokens: 获取第0个词（Hello）对应的token范围
start, end = encoding.word_to_tokens(0, sequence_id=0)
print(f"Word 0 -> tokens[{start}:{end}]: {encoding.tokens[start:end]}")
# 输出: Word 0 -> tokens[0:1]: ['Hello']

# word_to_chars: 获取第0个词在原文中的字符位置
char_start, char_end = encoding.word_to_chars(0, sequence_id=0)
print(f"Word 0 -> chars[{char_start}:{char_end}]")
# 输出: Word 0 -> chars[0:5]
```

### 4.2 Token级映射

```rust
/// token索引 -> char区间 + sequence_id
pub fn token_to_chars(&self, token: usize) 
    -> Option<(usize, Offsets)>;

/// token索引 -> word索引 + sequence_id
pub fn token_to_word(&self, token: usize) 
    -> Option<(usize, u32)>;
```

```python
# token_to_chars: 获取第1个token的字符位置
token_idx = 1
seq_id, (char_start, char_end) = encoding.token_to_chars(token_idx)
print(f"Token {encoding.tokens[token_idx]} -> chars[{char_start}:{char_end}]")

# token_to_word: 获取token属于哪个word
seq_id, word_id = encoding.token_to_word(token_idx)
print(f"Token {token_idx} belongs to word {word_id}")
```

### 4.3 字符级映射

```rust
/// 字符位置 -> token索引
pub fn char_to_token(&self, pos: usize, sequence_id: usize) 
    -> Option<usize>;

/// 字符位置 -> word索引
pub fn char_to_word(&self, pos: usize, sequence_id: usize) 
    -> Option<u32>;
```

```python
# char_to_token: 查询字符位置5属于哪个token
token_idx = encoding.char_to_token(5, sequence_id=0)
print(f"Char 5 is in token: {encoding.tokens[token_idx]}")

# char_to_word: 查询字符位置5属于哪个word
word_idx = encoding.char_to_word(5, sequence_id=0)
print(f"Char 5 is in word: {word_idx}")
```

---

## 5. Pair输入处理

### 5.1 Pair编码示例

```python
# 编码句子对
encoding = tokenizer.encode("First sentence.", "Second sentence.")

print(encoding.ids)           # [101, 2034, 6251, 1012, 102, 2117, 6251, 1012, 102]
print(encoding.type_ids)      # [0, 0, 0, 0, 0, 1, 1, 1, 1]
                               # ^^^^^^第一句^^^^  ^^^^^^第二句^^^^
print(encoding.tokens)        # ['[CLS]', 'first', 'sentence', '.', '[SEP]', 
                              #  'second', 'sentence', '.', '[SEP]']
```

### 5.2 序列范围查询

```python
# 获取各序列的token范围
seq0_range = encoding.sequence_ids.index(0)  # 第一句的token索引范围
seq1_range = encoding.sequence_ids.index(1)  # 第二句的token索引范围

# API方法
n_seq = encoding.n_sequences  # 序列数量
for i in range(n_seq):
    token_start, token_end = encoding.sequence_range(i)
    print(f"Sequence {i}: tokens[{token_start}:{token_end}]")
```

---

## 6. Overflow处理

### 6.1 长文本截断

```python
# 启用truncation
tokenizer.enable_truncation(max_length=10, stride=5)

# 长文本会产生overflow
encoding = tokenizer.encode("这是一个很长的文本，超过了max_length限制...")

print(len(encoding.ids))           # <= 10
print(len(encoding.overflowing))   # 可能有多个overflow片段

# 访问overflow片段
for i, overflow in enumerate(encoding.overflowing):
    print(f"Overflow {i}: {overflow.tokens}")
```

### 6.2 Stride机制

用于滑动窗口场景（如文档QA）：

```python
tokenizer.enable_truncation(
    max_length=512,
    stride=128,  # 相邻片段的重叠token数
    strategy="longest_first",
)

# 输入超过512时，产生按128步长滑动的多个片段
```

---

## 7. Padding与对齐

### 7.1 填充后的Encoding

```python
# 启用填充
tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=10)

encoding = tokenizer.encode("Hello")
print(encoding.ids)             # [101, 7592, 102, 0, 0, 0, 0, 0, 0, 0]
print(encoding.attention_mask)  # [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
                                 # ^^^有效token^^^
```

### 7.2 批量填充

```python
# 批量编码自动对齐到最长序列
encodings = tokenizer.encode_batch(["Hello", "This is longer text"])

for enc in encodings:
    print(f"IDs: {enc.ids}, Mask: {enc.attention_mask}")
```

---

## 8. 实用方法

### 8.1 属性访问

```python
# 基本信息
len(encoding)           # token数量
encoding.n_sequences    # 序列数量（pair输入时为2）
encoding.tokens         # token字符串列表
encoding.ids            # token ID列表
encoding.offsets        # 偏移量列表
encoding.type_ids       # segment IDs
encoding.words          # word ID列表
encoding.attention_mask # attention掩码
encoding.special_tokens_mask  # 特殊token掩码

# 特殊token查询
encoding.special_tokens_mask  # 1表示特殊token
```

### 8.2 切片操作

```python
# 获取子范围（如用于滑动窗口）
sub_encoding = encoding[0:10]  # 前10个token

# truncation后获取overflow
for overflow in encoding.overflowing:
    process(overflow)
```

### 8.3 转换为Python类型

```python
# 转换为常见的ML框架输入格式
import numpy as np

input_ids = np.array(encoding.ids)
attention_mask = np.array(encoding.attention_mask)
token_type_ids = np.array(encoding.type_ids)

# 或者转换为PyTorch/TensorFlow张量
import torch

inputs = {
    "input_ids": torch.tensor([encoding.ids]),
    "attention_mask": torch.tensor([encoding.attention_mask]),
}
```

---

## 9. 完整实战示例

### 9.1 命名实体识别(NER)标注

```python
from tokenizers import Tokenizer

# 加载BPE分词器
tokenizer = Tokenizer.from_pretrained("bert-base-cased")

# 文本和实体标注
text = "Apple Inc. was founded by Steve Jobs."
entities = [
    (0, 10, "ORG"),    # "Apple Inc."
    (29, 40, "PER"),   # "Steve Jobs"
]

# 编码
encoding = tokenizer.encode(text)

# 将字符级实体标注转换为token级BIO标注
def char_to_token_labels(encoding, entities, text_length):
    """将字符级实体标签转换为token级BIO标签"""
    labels = ["O"] * len(encoding.ids)
    
    for char_start, char_end, entity_type in entities:
        # 找到覆盖该字符范围的token
        start_token = None
        end_token = None
        
        for i, (tok_start, tok_end) in enumerate(encoding.offsets):
            # 跳过特殊token
            if tok_start == tok_end == 0:
                continue
            
            # 检查重叠
            if tok_start < char_end and tok_end > char_start:
                if start_token is None:
                    start_token = i
                end_token = i
        
        # 分配BIO标签
        if start_token is not None:
            labels[start_token] = f"B-{entity_type}"
            for i in range(start_token + 1, end_token + 1):
                labels[i] = f"I-{entity_type}"
    
    return labels

# 生成token级标签
token_labels = char_to_token_labels(encoding, entities, len(text))
for token, label in zip(encoding.tokens, token_labels):
    print(f"{token:15s} {label}")

# 输出:
# [CLS]           O
# Apple           B-ORG
# Inc             I-ORG
# .               I-ORG
# was             O
# founded         O
# by              O
# Steve           B-PER
# Jobs            I-PER
# .               I-PER
# [SEP]           O
```

### 9.2 问答任务对齐

```python
def get_answer_span(encoding, answer_text, context):
    """在tokenized序列中定位答案span"""
    # 尝试找到答案在token中的位置
    answer_tokens = tokenizer.encode(answer_text).tokens[1:-1]  # 去掉CLS/SEP
    
    for i in range(len(encoding.tokens) - len(answer_tokens) + 1):
        if encoding.tokens[i:i+len(answer_tokens)] == answer_tokens:
            return (i, i + len(answer_tokens))
    
    return None

# 使用
encoding = tokenizer.encode(question, context)
start_pos, end_pos = get_answer_span(encoding, "Apple", context)
```

---

## 10. 注意事项

### 10.1 偏移量注意事项

1. **特殊token偏移为(0,0)**：如[CLS]、[SEP]等特殊token的offsets通常是(0,0)
2. **字节vs字符**：根据编码选择正确的offset类型
3. **Unicode规范化**：Normalizer会改变偏移，需保持一致

### 10.2 性能优化

```python
# encode_char_offsets比encode慢（需要字符计数）
# 仅在需要时调用
if need_char_alignment:
    encoding = tokenizer.encode_char_offsets(text)
else:
    encoding = tokenizer.encode(text)
```
