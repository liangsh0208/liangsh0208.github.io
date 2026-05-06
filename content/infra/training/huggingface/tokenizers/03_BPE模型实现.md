# BPE模型实现

> **【源码定位】**
> - BPE模型核心: `tokenizers/src/models/bpe/model.rs`
> - BPE模块入口: `tokenizers/src/models/bpe/mod.rs`
> - BPE训练器: `tokenizers/src/models/bpe/trainer.rs`

---

## 1. BPE算法概述

**BPE (Byte Pair Encoding)** 是一种子词分词算法，最初用于数据压缩，后被引入NLP用于处理开放词汇问题。

### 核心思想

1. 从字符级别词汇表开始
2. 统计相邻字符对（或字节对）频率
3. 将最高频的对合并为新token
4. 重复直到达到目标词汇表大小

### 与GPT系列的关系

| 模型 | BPE变体 | 特点 |
|------|---------|------|
| GPT-2 | ByteLevel BPE | 直接使用字节，无需unk_token |
| GPT-3/GPT-4 | ByteLevel BPE | 更大词汇表，相同算法 |
| LLaMA | SentencePiece BPE | BPE + SentencePiece预处理 |

---

## 2. 核心数据结构

**文件**: `tokenizers/src/models/bpe/model.rs`

```rust
pub struct BPE {
    vocab: Vocab,                    // token -> id
    vocab_r: VocabR,                // id -> token
    merges: MergeMap,               // (id1, id2) -> (合并优先级, 新id)
    cache: Option<Cache<Vec<Token>>>,  // LRU缓存
    // 配置选项
    dropout: Option<f32>,          // 随机合并（用于训练数据增强）
    unk_token: Option<String>,     // 未知词标记
    continuing_subword_prefix: Option<String>,  // 子词前缀（如 ##）
    end_of_word_suffix: Option<String>,
    fuse_unk: bool,                // 是否合并连续unk
    byte_fallback: bool,           // 字节级回退
    ignore_merges: bool,           // 纯基于词汇表
}
```

### 类型定义

```rust
type Vocab = AHashMap<String, u32>;
type VocabR = AHashMap<u32, String>;
type MergeMap = AHashMap<(u32, u32), (u32, u32)>;  // (a,b) -> (rank, new_id)
```

---

## 3. Tokenize算法详解

### 3.1 核心流程

```rust
fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
    // 1. 检查缓存
    if let Some(cached) = self.cache.as_ref().and_then(|c| c.get(sequence)) {
        return Ok(cached.clone());
    }
    
    // 2. 按UTF-8字节拆分（启用byte_fallback时）
    let bytes = if self.byte_fallback {
        sequence.bytes().collect::<Vec<_>>()
    } else {
        sequence.as_bytes().to_vec()
    };
    
    // 3. 对每个字符/字节，查找vocab
    let mut tokens: Vec<Token> = Vec::new();
    for byte in bytes {
        let token_str = if self.byte_fallback {
            format!("<0x{:02X}>", byte)
        } else {
            String::from_utf8_lossy(&[byte]).to_string()
        };
        
        if let Some(&id) = self.vocab.get(&token_str) {
            tokens.push(Token::new(id, token_str, (0, 0)));
        } else {
            // 处理unk
            tokens.push(self.make_unk_token());
        }
    }
    
    // 4. 迭代合并
    loop {
        // 找到排名最高的可合并对
        let mut best_rank: u32 = u32::MAX;
        let mut best_pair: Option<(usize, u32)> = None;
        
        for i in 0..tokens.len().saturating_sub(1) {
            let pair = (tokens[i].id, tokens[i+1].id);
            if let Some(&(rank, new_id)) = self.merges.get(&pair) {
                if rank < best_rank {
                    best_rank = rank;
                    best_pair = Some((i, new_id));
                }
            }
        }
        
        // 没有可合并的对，退出
        if let Some((pos, new_id)) = best_pair {
            tokens.drain(pos..pos+2);
            tokens.insert(pos, Token::new(new_id, self.vocab_r[&new_id].clone(), (0, 0)));
        } else {
            break;
        }
    }
    
    // 5. 存入缓存
    if let Some(cache) = &self.cache {
        cache.set(sequence.to_string(), tokens.clone());
    }
    
    Ok(tokens)
}
```

### 3.2 算法可视化

```
输入："hello"
初始词汇: ['h', 'e', 'l', 'l', 'o']

迭代1: ('l', 'l') 合并 -> 'll'
       ['h', 'e', 'll', 'o']

迭代2: ('e', 'll') 合并 -> 'ell'
       ['h', 'ell', 'o']

迭代3: ('h', 'ell') 合并 -> 'hell'
       ['hell', 'o']

迭代4: ('hell', 'o') 合并 -> 'hello'
       ['hello']

最终: ['hello']
```

---

## 4. ByteLevel BPE详解

### 4.1 动机

传统BPE处理生僻字符和表情符号时会产生大量`<unk>`。ByteLevel BPE直接使用UTF-8字节作为基础单元，覆盖所有Unicode字符。

### 4.2 特点

- **无unk_token**：256字节覆盖所有字符
- **空格标记**：使用`Ġ` (U+0120) 标记词首空格
- **高效压缩**：常用Unicode字符会形成自己的token
- **占字节合并**：频繁合并的字符序列被压缩

### 4.3 字节映射

```python
from tokenizers import pre_tokenizers

# 获取256字节字母表
alphabet = pre_tokenizers.ByteLevel.alphabet()
# ['<0x00>', '<0x01>', ..., '<0xFF>']
```

### 4.4 训练配置

```python
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# 初始化ByteLevel BPE
tokenizer = Tokenizer(models.BPE(byte_fallback=True))

# ByteLevel预分词器
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

# 训练器
trainer = trainers.BpeTrainer(
    vocab_size=50000,
    min_frequency=2,
    special_tokens=["<|endoftext|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 256字节
)
```

---

## 5. 训练算法

### 5.1 训练流程

**文件**: `tokenizers/src/models/bpe/trainer.rs`

```rust
pub struct BpeTrainer {
    vocab_size: usize,
    min_frequency: usize,
    special_tokens: Vec<AddedToken>,
    limit_alphabet: Option<usize>,
    initial_alphabet: HashSet<char>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    show_progress: bool,
    max_piece_length: usize,
}
```

### 5.2 训练步骤

1. **收集词频**：统计语料中所有单词出现频率
2. **初始化词汇**：从初始字母表开始
3. **计算字节对**：使用优先队列选择最佳合并对
4. **迭代合并**：直到达到目标词汇表大小

### 5.3 训练示例

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 1. 创建模型
tokenizer = Tokenizer(models.BPE())

# 2. 配置预分词器
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

# 3. 创建训练器
trainer = trainers.BpeTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["<|endoftext|>", "<|pad|>"],
)

# 4. 训练
files = ["train1.txt", "train2.txt", "train3.txt"]
tokenizer.train(files, trainer)

# 5. 保存
tokenizer.save("bpe_tokenizer.json")
```

---

## 6. 高级特性

### 6.1 BPE Dropout

```rust
// 训练时随机跳过某些合并（数据增强）
pub struct BPE {
    dropout: Option<f32>,  // 0.0-1.0 跳过概率
}
```

```python
# Python配置
model = models.BPE(dropout=0.1)  # 10%概率跳过合并
```

### 6.2 字节回退

```rust
// 启用byte_fallback时，对未知字符使用字节表示
BPE {
    byte_fallback: true,
}

// "語"（不在vocab中）-> ["<0xE8>", "<0xAA>", "<0x9E>"]
```

### 6.3 缓存机制

```rust
// LRU缓存加速重复查询
pub struct Cache<T: Clone> {
    map: LruCache<String, T>,
    // 默认容量: 10,000
}
```

### 6.4 连续UNK合并

```rust
// fuse_unk: 连续unk合并为单个token
BPE {
    fuse_unk: true,
}

// 例如: [UNK] [UNK] [UNK] -> [UNK]
```

---

## 7. 完整实战示例

### 7.1 GPT-2风格BPE

```python
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# 1. 初始化ByteLevel BPE
tokenizer = Tokenizer(models.BPE(
    byte_fallback=True,
    cache_capacity=10000,
))

# 2. 配置ByteLevel
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

# 3. 训练
trainer = trainers.BpeTrainer(
    vocab_size=50000,
    min_frequency=2,
    special_tokens=["<|endoftext|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)

tokenizer.train(["corpus.txt"], trainer)

# 4. 使用
output = tokenizer.encode("Hello world")
print(output.tokens)  # ['Hello', 'Ġworld']  (Ġ表示空格前缀)
print(output.ids)     # [15496, 995]
```

### 7.2 与transformers集成

```python
from transformers import AutoTokenizer

# 底层使用tokenizers库
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 等价于：
# from tokenizers import Tokenizer
# tokenizer = Tokenizer.from_pretrained("gpt2")
```

---

## 8. 常见问题

### 8.1 性能优化

| 问题 | 原因 | 解决 |
|------|------|------|
| 首次编码慢 | BPE缓存冷启动 | 预热：提前编码常用词 |
| 内存占用高 | 缓存未限制 | 调整`cache_capacity` |
| 训练慢 | 大词汇表计算复杂 | 降低`max_piece_length` |

### 8.2 一致性排查

```python
# 问题：编码长度与transformers不一致
# 检查：
print(tokenizer.model.dropout)  # should be None for inference
print(tokenizer.pre_tokenizer)   # should match
print(tokenizer.normalizer)      # should match
```
