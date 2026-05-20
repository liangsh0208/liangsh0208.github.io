---
created: 2026-05-06
---

# WordPiece与Unigram

> **【源码定位】**
> - WordPiece模型: `tokenizers/src/models/wordpiece/mod.rs`
> - Unigram模型: `tokenizers/src/models/unigram/`
> - WordPiece训练器: `tokenizers/src/models/wordpiece/trainer.rs`
> - Unigram训练器: `tokenizers/src/models/unigram/trainer.rs`

---

## 1. WordPiece算法

### 1.1 算法概述

**WordPiece**是Google为神经机器翻译（NMT）引入的子词分词算法，后被BERT采用。

与BPE的核心区别：
- **BPE**：选择**频率最高**的字符对合并
- **WordPiece**：选择**似然最大化**的合并（最大化训练数据似然）

### 1.2 核心思想

```
# 初始化：从字符词汇表开始
# 每步选择使训练数据似然增加最大的合并
score = log(P(合并对)) - log(P(a) * P(b))
选择最小代价的合并（最大似然增益）
```

### 1.3 核心数据结构

**文件**: `tokenizers/src/models/wordpiece/mod.rs`

```rust
pub struct WordPiece {
    vocab: Vocab,                           // token -> id
    vocab_r: VocabR,                        // id -> token
    unk_token: String,
    unk_token_id: u32,
    continuing_subword_prefix: String,        // 子词前缀（如 ##）
    max_input_chars_per_word: usize,
}

// Tokenize过程
type Vocab = AHashMap<String, u32>;
type VocabR = AHashMap<u32, String>;
```

### 1.4 Tokenize算法

```rust
fn tokenize(&self, sentence: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    
    for word in sentence.split_whitespace() {
        let chars: Vec<char> = word.chars().collect();
        let mut start = 0;
        let mut is_bad = false;
        let mut sub_tokens = Vec::new();
        
        while start < chars.len() {
            let mut end = chars.len();
            let mut cur_substr = None;
            
            // 贪心最长匹配
            while start < end {
                let substr: String = if start == 0 {
                    chars[start..end].iter().collect()
                } else {
                    // 非首子词，添加前缀
                    self.continuing_subword_prefix.clone() + 
                    &chars[start..end].iter().collect::<String>()
                };
                
                if self.vocab.contains_key(&substr) {
                    cur_substr = Some(substr);
                    break;
                }
                end -= 1;
            }
            
            if let Some(substr) = cur_substr {
                let id = self.vocab[&substr];
                sub_tokens.push(Token::new(id, substr, (0, 0)));
                start = end;
            } else {
                is_bad = true;
                break;
            }
        }
        
        // 处理未知词
        if is_bad && !sub_tokens.is_empty() {
            tokens.push(Token::new(
                self.unk_token_id,
                self.unk_token.clone(),
                (0, 0)
            ));
        } else {
            tokens.extend(sub_tokens);
        }
    }
    
    Ok(tokens)
}
```

### 1.5 BERT风格配置

```python
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors, decoders

# 1. 创建WordPiece模型
tokenizer = Tokenizer(models.WordPiece(
    unk_token="[UNK]",
    continuing_subword_prefix="##",
    max_input_chars_per_word=100,
))

# 2. 配置Pipeline
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True, strip_accents=True)
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
tokenizer.post_processor = processors.BertProcessing(
    ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ("[CLS]", tokenizer.token_to_id("[CLS]")),
)
tokenizer.decoder = decoders.WordPiece(prefix="##")

# 3. 训练
trainer = trainers.WordPieceTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
)
tokenizer.train(["wiki.txt"], trainer)
```

---

## 2. Unigram算法

### 2.1 算法概述

**Unigram Language Model**（一元语言模型）是Google为SentencePiece引入的子词分词算法，用于XLNet、ALBERT等模型。

### 2.2 核心思想

与BPE/WordPiece的"从少到多"策略相反：
- **Unigram**：从庞大的候选集开始，逐步剪枝到目标大小
- 使用**Viterbi算法**在多个候选分段中选择最优
- 支持**概率采样**（子词正则化）

### 2.3 核心数据结构

**文件**: `tokenizers/src/models/unigram/mod.rs`

```rust
pub struct Unigram {
    vocab: Vec<(String, f64)>,    // (token, log_prob)
    token_to_ids: AHashMap<String, u32>,
    unk_id: u32,
    byte_fallback: bool,
    dps: Vec<Vec<usize>>,         // 实现细节
}

// Vocabulary由概率分布定义
// 每个token有对应的概率（从训练语料学习）
```

### 2.4 Tokenize算法

```rust
fn tokenize(&self, text: &str) -> Result<Vec<Token>> {
    // 使用Viterbi算法找到最优分段
    let lattice = self.build_lattice(text)?;
    let path = lattice.viterbi()?;  // 动态规划求最优路径
    
    let mut tokens = Vec::new();
    for node in path {
        let token = self.vocab[node.id].0.clone();
        let id = self.token_to_ids[&token];
        tokens.push(Token::new(id, token, (0, 0)));
    }
    
    Ok(tokens)
}

fn tokenize_with_dropout(&self, text: &str, dropout: f32) -> Result<Vec<Token>> {
    // 子词正则化：从多个候选中采样
    // 每次采样产生的分段可能不同
}
```

### 2.5 Viterbi算法可视化

```
输入："university"

构建Lattice（计算每个位置的最优分段）：
0: u/n/i/v/e/r/s/i/t/y
            ↓
1: un/i/v/e/r/s/i/t/y
            ↓
2: uni/v/e/r/s/i/t/y
            ↓
...
选择总概率最高的路径
```

### 2.6 子词正则化

**文件**: `tokenizers/src/models/unigram/lattice.rs`

```rust
// 支持采样的训练时增强
impl Lattice {
    /// 采样一个分段（用于训练数据增强）
    fn sample(&mut self, theta: f32) -> Result<Vec<Node>> {
        // 从后向前采样
        // 基于softmax分布选择下一个节点
    }
}
```

```python
# Python使用
from tokenizers import models

# 使用源码kouchim
model = models.Unigram()
model.enable_dropout(0.1)  # 10%概率使用替代分段

# 同一句话每次编码可能不同
tokens1 = model.tokenize("university")
tokens2 = model.tokenize("university")
# tokens1 可能不等于 tokens2
```

---

## 3. SentencePiece兼容

### 3.1 设计对比

| 特性 | SentencePiece | HuggingFace Tokenizers |
|------|---------------|------------------------|
| 预处理 | 独立库 | Pipeline组件 |
| 模型 | Unigram/BPE | BPE/WordPiece/Unigram/WordLevel |
| 特殊空格 | ▁前缀 | PreTokenizer处理 |
| 归一化 | NFKC内置 | Normalizer独立配置 |

### 3.2 模拟SentencePiece Unigram

```python
from tokenizers import Tokenizer, models, pre_tokenizers, normalizers, decoders

# 配置类似SentencePiece的Unigram
tokenizer = Tokenizer(models.Unigram())

# NFKC归一化（SentencePiece默认）
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFKC(),
    normalizers.Replace(" ", "▁"),  # 空格替换为▁
])

# Metaspace预分词器（处理▁）
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

# Metaspace解码器
tokenizer.decoder = decoders.Metaspace()
```

### 3.3 从SentencePiece模型加载

```python
# 从.protos模型文件加载
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("spiece.model")

# 或使用sentencepiece_processor转换
```

---

## 4. 训练算法对比

### 4.1 三种算法训练流程

| 步骤 | BPE | WordPiece | Unigram |
|------|-----|-----------|---------|
| 初始化 | 字符表 | 字符表 | 超大候选集 |
| 迭代方向 | 合并 | 合并 | 剪枝 |
| 选择标准 | 频率最高 | 似然最大 | 损失最小 |
| 终止条件 | 目标vocab大小 | 目标vocab大小 | 目标vocab大小 |
| 支持Dropout | BPE Dropout | 否 | 子词正则化 |

### 4.2 训练器对比

```python
from tokenizers import trainers

# BPE训练器
bpe_trainer = trainers.BpeTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["<s>", "</s>", "<unk>"],
)

# WordPiece训练器
wp_trainer = trainers.WordPieceTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["[CLS]", "[SEP]", "[UNK]", "[PAD]"],
)

# Unigram训练器
unigram_trainer = trainers.UnigramTrainer(
    vocab_size=32000,
    special_tokens=["<s>", "</s>"],
    unk_token="<unk>",
)
```

---

## 5. 实战示例

### 5.1 XLNet风格Unigram

```python
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, processors

# 1. 初始化Unigram
tokenizer = Tokenizer(models.Unigram())

# 2. 配置
# NFKC归一化（SentencePiece风格）
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFKC(),
])

# SentencePiece风格Metaspace
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True)
tokenizer.decoder = decoders.Metaspace(replacement="▁", add_prefix_space=True)

# 3. 训练
trainer = trainers.UnigramTrainer(
    vocab_size=32000,
    special_tokens=["<cls>", "<sep>", "<pad>", "<mask>", "<eod>"],
    unk_token="<unk>",
    shrinking_factor=0.75,  # 每次迭代保留比例
    show_progress=True,
)

tokenizer.train(["corpus.txt"], trainer)

# 4. 使用
output = tokenizer.encode("Hello world")
print(output.tokens)
```

### 5.2 ALBERT风格SentencePiece兼容

```python
# 完全模拟ALBERT分词器配置
tokenizer = Tokenizer(models.Unigram())

# ALBERT使用SentencePiece内置归一化
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFKC(),
    normalizers.Replace(" ", "▁"),
])

tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
    replacement="▁",
    add_prefix_space=True,
)

tokenizer.post_processor = processors.TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> </s> $B </s>",
    special_tokens=[
        ("<s>", 0),
        ("</s>", 1),
    ],
)
```

---

## 6. 选择指南

### 6.1 模型选择决策树

```
是否需要处理任意Unicode？
├── 是 -> ByteLevel BPE (GPT风格)
│
└── 否
    └── 是否需要子词正则化？
        ├── 是 -> Unigram (XLNet/ALBERT风格)
        └── 否
            └── 是否需要SentencePiece兼容？
                ├── 是 -> Unigram + Metaspace
                └── 否 -> WordPiece (BERT风格)
```

### 6.2 各模型特点总结

| 模型 | 优点 | 缺点 | 代表模型 |
|------|------|------|----------|
| BPE | 简单直观，无unk | 可能产生不合语义的合并 | GPT系列 |
| WordPiece | 似然最优，子词质量高 | 训练复杂度高 | BERT系列 |
| Unigram | 概率采样，分段灵活 | 实现复杂度最高 | XLNet, ALBERT |

---

## 7. 参考资料

### 核心论文
- **WordPiece**: [Google's Neural Machine Translation System](https://arxiv.org/abs/1609.08144) (Wu et al., 2016)
- **Unigram**: [Subword Regularization](https://arxiv.org/abs/1804.10959) (Kudo, 2018)
- **SentencePiece**: [SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) (Kudo & Richardson, 2018)
