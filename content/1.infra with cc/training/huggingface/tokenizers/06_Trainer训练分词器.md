---
created: 2026-05-06
---

# Trainer训练分词器

> **【源码定位】**
> - Trainer trait: `tokenizers/src/tokenizer/mod.rs` (lines 186-226)
> - BPE训练器: `tokenizers/src/models/bpe/trainer.rs`
> - WordPiece训练器: `tokenizers/src/models/wordpiece/trainer.rs`
> - Unigram训练器: `tokenizers/src/models/unigram/trainer.rs`

---

## 1. Trainer架构

### 1.1 核心Trait

```rust
pub trait Trainer {
    type Model: Model + Sized;
    
    /// 主训练方法：从语料学习词汇
    fn train(&self, model: &mut Self::Model) -> Result<Vec<AddedToken>>;
    
    /// 迭代 feeding 语料数据
    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where 
        I: Iterator<Item = S> + Send, 
        S: AsRef<str> + Send, 
        F: Fn(&str) -> Result<Vec<String>> + Sync;
}
```

### 1.2 Trainer层次

```
Trainer (trait)
    ├── TrainerWrapper (Enum)
    │       ├── BpeTrainer
    │       ├── WordPieceTrainer
    │       └── UnigramTrainer
    └── 各自实现train算法
```

---

## 2. BpeTrainer

### 2.1 配置参数

**文件**: `tokenizers/src/models/bpe/trainer.rs`

```rust
pub struct BpeTrainer {
    vocab_size: usize,                    // 目标词汇表大小
    min_frequency: usize,                 // token最小出现频率
    special_tokens: Vec<AddedToken>,      // 特殊token
    limit_alphabet: Option<usize>,       // 限制初始字母表大小
    initial_alphabet: HashSet<char>,    // 指定初始字符集
    continuing_subword_prefix: Option<String>,  // 子词前缀
    end_of_word_suffix: Option<String>,         // 词尾后缀
    show_progress: bool,                  // 显示进度条
    max_piece_length: usize,              // 最大子词长度
}
```

### 2.2 训练流程

```rust
impl BpeTrainer {
    fn do_train(&self, word_counts: &HashMap<String, u64>) -> (Vocab, Merges) {
        // 1. 初始化词汇：从字符开始
        let vocab = self.initialize_vocab(&word_counts);
        
        // 2. 计算初始词频统计
        let mut stats = self.compute_pair_stats(&word_counts, &vocab);
        
        // 3. 迭代合并直到目标大小
        while vocab.len() < self.vocab_size {
            // 选择频率最高的pair
            if let Some(((a, b), _)) = stats.iter().max_by_key(|(_, count)| *count) {
                // 更新词汇
                let new_token = format!("{}{}", a, b);
                vocab.insert(new_token, vocab.len() as u32);
                merges.push((a.clone(), b.clone()));
                
                // 更新统计
                self.update_stats(&mut stats, (*a, *b));
            } else {
                break;
            }
        }
        
        (vocab, merges)
    }
}
```

### 2.3 使用示例

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 1. 创建模型
tokenizer = Tokenizer(models.BPE())

# 2. 配置预分词器（影响训练时的word定义）
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

# 3. 创建训练器
trainer = trainers.BpeTrainer(
    vocab_size=50000,
    min_frequency=2,
    special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 256字节
    show_progress=True,
    max_piece_length=16,  # 限制子词长度
)

# 4. 训练
files = ["train1.txt", "train2.txt", "train3.txt"]
tokenizer.train(files, trainer)

# 5. 保存
tokenizer.save("bpe_tokenizer.json")
```

---

## 3. WordPieceTrainer

### 3.1 配置参数

**文件**: `tokenizers/src/models/wordpiece/trainer.rs`

```rust
pub struct WordPieceTrainer {
    vocab_size: usize,
    min_frequency: usize,
    special_tokens: Vec<AddedToken>,
    limit_alphabet: Option<usize>,
    initial_alphabet: HashSet<char>,
    continuing_subword_prefix: Option<String>,
    // WordPiece特有
    wordpieces_prefix: String,  // 默认 "##"
}
```

### 3.2 训练算法

与BPE的区别：选择标准不是频率，而是似然增益

```rust
impl WordPieceTrainer {
    fn do_train(&self, word_counts: &HashMap<String, u64>) -> Vocab {
        // 1. 初始化词汇（字符级别）
        let mut vocab = self.initialize_vocab();
        
        // 2. 计算词语分段
        let mut segmentations = self.initialize_segmentations(&word_counts, &vocab);
        
        // 3. 迭代选择最优合并
        while vocab.len() < self.vocab_size {
            // 计算每个候选pair的似然增益
            let candidates = self.compute_likelihood_scores(&segmentations, &word_counts);
            
            // 选择使似然增加最大的pair
            if let Some(best) = candidates.into_iter().max_by(|a, b| {
                a.likelihood_gain.partial_cmp(&b.likelihood_gain).unwrap()
            }) {
                vocab.insert(best.new_token);
                self.update_segmentations(&mut segmentations, &best);
            }
        }
        
        vocab
    }
}
```

### 3.3 使用示例

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, processors, decoders

# 1. 创建WordPiece模型
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# 2. 配置Pipeline（BERT风格）
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),
    normalizers.Lowercase(),
    normalizers.StripAccents(),
])
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

# 3. 创建训练器
trainer = trainers.WordPieceTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    continuing_subword_prefix="##",
    limit_alphabet=1000,  # 限制初始字母表
)

# 4. 训练
files = ["wiki.train.raw", "bookcorpus.txt"]
tokenizer.train(files, trainer)

# 5. 配置PostProcessor和Decoder
tokenizer.post_processor = processors.BertProcessing(
    ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ("[CLS]", tokenizer.token_to_id("[CLS]")),
)
tokenizer.decoder = decoders.WordPiece(prefix="##")

# 6. 保存
tokenizer.save("bert_tokenizer.json")
```

---

## 4. UnigramTrainer

### 4.1 配置参数

**文件**: `tokenizers/src/models/unigram/trainer.rs`

```rust
pub struct UnigramTrainer {
    vocab_size: usize,
    n_sub_iterations: usize,      // EM迭代次数
    shrinking_factor: f64,        // 每次剪枝比例 (默认0.75)
    special_tokens: Vec<AddedToken>,
    unk_token: Option<String>,
    max_piece_length: usize,
    seed_size: usize,             // 初始种子词汇大小
    show_progress: bool,
}
```

### 4.2 训练流程

Unigram使用**EM算法**交替优化：

```rust
impl UnigramTrainer {
    fn do_train(&self, sentences: Vec<String>) -> Vec<(String, f64)> {
        // 1. 构建初始大词汇表（包含所有子串）
        let mut vocab = self.make_seed_sentencepieces(sentences);
        
        // 2. EM迭代
        loop {
            // E-step: 使用Viterbi对所有句子分段，计算频率
            let freq = self.run_e_step(&vocab, &sentences);
            
            // M-step: 从频率重新估计概率
            let probs = self.run_m_step(&vocab, &freq);
            
            // 更新词汇概率
            self.update_vocab_probabilities(&mut vocab, &probs);
            
            // 剪枝
            let desirable_size = (vocab.len() as f64 * self.shrinking_factor) as usize;
            let _ = self.prune_vocab(&mut vocab, desirable_size);
            
            if vocab.len() <= self.vocab_size {
                break;
            }
        }
        
        vocab
    }
}
```

### 4.3 使用示例

```python
from tokenizers import Tokenizer, models, trainers, normalizers, pre_tokenizers, decoders

# 1. 创建Unigram模型
tokenizer = Tokenizer(models.Unigram())

# 2. 配置（SentencePiece风格）
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFKC(),
])
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
    replacement="▁", 
    add_prefix_space=True,
)
tokenizer.decoder = decoders.Metaspace(
    replacement="▁",
    add_prefix_space=True,
)

# 3. 创建训练器
trainer = trainers.UnigramTrainer(
    vocab_size=32000,
    shinrinking_factor=0.75,  # 每次保留75%
    special_tokens=["<s>", "</s>", "<pad>", "<mask>", "<unk>"],
    unk_token="<unk>",
    max_piece_length=64,
    seed_size=500000,  # 初始候选集大小
    show_progress=True,
)

# 4. 训练
tokenizer.train(["corpus.txt"], trainer)

# 5. 保存
tokenizer.save("unigram_tokenizer.json")
```

---

## 5. 训练语料准备

### 5.1 语料格式

```python
# 文本文件：一行一个句子/文档
# train.txt:
# This is the first line.
# This is the second line.
# ...

# 支持多个文件
files = ["train1.txt", "train2.txt", "train3.txt"]
tokenizer.train(files, trainer)
```

### 5.2 从内存训练

```python
from tokenizers import Tokenizer, models, trainers

# 从字符串列表训练
sequences = [
    "This is the first sentence.",
    "This is the second sentence.",
    "Another document here.",
]

# 使用from_memory接口（部分版本支持）
# 或使用迭代器
tokenizer.train_from_iterator(sequences, trainer=trainer)
```

### 5.3 大型语料处理

```python
# 使用生成器处理大文件
def read_large_file(file_path, chunk_size=10000):
    """大文件生成器"""
    with open(file_path, 'r', encoding='utf-8') as f:
        chunk = []
        for line in f:
            chunk.append(line.strip())
            if len(chunk) >= chunk_size:
                yield from chunk
                chunk = []
        if chunk:
            yield from chunk

# 训练（自动流式处理）
files = ["large_corpus_1.txt", "large_corpus_2.txt"]
tokenizer.train(files, trainer)
```

---

## 6. 训练最佳实践

### 6.1 参数调优指南

| 参数 | BPE | WordPiece | Unigram |
|------|-----|-----------|---------|
| vocab_size | 通常30k-50k | 通常30k | 通常32k |
| min_frequency | 2-5 | 2-5 | 不限（概率控制） |
| max_piece_length | 16-32 | 不涉及 | 16-64 |

### 6.2 多文件训练

```python
# 包含多个数据源
data_files = [
    "data/wiki.txt",
    "data/books.txt",
    "data/code.txt",  # 代码语料
    "data/chat.txt",  # 对话语料
]

# 使用不同权重（部分训练器支持）
# 或者先合并文件
```

### 6.3 特殊token设计

```python
# 语言模型特殊token
lm_tokens = ["<|endoftext|>", "<|pad|>", "<|mask|>"]

# BERT特殊token
bert_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

# T5特殊token
t5_tokens = ["<pad>", "</s>", "<unk>", "<extra_id_0>", "<extra_id_1>"]

# 训练器配置
trainer = trainers.BpeTrainer(
    vocab_size=50000,
    special_tokens=lm_tokens,
)
```

---

## 7. 完整实战：训练自定义分词器

### 7.1 中文BPE分词器

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, processors, decoders

# 1. 创建模型
tokenizer = Tokenizer(models.BPE(
    unk_token="<|unk|>",
    byte_fallback=True,  # 字节回退处理生僻字
))

# 2. 配置
# 中文通常不需要复杂normalizer
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFKC(),  # 统一中日韩字符
    normalizers.Strip(),
])

# ByteLevel处理中文
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

# 3. 训练器
trainer = trainers.BpeTrainer(
    vocab_size=50000,
    min_frequency=2,
    special_tokens=["<|endoftext|>", "<|unk|>", "<|pad|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    max_piece_length=8,  # 中文token不宜过长
)

# 4. 训练
tokenizer.train(
    ["zh_wiki.txt", "zh_news.txt", "zh_books.txt"],
    trainer,
)

# 5. 测试
text = "自然语言处理是人工智能的重要领域。"
encoded = tokenizer.encode(text)
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")

# 6. 保存
tokenizer.save("chinese_bpe_tokenizer.json")
```

### 7.2 代码专用分词器

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 代码需要保留空白和缩进
tokenizer = Tokenizer(models.BPE())

# 不添加prefix space，保留原始空白
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False,
    trim_offsets=False,
)

# 训练器：允许更长子词
trainer = trainers.BpeTrainer(
    vocab_size=50000,
    min_frequency=2,
    special_tokens=["<|endoftext|>", "<|pad|>"],
    max_piece_length=32,  # 代码token可以更长
)

# 使用代码语料（GitHub, StackOverflow等）
code_files = [
    "python_code.txt",
    "javascript_code.txt",
    "java_code.txt",
]
tokenizer.train(code_files, trainer)
```

---

## 8. 训练后处理

### 8.1 添加特殊token

```python
# 训练后添加额外token
tokenizer.add_special_tokens(["[NEW_SPECIAL]"])
tokenizer.add_tokens(["custom_token_1", "custom_token_2"])

# 保存更新后的版本
tokenizer.save("updated_tokenizer.json")
```

### 8.2 合并词汇表

```python
# 加载多个分词器并合并（高级用法）
# 需要手动处理vocab和merges的合并
```

---

## 9. 参考资料

### 核心论文
- **BPE**: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (Sennrich et al., 2016)
- **BPE Dropout**: [Training BPE with Dropout](https://arxiv.org/abs/1910.13267) (Provilkov et al., 2019)
- **WordPiece**: [Google's Neural Machine Translation System](https://arxiv.org/abs/1609.08144) (Wu et al., 2016)
- **Unigram**: [Subword Regularization](https://arxiv.org/abs/1804.10959) (Kudo, 2018)
