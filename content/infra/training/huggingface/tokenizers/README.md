# HuggingFace Tokenizers 架构文档

本目录包含HuggingFace Tokenizers库的细粒度架构文档，从源码层面深入解析分词器的实现原理。

---

## 文档导航

| 文档 | 内容概览 |
|------|----------|
| [00_整体架构与设计理念.md](./00_整体架构与设计理念.md) | Pipeline模式、Rust核心实现、Wrapper多态、偏移量追踪设计 |
| [01_Tokenizer核心API.md](./01_Tokenizer核心API.md) | Tokenizer类、from_pretrained、encode/decode、序列化 |
| [02_五大处理阶段.md](./02_五大处理阶段.md) | Normalizer、PreTokenizer、Model、PostProcessor、Decoder详解 |
| [03_BPE模型实现.md](./03_BPE模型实现.md) | BPE算法、ByteLevel BPE、训练流程、缓存机制 |
| [04_WordPiece与Unigram.md](./04_WordPiece与Unigram.md) | WordPiece算法、UnigramLM、SentencePiece兼容、子词正则化 |
| [05_Encoding数据结构.md](./05_Encoding数据结构.md) | Encoding结构、offset映射、word_to_tokens、Pair输入处理 |
| [06_Trainer训练分词器.md](./06_Trainer训练分词器.md) | BpeTrainer、WordPieceTrainer、UnigramTrainer、语料准备 |
| [07_添加特殊标记.md](./07_添加特殊标记.md) | AddedToken、AddedVocabulary、DoubleArrayTrie匹配、对话系统模板 |
| [08_并行处理与性能.md](./08_并行处理与性能.md) | Rayon并行、缓存优化、线程池控制、性能调优 |

---

## 阅读建议

### 入门路径

1. 先读 [00_整体架构与设计理念.md](./00_整体架构与设计理念.md) 理解整体架构
2. 浏览 [01_Tokenizer核心API.md](./01_Tokenizer核心API.md) 熟悉基本使用
3. 根据使用场景选择特定处理阶段深入（02-04）

### 源码定位速查

| 模块 | 源码路径 |
|------|----------|
| Tokenizer核心 | `tokenizers/src/tokenizer/mod.rs` |
| Encoding结构 | `tokenizers/src/tokenizer/encoding.rs` |
| AddedVocabulary | `tokenizers/src/tokenizer/added_vocabulary.rs` |
| BPE模型 | `tokenizers/src/models/bpe/` |
| WordPiece模型 | `tokenizers/src/models/wordpiece/` |
| Unigram模型 | `tokenizers/src/models/unigram/` |
| PreTokenizers | `tokenizers/src/pre_tokenizers/` |
| Normalizers | `tokenizers/src/normalizers/` |
| PostProcessors | `tokenizers/src/processers/` |
| Decoders | `tokenizers/src/decoders/` |
| Python绑定 | `bindings/python/src/` |

---

## 核心概念速览

### Pipeline架构

```
Raw Text -> [Normalizer] -> [PreTokenizer] -> [Model] -> [PostProcessor] -> [Decoder]
                ↓                ↓              ↓              ↓              ↓
           文本清洗        初步分割        ID映射      添加特殊token    还原文本
```

### 模型对比

| 模型 | 算法 | 特点 | 代表模型 |
|------|------|------|----------|
| BPE | 子词合并 | 高效，字节级支持 | GPT系列 |
| WordPiece | 最大似然合并 | BERT原生 | BERT系列 |
| Unigram | 概率剪枝 | 支持子词正则化 | XLNet/ALBERT |

### 关键设计模式

- **Wrapper多态**：Enum Wrapper实现Rust组件多态
- **偏移量追踪**：Encoding维护字符<->token双向映射
- **Double-Array Trie**：O(n)多模式匹配特殊token

---

## 官方资源

- **官方文档**: https://huggingface.co/docs/tokenizers
- **GitHub仓库**: https://github.com/huggingface/tokenizers
- **API参考**: `docs/source/api/` (本地)

---

## 核心论文

- **BPE**: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (Sennrich et al., 2016)
- **WordPiece**: [Google's Neural Machine Translation System](https://arxiv.org/abs/1609.08144) (Wu et al., 2016)
- **Unigram**: [Subword Regularization](https://arxiv.org/abs/1804.10959) (Kudo, 2018)
- **BPE Dropout**: [Training BPE with Dropout](https://arxiv.org/abs/1910.13267) (Provilkov et al., 2019)
- **SentencePiece**: [SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) (Kudo & Richardson, 2018)

---

*文档编写日期: 2026/04/20*
*基于HuggingFace Tokenizers源码分析*
