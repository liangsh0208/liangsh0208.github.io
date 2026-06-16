---
created: 2026-06-10
published: 2020-09-28
paper: https://arxiv.org/abs/2009.13081
code: https://github.com/jind11/MedQA
authors: Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, Peter Szolovits (MIT CSAIL)
tags:
  - medical-QA
  - benchmark
  - open-domain-QA
  - USMLE
  - reading-comprehension
---

# What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams

## 一句话总结

MedQA 是首个基于真实医学执业考试的**开放域多选问答数据集**，涵盖英文（USMLE）、简体中文（MCMLE）、繁体中文（TWMLE）三语共 61,097 道题目，配套医学教科书语料库，当时最佳方法仅达 36.7%（英文）准确率，揭示了医学 QA 的巨大挑战。

---

## 1. 研究背景与动机

### 1.1 问题定义

**开放域问答（Open-domain QA）** 要求系统在没有预先给定上下文段落的情况下，从大规模知识库中检索相关信息并回答问题。医学领域的 QA 尤其困难，因为：
- 需要广泛的专业领域知识
- 问题往往涉及多步推理（multi-hop reasoning）
- 临床场景复杂，需要综合多个知识点

### 1.2 现有方法的不足

| 现有数据集 | 局限性 |
|-----------|--------|
| BioASQ | 仅 biomedical 文献，非临床问题 |
| emrQA | 基于电子病历，非开放域 |
| HeadQA | 规模较小，仅西班牙语 |
| 一般 MRC 数据集 (SQuAD 等) | 答案直接存在于给定段落中，非开放域 |

**核心差距**：此前没有一个大规模的、来自真实临床考试的、需要从外部知识库检索信息的 **free-form multiple-choice OpenQA** 数据集。

![](MedQA_table2_comparison.png)

> **Table 2**: MedQA 与现有医学 QA 数据集对比。MedQA 是唯一同时满足"大规模 + 多选 + 开放域"的医学问答数据集。

---

## 2. 方法

### 2.1 数据集构建

![](MedQA_table1_examples.png)

> **Table 1**: MedQA 题目示例。展示了两道典型的 USMLE 风格问题：需要从长篇临床描述中提取关键信息，结合医学知识推理出正确诊断。

#### 数据来源

| 数据集 | 来源 | 题目数 | 语言 |
|--------|------|--------|------|
| USMLE | 美国医师执照考试 | 12,723 | 英文 |
| MCMLE | 中国国家医师资格考试 | 34,251 | 简体中文 |
| TWMLE | 台湾医师资格考试 | 14,123 | 繁体中文 |

#### 数据统计

![439](Pasted%20image%2020260610115059.png)

> **Table 3**: MedQA 三语数据集整体统计。USMLE 问题平均长度 116.6 tokens，远超中文数据集，体现其临床案例描述的复杂性。

| 指标        | USMLE       | MCMLE      | TWMLE       |
| --------- | ----------- | ---------- | ----------- |
| 每题选项数     | 4           | 4          | 4           |
| 平均/最大选项长度 | 3.5 / 45    | 7.3 / 100  | 20.6 / 210  |
| 平均/最大问题长度 | 116.6 / 530 | 45.7 / 333 | 61.0 / 1950 |
| 词汇/字符量    | 63,317      | 3,263      | 3,588       |
| 训练集       | 10,178      | 27,400     | 11,298      |
| 开发集       | 1,272       | 3,425      | 1,412       |
| 测试集       | 1,273       | 3,426      | 1,413       |

#### 配套语料库

论文同时收集了医学教科书作为知识库：
- **英文**：18 本广泛使用的医学教科书（如 Harrison's Principles of Internal Medicine 等）
- **中文**：对应中国和台湾医学教材
![478](Pasted%20image%2020260610113857.png)
> **Table 4**: 文档集合统计。英文语料含 18 本教科书，共 12.7M tokens；中文语料含 33 本教科书，共 14.7M 字符。

语料库按段落或句子切分，存入 Elasticsearch 作为检索后端。

### 2.2 基线系统设计：两阶段流水线

```
问题 + 选项 → [Document Retriever] → 候选段落 → [Document Reader] → 答案选择
```

整体架构遵循经典 OpenQA 范式：**检索器 + 阅读器**。

#### 阶段一：Document Retriever（文档检索）

| 方法 | 描述 |
|------|------|
| **IR-ES** | 基于 Elasticsearch 的 BM25 检索，直接用问题文本查询 |
| **IR-CUSTOM** | 定制化检索：对问题进行实体提取（MetaMap），构建更精准的查询 |
| **PMI** | 点互信息（Pointwise Mutual Information），计算选项与检索文档的共现统计 |

**IR-CUSTOM 的关键设计**：
1. 使用 MetaMap 从问题中提取医学实体
2. 将实体作为查询关键词检索
3. 对检索结果进行重排序
4. 返回 top-N 段落作为阅读器输入

#### 阶段二：Document Reader（阅读理解）

将问题转化为多选阅读理解任务：

$$P(a_i | q, D) = \text{softmax}(\text{MLP}(\text{Encoder}([q; d; a_i])))$$

其中：
- $q$：问题文本
- $D$：检索得到的文档集合
- $a_i$：第 $i$ 个候选答案
- Encoder：预训练语言模型

| 模型 | 说明 |
|------|------|
| **BERT-base** | 通用 BERT 基础模型 |
| **BioBERT** | 在生物医学文献上预训练的 BERT |
| **BERT-base-zh** | 中文 BERT |
| **BERT-base-WWM-EXT** | 全词遮蔽中文 BERT |
| **RoBERTa-Large-WWM-EXT** | 中文 RoBERTa Large |
| **mBERT** | 多语言 BERT |

**MAX-OUT 策略**：训练时使用最大化正确答案得分与最高错误答案得分之间 margin 的损失函数，而非标准交叉熵。

### 2.3 关键设计选择

1. **为什么用两阶段而非端到端？** 医学知识库庞大，直接端到端不可行，需要先缩小搜索范围
2. **为什么用 MetaMap？** 医学文本含大量专业术语，通用分词/关键词提取效果差
3. **为什么提供教科书语料而非让模型直接回答？** 验证模型是否真的"理解"了医学知识，而非依赖参数记忆

---

## 3. 实验结果

### 3.1 实验设置

- **检索**：Elasticsearch 2.4.1，top-N 段落（N=1~10）
- **阅读器训练**：在各自训练集上微调预训练模型
- **评估指标**：准确率（Accuracy）
- **对照**：CHANCE baseline（随机选择，25%）

### 3.2 主实验结果

#### MCMLE 数据集（简体中文）

| 方法 | Dev | Test |
|------|-----|------|
| CHANCE | 25.0 | 25.0 |
| PMI | 36.6 | 36.9 |
| IR-ES | 38.3 | 37.2 |
| IR-CUSTOM | 39.1 | 37.8 |
| MAX-OUT | 51.8 | 50.9 |
| BERT-BASE-ZH | 66.5 | 65.8 |
| BERT-BASE-WWM-EXT | 64.4 | 64.0 |
| MBERT-BASE | 62.1 | 62.3 |
| **RoBERTa-LARGE-WWM-EXT** | **69.3** | **70.1** |

![](MedQA_table7_MCMLE_results.png)

> **Table 7**: MCMLE 基线实验结果。RoBERTa-Large 达 70.1% 测试准确率，远超非预训练模型，在中文数据上预训练模型优势显著。

#### USMLE 数据集（英文）

| 方法 | Dev | Test |
|------|-----|------|
| CHANCE | 25.0 | 25.0 |
| IR-ES | 33.1 | 32.6 |
| IR-CUSTOM | 34.8 | 33.7 |
| BERT-BASE | 34.3 | 34.1 |
| BioBERT-BASE | 34.1 | 36.7 |
| BioBERT-LARGE | **36.6** | **36.7** |

#### TWMLE 数据集（繁体中文）

| 方法 | Dev | Test |
|------|-----|------|
| CHANCE | 25.0 | 25.0 |
| IR-ES | 28.7 | 28.9 |
| IR-CUSTOM | 28.3 | 29.5 |
| BERT-BASE-ZH | 40.3 | 40.0 |
| MBERT-BASE | 41.2 | 42.0 |

![](MedQA_table8_USMLE_TWMLE_results.png)

> **Table 8**: USMLE 和 TWMLE 基线实验结果。USMLE 最佳仅 36.7%（BioBERT-Large），远逊于 MCMLE 的 70.1%，说明英文临床推理型问题难度远高于中文知识记忆型问题。

### 3.3 关键发现

1. **英文最难**：USMLE 最佳仅 36.7%，因为问题更长更复杂（平均 116.6 tokens），且多为临床推理型
2. **简体中文相对"容易"**：RoBERTa-Large 达 70.1%，可能因为中国考试题型更偏知识记忆
3. **检索器是瓶颈**：定制检索 (IR-CUSTOM) 相比通用检索 (IR-ES) 仅提升 1-2 个百分点
4. **BioBERT 优于通用 BERT**：在英文数据上，领域预训练带来 ~2.5% 提升
5. **大模型有帮助但有限**：从 base 到 large 的提升不显著

### 3.4 错误分析

论文通过分析指出性能瓶颈在于：

![](MedQA_table9_retrieval_analysis.png)

> **Table 9**: 检索质量分析（top-25 段落中证据覆盖情况）。USMLE 高达 68% 的问题检索不到任何相关证据，而 MCMLE 仅 4%——这直接解释了为什么 USMLE 准确率远低于 MCMLE。

- **检索器无法完成多跳推理**：很多问题需要综合多个知识片段，当前检索只能找到部分相关段落
- **选项干扰性强**：医学选项之间语义高度相似，需要精细辨别
- **问题需要高阶推理**：不仅是找到答案所在段落，还需要逻辑推理和临床判断

![](MedQA_table10_topN.png)

> **Table 10**: Top-N 检索召回率分析。USMLE 在 top-15 才有 81.2% 的证据覆盖率，而 MCMLE 仅需 top-1 就达 66.7%，再次验证英文题目的检索难度大得多。

---

## 4. 局限性与未来方向

### 局限性
- 基线方法较简单，未探索更复杂的知识图谱方法
- 英文语料仅 18 本教科书，覆盖面有限
- 未评估人类医生在此数据集上的表现作为 upper bound
- 检索与阅读分离，无法端到端优化

### 未来方向（论文发表后的发展）
- GPT-4 在 USMLE 上已达 ~86%，远超原论文基线
- MedQA 已成为医学 LLM 评估的标准 benchmark
- 后续工作探索了 RAG、知识图谱增强、Chain-of-Thought 等方法

---

## 5. 个人思考

1. **数据集的持久价值**：尽管基线性能已被大幅超越，MedQA 作为 benchmark 的价值依然巨大——它定义了一个清晰的问题格式和评估标准，使得不同方法可以公平比较。这是好 benchmark 的核心品质。

2. **开放域设定的前瞻性**：论文坚持"开放域"设定（不直接给段落）是明智的，因为这更接近真实临床场景——医生需要从庞大的知识中调取相关信息。

3. **多语言设计的启示**：三语数据集揭示了有趣的跨语言差异：中文考试偏记忆、英文偏推理。这对评估模型的"真实能力"很有参考价值。

4. **作为 LLM 评估基准的意义**：MedQA-USMLE 现在几乎是每篇医学 LLM 论文的标配评测集，说明了选择真实执业考试作为题源的战略眼光。

5. **瓶颈分析的洞察力**：论文指出"检索器是瓶颈"的结论在 RAG 时代尤为relevant——检索质量直接决定最终性能上限。

---

## 6. 关键引用

```bibtex
@article{jin2021disease,
  title={What disease does this patient have? a large-scale open domain question answering dataset from medical exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={Applied Sciences},
  volume={11},
  number={14},
  pages={6421},
  year={2021},
  publisher={MDPI}
}
```
