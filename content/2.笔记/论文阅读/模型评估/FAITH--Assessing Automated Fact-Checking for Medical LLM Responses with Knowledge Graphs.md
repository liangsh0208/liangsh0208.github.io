---
created: 2026-06-09
paper: https://arxiv.org/abs/2511.12817
code: https://github.com/COLA-Laboratory/FAITH
authors: Shasha Zhou, Mingyu Huang, Jack Cole, Charles Britton, Ming Yin, Jan Wolber, Ke Li (University of Exeter, Purdue University, GE HealthCare)
published: 2025-11-16
tags:
  - 医学LLM
  - 事实核查
  - 知识图谱
  - 幻觉检测
  - AAAI
---
<u>- 基于医疗知识图谱，对模型的医疗知识回复进行事实性评估。</u>


# FAITH: Assessing Automated Fact-Checking for Medical LLM Responses with Knowledge Graphs

## 一句话总结

<font color="#ff0000">本文提出 FAITH 框架，利用医学知识图谱（KG）对 LLM 生成的医学回复进行无参考、可解释的事实性评估</font>；实验表明 FAITH 与临床医生判断的相关性（Pearson ρ=0.696）远超传统 NLP 指标（BLEU-4 仅 0.081），且能有效区分不同 LLM 的能力差异，鲁棒性高（CDV 仅 0.014±0.005）。

![](faith_fig1_overview.png)

> **Figure 1**: FAITH 框架总览。(a) LLM 生成的医学回复；(b) 将回复分解为原子级医学声明（claim）；(c) 通过实体匹配将声明链接到医学 KG 中的节点和中间节点；(d) 基于路径特征和边语义计算每条声明的事实性分数并聚合为整体回复分数。

---

## 1. 研究背景与动机

### 1.1 问题定义

大语言模型在医学领域展现出强大能力，但在高风险的医疗场景中，模型可能产生看似合理实则危险的错误信息（幻觉）。例如错误的因果关联"dry cough is a symptom of bronchiectasis"。

**核心问题**：如何在不依赖人工参考答案的情况下，自动、可解释地评估 LLM 医学回复的事实性（factual correctness）？

### 1.2 现有方法的不足

| 方法类型 | 代表工作 | 主要局限 |
|---------|---------|---------|
| NLP 指标 | BLEU, BERTScore, ROUGE | 聚焦词汇/语义相似，与事实性关联弱（如 ρ=0.081） |
| 有监督指标 | MEDCON, GPT-based evaluators | 需要参考回复或标注数据，在医学场景难以获取 |
| LLM as Judge | GPTScore, LLM-Eval | 评判模型自身易幻觉，缺乏可解释性 |
| 传统 KG fact-checking | KL, KL-REL, TransE | 针对单条 triple 设计，不适用于自由文本输出 |

本文**首次提出了面向医学 LLM 回复的无参考 KG-based fact-checking 框架**，并通过系统性评估验证其可靠性与实用性。

---

## 2. 方法

### 2.1 核心思想

FAITH 的核心洞察：**医学回复的事实性可以通过医学知识图谱中的结构化证据路径来量化**。回复中涉及的医学实体、关系和已知医学知识之间的路径契合度可作为事实性的代理信号（proxy）。

方法流程包含 4 个关键模块：声明提取 → 实体匹配 → KG 遍历与事实评估 → 打分与解释。

### 2.2 规范定义

**声明（Claim）**：三元组 $t = (s, r, o)$，其中 $s$ 为主体（subject），$o$ 为客体（object），$r$ 为谓词（predicate）。

**知识图谱（KG）**：图 $\mathcal{G} = (\mathcal{E}, \mathcal{R})$，节点 $\mathcal{E}$ 为实体，边 $\mathcal{R}$ 为关系。

**知识路径**：连接 $e_1$ 与 $e_k$ 的路径 $p = (e_1, r_1, e_2, r_2, \dots, r_{k-1}, e_k)$，包含 $k$ 个实体和 $k-1$ 条关系。

### 2.3 模块详解

#### 模块 1：医学声明提取（Medical Claim Extraction）

使用 GPT-4o（temperature=0, max tokens=1024）将 LLM 回复 $\mathcal{D}$ 分解为原子级医学声明 $\{t_i\}_{i=1}^{n}$。

Prompt 策略：
- **两阶段 prompt**：先识别所有医学实体，再确定实体间的关系
- **多轮对话（multi-round）**：增加 claim 提取的召回率
- **批判性分析 prompt（critical analysis）**：加入诱导幻觉的提示，模型需重新以怀疑态度审视提取出的声明

> 实验发现：多轮对话主要提升召回率，批判性分析主要提升精度，两者组合最有效。

#### 模块 2：医学实体匹配（Medical Entity Matching）

将声明中的实体通过 UMLS（Unified Medical Language System）映射到 KG 节点：

1. 使用 UMLS API 将医生和 KG 实体统一标准化为 Concept Unique Identifiers (CUI)
2. 无法匹配的实体标记为 "unverifiable"（保守设计：只评估高置信度实体）
3. UMLS 集成 200+ 生物医学词表（MeSH, SNOMED CT, Human Phenotype Ontology），涵盖 340 万+医学概念

#### 模块 3：KG 遍历与事实评估（KG Traversal & Factual Evaluation）

**路径搜索**：在 $e_1$ 与 $e_k$ 之间搜索 KG 中的最短路径（shortest path），假设最短路径最能代表主要因果/语义连接。

**关系语义相似性（Step 1）**：评估声明中的谓词与 KG 路径中关系的对齐程度

$$S(p, t) = \frac{1}{k-1} \times \sum_{i=1}^{k-1} \cos\_sim(r_i, \hat{r})$$

其中 $\hat{r}$ 是声明中的谓词，$r_i$ 是路径 $p$ 中的第 $i$ 条关系，$k-1$ 为路径长度。

局限性：仅语义相似性不足以评估事实性（如 "indication" 与 "contraindication" 的关系相似但语义完全相反）。

**事实性评分（Step 2）**：综合语义相似性与 KG 拓扑特征

$$W(p, t) = S(p, t) \times \left[\sum_{i=2}^{k-1} \frac{e^{\alpha PR(e_i)}}{u(r_i, \hat{r})} + \frac{1}{u(r_{k-1}, \hat{r})}\right]^{-1}$$

| 符号 | 含义 |
|------|------|
| $S(p,t)$ | 关系语义相似性 |
| $PR(e_i)$ | 实体 $e_i$ 的 PageRank 值（实体中心性） |
| $\alpha$ | 缩放常数（本研究中为 100） |
| $u(r_i, \hat{r})$ | 基于共现图中的关系共现权重 |

**实体中心性（Entity Centrality）**：高 PageRank 值表示该实体出现在更多医学声明中，提供更可靠的支持。

**关系共现（Relation Co-occurrence）**：高频共现关系 $u(r_i, \hat{r}) \approx 1$ 表示关系密切，
low co-occurrence 则提供微弱支持，应在评分中放大距离。

#### 模块 4：打分与解释（Scoring & Interpretation）

聚合所有声明的事实性分数为整体回复评分：

$$\hat{W}(\mathcal{P}, \mathcal{D}) = \frac{1}{n} \times \sum_{\substack{p_i \in \mathcal{P} \\ t_i \in \mathcal{D}}} W(p_i, t_i)$$

- 取值范围：$[-1, 1]$，正值表示与医学知识一致（supported），负值表示矛盾（contradicted）
- 绝对值反映实体联系的强度
- 每个分数与具体的 KG 路径绑定，提供**细粒度的可解释性**

---

## 3. 实验结果

### 3.1 实验设置

| 要素 | 配置 |
|------|------|
| 医学 QA 数据集 | MedQA, MMLU（医学子集）, MS-AKT, LiveQA |
| 医疗摘要 | FactPICO, HealthFC, BEAR-FACT |
| 事实验证 | MFV (Medical Fact Verification) |
| 评估模型 | GPT-4o, GPT-4o-mini, Llama 3 8B, Llama 3.1 8B, OpenBioLLM |
| KG | UMLS 2025AA 版（340万+概念，2300万+关系） |
| NLP 基线 | BLEU-4, ROUGE-L, BERTScore |
| 计算 fact-checking 基线 | KL, KL-REL, TransE |
| LLM 评判基线 | MEDCON, FActScore, ImapScore |
| 临床医生评估 | 20 名 UK 临床医生，每人至少 5 年临床经验 |

### 3.2 区分 LLM 能力

![](faith_fig2_distinguish_llms.png)

> **Figure 2**: FAITH 区分不同 LLM 的能力及其对噪声的鲁棒性。(a-d) MedQA, MMLU, MS-AKT, LiveQA 上的平均事实性分数；(e-h) 对应的数据集上各指标在 10 种改写（paraphrasing）版本下的变异系数（CV）。FAITH 的 CV 最低（0.014±0.005），表明对文本变化高度鲁棒。

核心发现：
- FAITH 在 4 个数据集上能有效区分 5 个 LLM 的能力差异
- 对 ChatGPT-4o 与其他模型的区分度尤为明显（Fig. 2a）
- BERTScore、ROUGE-L 等 NLP 指标在区分不同 LLM 上的表现混乱
- 其他 fact-checking 基线（KL, KL-REL, TransE）虽有区分度但不如 FAITH 稳定

### 3.3 变异鲁棒性（Robustness to Variance）

通过 GPT-4o 对每次回复生成 10 种不同改写版本，计算变异系数（Coefficient of Variation, CV）。

| 指标 | 平均 CV (mean ± std) |
|------|---------------------|
| **FAITH** | **0.014 ± 0.005** |
| BLEU-4 | 0.910 ± 0.862 |
| ROUGE-L | 0.42 ± 0.31 |
| BERTScore | 0.15 ± 0.12 |
| FActScore | 0.25 ± 0.20 |
| KL | 0.18 ± 0.15 |

FAITH 的 CV 比 BLEU-4 低 **65 倍**，比第二好的指标（BERTScore）低 **10 倍**。

### 3.4 与临床医生的相关性

![](faith_fig3_clinician_corr.png)

> **Figure 3**: FAITH 与临床医生的判断展现出最高的相关性。(a) 官方答案、Llama 3.1、GPT-4o 的临床评分对比；(b) FAITH 与临床评估的散点图（Pearson ρ=0.696）；(c) FAITH 与各基线方法的相关系数对比。

临床医生按照 3 个维度评估：
- **factuality**（事实性）
- **relevance**（相关性）
- **potential harm**（潜在危害），均使用 5 分制 Likert 量表

评估了 16 个 MS-AKT 数据集的问答对，每个回答至少由 2 名医生评分，Cohen's κ = 0.64（显著一致性）。

| 方法 vs 临床医生 | Pearson ρ |
|----------------|-----------|
| BLEU-4 | 0.081 |
| ROUGE-L | -0.05 |
| BERTScore | 0.25 |
| FActScore | 0.06 |
| MEDCON | 0.05 |
| ImapScore | -0.04 |
| **FAITH** | **0.696** |

FAITH 与临床判断的 ρ=0.696 显著高于所有基线，且具备统计显著性（95% CI）。

### 3.5 可解释性分析

![](faith_fig4_explainability.png)

> **Figure 4**: FAITH 的可解释性分析。(a) 混淆矩阵展示临床医生标注的错误声明与 FAITH 最低分声明的对齐情况；(b) GPT-4o 错误回复中前 5 个最常见 KG 关系类型的分布。

**两方面可解释性评估**：

1. **解释忠实性（Faithfulness to medical consensus）**：判断 FAITH 发现的具体错误声明是否被临床医生认可。结果表明 FAITH 的 Precision=0.65, Recall=0.59, F1=0.62。在 83.6% 的病例中，临床医生发现的错误声明被 FAITH 排在前 5 最低分中。

2. **LLM 限制分析效用**：分析 GPT-4o 回复中的错误模式，发现近半数错误涉及疾病的表型特征（phenotipical features），尤其体现在病情鉴定和症状匹配上。

### 3.6 干预实验（Practical Utility）

![](faith_fig5_intervention.png)

> **Figure 5**: FAITH 通过选择性干预增强 LLM 事实性。(a) Reject-to-Answer (RTA) 策略下的准确率；(b) RAG 策略下的事实性分数。

**干预场景 1：Reject-to-Answer (RTA)**
- 设置 FAITH 分数阈值，拒绝低事实性的回复
- 在 MedQA 上，5%-50% 数据集的 RTA 显著提升了 GPT-4o 的准确率和事实性分数

**干预场景 2：与 RAG 结合**
- 对低事实性分数的回复触发 RAG 检索
- 模型不确定性作为触发机制（基于百分位数，5%-50%）
- 在 RAG 场景中，FAITH 作为 thresholding 机制比将所有问题都用 RAG 处理更 cost-effective

### 3.7 更广适用性

![](faith_fig6_broader_applicability.png)

> **Figure 6**: FAITH 在医学摘要和事实验证上的广泛适用性。(a) 基于 FactPICO benchmark 的医学摘要事实性分数；(b) FactPICO 上专家评分与 FAITH 的相关性 (Pearson ρ=0.61)；(c) HealthFC 上真假声明的区分度；(d) MFV (Medical Fact Verification) 上的真假声明区分度。

- **医学摘要**（FactPICO）：对 115 个从 Alpaca, Llama-2, GPT-4 生成的临床试验摘要进行排名，FAITH 排名与专家排名高度一致（Pearson ρ=0.61）
- **事实验证**（HealthFC & BEAR-FACT）：FAITH 能有效区分真假医学声明（Fig. 6c, d）

---

## 4. 消融实验

### 4.1 对 KG 选择和完整性的敏感性

**不同 KG**：集成 PrimeKG、OGBL-biokg 等不同规模和来源的 KG。结果显示 LLM 相对排名保持一致，但绝对分数受 KG 质量影响。

**KG 噪声鲁棒性**：施加 3 种噪声：
1. 20% 随机边删除
2. 20% 随机节点删除（含关联边）
3. 20% 随机噪声边插入

结果显示删除噪声节点/边造成的性能下降远大于插入噪声边，证明 KG 质量（尤其是精确度）至关重要。

### 4.2 声明提取模块的敏感性

| 提取策略 | 效果 |
|---------|------|
| Base prompt | 基准 |
| Base + critical analysis | 精度提升 |
| Base + multi-round | 召回率提升 |
| **FAITH** (full prompting) | **两者兼顾，最佳 F1** |

**LLM 选择敏感性**：
- 将 GPT-4o 替换为 Llama 3 / Llama 3.1 等开源模型：轻微性能下降但仍显著优于 QuickUMLS
- GPT-4o vs GPT-4o-mini：差异很小，GPT-4o-mini 已足够胜任

---

## 5. 局限性与未来方向

1. **KG 质量依赖**：对 KG 的覆盖度和精确度高度依赖；知识盲区的声明会被标记为 "unverifiable"，可能导致假阴性
2. **上游模块依赖**：pipeline 依赖声明提取模块的准确性，该阶段的错误会在系统中传播
3. **单条最短路径假设**：当前仅使用实体间最短路径，可能遗漏其他有效证据路径
4. **多跳推理挑战**：对于涉及复杂多跳推理的声明（如 3-hop+），路径搜索和评分仍具挑战

**未来方向**：
- 整合多个 KG 提高覆盖率
- 改进 claim extraction 的 precision/recall
- 探索最优路径选择策略（不止最短路径）
- 扩展到法律和金融等其他敏感领域

---

## 6. 个人思考

1. **方法定位的精准性**：FAITH 并非要完全取代人类临床医生，而是作为"第一道防线"（guardrail），在模型部署前自动筛选高风险回复。文中"阈值机制"（RTA 和 RAG-triggered）的设计体现了从"评估工具"到"安全系统"的工程思路。

2. **KG 的回归**：在 LLM 时代，结构化知识的价值被重新发现。本文证明 KG 不仅可用于 RAG 增强生成，还可作为独立的评估工具——且该工具比 LLM-based judge 更可靠（因为 judge 本身也会幻觉）。这提示我们：**评估 LLM 不一定非要用更强的 LLM**。

3. **可解释性的落地**：医学场景对可解释性有天然刚需（医生需要知道为什么AI认为回复有误）。FAITH 将每个分数与具体的 KG 路径绑定，这种"X 错误因为 KG 中不存在 A→B→C 的证据路径"的解释方式比黑盒 LLM judge 更实用。

4. **与传统 NLP 指标的对比冲击**：BLEU-4 与临床医生判断的 ρ=0.081 是一个值得深思的数据。如果主流的自动评估指标与真实场景的需求如此脱节，那 NLP 社区需要重新审视评价标准。

5. **可扩展性考虑**：当前方法依赖 UMLS（医学专用 KG）。如果迁移到法律或金融等其他垂直领域，核心挑战不是方法本身，而是如何获得领域高质量的结构化知识图谱。

---

## 7. 关键引用

```bibtex
@article{zhou2025faith,
  title={Assessing Automated Fact-Checking for Medical LLM Responses with Knowledge Graphs},
  author={Zhou, Shasha and Huang, Mingyu and Cole, Jack and Britton, Charles and Yin, Ming and Wolber, Jan and Li, Ke},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={1},
  year={2026}
}
```
