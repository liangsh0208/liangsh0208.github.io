---
created: 2026-06-10
published: 2024-08-15
paper: https://arxiv.org/abs/2408.08422
authors: Guanchu Wang, Junhao Ran, Ruixiang Tang, Chia-Yuan Chang, Yu-Neng Chuang, Zirui Liu, Vladimir Braverman, Zhandong Liu, Xia Hu (Rice University; Baylor College of Medicine; Texas A&M University)
tags:
  - LLM-Benchmark
  - Medical-AI
  - Rare-Disease
  - RAG
  - Question-Answering
  - Dataset
---

# ReDis-QA: Assessing and Enhancing Large Language Models in Rare Disease Question-answering

## 一句话总结
ReDis-QA 是首个面向开源 LLM 的罕见病问答基准数据集（1,360 题、205 种罕见病），并同步开源了首个罕见病专用检索语料库 ReCOP（1,324 种疾病、9,268 知识块）；实验证明，基于 ReCOP 的检索增强生成（RAG）可平均提升 LLM 罕见病诊断准确率 **8%**，并引导模型生成可追溯文献的可信解释。

![](ReDisQA_fig1_pipeline.png)

> **Figure 1**: ReDis-QA 数据集构建流程：从 MedMCQA、MedQA、MMLU 三个医学基准中筛选罕见病相关问题，经人工清洗和属性标注，最终形成 1,360 题、覆盖 205 种罕见病的 QA 数据集。

---

## 1. 研究背景与动机

### 1.1 问题定义

罕见病（Rare Diseases）全球已知超过 7,000 种，影响约 3 亿人。尽管 LLM 在通用医学领域表现出色（如 MedQA 上超过人类），但**罕见病诊断仍是 LLM 的重大挑战**：

1. **泛化困难**：罕见病文献稀少，LLM 在预训练语料中接触有限，容易将罕见病与常见病混淆，产生幻觉
2. **遗传复杂性**：多数罕见病由基因突变引起，需要精细的遗传知识，而这难以通过现有提示策略弥补
3. **评估缺失**：此前没有专门面向开源 LLM 的罕见病问答基准

### 1.2 现有方法的不足

| 不足维度 | 具体表现 |
|---------|---------|
| **数据集封闭** | 多数罕见病研究使用闭源模型（如 GPT-4）和封闭数据，不可复现 |
| **缺乏专用语料** | 虽有 PubMed、StatPearls 等医学语料，但缺乏系统化、结构化的罕见病专用知识库 |
| **QA 形式单一** | 现有医学 QA 基准（MedQA、MedMCQA）包含大量常见病问题，罕见病占比极低 |
| **RAG 效果未知** | 检索增强生成（RAG）在罕见病领域的有效性缺乏系统评估 |

---

## 2. 核心贡献

本文的两大核心贡献：

### 2.1 ReDis-QA：罕见病问答基准数据集

- **规模**：1,360 个高质量选择题，覆盖 **205 种罕见病**
- **来源**：从 MedMCQA、MedQA、MMLU 三大医学基准中筛选并清洗
- **标注**：每题标注疾病名称和**知识属性**（symptoms, causes, affects, related disorders, diagnosis, others）
- **用途**：评估开源 LLM 在罕见病诊断中的能力

![](ReDisQA_fig2_stats.png)

> **Figure 2**: (a) ReDis-QA 中 Top-50 罕见病的题目数量分布；(b) 各知识属性占比：Causes 34%、Diagnosis 18%、Related-disorders 15%、Affects 13%、Symptoms 11%、Others 9%；(c) 五种开源 LLM 在不同属性子集上的准确率。

### 2.2 ReCOP：首个罕见病专用检索语料库

- **来源**：NORD（National Organization for Rare Disorders）数据库
- **覆盖**：1,324 种罕见病的专业报告
- **结构**：每份报告切分为 **7 个属性块**（chunk），共 9,268 个知识块
  - Overview, Symptoms, Causes, Affects, Related disorders, Diagnosis, Standard therapies
- **特点**：非技术语言撰写，兼顾专业性与可读性；包含遗传突变、疾病同义词、参考文献等元信息

![](ReDisQA_fig3_recop.png)

> **Figure 3**: ReCOP（Rare Disease Corpus）构建流程：从 NORD 数据库收集原始报告，按 7 个知识属性切分为结构化知识块，每个属性块与 ReDis-QA 中的问题属性一一对应，确保检索内容高度相关。

---

## 3. 实验设置

### 3.1 评估模型

| 模型 | 参数规模 | 类型 |
|------|---------|------|
| Llama-2-7B | 7B | 通用开源 |
| Mistral-7B-v0.2 | 7B | 通用开源 |
| Phi-3-7B | 7B | 通用开源 |
| Gemma-1.1-7B | 7B | 通用开源 |
| Qwen-2-7B | 7B | 通用开源 |

> 全部使用 HuggingFace 上的 instruction-tuned 版本

### 3.2 评估指标

**准确率（Accuracy）**：四选一多项选择题的直接判断正确率

### 3.3 RAG 配置

| 组件 | 选项 |
|------|------|
| **检索器** | Meta-data（疾病名关键词匹配）、MedCPT（稠密向量检索）、BM25（稀疏词匹配） |
| **检索数量** | k = 5 或 7 |
| **语料库** | ReCOP（本文）、Textbooks、StatPearls、PubMed、Wikipedia |

---

## 4. 实验结果

### 4.1 无 RAG 基线（Benchmark on ReDis-QA）

| 模型 | 准确率 | 表现定位 |
|------|--------|---------|
| **Phi-3-7B** | **71.6%** | 最优 |
| Qwen-2-7B | 61.9% | 次优 |
| Mistral-7B-v0.2 | 47.3% | 中等 |
| Gemma-1.1-7B | 46.6% | 中等 |
| Llama-2-7B | 36.4% | 最差 |
| **平均** | **52.8%** | — |

**核心发现**：
- **Phi-3-7B 一枝独秀**，准确率超过 70%，显著领先其他模型
- Llama-2-7B、Mistral、Gemma 在罕见病诊断上均不足 50%
- **症状（Symptoms）、病因（Causes）、影响（Affects）类问题相对容易**，模型准确率较高
- **相关疾病（Related disorders）、诊断（Diagnosis）、其他（Others）类问题更具挑战性**

### 4.2 ReCOP 检索增强效果

**Table 2: 有无 ReCOP 的准确率对比（%）**

| Retriever | k | Llama-2 | Mistral | Phi-3 | Gemma | Qwen | 平均 | 提升 |
|-----------|---|---------|---------|-------|-------|------|------|------|
| N/A (无 RAG) | 0 | 36.4 | 47.3 | 71.6 | 46.6 | 61.9 | 52.8 | — |
| Meta-data | 5 | 40.8 | 57.1 | 74.6 | 58.2 | 66.2 | 59.4 | +6.6 |
| **Meta-data** | **7** | **39.8** | **60.9** | **75.1** | **61.2** | **67.6** | **60.9** | **+8.1** |
| MedCPT | 7 | 43.1 | 54.4 | 71.8 | 55.8 | 65.9 | 58.2 | +5.4 |
| BM25 | 7 | 41.5 | 55.5 | 72.5 | 53.5 | 63.8 | 57.4 | +4.6 |

**核心发现**：

1. **Meta-data retriever 效果最佳**：使用疾病名作为关键词直接匹配，效果优于稠密向量检索（MedCPT）和稀疏词匹配（BM25）
   - 说明 ReDis-QA 的元数据标注与 ReCOP 的 chunk 结构对齐是关键优势
2. **k=7 优于 k=5**：更多检索块带来更丰富的上下文，提升模型判断
3. **ReCOP 平均提升 8%**，对 Llama-2-7B 提升最大（+8.1%），对 Phi-3-7B 也有 3.5% 提升

### 4.3 ReCOP 相比其他语料库

**Table 4: ReCOP 与现有语料库的对比（k=7, MedCPT / BM25 检索器）**

| 语料库 | 平均准确率（MedCPT） | 平均准确率（BM25） |
|--------|---------------------|-------------------|
| Textbooks | 58.5 | 58.8 |
| Textbooks + ReCOP | **63.1** | **62.5** |
| StatPearls | 58.4 | 58.4 |
| StatPearls + ReCOP | **61.9** | **62.3** |
| PubMed | 60.4 | 60.4 |
| PubMed + ReCOP | **63.4** | **63.6** |
| Wikipedia | 58.0 | 58.2 |
| Wikipedia + ReCOP | **61.9** | **62.8** |

**核心发现**：
- **ReCOP 单独使用优于所有 baseline 语料库**（除了 PubMed）
- **ReCOP + 任何 baseline 语料库都能进一步提升效果**，说明 ReCOP 提供了独特的罕见病知识
- 最佳组合为 **PubMed + ReCOP**（63.4%~63.6%），优于单独 ReCOP

### 4.4 RAG 效果可视化

![](ReDisQA_fig4_rag_radar.png)

> **Figure 4**: (a)-(d) 各模型在有无 ReCOP 下，不同知识属性的准确率雷达图（红线=无 RAG，蓝线=有 RAG）。(e)-(h) ReCOP 分别与 Textbooks、StatPearls、PubMed、Wikipedia 组合的效果对比。可见 ReCOP 全面提升了各属性上的表现，且与现有语料库互补。

---

## 5. 自然语言解释的可信度分析

### 5.1 核心发现：RAG 不止提升准确率，还提升**可信度**

**无 RAG 时**：LLM 会产生看似合理但实际错误的解释，即**"自信的幻觉"**

**ReCOP-RAG 时**：模型能引用 ReCOP 中的知识，产生**可追溯文献**的正确解释

![](ReDisQA_fig5_explanation.png)

> **Figure 5**: Abetalipoproteinemia 诊断案例对比。无 RAG 时模型误选 A（LCAT），并给出了看似合理的错误解释；ReCOP-RAG 正确选择了 C（MTP），且解释与 NORD 文献完全一致。

### 5.2 更多案例分析

| 案例 | 疾病 | 无 RAG 答案 | ReCOP-RAG 答案 |  Golden |
|------|------|------------|----------------|---------|
| 感染部位 | Cryptococcosis | CNS (错误) | **Lungs (正确)** | D |
| 遗传方式 | Achondroplasia | Autosomal Recessive (错误) | **Autosomal Dominant (正确)** | A |
| 关联疾病 | Aniridia + Hemihyperplasia | None (错误) | **Wilms' tumour (正确)** | B |
| 治疗药物 | Achalasia cardia | Bethanechol (错误) | **Nifedipine (正确)** | A |

> 这些案例充分说明，罕见病领域 LLM 的**先验知识严重不足**，必须依赖外部检索才能给出可靠答案。

---

## 6. Entropy-Aware 多语料库 RAG

### 6.1 算法设计

作者提出一种**熵感知的多语料库融合策略**（Algorithm 1）：

1. **分别检索**：从语料库 X 和语料库 Y 各检索 top-k 文档
2. **分别生成**：LLM 基于两组文档分别生成回答和选项概率分布
3. **计算熵**：$H_X = \text{CalculateEntropy}(\text{Probs}_X)$，$H_Y = \text{CalculateEntropy}(\text{Probs}_Y)$
4. **选择答案**：选择熵较低（概率分布更集中、模型更确定）的那一组的答案

> **直觉**：当 LLM 对答案非常确定时（低熵），直接采纳；当不确定时（高熵），换另一语料库的答案。这就像让模型"自我评估信心度"。

---

## 7. 局限性与未来方向

| 局限 | 说明 |
|------|------|
| **仅 MCQ 格式** | ReDis-QA 采用四选一形式，未覆盖开放式诊断和鉴别诊断 |
| **开源模型局限** | 仅评估了 <10B 参数的开源模型，未与 GPT-4 等闭源模型对比 |
| **数据规模** | 1,360 题、205 种疾病，规模相对有限 |
| **英语单语** | 仅覆盖英语内容，未涉及多语言罕见病诊断 |
| **静态知识** | ReCOP 基于 NORD 数据库，知识更新频率有限 |

**未来方向**：
- 扩展至开放式问答和鉴别诊断场景
- 与闭源模型（GPT-4、Claude）进行对比
- 构建多语言罕见病语料库
- 结合多模态数据（如基因测序结果、医学影像）
- 探索更复杂的检索策略（如多跳推理、知识图谱增强 RAG）

---

## 8. 个人思考

### 方法的亮点
1. **双轮驱动**：ReDis-QA（评估）+ ReCOP（增强）相辅相成，兼具诊断和解释双重价值
2. **结构化切分策略**：将 NORD 报告按 7 个属性切分，确保检索块与问题属性精确对齐
3. **Meta-data retriever 的简单有效**：直接用疾病名匹配，效果竟优于向量检索——在罕见病领域，精确匹配语义的稀缺性使得"关键词优先"成为最优策略
4. **熵感知融合**：多语料库场景下的简单熵机制，为小模型提供了"自我校准"能力

### 与 RareBench 的关联和差异
- **RareBench**（KDD 2024）评估的是**多任务诊断能力**（表型提取、疾病筛查、鉴别诊断），基于 EHR 真实病例，侧重**临床复杂度**
- **ReDis-QA** 评估的是**知识问答能力**（MCQ 形式），基于医学考试题和 NORD 知识库，侧重**知识覆盖和检索增强**
- **互补性**：RareBench 更像"临床实战考试"，ReDis-QA 更像"医学知识测验"

### 对 RAG 在医学领域应用的启示
- **语料库质量 > 检索算法**：在罕见病这种知识极度稀缺的领域，拥有一个高质量、结构化的专用语料库比优化检索算法更重要
- **可解释性至关重要**：医学场景中用户不仅需要正确答案，还需要可追溯的解释，RAG 恰好满足这一需求

---

## 9. 关键引用

```bibtex
@article{wang2024redisqa,
  title={Assessing and Enhancing Large Language Models in Rare Disease Question-answering},
  author={Wang, Guanchu and Ran, Junhao and Tang, Ruixiang and Chang, Chia-Yuan and Chuang, Yu-Neng and Liu, Zirui and Braverman, Vladimir and Liu, Zhandong and Hu, Xia},
  journal={arXiv preprint arXiv:2408.08422},
  year={2024}
}
```
