---
created: 2026-06-09
paper: https://doi.org/10.1145/3696410.3714535
code: https://github.com/ZihengZZH/MedKGEval
authors: Ziheng Zhang, Zhenxi Lin, Yefeng Zheng, Xian Wu (Tencent)
published: 2025-04-28
tags:
  - Medical-AI
  - LLM-Evaluation
  - Knowledge-Graph
  - WWW-2025
  - Benchmark
---

# How much Medical Knowledge do LLMs have? An Evaluation of Medical Knowledge Coverage for LLMs

## 一句话总结

MedKGEval 提出了首个基于医学知识图谱（CPubMedKG / CMeKG）的 LLM 医学知识覆盖度评估框架，从**任务级**（实体、关系、子图）和**知识级**（覆盖度、拓扑结构）两个维度系统评测了 21 个主流 LLM（含 GPT-4o、DeepSeek-V3、多个医学专用模型），揭示出**通用大模型在医学知识覆盖上显著优于多数医学专用模型**的反直觉现象。

---

## 1. 研究背景与动机

### 1.1 问题定义

LLM 在医学问答、临床决策等任务中表现亮眼，但其**医学知识储备到底有多少**、**覆盖是否系统**、**专用医学模型是否真比通用模型强**，一直缺乏精细的量化评估。

现有评估的局限：
- **医学考试型评估**（如 USMLE、MedMCQA）只测问答能力，不测量知识覆盖广度。
- **知识图谱补全**任务只测局部关系推理，不涉及实体语义、子图结构。
- **医学专用模型**（如 HuatuoGPT、PULSE）声称在医学场景更优，但缺乏基于标准化知识图谱的横向对比。

### 1.2 核心洞察

> "用知识图谱作为评估 LLM 医学知识覆盖的镜子，而非只用考试题作为筛子。"

MedKGEval 的评估不依赖人工标注问答对，而是**让 LLM 直接补全/解释医学知识图谱中的实体、关系和子图结构**，通过 LLM 输出与图谱真实结构的对齐程度来衡量其医学知识储备。

---

## 2. 方法

### 2.1 整体框架

> **Figure 1**: MedKGEval 整体评测框架三层结构——(1) Entity-Level（实体分类/纠错/定义）→ (2) Relation-Level（关系识别/事实核验/知识补全）→ (3) Subgraph-Level（错误识别/多跳推理），两条评估路径——Task-Oriented（答对率）和 Knowledge-Oriented（覆盖度）并行输出。

MedKGEval 的评测基于两个权威中文医学知识图谱：

| KG            | 规模      | 特点                          |
| :------------ | :------ | :-------------------------- |
| **CPubMedKG** | 约百万级三元组 | 从 PubMed 摘要和医学文献抽取，覆盖广但噪声较高 |
| **CMeKG**     | 十万级三元组  | 从中国医学数据库手工整理，结构化程度高，质量更优    |
|               |         |                             |

KG 中的每个三元组表示为 $(h, r, t)$，其中 $h$ 为头实体，$r$ 为关系，$t$ 为尾实体。MedKGEval 围绕 KG 的真实结构，设计了 9 个评测子任务。

### 2.2 任务级评估（Task-Oriented Evaluation）

面向 LLM 在 KG 上的**直接任务执行能力**，分为三个层级：

#### 2.2.1 实体级任务（Entity-Level）

| 子任务 | 全称 | 评估内容 |
|:---|:---|:---|
| **ET** | Entity Type | 给定实体名称，判断其所属医学类别（如疾病、药物、手术） |
| **EC** | Entity Correction | 给定一个包含错误的实体描述，修正为准确表述 |
| **ED** | Entity Description | 给定实体，生成其标准医学定义 |

直觉：这三个任务分别测试模型对医学实体的**分类能力**、**纠错能力**和**定义生成能力**。

#### 2.2.2 关系级任务（Relation-Level）

| 子任务 | 全称 | 评估内容 |
|:---|:---|:---|
| **RT** | Relation Type | 给定 $(h, t)$，判断头尾实体之间的关系类型 |
| **FC** | Factual Consistency | 给定三元组 $(h, r, t)$，判断其医学正确性 |
| **RP** | Relation Prediction | 给定 $h$ 和 $r$，预测尾实体 $t$ |

直觉：这三个任务分别测试模型的**关系识别**、**事实核验**和**知识补全**能力。

#### 2.2.3 子图级任务（Subgraph-Level）

| 子任务 | 全称 | 评估内容 |
|:---|:---|:---|
| **ER** | Entity Retrieval | 给定查询实体，从候选中召回所有与之有 KG 关系的实体 |
| **R1** | One-hop Concept Recall | 给定实体，召回其所有 1-hop 邻居概念 |
| **R2** | Two-hop Relationship Recall | 给定实体，召回其 2-hop 范围内的关系路径 |

直觉：子图级任务测试模型的**多跳推理**和**结构化知识组织**能力，比单点任务更接近真实临床决策。

### 2.3 知识级评估（Knowledge-Oriented Evaluation）

任务级评估只测"能做对多少题"，但无法回答"模型知识覆盖到知识图谱的哪些角落"。知识级评估通过将 LLM 的输出投影回知识图谱，测量其与真实 KG 在**结构**和**拓扑**上的重合程度。

设 KG 为 $G=(\mathcal{E}, \mathcal{R}, \mathcal{T})$，其中 $\mathcal{E}$ 为实体集合，$\mathcal{R}$ 为关系集合，$\mathcal{T}$ 为三元组集合。LLM 在某一任务上的输出构成预测集合 $\hat{G} = (\hat{\mathcal{E}}, \hat{\mathcal{R}}, \hat{\mathcal{T}})$。

定义以下覆盖度指标：

| 指标 | 公式直觉 | 含义 |
|:---|:---|:---|
| **CovAvg($\mathcal{E}$)** | $\frac{1}{\|\mathcal{E}\|} \sum_{e \in \mathcal{E}} \mathbb{I}(e \in \hat{\mathcal{E}})$ | 实体平均覆盖率：KG 中有多少实体被 LLM **至少提到过一次** |
| **CovDeg($\mathcal{E}$)** | 按实体度数加权的覆盖率 | 高度数（更核心）实体的覆盖程度，避免只覆盖边缘实体 |
| **CovAvg($\mathcal{R}$)** | $\frac{1}{\|\mathcal{R}\|} \sum_{r \in \mathcal{R}} \mathbb{I}(r \in \hat{\mathcal{R}})$ | 关系平均覆盖率 |
| **CovDeg($\mathcal{R}$)** | 按关系出现频率加权的覆盖率 | 高频（更常见）关系的覆盖程度 |
| **Cov($\mathcal{T}$)** | 整体拓扑覆盖度 | 综合实体和关系覆盖的三元组层面覆盖度 |

> **关键设计**：CovDeg 对高度数实体/高频关系加权，避免模型只覆盖"冷门边角料"却在平均指标上好看的偏差。

### 2.4 模型评测列表

共评测 **21 个 LLM**，覆盖通用、医学、推理三类：

| 类别 | 模型 |
|:---|:---|
| **通用 tiny** | Qwen2-0.5B, Qwen2-1.5B, DeepSeek-R1-1.5B, Qwen3-1.7B |
| **通用 small** | Qwen2-7B, Baichuan2-7B, DeepSeek-R1-7B, Qwen3-4B/8B |
| **通用 medium** | Baichuan2-13B, DeepSeek-R1-14B, Qwen3-14B |
| **通用 large / API** | GPT-4o, DeepSeek-V3, Hunyuan-Large |
| **医学专用** | DISC-MedLLM (13B), HuatuoGPT2-7B/13B, PULSE-7B, WiNGPT2 (8B), HuatuoGPT-o1-7B |

选题策略：包含不同参数规模（0.5B ~ API 级）、不同领域（通用 vs. 医学）、不同推理模式（标准 vs. R1/o1 推理增强）的模型，保证结论的泛化性。

---

## 3. 实验结果

### 3.1 实验设置

- **数据集**：CPubMedKG_large、CPubMedKG_small、CMeKG_large、CMeKG_small
- **评测方式**：基于 LLM API / 本地推理，输出答案后与 KG 标准答案精确匹配或语义匹配
- **扫描方式**：由于完整 KG 规模巨大，采用论文中设计的采样策略构造评测子集，保证覆盖核心实体和长尾实体

### 3.2 主实验结果：CPubMedKG_small（任务级）

task-oriented evaluation（CPubMedKG_small）精选结果：

| 模型 | ET | EC | ED | 实体 Avg | RT | FC | RP | 关系 Avg | ER | R1 | R2 | 子图 Avg | **Overall** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [G] GPT-4o | 97.8 | 90.0 | 71.0 | **86.3** | 84.1 | 59.3 | 62.2 | **68.5** | 46.5 | 41.5 | 89.3 | **59.1** | **71.3** |
| [G] Hunyuan-Large | 96.7 | **100** | 72.0 | 89.6 | 60.4 | 49.9 | 58.2 | 56.2 | **47.4** | 58.5 | **85.5** | **63.8** | 69.8 |
| [G] DeepSeek-V3 | 97.8 | 87.5 | 71.0 | 85.4 | **78.4** | **58.5** | **60.9** | **65.9** | 42.8 | 45.7 | 87.1 | 58.5 | 70.0 |
| [G] Qwen3-8B | 97.8 | 82.5 | 74.0 | 84.8 | **87.3** | 51.4 | 59.9 | 66.2 | 42.9 | 58.3 | 89.7 | **63.6** | 71.5 |
| [G] DeepSeek-R1-14B | 98.9 | **100** | 63.0 | **87.3** | 83.6 | 59.4 | 60.7 | 67.9 | **54.1** | 47.7 | 90.0 | **63.9** | **73.0** |
| [M] WiNGPT2 | 97.8 | 85.0 | 73.0 | 85.3 | 75.2 | 31.9 | 58.0 | 55.0 | 33.8 | 68.5 | 73.7 | 58.7 | 66.3 |
| [M] HuatuoGPT-o1-7B | 96.7 | 97.5 | 56.0 | 83.4 | 71.5 | 33.0 | 56.4 | 53.6 | **46.5** | 31.5 | 77.0 | 51.7 | 62.9 |
| [M] HuatuoGPT2-13B | 85.9 | 62.5 | 73.0 | 73.8 | 34.6 | 32.7 | 42.7 | 36.7 | 32.9 | 55.4 | 41.6 | 43.3 | 51.3 |
| [M] DISC-MedLLM | 71.7 | 15.0 | 53.0 | 46.6 | 31.5 | 8.1 | 39.9 | 26.5 | 22.8 | 0.2 | 52.2 | 25.1 | 32.7 |

### 3.3 主实验结果：CPubMedKG_small（知识级）

| 模型 | CovAvg($\mathcal{E}$) | CovDeg($\mathcal{E}$) | CovAvg($\mathcal{R}$) | CovDeg($\mathcal{R}$) | **Cov($\mathcal{T}$)** |
|:---|:---:|:---:|:---:|:---:|:---:|
| [G] Hunyuan-Large | **69.9** | **68.8** | **60.4** | 57.3 | **64.9** |
| [G] GPT-4o | 69.4 | 67.7 | 61.3 | **60.7** | 65.4 |
| [G] DeepSeek-V3 | 68.9 | 67.3 | 59.4 | 59.7 | 64.7 |
| [G] Qwen3-8B | 69.4 | 69.6 | **61.5** | **62.9** | 66.7 |
| [G] DeepSeek-R1-14B | **70.5** | 68.9 | **63.1** | 61.5 | **66.4** |
| [M] WiNGPT2 | 63.2 | 62.6 | 49.3 | 53.9 | 59.7 |
| [M] HuatuoGPT-o1-7B | 60.4 | 59.4 | 50.8 | 49.9 | 56.2 |
| [M] HuatuoGPT2-13B | 50.7 | 50.3 | 32.2 | 37.1 | 45.9 |
| [M] DISC-MedLLM | 32.2 | 32.9 | 20.3 | 22.7 | 29.5 |

### 3.4 核心发现

1. **通用大模型 > 多数医学专用模型**  
   GPT-4o、DeepSeek-V3、Hunyuan-Large、Qwen3 系列在整体得分和知识覆盖度上全面压制 DISC-MedLLM、HuatuoGPT2、PULSE 等医学专用模型。唯一接近通用模型的医学模型是 **WiNGPT2** 和 **HuatuoGPT-o1-7B**。

2. **医学模型在关系级任务上尤其薄弱**  
   DISC-MedLLM 的 FC（事实一致性）仅 8.1，RP（关系预测）仅 39.9；HuatuoGPT2-13B 的 RT（关系识别）仅 34.6。说明医学模型在预训练阶段偏重于"生成流畅的医学回答"，而非"储备系统的医学知识结构"。

3. **推理增强模型（R1/o1）显著提升**  
   DeepSeek-R1-14B 在 Overall（73.0）和 Cov($\mathcal{T}$)（66.4）上均为所有开源模型中最高；HuatuoGPT-o1-7B 也明显优于同系列非推理版 HuatuoGPT2-7B。说明推理能力对医学知识结构化应用有直接增益。

4. **参数规模仍是硬道理**  
   Qwen3 系列从 1.7B → 4B → 8B → 14B，Overall 从 63.6 → 67.7 → 71.5 → 70.0，基本随规模递增；小参数模型（0.5B/1.5B）几乎不具备实用医学知识能力。

---

## 4. 局限性

1. **知识图谱质量依赖**：CPubMedKG 存在文献抽取噪声，CMeKG 规模相对较小，结论受 KG 覆盖范围限制。
2. **中文为主**：两个 KG 均为中文医学知识图谱，未覆盖英文医学资源（如 UMLS/SNOMED CT 英文版）。
3. **评估未考虑时序变化**：医学指南和药物适应症持续更新，静态 KG 无法反映知识的时效性。
4. **评测方式偏结构匹配**：CovDeg 等指标基于精确匹配，对同义词、近义表述的容错性不足。

---

## 5. 个人思考

1. **"医学专用模型"神话的破除**  
   这篇论文最有冲击力的结论是：多数医学专用模型在系统医学知识覆盖上远不如通用大模型。这提醒我们：医学场景的"专用"不能只看生成风格是否像医生，更要看底层知识结构是否扎实。当前很多医学模型只是在通用基座上做了少量医学对话 SFT，并未真正注入系统的 KG 结构知识。

2. **知识图谱作为评测工具的价值**  
   相比问答题评测，KG 评测有两个独特优势：(1) 可以精确测量"覆盖盲区"（哪些实体/关系/子图是模型完全不知道的）；(2) 覆盖度指标（CovDeg）天然具有结构解释性。这为未来"诊断 LLM 知识短板"提供了方法论。

3. **与 MediEval 的对比**  
   MediEval（[[medieval--patient-contextual-medical-reasoning-benchmark|MediEval]]）关注"语境一致性 + 知识正确性"，MedKGEval 关注"知识覆盖广度 + 结构深度"。两者互补：MediEval 回答"模型在临床记录中会不会犯错"，MedKGEval 回答"模型到底懂多少医学知识"。未来结合两者——既看覆盖面、又看应用面——才能对医学 LLM 形成完整画像。

4. **对医学 LLM 训练的启示**  
   如果医学专用模型真想超越通用模型，预训练阶段就需要大规模注入结构化 KG 知识（而非仅靠对话 SFT）。

---

## 6. 关键引用

```bibtex
@inproceedings{zhang2025medkgeval,
  author    = {Zhang, Ziheng and Lin, Zhenxi and Zheng, Yefeng and Wu, Xian},
  title     = {How much Medical Knowledge do LLMs have? An Evaluation of Medical Knowledge Coverage for LLMs},
  booktitle = {Proceedings of the ACM Web Conference 2025},
  series    = {WWW '25},
  year      = {2025},
  publisher = {ACM},
  address   = {New York, NY, USA},
  doi       = {10.1145/3696410.3714535},
  pages     = {5330--5341},
  numpages  = {12},
  location  = {Sydney, NSW, Australia}
}
```

---

## 7. 附录：完整实验表格（CMeKG_small）

### CMeKG_small 任务级评估

| 模型 | ET | ED | 实体 Avg | RT | FC | RP | 关系 Avg | ER | R1 | R2 | 子图 Avg | Overall |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [G] GPT-4o | 95.0 | 100 | 97.5 | 64.5 | 74.2 | 69.4 | 70.5 | 56.6 | 70.9 | 83.9 | 70.5 | **74.7** |
| [G] DeepSeek-V3 | 92.8 | 83.3 | 88.1 | 65.0 | 74.9 | 70.0 | 70.0 | 51.1 | 69.7 | 80.7 | 67.2 | 71.1 |
| [G] Qwen3-8B | 96.6 | 100 | 98.3 | 57.3 | 68.9 | 63.1 | 63.1 | **53.8** | 63.7 | 83.6 | 67.0 | 71.4 |
| [G] DeepSeek-R1-14B | 91.9 | 66.7 | 79.3 | 62.5 | 72.2 | 67.3 | 67.3 | **53.4** | 53.2 | 76.8 | 61.1 | 64.9 |
| [M] WiNGPT2 | 87.2 | 100 | 93.6 | 37.1 | 68.5 | 52.8 | 52.8 | 43.0 | 77.9 | 91.8 | 70.9 | 68.9 |
| [M] HuatuoGPT-o1-7B | 89.4 | 83.3 | 86.4 | 44.0 | 75.5 | 59.7 | 59.7 | 48.1 | 43.6 | 72.8 | 54.8 | 61.3 |

### CMeKG_small 知识级评估

| 模型 | CovAvg($\mathcal{E}$) | CovDeg($\mathcal{E}$) | CovAvg($\mathcal{R}$) | CovDeg($\mathcal{R}$) | Cov($\mathcal{T}$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| [G] GPT-4o | 72.8 | 73.3 | 58.8 | 68.1 | 71.6 |
| [G] DeepSeek-V3 | 73.2 | 72.6 | 61.1 | 67.0 | 70.7 |
| [G] Qwen3-8B | **72.8** | **71.1** | **59.6** | **65.2** | **69.1** |
| [G] DeepSeek-R1-14B | 69.9 | 68.9 | 63.1 | 61.5 | 66.4 |
| [M] WiNGPT2 | 64.1 | 66.3 | 51.9 | 62.5 | 65.1 |
| [M] HuatuoGPT-o1-7B | 62.0 | 63.0 | 48.7 | 56.9 | 60.9 |

---

---

## 8. 图表与公式详解（基于完整 PDF）

### 8.1 Figure 1：多粒度任务设计（PDF 第 3 页）

> **Figure 1**（MedKGEval 框架总览）：左侧展示从 Medical Knowledge Graph 提取知识，通过 Task-Level Evaluation（实体/关系/子图任务）和 Knowledge-Level Evaluation 两个通道并行评测 LLM。右侧展示 9 个具体任务的 Q/A 格式示例。左侧展示从 Medical Knowledge Graph 提取知识，通过 Task-Level Evaluation（实体/关系/子图任务）和 Knowledge-Level Evaluation 两个通道并行评测 LLM。右侧展示 9 个具体任务的 Q/A 格式示例。

**图表结构解析**：

整个框架分为三层**自上而下的任务粒度**，对应医学知识从微观到宏观的层层递进：

| 层级 | 任务数 | 知识粒度 | 为什么这样设计 |
|:---|:---:|:---|:---|
| **Entity-Level** | 3 个 | 单个概念 | 检验模型是否能识别、纠正、定义一个医学实体 |
| **Relation-Level** | 3 个 | 两元连接 | 检验模型是否理解实体之间的医学关系 |
| **Subgraph-Level** | 3 个 | 区域网络 | 检验模型是否能从局部拓扑结构中推理出完整知识 |

> **设计直觉**：从"知道一个词"到"理解一个词与另一个词的关联"再到"理解一片局部网络"，构成了人类学习医学知识的认知路径。MedKGEval 用同样的认知阶梯来评估 LLM。

---

### 8.2 Figure 2：按度数的精细分析（PDF 第 8 页）

> **Figure 2**（15 个度数最高实体 / 关系的柱状 + 折线双轴图）：上图展示 15 个度数最高的医学实体，下图展示 15 个度数最高的医学关系。柱状图表示 $deg(·)$，折线表示各 LLM 的准确率。
> - 柱状图（淡蓝色）：实体的度数 $deg(\cdot)$，越高表示该实体/关系在 KG 中越核心。
> - 折线（不同颜色）：不同 LLM 在这些实体/关系上的准确率。
> - 关键用途：揭示 LLM 在"核心知识"和"边缘知识"上的能力差异。

**图 (a) — 实体层面**：
- 横轴：15 个度数最高的医学实体（如疾病名称）
- 左侧纵轴（柱状）：该实体在 CPubMedKG 中的度数（连接数）
- 右侧纵轴（折线）：各模型在该实体相关 ET/EC/ED 任务上的平均准确率

**图 (b) — 关系层面**：
- 横轴：15 个度数最高的医学关系（如"治疗"、"导致"）
- 柱状图：关系的度数（该关系在整个 KG 中出现的次数）
- 折线：各模型在该关系相关 RT/FC/RP 任务上的平均准确率

**关键发现**（从图中读出的模式）：
1. **核心实体 ≠ 高准确率**：某些度数非常高的实体（如在 KG 中连接数百条边的常见疾病），模型反而表现一般，可能因为"太常见反而被训练数据稀释"
2. **关系级折线波动大**：相比实体级的相对平滑，关系级各模型之间差异显著，尤其在低频关系上
3. **GPT-4o 并非处处领先**：某些特定实体（如罕见疾病的特有并发症），更小模型反而超过 GPT-4o

---

### 8.3 公式逐一详解

#### ① 实体度数 (Eq. 1)

$$\text{deg}(e) = |\{(e, r, t) \in \mathcal{T} \mid r \in \mathcal{R}, t \in \mathcal{E}\}| + |\{(h, r, e) \in \mathcal{T} \mid h \in \mathcal{E}, r \in \mathcal{R}\}|$$

| 符号 | 含义 |
|:---|:---|
| $\text{deg}(e)$ | 实体 $e$ 在 KG 中的**总度数** |
| 第一项 | $e$ 作为**头实体**参与的三元组数量（出度） |
| 第二项 | $e$ 作为**尾实体**参与的三元组数量（入度） |

> **通俗理解**：一个实体在 KG 中连接了多少条边。出度 + 入度 = 总度数。度数越高，该实体越"核心"。
> **示例**：实体"高血压"在 CPubMedKG 中可能连接到数百种并发症和治疗方案，因此度数为几百。

#### ② 整体度数 (Eq. 2)

$$\text{deg}(\mathcal{E}) = \sum_{e \in \mathcal{E}} \text{deg}(e)$$

> 所有实体的度数之和（等价于 KG 中三元组总数 × 2），用于归一化。

#### ③ 关系度数 (Eq. 3)

$$\text{deg}(r) = |\{(h, r, t) \in \mathcal{T}\}|$$

> 关系 $r$ 在 KG 中出现的**总次数**。关系越常出现，说明它是越基础的医学概念（如"治疗"可能比"罕见并发症"的度数高得多）。

#### ④ 准确率 (Eq. 4)

$$\text{Acc} = \frac{|Q^c|}{|Q|}$$

| 符号 | 含义 |
|:---|:---|
| $Q$ | 该任务的全部问题集合 |
| $Q^c$ | 被正确回答的问题子集 |
| $|Q|$ | 问题总数 |

> **通俗理解**：就是最基本的 Accuracy。但注意——不同任务的"正确"定义不同：
> - multi-choice（选择题）：LLM 输出的选项与标准答案一致即正确
> - true-false（判断题）：输出"yes"或"no"与事实一致即正确
> - free generation（自由生成）：需要**精确匹配**或**语义匹配**（论文中如 RP/EC/ED 需更复杂的评估）

#### ⑤ 平均实体覆盖度 (Eq. 5)

$$\text{CovAvg}(\mathcal{E}) = \frac{1}{|\mathcal{E}|} \sum_{e \in \mathcal{E}} \sum_{i=1}^{9} \frac{|Q_i^c(e)|}{|Q_i(e)|}$$

| 符号 | 含义 |
|:---|:---|
| $\mathcal{E}$ | KG 中的全部实体集合 |
| $Q_i(e)$ | 第 $i$ 个任务中以 $e$ 为核心实体的问题集合 |
| $Q_i^c(e)$ | 其中被 LLM 正确回答的子集 |
| $\frac{|Q_i^c(e)|}{|Q_i(e)|}$ | 实体 $e$ 在第 $i$ 个任务上的**局部正确率** |
| 外层的 $\frac{1}{|\mathcal{E}|}\sum$ | 对所有实体取**简单算术平均** |

> **关键直觉**：每个实体在 9 个任务上都有本题率。CovAvg 将所有实体的正确率简单平均。**问题**：如果模型只在大量低度数（边缘）实体上表现好，而核心实体很差，CovAvg 仍可能好看。

#### ⑥ 度数感知实体覆盖度 (Eq. 6) — 核心创新

$$\text{CovDeg}(\mathcal{E}) = \sum_{e \in \mathcal{E}} \sum_{i=1}^{9} \frac{|Q_i^c(e)|}{|Q_i(e)|} \times \frac{\text{deg}(e)}{\text{deg}(\mathcal{E})}$$

> 与 Eq. 5 的区别：将简单平均 $\frac{1}{|\mathcal{E}|}$ 替换为**按度数加权** $\frac{\text{deg}(e)}{\text{deg}(\mathcal{E})}$。

| 权重项 | 效果 |
|:---|:---|
| $\frac{\text{deg}(e)}{\text{deg}(\mathcal{E})}$ | 度数越高的实体，在求和中占据越大的比重 |
| 度数低的边缘实体 | 权重很小，即使全对也拉不高总分 |
| 度数高的核心实体 | 权重很大，表现差会显著拉低总分 |

> **公式设计动机**：避免模型通过"只会背冷门知识"来刷分。希望模型真正掌握**核心医学知识**（如高血压、糖尿病等常见疾病）。

> **箭头含义**（PDF Table 4 中 $\uparrow/\downarrow$ 的作用）：
> - $\text{CovAvg}(\mathcal{E}) < \text{CovDeg}(\mathcal{E})$：模型在**高度数核心实体**上表现更好（$\uparrow$）
> - $\text{CovAvg}(\mathcal{E}) > \text{CovDeg}(\mathcal{E})$：模型在**低度数边缘实体**上表现更好（$\downarrow$）

#### ⑦ 平均关系覆盖度 (Eq. 7)

$$\text{CovAvg}(\mathcal{R}) = \frac{1}{|\mathcal{R}|} \sum_{r \in \mathcal{R}} \sum_{i=1}^{9} \frac{|Q_i^c(r)|}{|Q_i(r)|}$$

> 结构与 Eq. 5 对称，对象从实体换成关系。$Q_i(r)$ 表示第 $i$ 个任务中以关系 $r$ 为核心考察对象的问题集合。

#### ⑧ 度数感知关系覆盖度 (Eq. 8)

$$\text{CovDeg}(\mathcal{R}) = \sum_{r \in \mathcal{R}} \sum_{i=1}^{9} \frac{|Q_i^c(r)|}{|Q_i(r)|} \times \frac{\text{deg}(r)}{|\mathcal{T}|}$$

> 与 Eq. 6 对称，按关系度数 $deg(r)$ 加权。这里归一化分母用 $|T|$（三元组总数）而非 $deg(\mathcal{R})$，因为每条三元组恰好对应一个关系。

#### ⑨ 三元组整体覆盖度 (Eq. 9) — 综合指标

$$\text{Cov}(\mathcal{T}) = \frac{1}{3} \sum_{(h,r,t) \in \mathcal{T}} \sum_{i=1}^{9} \left[ \frac{|Q_i^c(h)|}{|Q_i(h)|} + \frac{|Q_i^c(r)|}{|Q_i(r)|} + \frac{|Q_i^c(t)|}{|Q_i(t)|} \right]$$

| 符号 | 含义 |
|:---|:---|
| $(h, r, t)$ | KG 中一个具体三元组 |
| $\frac{|Q_i^c(h)|}{|Q_i(h)|}$ | 头实体 $h$ 在第 $i$ 个任务上的正确率 |
| $\frac{|Q_i^c(r)|}{|Q_i(r)|}$ | 关系 $r$ 在第 $i$ 个任务上的正确率 |
| $\frac{|Q_i^c(t)|}{|Q_i(t)|}$ | 尾实体 $t$ 在第 $i$ 个任务上的正确率 |
| $\frac{1}{3}$ | 对三项取平均 |

> **通俗理解**：每个三元组由"头实体 + 关系 + 尾实体"三块砖组成。Cov($T$) 测量模型掌握了多少完整的"砖组合"——不是分别知道 $h$、$r$、$t$，而是有结构地掌握 $(h,r,t)$ 整体。

---

### 8.4 Table 2 & Table 5：Benchmark 设计矩阵

| Level | Task | Q/A Format | CPubMedKG(small) #QA | CPubMedKG(large) #QA | CMeKG(small) #QA | CMeKG(large) #QA |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| Entity | ET | multi-choice | 92 | 166 | 486 | 1,636 |
| Entity | EC | multi-choice | 40 | 8 | 0 | 6 |
| Entity | ED | true-false | 100 | 20 | 0 | 78 |
| Relation | RT | multi-choice | 3,142 | 7,322 | 100 | 26 |
| Relation | FC | true-false | 9,426 | 21,906 | 328 | 19,881 |
| Relation | RP | multi-choice | 3,142 | 7,322 | 0 | 6,637 |
| Subgraph | ER | multi-choice | 6,284 | 14,644 | 0 | 0 |
| Subgraph | R1 | true-false | 6,284 | 14,644 | 964 | 12,906 |
| Subgraph | R2 | multi-choice | 3,148 | 7,332 | 2,546 | 6,453 |

**设计洞察**：
1. **EC 和 ED 在 CMeKG 上数量极少甚至为 0**：因为 CMeKG 的实体同义词和类型标注不如 CPubMedKG 完备，直接限制了这两个任务的构造。
2. **large 版本是 small 的约 2.3 倍**：从采样策略看，$deg(e) > \theta$ 的阈值决定了 small 和 large 的划分。论文提到 $\theta$ 可调。
3. **true-false vs. multi-choice 的分配**：FC（事实检查）用判断题最自然；ET/RP/ER 用选择题适合自动评估。

---

### 8.5 Table 6：统一 Prompt 设计

论文附录 Table 6 展示了每个任务的**示例问题和统一 Prompt**：

```
You are a medical AI assistant focused on providing accurate and reliable 
medical information. Please answer the following medical-related questions 
directly, ensuring that your responses are concise and clear, without any 
additional reasoning or explanation. Please note that all answers should 
be based on the latest medical knowledge and research.
```

> **复现关键**：这个 Prompt 是论文所有 LLM 的统一输入模板。复现时必须**逐字使用**，不能添加额外的"角色扮演"（如"你是一位资深医生"），否则可能引入偏差。

| Task | 示例问题（核心模式） |
|:---|:---|
| ET | "What is the entity type of 'arrhythmia'? Please select..." |
| EC | "Which ... has a different entity type compared to the others?" |
| ED | "Are 'primary hypertension' and 'hypertensive disease' synonyms? yes/no" |
| RT | "Which combinations of head entity type and tail entity type can be correctly connected?" |
| FC | "Is there a medical relationship ... between ... and ...? yes/no" |
| RP | "What is the medical relationship between...? select..." |
| ER | "Which of the following five medical relationships is incorrect?" |
| R1/R2 | 多跳推理，基于已知事实链推断新关系 |

---

### 8.6 Table 4 趋势箭头解读

PDF Table 4（Knowledge-level, small）中的 $\uparrow$ 和 $\downarrow$ 列是论文的一个重要分析维度。

**比较 CovAvg vs. CovDeg 的含义**：

| 趋势 | 条件 | 模型行为特征 |
|:---|:---|:---|
| $\uparrow$ | CovDeg > CovAvg | 模型在**核心知识**（高度数实体/高频关系）上更强，低度数实体相对较弱 |
| $\downarrow$ | CovDeg < CovAvg | 模型在**边缘知识**（低度数实体/低频关系）上更强，核心知识相对较弱 |
| $=$ | 两者接近 | 核心和边缘知识覆盖较为均衡 |

**结果模式**（从 PDF 中读出的）：
- **GPT-4o**：CMeKG 上实体 $\downarrow$，但关系 $\downarrow$ —— 说明 GPT-4o 对 CMeKG 的边缘关系掌握更多
- **Qwen2-0.5B**：四处都是 $\downarrow$ —— 小模型根本没有"核心知识"的概念，只会背碎片
- **DeepSeek-R1-14B**：大部分 $\uparrow$ —— 推理增强模型能够更好地理解核心知识结构

---

## 9. 复现指南

### 9.1 环境准备

```bash
git clone https://github.com/ZihengZZH/MedKGEval.git
cd MedKGEval
pip install -r requirements.txt  # 假设存在，否则根据 scripts/ 中的 import 安装
```

### 9.2 数据准备

论文使用了两个 KG 来源：

1. **CPubMedKG**：从 [官网](https://cpubmed.openi.org.cn/graph/wiki) 下载原始数据
2. **CMeKG**：从相关渠道获取

然后运行采样脚本：
```bash
python utils/kg_sample.py --input CPubMedKG_raw --output CPubMedKG_small --theta 2.0
python utils/kg_sample.py --input CPubMedKG_raw --output CPubMedKG_large --theta 1.0
```

> $\theta$ 为度阈值：$deg(e) > \theta$ 的实体被保留（配合其对应三元组）。$\theta$ 越大，保留实体越少，子集越小。

### 9.3 QA 构造

```bash
python utils/qa_construct.py --kg CPubMedKG_small --output benchmarks/CPubMedKG_small
```

> **关键逻辑**：每个任务根据 KG 结构自动生成问题集。
> - ET：遍历每个实体，构造"这是什么类型"的多选题
> - FC：遍历每个三元组，构造"是否存在该关系"的判断题，同时生成错误三元组作为反面
> - R1/R2：从特定实体出发，沿 KG 边构造多跳推理题

### 9.4 LLM 评测

```bash
# 示例：评测 Qwen2-7B on CPubMedKG_small
python scripts/run_eval.py --model Qwen/Qwen2-7B-Instruct \
  --benchmark benchmarks/CPubMedKG_small \
  --output results/CPubMedKG_small_qwen2-7b.json
```

> **Prompt 严格性**：确保脚本内使用的 System Prompt 与 Table 6 完全一致（英文模板，角色为"medical AI assistant"）。
> **输出解析**：
> - multi-choice 任务：解析 LLM 输出中的 A/B/C/D 选项
> - true-false 任务：解析 "yes"/"no"（论文中通过正则匹配，非语义理解）

### 9.5 评估指标计算

```bash
python utils/eval.py --results results/CPubMedKG_small_qwen2-7b.json \
  --kg CPubMedKG_small \
  --mode task    # 输出 Acc 和 Overall

python utils/eval.py --results results/CPubMedKG_small_qwen2-7b.json \
  --kg CPubMedKG_small \
  --mode knowledge    # 输出 CovAvg, CovDeg, Cov
```

> **精确匹配 vs. 语义匹配**：
> - ET/EC/RT/RP/ER 等选择题：用**精确选项匹配**判断正误
> - FC 判断题：用**关键词匹配**（如"yes"/"no"）
> - ED 定义生成：可能需要**语义相似度**或**人工审核**（论文主要用精确匹配）

### 9.6 复现核心注意点

1. **KG 采样策略必须可复现**：原始 KG 版本会影响所有指标。作者使用了特定版本的 CPubMedKG 和 CMeKG，不同版本的三元组数可能不同。
2. **LLM Temperature = 0**：为了保证结果可复现，所有 LLM 推理应将 temperature 设为 0（贪婪解码）。
3. **答案后处理标准化**：LLM 输出可能有格式差异（如"A." vs "A" vs "answer: A"），复现时后处理逻辑需与论文完全一致。
4. **实体链接**：如果 LLM 输出中出现同义词（如"心肌梗塞"和"心肌梗死"），精确匹配会判为错误。论文未详细说明是否使用了实体链接或同义词映射——这是一个潜在差异点，建议按严格精确匹配复现。

---

## 10. 补充：Large 版本完整实验（PDF 附录 Table 8–9）

### Table 8：Task-Level (Large) 精选

| 模型 | 实体 Avg | 关系 Avg | 子图 Avg | **Overall** |
|:---|:---:|:---:|:---:|:---:|
| Qwen2-7B | 75.83 | 70.69 | 51.04 | **63.90** |
| Baichuan2-13B | 70.77 | 66.65 | 47.86 | 59.33 |
| DISC-MedLLM | 47.19 | 38.49 | 18.02 | 33.87 |
| HuatuoGPT2-7B | 41.79 | 33.63 | 14.20 | 29.19 |
| WiNGPT2 | 70.13 | 55.67 | 43.22 | 55.86 |
| GPT-4o | **86.28** | **68.54** | **55.66** | **70.65** |

### Table 9：Knowledge-Level (Large) 精选

| 模型 | CovAvg($\mathcal{E}$) | CovDeg($\mathcal{E}$) | CovAvg($\mathcal{R}$) | CovDeg($\mathcal{R}$) | **Cov($\mathcal{T}$)** |
|:---|:---:|:---:|:---:|:---:|:---:|
| Qwen2-7B | 69.24 | 70.73 | 54.51 | 55.55 | **65.43** |
| Baichuan2-13B | 64.54 | 63.88 | 46.89 | 49.59 | 54.98 |
| GPT-4o | **66.75** | **65.66** | **55.60** | **54.98** | **62.31** |

> **Small vs. Large 对比**：Large 版本 Overall 普遍低于 Small，因为 Large 包含更多低度数、低频知识和长尾关系，对模型是更大考验。这验证了评估框架的"难度可调节"设计。
