---
created: 2026-06-09
published: 2026-05-13
paper: https://arxiv.org/abs/2605.13045
code: https://github.com/GuanZihan/TempoMed-Bench
authors: Zihan Guan, Qiao Jin, Guangzhi Xiong, Fangyuan Chen, Mengxuan Hu, Qingyu Chen, Yifan Peng, Zhiyong Lu, Anil Vullikanti (University of Virginia, NIH, Dana-Farber Cancer Institute, Yale University, Weill Cornell Medicine)
tags:
  - 模型评估
  - 医学NLP
  - 时间感知
  - 大语言模型
  - 基准测试
---

# Large Language Models Lack Temporal Awareness of Medical Knowledge

## 一句话总结

本文提出首个评估大语言模型医学知识**时间感知（temporal awareness）**的基准测试 TempoMed-Bench，揭示了当前 LLM 在医学领域存在严重的时间感知缺陷：对最新知识的掌握随时间呈线性衰退（而非知识截断式的骤降）、历史知识回忆准确率仅为最新知识的 25%-54%、且在不同年份间的预测行为高度不一致；即使用 agentic RAG 增强，改善也非常有限。

![](TempoMed_fig1_motivation.png)

> **Figure 1**: 以韩国肥胖治疗指南为例（2018 vs 2024），说明医学知识随时间演变的现实：新药获批后治疗推荐发生显著变化。现有的无时间（atemporal）评估方法无法捕捉 LLM 是否能正确对齐特定时间点的医学知识。

---

## 1. 研究背景与动机

### 1.1 问题定义

**时间感知（Temporal Awareness）**：指 LLM 能够正确回忆医学知识，并将其与**指定时间点**对齐的能力。即模型不仅要"知道"某一医学事实，还要知道该事实在何时是正确/过时的。

为什么时间感知在医学领域尤其关键？
1. **用户咨询场景**：患者可能询问基于最新证据的治疗建议，模型必须知道当前（而非过时）的推荐。
2. **临床回溯场景**：医生/研究员可能需要解读历史研究、回顾性分析过去的治疗决策，模型必须区分"当时正确"与"现在正确"。
3. **数据本身的时序性**：大部分医学数据本质上是历史性和纵向的（historical & longitudinal），来自过去的患者就诊和临床试验记录。

### 1.2 现有方法的不足

| 维度 | 现有评估基准 | 本工作 |
|------|-----------|--------|
| 评估方式 | 无时间语境的考试式问答（MedQA、PubMedQA、MedMCQA、MultiMedQA 等） | 基于临床指南版本的**时间锚定**MCQ |
| 知识假设 | 静态（static） | 动态演化（evolving） |
| 知识来源 | 综合医学考题 | 真实发布的临床指南 |
| 过时知识 | 未评估 | 显式评估历史知识回忆 |

现有基准的问题在于：即使 LLM 在 atemporal 基准上取得高分，也可能在时间维度上"张冠李戴"——以当前知识回答历史问题，或以过时知识回答当前问题。

---

## 2. TempoMed-Bench 数据构建

### 2.1 核心思想

利用**临床指南**作为医学知识演化的自然快照。指南由专业学会定期更新，天然蕴含"同一主题在不同时间的推荐差异"。通过提取指南版本间的差异，构建时间锚定的多项选择题（MCQ），探测模型知识更偏向"最新版"还是"旧版"。

### 2.2 构建流程（四阶段）

![](TempoMed_fig2_pipeline.png)

> **Figure 2**: 数据构建流程概览。从 PMC 数据库收集指南论文，经 LLM 筛选和轨迹构建，再通过差异提取和 MCQ 生成，得到 721 道时间锚定的评测题。

#### Stage 1: 指南收集与过滤

- **数据源**：PubMed Central (PMC)，含 750 万+ 生物医学论文
- **粗筛策略**：标题或 `<article-type>` 标签中含 "guideline"
- **结果**：23,195 篇候选指南论文（comm: 13,002；noncomm: 8,790；other: 1,403）

#### Stage 2: 指南轨迹构建（Guideline Trajectory Building）

- **定义**：一条轨迹 = 同一组织、同一主题、按时间排序的指南版本序列
- **LLM 筛选**：判断是否为专业学会发布的真正指南；提取引用的早期版本
- **后处理**：利用 PubMed "Similar Articles" 补全遗漏的先前版本；恢复缺失的 PMID；过滤子轨迹
- **结果**：**3,411 条轨迹**（TempoMed-Traj）

#### Stage 3: 轨迹差异提取

![](TempoMed_fig3_difference.png)

> **Figure 3**: 差异提取流程。对每条轨迹中的最新版与旧版指南，使用 LLM 提取有意义的变化，再经独立的 verifier LLM 评分（GOOD=1, OK=0.5, BAD=0），确保差异是"严格 head-to-head 对比"且"非微小增量更新"。

- 要求：差异必须代表直接对比、意义显著、排除微小更新
- **结果**：**721 对有效差异**

#### Stage 4: MCQ 生成

![](TempoMed_fig4_mcq.png)

> **Figure 4**: MCQ 生成流程。每对差异生成一道 NBME（美国国家医学考试委员会）风格的 5 选项 MCQ，选项构成如下：
> - 1 个**最新知识**选项（up-to-date）
> - 1 个**过时知识**选项（outdated）
> - 2 个**混淆项**（distractors，与两者均不符但临床合理）
> - 1 个 **"我不知道"**（IDK）选项，用于评估模型弃权能力

**最终数据集**：721 道 MCQ，覆盖 2008–2026 年的最新指南与 2001–2025 年的过时指南。

### 2.3 数据统计

![](TempoMed_fig5_stats.png)

> **Figure 5**: 数据集统计分布。(a) 最新指南年份分布：2008–2026；(b) 过时指南年份分布：2001–2025；(c) 问题长度近似高斯分布。

---

## 3. 评估方法与实验设置

### 3.1 评测模型

共 14 个模型，含 10 个开源 + 4 个闭源：

| 类型 | 模型 |
|------|------|
| 闭源（OpenAI） | GPT-5, GPT-4.1, GPT-4o |
| 开源通用 | Llama-3.1-8B, Llama-3.2-3B, Qwen2.5-14B, Qwen2.5-7B, Qwen3-4B, GPT-OSS-20B, OLMo-3-7B |
| 开源医学 | MedGemma-27B, MedGemma-1.5-4B, MedGemma-4B, Gemma-3-4B |

### 3.2 位置扰动（Positional Perturbation）

为消除选项位置偏差（positional bias），每道题额外生成 2 个变体：
- **Reorder-only**：选项顺序重排，标签-内容绑定不变
- **Label Shuffling**：内容跨位置打乱，改变正确项位置
- **Alternate Labels**：标签替换（如 X/Y/Z/W）

### 3.3 四大研究问题（RQs）

| RQ | 问题 | 核心发现 |
|----|------|----------|
| RQ1 | LLM 是否已掌握最新医学知识？ | 即使最强模型也远非全知；知识随时间线性衰退 |
| RQ2 | LLM 能否正确回忆历史（过时）知识？ | 历史知识准确率仅为最新知识的 25%-54% |
| RQ3 | LLM 跨时间维度的行为是否一致？ | 高度不一致，呈现多种错误模式 |
| RQ4 | Agentic RAG 能否缓解？ | 改善有限（-3.15% ~ +14.14%），甚至可能反效果 |

---

## 4. 实验结果

### 4.1 RQ1: 最新知识掌握程度

**Table 1: 主实验结果——各模型在 TempoMed-Bench 上的准确率与选项分布**

| Model | Accuracy (%) | Up-to-date (%) | Outdated (%) | Distractor (%) | Invalid (%) | Unknown (%) |
|-------|-------------|----------------|--------------|----------------|-------------|-------------|
| GPT-5 | 70.69 | 70.69 | 15.05 | 2.87 | 8.89 | 2.50 |
| GPT-4.1 | 71.11 | 71.11 | 16.85 | 3.06 | 8.98 | 0.00 |
| GPT-4o | 44.95 | 44.95 | 7.59 | 1.67 | 9.12 | 36.67 |
| GPT-OSS-20B | 54.05 | 54.05 | 18.49 | 6.33 | 6.47 | 14.66 |
| Llama-3.1-8B | 54.92 | 54.92 | 25.29 | 14.01 | 0.18 | 5.59 |
| Llama-3.2-3B | 53.63 | 53.63 | 27.14 | 15.63 | 0.05 | 3.56 |
| Qwen2.5-14B | 46.32 | 46.32 | 15.86 | 6.20 | 0.00 | 31.62 |
| Qwen2.5-7B | 59.64 | 59.64 | 23.81 | 16.09 | 0.00 | 0.46 |
| Qwen3-4B | 46.28 | 46.28 | 17.89 | 10.59 | 0.00 | 25.24 |
| MedGemma-27B | 53.68 | 53.68 | 15.58 | 6.33 | 0.05 | 24.36 |
| MedGemma-1.5-4B | 55.43 | 55.43 | 24.27 | 17.85 | 0.92 | 1.53 |
| MedGemma-4B | 60.10 | 60.10 | 24.97 | 14.75 | 0.14 | 0.05 |
| Gemma-3-4B | 58.16 | 58.16 | 22.65 | 18.82 | 0.00 | 0.37 |
| OLMo-3-7B | 57.51 | 57.51 | 22.98 | 14.98 | 0.00 | 4.53 |

**关键发现 1：最强模型也非全知**
- GPT-4.1 和 GPT-5 的准确率仅约 **71%**，说明即使前沿模型也无法掌握所有最新医学知识。

**关键发现 2：小模型倾向于选混淆项**
- Llama-3.2-3B、Gemma-3-4B 等模型的**混淆项选择率**高达 15%-18%，说明这些模型的参数知识可能偏离正确轨迹。

**关键发现 3：部分模型频繁弃权**
- GPT-4o 的 **Unknown（IDK）率达到 36.67%**，Qwen2.5-14B 为 31.62%，Qwen3-4B 为 25.24%。模型对自身不确定的知识倾向于直接弃权。

### 4.2 时间退化趋势（核心发现）

![](TempoMed_fig6_degradation.png)

> **Figure 6**: 最新医学知识准确率随时间（指南年份）的变化趋势。纵轴为准确率，横轴为指南发布年份。红线标记各模型的知识截断日期（knowledge cutoff）。**关键结论**：所有模型对更近年份的知识掌握得更差，且衰退呈**近似线性趋势**，而非在截断日期附近骤降。这说明参数化的医学知识并非严格受限于知识截断——更合理的解释是：新指南在预训练语料中出现频率较低，而较早的成熟指南被更广泛引用。

### 4.3 RQ2: 历史知识回忆

![](TempoMed_fig7_accuracy.png)

> **Figure 7**: 历史知识目标问题与最新知识目标问题的准确率对比。将问题模板中的目标年份从最新版改为历史版后，模型准确率出现明显"崩塌"。**历史知识准确率仅为最新知识的 25.37% - 53.89%**。

**实验方法**：修改问题模板为：
> *"According to the xxx guideline issued in {historical year}, which option do you think is correct?"*

**解读**：这表明 LLM 在预训练/后训练阶段可能存在**知识遗忘效应（knowledge forgetting）**——模型在更新知识的过程中，对旧知识的编码被削弱。

### 4.4 RQ3: 时间一致性

**定义**：时间一致性不要求模型在每个年份都答对，而是要求预测在时间上**逻辑一致地演进**。理想情况下，当指南在某年更新后，模型对"旧推荐"的态度应从"同意"变为"不同意"，对"新推荐"的态度应反之（ zig-zag 模式）。

**实验方法**：对每条含两个版本的轨迹，构造时间探测问题：
> *"According to the most recent guideline on or before {target year}, do you agree with the statement that {statement}?"*
> 
> 目标年份从 2000 年到 2026 年逐步变化，观察模型回答的"是/否"热图。

**六种一致性模式**：

| 模式 | 描述 | 含义 |
|------|------|------|
| **All-True** | 对最新和过时陈述在所有年份都答"是" | 强烈肯定偏差 |
| **All-False** | 对所有陈述在所有年份都答"否" | 缺乏信心或知识识别失败 |
| **Only-Know-Latest** | 始终偏好最新推荐，无视年份 | 只知道最新知识，无时间边界 |
| **Only-Know-Prior** | 始终偏好过时推荐 | 无法吸收新知识更新 |
| **Wrong-Transition-Point** | zig-zag 模式出现，但转折点错误 | 感知到变化但时机错位 |
| **Correct-Transition-Point** | 正确 zig-zag 模式 | 理想行为（极少出现） |

**Table 2: 时间一致性模式分布（55 条双版本轨迹）**

| Model | Inconsistency | All-True | Only-Know-Latest | Wrong-Transition-Point | Correct-Transition-Point |
|-------|--------------|----------|------------------|----------------------|------------------------|
| GPT-5 | **78.18%** | 0.00% | 12.73% | 5.45% | 3.64% |
| GPT-4.1 | **63.64%** | 5.45% | 21.82% | 1.82% | 1.82% |
| Llama-3.1-8B | **49.09%** | 14.55% | 21.82% | 0.00% | 0.00% |
| Qwen2.5-7B | **58.18%** | 9.09% | 14.55% | 0.00% | 0.00% |
| MedGemma-4B | **34.55%** | **52.73%** | 9.09% | 0.00% | 0.00% |

**核心发现**：
- **GPT-5 的不一致性高达 78.18%**，即使是顶尖模型，其行为在时间维度上几乎完全不可预测。
- **Only-Know-Latest 和 All-True 是前两大主要模式**，再次印证"模型可能编码了最新知识，但缺乏时间感知"。
- **Correct-Transition-Point（理想模式）在所有模型中均低于 4%**，说明几乎没有任何模型能在时间维度上做出逻辑一致的演进。
- MedGemma-4B 的 All-True 率高达 52.73%，说明该模型对医学陈述存在强烈的盲目肯定倾向。

### 4.5 RQ4: Agentic RAG 的效果

**Table 3: Agentic RAG 前后对比**

| Backend LLM | Agentic RAG | Up-to-date (%) | Δ | Outdated (%) | Δ |
|-------------|-------------|---------------|-----|--------------|-----|
| GPT-4.1 | (baseline) | 78.1 | — | 27.35 | — |
| | ToolUniverse | 76.9 | **-1.2** | 30.93 | +3.58 |
| | Biomni | 79.20 | +1.10 | 29.56 | +2.21 |
| GPT-5 | (baseline) | 79.7 | — | 37.73 | — |
| | ToolUniverse | 85.3 | +5.6 | 51.87 | **+14.14** |
| | Biomni | 76.55 | **-3.15** | 43.62 | +5.89 |

**核心发现**：
- **改善非常有限**：ToolUniverse 对 GPT-5 的 up-to-date 准确率仅提升 +5.6%，Biomni 甚至导致 GPT-5 下降 -3.15%。
- **失败原因分析**：虽然 agent 能通过 `PMC_search_papers` 正确检索到目标指南，但 `Web_Search` 工具同时引入了**额外的、可能相互冲突的文档**。多个信息源之间的竞争导致模型认知负荷增加，最终反而降低准确率。
- 这与现有研究中"检索噪声/冲突信息会干扰 LLM 决策"的发现一致（Amiraz et al., 2025; Shi et al., 2023）。

---

## 5. 局限性与未来方向

### 5.1 本文局限

1. **数据规模有限**：严格的差异提取标准导致仅 721 条 MCQ，覆盖范围有限。
2. **未探索微调与高级 Agent 架构**：仅使用 off-the-shelf agentic RAG 工具，未尝试针对时间感知任务进行模型微调或设计更先进的 agent 工作流。
3. **全文获取受限**：仅 PMC 提供全文访问，PubMed 中的大量摘要-only 论文未被纳入。

### 5.2 未来方向

- **扩大规模**：放宽差异提取标准或引入更多数据源，增大数据集覆盖面。
- **针对性训练**：探索在预训练或后训练阶段显式注入"时间标签"，让模型学习知识的时间边界。
- **改进检索策略**：设计能够识别并利用指南版本链的专用检索器，避免引入无关冲突信息。
- **知识编辑（Knowledge Editing）**：研究如何在不遗忘旧知识的前提下，精确更新模型中的医学知识。
- **实时更新机制**：探索结合 PubMed/PMC 实时新论文的增量知识注入管道。

---

## 6. 个人思考

### 6.1 方法的优雅之处

1. **指南作为自然时间戳（timestamped snapshots）**：选择临床指南作为知识演化载体非常巧妙——指南不仅涉及事实性知识，还包含推荐等级、禁忌症、剂量等复杂的决策知识，比 Wiki/factoid 问答更接近真实临床场景。
2. **Minimal but rigorous construction**：整个构建流程仅使用 LLM + verifier + 标准 prompt，无需繁重的人工标注，却通过严格的评分规则（GOOD/OK/BAD）和后处理保证了质量。
3. **IDK 选项的设计**：引入"我不知道"选项是一个被低估的聪明设计，它同时评估了模型的**自知之明（self-awareness）**和**校准能力（calibration）**。

### 6.2 对当前 LLM 评估的启示

当前医疗 LLM 的评估存在"静态幻觉"——我们在静态基准上刷出了越来越高的分数，却忽视了医学知识本质上是流变的。本文提醒我们：

> **在医学这种高时效性领域，知道"什么时候正确"和知道"正确答案"同等重要。**

### 6.3 对 Agentic RAG 的冷思考

RQ4 的结果给我很大触动。业界普遍认为"检索增强 = 知识过时问题的银弹"，但本文表明：
- 如果检索器引入的信息存在冲突，LLM 并未表现出预期的"信息整合"能力，反而被噪音干扰。
- Agentic RAG 的复杂性（多工具调用、信息聚合）本身可能成为新的失败来源。

这提示我们在设计临床 RAG 系统时，**检索精度（precision）可能比召回率（recall）更重要**——宁可少检索几条，也要确保引入的信息高度相关且一致。

### 6.4 与其他工作的关联

- **与医学基准的对比**：相比 MedQA/PubMedQA 等 atemporal 基准，TempoMed-Bench 填补了一个关键空白。可以思考如何将时间感知维度拓展到其他垂直领域（如法律判例、金融产品规范、软件安全漏洞等）。
- **与知识编辑（Knowledge Editing）的关联**：本文发现的历史知识遗忘问题，正好指向了知识编辑领域的一个核心挑战——如何在"更新"与"保留"之间取得平衡。TempoMed-Bench 可作为该方向的有力评测工具。

---

## 7. 关键引用

```bibtex
@article{guan2026temporal,
  title={Large Language Models Lack Temporal Awareness of Medical Knowledge},
  author={Guan, Zihan and Jin, Qiao and Xiong, Guangzhi and Chen, Fangyuan and Hu, Mengxuan and Chen, Qingyu and Peng, Yifan and Lu, Zhiyong and Vullikanti, Anil},
  journal={arXiv preprint arXiv:2605.13045},
  year={2026}
}
```

**核心参考文献**：
- MedQA (Jin et al., 2021): 医学 QA 基准
- PubMedQA (Jin et al., 2019): 生物医学文献问答
- MedHELM (Bedi et al., 2026): 全面医学基准套件
- ToolUniverse (Gao et al., 2025): 生物医学 agentic 工具集合
- Biomni (Huang et al., 2025): 通用生物医学 AI Agent
- Positional Bias (Zheng et al., 2023; Pezeshkpour et al., 2024)
- Distracting Retrieval Context (Amiraz et al., 2025; Shi et al., 2023)


![](Pasted%20image%2020260609192452.png)

- 关键是时间感知能力。
