---
created: 2026-06-09
paper: https://arxiv.org/abs/2506.04178
code: https://openthoughts.ai
authors: Etash Guha*, Ryan Marten*, Sedrick Keh*, Negin Raoof*, Georgios Smyrnis*, Hritik Bansal†, Marianna Nezhurina†, Jean Mercat†, Trung Vu†, Zayne Sprague†, et al. (Stanford, UW, BespokeLabs, Toyota Research, UC Berkeley, UT Austin, UCLA, NYU, UNC, ASU, etc.)
published: 2025-06-04
tags:
  - reasoning-model
  - distillation
  - data-curation
  - SFT
  - open-source
---

# OpenThoughts: Data Recipes for Reasoning Models

## 一句话总结

OpenThoughts 项目通过**1000+ 受控实验系统拆解了推理模型 SFT 数据构建的每一步**，发现了五条反直觉的"数据配方"——最重要的是：**更弱的教师模型（QwQ-32B）反而比更强的 DeepSeek-R1 更适合蒸馏；每道题采样 16 个答案比扩展题目来源更有效。** 基于这些发现构建的 OpenThoughts3-1.2M 数据集训练出的 OpenThinker3-7B，在 AIME25 / LiveCodeBench / GPQA 上分别取得 **53% / 51% / 54%**，超越 DeepSeek-R1-Distill-Qwen-7B 达 **15~20 个百分点**。

![](OpenThoughts_x1.png)

> **Figure 1**: OpenThoughts3 与现有开源推理数据集在不同数据规模下的性能对比。红线（OpenThoughts3）在 AIME 2025、LiveCodeBench、GPQA Diamond 三个 benchmark 上的 scaling 曲线均显著优于其他数据集（Nemotron Nano、LIMO、s1.1、AM）。横轴为数据集大小（1K → 1M），纵轴为准确率（%）。

---

## 1. 研究背景与动机

### 1.1 问题定义

当前前沿推理模型（DeepSeek-R1、o3、Gemini 等）的训练 recipe 高度依赖**专有数据集**，公开信息极少。这使得开源社区难以复现和理解推理模型的构建过程。虽然 SFT（监督微调）已被证明是构建推理模型的有效手段（如 R1-Distill 系列），但大多数项目：
<font color="#ff0000">- 只探索了有限的设计空间（如仅使用人工编写问题或单一教师模型）</font>
<font color="#ff0000">- 缺乏系统性的消融实验来验证每个数据构建步骤的影响</font>

### 1.2 现有不足

| 现有数据集 | 局限 |
|-----------|------|
| **OpenR1 / LIMO / s1.1** | 规模小（<100K），设计空间探索不完整 |
| **Nvidia Nemotron** | 未公开完整的 pipeline 设计决策 |
| **DeepMath-103K / Skywork-OR1** | 局限于单领域（数学或代码） |
| **AceReason / Natural Reasoning** | 未系统对比不同问题来源/过滤策略的影响 |

### 1.3 核心贡献

1. **首个系统拆解推理 SFT 数据构建全流程**的项目：通过 1000+ 受控实验，逐一检验问题来源、混合策略、过滤方法、去重、答案过滤、教师模型选择等每一步
2. **五条反直觉发现**（见下方「关键发现」），彻底挑战了关于数据质量的既有认知
3. **OpenThoughts3-1.2M**：当前最强开源推理数据集，覆盖数学（850K）、代码（250K）、科学（100K）
4. **OpenThinker3-7B**：首个在公开数据上训练的 7B 推理模型，全面超越 DeepSeek-R1-Distill-Qwen-7B

---

## 2. 关键发现（Top 5 Takeaways）

本文通过 1000+ 受控实验（每个实验固定 31,600 数据点，训练 Qwen2.5-7B-Instruct）得出了五条核心结论：

### 🔍 Finding 1：每道题多采样答案 > 扩展题目来源

> **每道题采样 16 个答案**（16× answers per question）是将数据源扩展至少 **16 倍**的有效技术，性能与使用更多不重复题目相当甚至更优。

| 策略 | Avg | 直觉 |
|------|-----|------|
| Exact Dedup + 16× | **36.2** | 少题多答，答案多样性带来增益 |
| No Dedup + 4× | 35.8 | 多题少答，但题目质量参差不齐 |
| No Dedup + 1× | 35.5 | 基线 |

### 🔍 Finding 2：更弱的教师 ≠ 更差的蒸馏效果

> **QwQ-32B 是比 DeepSeek-R1 更强的教师模型**，尽管 QwQ 在目标 benchmark 上的直接得分低于 DeepSeek-R1。

| Teacher (Code) | Avg | CodeElo | 直接得分 vs R1 |
|---------------|-----|---------|---------------|
| **QwQ-32B** | **44.2** | **29.5** | R1 直接得分 > QwQ by 9% |
| DeepSeek-R1 | 42.3 | 27.2 | QwQ 蒸馏效果更好 |

| Teacher (Math) | Avg | MATH500 | 直接得分 vs R1 |
|---------------|-----|---------|---------------|
| **QwQ-32B** | **44.2** | **71.6** | R1 蒸馏仅 41.6 |
| DeepSeek-R1 | 41.6 | 64.8 | — |

**关键洞察**：教师模型在推理 benchmark 上的绝对性能 ≠ 蒸馏后学生的性能。QwQ 的推理 trace 可能更结构化、更可迁移。

### 🔍 Finding 3：答案过滤几乎无用

> 实验了多种答案过滤方法（随机过滤、最短答案、去除非英语、GPT 验证等），**没有任何方法显著优于"不过滤"基线**。

| Math Answer Filtering | Avg |
|----------------------|-----|
| No Filtering | **41.9** |
| Random Filtering | 41.6 |
| Shortest Answers | 41.1 |
| GPT Verification | 40.0 |
| Removing Long Paragraphs | 38.0 |

**Takeaway**：教师模型生成的所有答案（无论长短、语言）都包含有价值的推理信号。

### 🔍 Finding 4：少而精的问题来源 > 多样但杂的来源

> 从 **Top 1~2 个高质量来源**混合问题，比从 Top 8~16 个来源混合获得更好的下游性能。

| Code Mixing | Avg | 下降幅度 |
|------------|-----|---------|
| Top 1 | 39.9 | — |
| Top 2 | **41.3** | 最优 |
| Top 4 | 38.6 | -2.7 |
| Top 8 | 37.0 | -4.3 |
| Top 16 | 36.4 | -4.9 |

Top 2 vs Top 16：**5% 准确率差距**。引入低质量来源稀释了数据集的整体质量。

### 🔍 Finding 5：LLM-based 过滤 > 传统预训练过滤方法

> 使用 LLM 标注的难度或回答长度来选择问题，显著优于基于 embedding 或 fastText 的经典预训练数据过滤方法。

| Math Filtering | Avg | vs fastText 提升 |
|---------------|-----|------------------|
| Response Length (GPT-4.1-mini) | **41.9** | **+6.3** |
| fastText (P:Numina; N:Lap1official) | 35.6 | — |

| Code Filtering | Avg | vs AskLLM 提升 |
|---------------|-----|-----------------|
| Difficulty-based (GPT-4o-mini) | **43.0** | **+1.4** |
| AskLLM Selection | 41.6 | — |

---

## 3. OpenThoughts3 数据 Pipeline

![](OpenThoughts_x2.png)

> **Figure 2**: OpenThoughts3 数据构建六步流水线：(1) 问题来源 → (2) 问题混合 → (3) 问题过滤 → (4) 去重与多答案采样 → (5) 答案过滤 → (6) 教师模型选择。

### 3.1 问题来源（Question Sourcing）

问题来源分为三类：
- **完全合成**（Fully synthetic）：LLM 生成的问题
- **半合成**（Semi-synthetic）：从现有数据改写/扩展
- **非合成**（Non-synthetic）：人类编写的原始问题

**各域名 Top 3 来源**：

| 领域 | Top 1 来源 | Top 2 来源 | Top 3 来源 |
|------|-----------|-----------|-----------|
| **Code** | StackExchange-CodeGolf (38.8) | OpenCodeReasoning (38.4) | KodCode-V1 (37.7) |
| **Math** | OpenMath-2-Math (38.1) | NuminaMath-1.5 (37.4) | MathPile (36.2) |
| **Science** | StackExchange-Physics (34.3) | OrganicChemistry-PDF (34.0) | CQADupStack-Physics (33.3) |

> 来源间最大差距达 17.2 个百分点（code 领域最好 vs 最差）。

### 3.2 问题混合（Mixing Questions）

**最终选择**：
- **Math**：OpenMath-2-Math（单一来源）
- **Code**：CodeGolf + OpenCodeReasoning（2 个来源）
- **Science**：StackExchange-Physics + OrganicChemistry-PDFs（2 个来源）

### 3.3 问题过滤（Question Filtering）

**最终选择**：
- **Code**：Difficulty-based filtering with GPT-4o-mini
- **Math & Science**：Response length filtering with GPT-4.1-mini

### 3.4 去重与多答案采样（Deduplication & Multiple Answers）

**最终选择**：
- **所有领域**：每题采样 **16 个答案**
- **Math & Science**：精确去重（Exact dedup）
- **Code**：不去重（No dedup）——代码问题的重复对性能影响很小

### 3.5 答案过滤（Answer Filtering）

**最终选择**：**不执行答案过滤**（No filtering）——所有答案都被保留。

### 3.6 教师模型选择（Teacher Model）

**最终选择**：**QwQ-32B** 作为教师模型，用于所有领域。

### 3.7 去污染（Decontamination）

![](OpenThoughts_x6.png)

> **Figure 3（补充）**: 去污染结果混淆矩阵。2958 个真实无泄漏样本被正确识别，3080 个真实泄漏样本也被正确识别。仅 42 个假阳性（误判为泄漏）和 12 个假阴性（漏判）——去污染精度极高。

---

## 4. Scaling 到 1.2M

### 4.1 数据量 Scaling

OpenThoughts3-1.2M 的组成：
- **Math**：850,000 条
- **Code**：250,000 条
- **Science**：100,000 条

### 4.2 最终模型性能

![](OpenThoughts_x3.png)

> **Figure 3**: 各 pipeline 步骤的最佳策略随数据量 scaling 的性能曲线。横轴为数据量（1K → 1M），纵轴为准确率。不同颜色代表不同步骤的 top 策略，红线（综合所有最优策略）持续领先。

**OpenThinker3-7B vs DeepSeek-R1-Distill-Qwen-7B**：

| Benchmark | OpenThinker3-7B | DS-R1-Qwen-7B | Δ |
|-----------|----------------|---------------|-----|
| AIME24 | **69.0** | 51.3 | **+17.7** |
| AIME25 | **53.3** | 38.0 | **+15.3** |
| AMC23 | **93.5** | 92.0 | +1.5 |
| MATH500 | **90.0** | 88.0 | +2.0 |
| CodeElo | **31.0** | 19.9 | **+11.1** |
| LCB 05/23-05/24 | **64.5** | 48.7 | **+15.8** |
| LCB 06/24-01/25 | **51.7** | 34.5 | **+17.2** |
| CodeForces | **32.2** | 21.1 | **+11.1** |
| GPQA-D | **53.7** | 33.2 | **+20.5** |
| JEEBench | **72.4** | 50.4 | **+22.0** |
| HMMT 02/25 | **42.7** | 25.0 | **+17.7** |
| HLE MCQ | 10.2 | **12.4** | -2.2 |
| **Average** | **55.3** | **42.9** | **+12.4** |

**关键结论**：
- OpenThinker3-7B 是 **7B 级别最强开源数据推理模型**
- 平均提升 **12.4 个百分点**，在 GPQA 和 JEEBench 上提升超过 **20 个百分点**
- 仅在 HLE MCQ（Humanity's Last Exam）上略低（-2.2），可能因为 HLE 需要极广泛的知识覆盖

---

## 5. 与相关工作的对比

| 数据集 | 规模 | 公开 recipe | 系统消融 | 7B 模型 AIME24 |
|--------|------|------------|---------|---------------|
| **OpenThoughts3** | **1.2M** | ✅ 完整 | ✅ 1000+ 实验 | **69.0** |
| DeepSeek-R1-Distill | 未知 | ❌ 专有 | ❌ | 51.3 |
| OpenR1 | ~100K | ✅ | ❌ | ~55 |
| LIMO | ~100K | ✅ | ❌ | ~55 |
| s1.1 | ~100K | ✅ | ❌ | ~60 |
| Nemotron Nano | 1M | ❌ | ❌ | ~50 |
| OpenThinker2-32B | 1M | ✅ | ⚠️ 部分 | 60.7 |

> OpenThinker2-32B 是首个在公开数据上训练并匹配 DeepSeek-R1-Distill-32B 的模型；OpenThinker3-7B 在此基础上进一步缩小了 7B 级别的差距。

---

## 6. 局限性与未来方向

### 6.1 本文局限性
1. **未探索 RL**：仅使用 SFT，未对比 RL（如 PPO、DPO、GRPO）的效果
2. **未探索 curriculum learning / staged SFT**：所有数据一次性混合训练，未探索分阶段训练策略
3. **教师模型固定**：主要对比了 QwQ-32B、DeepSeek-R1 和 Phi-4，未覆盖更多教师模型
4. **答案过滤策略有限**：虽然测试了 10+ 种方法，但未探索如"基于 reasoning trace 质量"的高级过滤

### 6.2 未来方向
1. **RL 数据构建**：将 OpenThoughts 数据用于 RL 训练（GRPO/DPO）
2. **跨领域迁移**：探索数学/代码/科学之间的跨领域知识迁移
3. **学生接近教师时的 scaling**：探索当学生性能接近教师时，数据增量的边际收益
4. **问题多样性 vs 答案多样性的交互**：进一步理解两者对最终性能的贡献
5. **课程学习**：按难度从易到难的 staged SFT

---

## 7. 个人思考

### 7.1 最反直觉的发现：QwQ-32B vs DeepSeek-R1

这条发现对我冲击最大：**更强的教师模型不一定是更好的蒸馏来源。**

QwQ-32B 在 CodeElo 上比 DeepSeek-R1 直接低 9%，在 GPQA 上低 8%，在 JEEBench 上低 23%——但蒸馏到 7B 学生上，QwQ 的学生全面超越 R1 的学生。

**可能的解释**：
- QwQ 的 reasoning trace 可能更结构化、更可迁移
- DeepSeek-R1 的答案可能包含更多"内部知识"（即依赖大模型参数记忆而非显式推理），这部分知识小模型无法继承
- QwQ 可能在"教会学生如何思考"方面做得更好，而不是展示"我能做什么"

> 这与教育心理学中的"专家盲点"（expert blind spot）类似：顶尖专家往往不善于教学，因为他们已经内化了太多中间步骤，无法清晰地外化推理过程。

### 7.2 数据配方的工程智慧

本文最大的价值不是「某个具体数字」，而是**系统拆解了数据构建的每一步**并给出了可复现的 recipe：

| Pipeline 步骤 | 关键决策 |
|-------------|---------|
| 来源选择 | 少而精（Top 1~2），拒绝"多而杂" |
| 问题过滤 | LLM 标注的难度/长度 > embedding/fastText |
| 答案生成 | 16× 答案 per 题 > 扩展题目来源 |
| 答案过滤 | 不过滤 > 任何过滤策略 |
| 教师选择 | QwQ-32B > DeepSeek-R1（反直觉） |

这些决策每一条都足以让数据科学家少走弯路，合在一起构成了**首个可复现的推理模型 SFT 最佳实践**。

### 7.3 对当前实践的启示

1. **数据构建优先级**：在做推理模型 SFT 时，应该花 80% 的时间在「问题来源筛选 + 教师模型选择」上，而非「答案过滤和格式清洗」上
2. **教师模型选择的建议**：不要只看 benchmark 排行榜选教师，而要做小规模蒸馏实验对比不同教师的实际效果
3. **Scaling 策略**：在数据有限时，优先对每个问题采样更多答案，而非扩展更多问题来源

---

## 8. 关键引用

```bibtex
@article{guha2025openthoughts,
  title={Data Recipes for Reasoning Models},
  author={Guha, Etash and Marten, Ryan and Keh, Sedrick and Raoof, Negin and Smyrnis, Georgios and Bansal, Hritik and Nezhurina, Marianna and Mercat, Jean and Vu, Trung and Sprague, Zayne and Suvarna, Ashima and Feuer, Benjamin and Chen, Liangyu and Khan, Zaid and Frankel, Eric and Grover, Sachin and Choi, Caroline and Muennighoff, Niklas and Su, Shiye and Zhao, Wanjia and Yang, John and Pimpalgaonkar, Shreyas and Sharma, Kartik and Ji, Charlie Cheng-Jie and Deng, Yichuan and Pratt, Sarah and Ramanujan, Vivek and Saad-Falcon, Jon and Li, Jeffrey and Dave, Achal and Albalak, Alon and Arora, Kushal and Wulfe, Blake and Hegde, Chinmay and Durrett, Greg and Oh, Sewoong and Bansal, Mohit and Gabriel, Saadia and Grover, Aditya and Chang, Kai-Wei and Shankar, Vaishaal and Gokaslan, Aaron and Merrill, Mike A and Hashimoto, Tatsunori and Choi, Yejin and Jitsev, Jenia and Heckel, Reinhard and Sathiamoorthy, Maheswan and Dimakis, Alexandros G and Schmidt, Ludwig},
  journal={arXiv preprint arXiv:2506.04178},
  year={2025}
}
```

---

**项目资源**：
- 数据集和模型：https://openthoughts.ai
- HuggingFace：https://huggingface.co/open-thoughts/OpenThinker3-7B
- GitHub：https://github.com/open-thoughts

**相关论文**：
- [EvalTree](EvalTree--Profiling%20Language%20Model%20Weaknesses%20via%20Hierarchical%20Capability%20Trees.md) — 推理模型评估中的弱点定位
- [SkillVerse](SkillVerse--Assessing-and-Enhancing-LLMs-with-Tree-Evaluation.md) — 推理能力的树形评估
- [QualEval](QualEval--Qualitative-Evaluation-for-Model-Improvement.md) — 定性评估框架
