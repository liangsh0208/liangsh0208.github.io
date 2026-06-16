---
created: 2026-06-10
published: 2025-01-24
paper: https://arxiv.org/abs/2501.14249
code: https://lastexam.ai
authors: Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, et al. (1,119 authors); Center for AI Safety, Scale AI, and 500+ institutions
tags:
  - Benchmark
  - LLM-Evaluation
  - Multi-modal
  - Academic-Questions
  - Center-for-AI-Safety
  - Scale-AI
---

# Humanity's Last Exam (HLE)

## 一句话总结

Center for AI Safety 与 Scale AI 联合全球 1,000+ 学科专家构建了 HLE —— 一个包含 2,500 道高难度多模态学术问题的 benchmark；当前最强模型（o3-mini high）准确率仅 13.4%，校准误差高达 80%，揭示了前沿 LLM 在专家级知识前沿上仍有巨大鸿沟。

![](HLE_fig1_benchmarks.png)

> **Figure 1**: HLE 与现有 benchmark 饱和度的对比。在 MMLU 上已有模型接近满分（>90%），但在 HLE 上所有前沿模型的准确率均低于 15%，说明 HLE 有效区分了当前最先进的模型能力边界。

---

## 1. 研究背景与动机

### 1.1 现有 Benchmark 的饱和危机

追踪 LLM 能力进展的 benchmark 正在快速失效：

| Benchmark | 当前 SOTA 表现 | 状态 |
|-----------|---------------|------|
| MMLU | >90% | 基本饱和 |
| GPQA | ~60-70% | 接近饱和 |
| MATH | ~80-90% | 快速逼近天花板 |

> 当 benchmark 被刷到接近 100% 时，它就无法再指导模型研发方向。**模型越好，benchmark 越急需升级** —— 这是一个永恒的评估军备竞赛。

### 1.2 HLE 的核心定位

HLE 的三重设计目标：

1. **高难度**：每道题必须能难倒当前前沿 LLM（GPT-4o、o1 等）
2. **广覆盖**：横跨数学、人文、自然科学等上百个学科
3. **可验证**：有明确、唯一、易于自动验证的正确答案

> 与开放式生成评测（如 SWE-bench、Agent 评测）不同，HLE 专注于 **closed-ended academic questions** —— 这类问题有客观正误，能更精确地测量模型的"知识边界"和"推理深度"。

---

## 2. 数据集构建

### 2.1 全球协作网络

HLE 是人类历史上规模最大的学术众测项目之一：

| 指标 | 数据 |
|------|------|
| 贡献专家 | 近 1,000 人 |
| 机构覆盖 | 500+ 所大学/研究机构 |
| 国家覆盖 | 50 个 |
| 专家背景 | 教授、研究人员、博士生为主 |

**激励机制**：设立 **$500,000** 奖金池，前 50 名入选问题各奖励 $5,000，接下来 500 名各 $500。此外，所有被接受问题的提交者获得论文合著权。

### 2.2 问题格式

| 格式 | 占比 | 说明 |
|------|------|------|
| Exact-match | ~76% | 模型输出精确字符串答案 |
| Multiple-choice | ~24% | 5 个以上选项，需选出正确项 |
| **Multi-modal** | **~14%** | 需同时理解文字和图像 |

> v1 版本中曾提到 80% exact-match / 10% 多模态，v10 的最终版调整为：约 14% 多模态、24% 多选、其余精确匹配。这反映了在数据优化过程中对多样性和难度的平衡调整。

### 2.3 数据创建流水线（Figure 4）

![](HLE_fig4_pipeline.png)

> **Figure 4**: HLE 的数据创建流程：
> 1. **70,000 次尝试** —— 领先 LLM 自动测试每道提交题
> 2. **LLM 难度检查** —— 只有能难住前沿模型的问题才进入下一阶段
> 3. **13,000 项提交** —— 来自全球专家的原始问题
> 4. **专家评审与改进** —— 多轮学术评审，迭代精炼问题
> 5. **6,000 项候选** —— 高质量候选问题
> 6. **组织者与专家评审审核** —— 最终把关
> 7. **2,500 公开集 + 私有保留集** —— 防 overfitting

**关键筛选标准**：
- 精确（precise）、无歧义（unambiguous）、可解（solvable）
- **不可搜索**（non-searchable）：不能通过简单互联网检索快速找到答案
- 原创或基于现有信息的非平凡综合
- 排除开放式问题、主观解释、大规模杀伤性武器相关内容

### 2.4 质量控制：专家评审与分歧率

**双轮评审机制**：

| 轮次 | 内容 | 参与人数 |
|------|------|---------|
| Round 1 | 学科专家评审，标准化打分，逐条反馈 | 数百人 |
| Round 2 | 组织者与精选评审员精选最佳问题 | 数十人 |
| Audit | 招募美国顶尖大学学生完全求解样本问题 | - |

**专家分歧率**：
- 公开集整体：**15.4%**
- 生物/化学/健康子集：**~18%**

> 这个分歧率与复杂医疗 benchmark（如 HealthBench）中的专家分歧率相当，说明 HLE 问题确实处于**人类知识的不确定性前沿**。

### 2.5 学科分布

![](HLE_fig3_subjects.png)

> **Figure 3**: HLE 的学科分布。数学独占 41%，生物学/医学 11%，计算机科学/AI 10%，人文/社会科学 9%，物理 9%，化学 7%，工程 4%，其他 9%。覆盖超过 **100 个细分学科**。

---

## 3. 实验结果

### 3.1 主实验：各模型表现

前沿模型在 HLE 上的**准确率极低**：

| 模型 | 准确率 (%) ↑ | RMS 校准误差 (%) ↓ |
|------|------------|-----------------|
| GPT-4o | 2.7 | 89 |
| Grok 2 | 3.0 | 87 |
| Claude 3.5 Sonnet | 4.1 | 84 |
| Gemini 1.5 Pro | 4.6 | 88 |
| Gemini 2.0 Flash Thinking | 6.6 | 82 |
| o1 | 8.0 | 83 |
| DeepSeek-R1 | 8.5 | 73 |
| **o3-mini (high)** | **13.4** | **80** |

**纯文本子集对比**（验证多模态对难度的影响）：

| 模型 | 纯文本准确率 (%) | 完整集准确率 (%) | 差异 |
|------|----------------|---------------|------|
| GPT-4o | 2.3 | 2.7 | +0.4 |
| Claude 3.5 Sonnet | 4.3 | 4.1 | -0.2 |
| o1 | 7.8 | 8.0 | +0.2 |
| o3-mini (high) | 13.4 | 13.4 | 0 |

> 去掉多模态题目后，**多数模型表现几乎没有提升**，说明 HLE 的难度主要来自于**问题本身的深度和知识门槛**，而非模态融合。纯文本 HLE 就足以难倒所有前沿模型。

### 3.2 校准分析：模型不仅答错，还过度自信

校准误差的衡量方式：让模型在给出答案时同时报告 **0%-100% 的置信度**。理想校准下，声称 50% 置信度的题目应有 50% 的实际准确率。

然而，所有模型的 **RMS 校准误差都在 73%-89% 之间** —— 这意味着：
- 模型经常**以高置信度输出错误答案**
- 模型无法识别自己在哪些题目上"不懂"
- 这种 "confabulation/hallucination combined with overconfidence" 是当前的普遍现象

> DeepSeek-R1 的校准误差最低（73%），o3-mini 的校准误差为 80%。而 GPT-4o 的校准误差高达 89%，说明它最不能正确评估自己的能力边界。

### 3.3 分学科表现（Table 3）

纯文本子集上各学科的 top 模型（o3-mini high）表现：

| 学科 | 准确率 (%) |
|------|-----------|
| 数学 | **18.6** |
| 物理 | **15.3** |
| 生物/医学 | 10.0 |
| 化学 | 9.0 |
| 计算机/AI | 8.4 |
| 人文/社科 | 5.2 |
| 工程 | 6.5 |
| 其他 | 6.9 |

完整数据集上 o1 的表现：

| 学科 | 准确率 (%) |
|------|-----------|
| 人文/社科 | **8.7** |
| 化学 | 9.7 |
| 物理 | 7.0 |
| 计算机/AI | 8.2 |
| 数学 | 7.4 |
| 生物/医学 | 10.4 |
| 工程 | 6.3 |

**关键发现**：
- **数学和物理**是推理模型（o3-mini/o1）相对最强的领域，可能与强化学习训练中的 STEM 侧重有关
- **人文/社科**在多模态完整数据集中反而得分不低（o1 达 8.7%），但在纯文本子集中表现最差（o3-mini 仅 5.2%）
- **生物/医学**在多模态数据中得分较高（10.4%），可能受益于大量生物医学训练数据

### 3.4 Token 消耗分析

![](HLE_fig5_reasoning_tokens.png)

> **Figure 5**: 推理模型（o1, DeepSeek-R1, Gemini 2.0 Flash Thinking）的平均 completion token 数。数学和物理领域的 token 消耗最大，说明模型在这些学科上"思考"最多。

![](HLE_fig6_nonreasoning_tokens.png)

> **Figure 6**: 非推理模型（GPT-4o, Grok 2, Claude 3.5 Sonnet, Gemini 1.5 Pro）的平均 output token 数。相比推理模型，非推理模型的输出 token 数显著更少，但准确率也更低。

**关键观察**：
- 推理模型在所有学科上的 token 消耗都显著高于非推理模型
- 数学和工程学科的 token 消耗最高 —— 这是模型"深度思考"最集中的领域
- 未来模型不仅要提升准确率，还应追求 **compute-optimal**（在更少的推理预算内获得更高的正确率）

### 3.5 HLE 的"不饱和"设计

> "Few problems in HLE seem unsolved... but also few problems are already solved by all models..."

- **没有"所有模型都解决"的问题** —— 说明 benchmark 远未饱和
- **存在所有模型都失败的题目** —— 验证了其有效区分度
- HLE 公开集 + 私有保留集的分离设计，防止模型"刷题"和 overfitting

---

## 4. 讨论

### 4.1 局限性与挑战

**1. 数据审查的人力投入巨大**
- 耗时几个月、涉及 1,119 名作者的人力投入
- 15.4%-18% 的专家分歧率说明问题本身存在合理的主观性
- 不适用于快速迭代场景

**2. 反搜索设计的双刃剑**
- HLE 强调问题不能被简单网络搜索到
- 但随着搜索引擎（如 Perplexity）与 LLM 的深度融合，"不可搜索"的边界在模糊化
- 论文在 B.2 节专门审计了"潜在可搜索"问题并移除了部分

**3. 封闭问答 ≠ 真正智能**
- HLE 是**最后一道封闭学术考试**，但不是 AI 的"终考"
- 作者明确承认："HLE may be the last academic exam we need to give to models, but it is far from the last benchmark for AI"
- 开放式研究能力、创造力、社会推理等仍需要其他 benchmark 衡量

**4. 化学、生物问题的潜在答案错误率**
- 外部审计发现约 **30%** 的化学/生物答案可能错误（Future House Research，见参考文献 [47]）
- 这在科学与医学类 benchmark 中是系统性风险

### 4.2 未来展望

**HLE-Rolling 动态更新**：
- 随着模型能力提升，HLE 原始数据集会被逐渐攻克
- HLE-Rolling 将持续接收新的高难度问题
- 目标是在前沿模型 hit the ceiling 时无缝迁移到新版本

**预测**：论文推测到 **2025 年底**，模型可能在 HLE 上达到 **50%+ 准确率**。如果成真，将意味着模型在封闭学术问题上达到了专家级水平。

---

## 5. 个人思考

### 5.1 对 Benchmark 设计的启发

HLE 最让我钦佩的设计有以下几点：

1. **"LLM-first difficulty check"**：在专家评审之前，先用前沿 LLM 测试问题的难度，只有 LLM 做不出来的问题才进入人工流程。这比传统 benchmark 的"人工出题 → 人工审核 → 模型测试"流程更高效，也更贴合实际评估目标。

2. **"不可搜索"作为核心质量标准**：在一个 AI + 搜索高度融合的时代，"不能通过简单 Google 找到答案"是一个极有价值的筛选条件。它迫使问题设计者基于前沿研究或深度专业知识出题，而非汇编已有知识。

3. **公开集 + 私有保留集** 的架构是防止 overfitting 的黄金标准。相比某些 benchmark 的"全公开"策略，HLE 的这一设计让声称的得分更具可信度。

### 5.2 对模型能力评估的反思

当前主流 benchmark（MMLU、GSM8K、HumanEval）已经被刷爆，模型厂商的" leaderboard 数字"越来越失去说服力。HLE 的出现提醒我们：

- **准确率 <15% 的 benchmark 才是好 benchmark** —— 它留出了足够的能力梯度空间
- **校准误差比准确率更可怕** —— 模型不知道"自己不知道"，这对医疗、法律等高风险应用是致命缺陷
- **数学/物理不是全部** —— 人文社科类问题的低准确率说明，跨领域的泛化能力仍是短板

### 5.3 关于"Humanity's Last Exam"名字的争议

HLE 的名字带有一定夸张和宣言性质。作为一个 benchmark 论文，它确实：
- **不是**"最后的"benchmark（作者自己也承认）
- **是** 当前"封闭学术问答"类别中最难、最全面的 benchmark
- 其真正价值不在于"难倒模型"，而在于建立了一个**持续追踪专家级知识边界的基准**

### 5.4 与当前研究的关联

HLE 与 HealthBench、SWE-bench、Agentic RL benchmark 等共同构成了 LLM 评估的新范式：**从"能做什么"到"在哪里失败"**。只有当 benchmark 能指示模型的明确能力天花板时，它才能真正指导研究方向。HLE 让我重新思考：在自己的研究领域中，是否也需要建立类似"专家知识前沿"的评估标准？

---

## 6. 关键引用

```bibtex
@article{phan2025hle,
  title={Humanity's Last Exam},
  author={Phan, Long and Gatti, Alice and Han, Ziwen and Li, Nathaniel and Hu, Josephina and Zhang, Hugh and Zhang, Chen Bo Calvin and Shaaban, Mohamed and Ling, John and Shi, Sean and Choi, Michael and Agrawal, Anish and Chopra, Arnav and Khoja, Adam and Kim, Ryan and Ren, Richard and Hausenloy, Jason and Zhang, Oliver and Mazeika, Mantas and Yue, Summer and Wang, Alexandr and Hendrycks, Dan and et al.},
  journal={arXiv preprint arXiv:2501.14249},
  year={2025},
  url={https://arxiv.org/abs/2501.14249},
  code={https://lastexam.ai}
}
```

Nature 版本引用（如适用）：
```bibtex
@article{phan2026hle,
  title={A benchmark of expert-level academic questions to assess AI capabilities},
  author={{Center for AI Safety} and {Scale AI} and {HLE Contributors Consortium}},
  journal={Nature},
  volume={649},
  pages={1139--1146},
  year={2026},
  doi={10.1038/s41586-025-09962-4}
}
```
