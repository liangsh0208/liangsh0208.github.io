---
created: 2026-06-10
published: 2024
paper: https://arxiv.org/abs/2406.16714
code: https://github.com/thu-coai/AutoDetect
authors: Jiale Cheng, Yida Lu, Xiaotao Gu, Pei Ke, Xiao Liu, Yuxiao Dong, Hongning Wang, Jie Tang, Minlie Huang (Tsinghua University, Zhipu AI)
tags:
  - LLM-Safety
  - Red-Teaming
  - Model-Evaluation
  - Multi-Agent
  - EMNLP2024
---

# AutoDetect: Towards a Unified Framework for Automated Weakness Detection in Large Language Models

## 一句话总结

AutoDetect 是一个受教育体系启发的自动化 LLM 弱点发现框架，通过 Examiner、Questioner、Assessor 三个 LLM Agent 的协作循环，在指令遵循、数学推理和代码生成三大通用任务上实现了超过 **50%** 的弱点识别成功率（ISR），并利用发现的数据对 Mistral-7B 和 Llama 系列进行针对性微调，在多个基准上带来超过 **10%** 的性能提升。

![](AutoDetect_fig1_isr.png)

> **Figure 1**: AutoDetect 在指令遵循、数学和代码三大任务上的识别成功率（ISR）以及在多个基准上的模型增强效果。最强模型（如 ChatGPT）的 ISR 超过 30%，较弱的 Llama2-7B-Chat 在代码任务上 ISR 高达 89.8%。利用 AutoDetect 发现的数据对 Llama2-7B-Chat 进行微调后，IFEval-p 从 32.3% 提升至 42.5%，HumanEval 从 13.4% 提升至 18.7%。

---

## 1. 研究背景与动机

### 1.1 问题定义

如何自动化、系统化地识别单个 LLM 在通用任务中的**个性化弱点**，并利用这些发现指导模型改进？

### 1.2 现有方法的不足

**传统基准测试（Benchmarking）**
- 设计目标是**比较排名**不同模型，而非针对单个模型发现其特异性缺陷
- 静态基准存在数据泄露、Leaderboard 饱和等问题
- 动态基准覆盖范围有限，构造方法缺乏通用性

**人工检查（Manual Inspection / Red Teaming）**
- 劳动密集、成本高昂，难以规模化
- 早期的人工 Red Teaming（如 Dinan et al., 2019）已被证明效率低下
- 自动 Red Teaming（如 Perez et al., 2022）多用于安全领域，对通用任务的覆盖不足

**数据增强方法（如 Self-Instruct）**
- 无针对性扩充训练数据，无法精准修复模型弱项

---

## 2. 方法

### 2.1 核心思想

AutoDetect 借鉴了**教育评估体系**的设计哲学：像老师出考题检查学生学习效果一样，为 LLM 设计考题、评分并找出薄弱知识点，然后针对性补强。框架包含**三个 LLM Agent** 和**两个循环**：

![](AutoDetect_fig2_framework.png)

> **Figure 2**: AutoDetect 框架流程。包含两个循环：(1)  Examiner → Questioner → Assessor 的主循环，负责构建 taxonomy、生成考题、分析弱点；(2) Questioner 内部的迭代搜索循环，通过历史低分样本不断生成更难的考题。

### 2.2 三个智能体（Agents）

| Agent          | 职责                          | 核心能力                                  |
| -------------- | --------------------------- | ------------------------------------- |
| **Examiner**   | 构建任务分类体系（Taxonomy），动态优化测试框架 | 将任务 (Task, Description) 分解为层次化的类别和知识点 |
| **Questioner** | 根据每个知识点生成考题                 | 在迭代搜索中基于历史答题记录生成越来越难的问题               |
| **Assessor**   | 分析目标模型低分案例                  | 推测新颖潜在弱点，引导 Examiner 扩展 taxonomy      |

### 2.3 形式化定义

**输入**：任务描述对 $(\mathcal{T}, \mathcal{D})$  
**输出**：模型弱点集合 $\mathcal{W}$

$$
\boxed{ \mathcal{W} = \textsc{AutoDetect}(\mathcal{T}, \mathcal{D}) }
$$

内部三个 Agent 的运作形式化如下：

$$
\mathcal{C} = \text{Examiner}(\mathcal{T}, \mathcal{D}) \quad \text{(构建分类体系)} \\
\mathcal{Q} = \text{Questioner}(\mathcal{H}) \quad \text{(基于历史生成新题)} \\
k_{\text{new}} = \text{Assessor}(\mathcal{H}_{\text{low}}) \quad \text{(从低分案例发现新弱点)}
$$

其中 $\mathcal{H}$ 为搜索历史，$\mathcal{H}_{\text{low}}$ 为得分低于阈值的历史样本集。

### 2.4 迭代搜索（Iterative Search）

1. **初始化**：每个知识点生成 5 道 seed 问题
2. **答题与评分**：目标模型回答，使用 GPT-4 作为参考评分器（MT-bench 评分提示）
3. **迭代生成**：Questioner 依据低分历史样本生成新的更有挑战性的问题
4. **一轮终止条件**：达到预设迭代步数（实验中为 10 步）
5. **循环更新**：Assessor 分析低分案例 → 发现新弱点 → 反馈给 Examiner 补充 taxonomy → 下一轮循环

> **评分机制**：采用 MT-bench 的 1-10 分制评分提示。要求 LLM 评分器在答案错误时**不得超过 3 分**。实验证实人机一致率达 **88.7%**。

### 2.5 关键设计选择

| 设计 | 理由 |
|------|------|
| 动态 taxonomy 而非静态题库 | 根据目标模型表现自适应调整测试重点，实现针对性评估 |
| 迭代搜索而非一次性生成 | 类似于对抗测试中的渐进式探针，从易到难逐步暴露弱点 |
| GPT-4 作为评分参考 | 利用强模型的高质量答案和评分能力确保评估可靠性 |
| 三大 Agent 协作 | Examiner 保证广度，Questioner 保证深度，Assessor 保证发现新方向 |

### 2.6 模型增强（Model Enhancement）

发现弱点后将问题和参考答案用于微调：

$$
\mathcal{L} = -\frac{1}{N} \sum_{t=1}^{N} \log P(R_t \mid Q, R_{<t})
$$

实际采用 **LoRA**（rank=8）+ **DeepSpeed Zero-2** + AdamW 对发现的问题数据进行微调，所有模型训练 5 epoch，学习率 2e-5，batch size=4。

---

## 3. 评估指标

### 3.1 ISR（Identification Success Rate）

$$
\text{ISR} = \frac{\text{Num}_{<4}}{\text{Num}_{\text{total}}}
$$

- $\text{Num}_{<4}$：评分 ≤3（即模型答错）的样本数
- $\text{Num}_{\text{total}}$：总评估样本数
- 取 3 为分界线的依据：MT-bench 评分实验中，GPT-4 在答案明确错误时极少给出超过 3 分的评分

### 3.2 人工评估

对 150 条样本（每项任务 50 条，跨所有模型）进行人工评估：

| 维度 | 合理率 | Fleiss Kappa（一致性） |
|------|--------|---------|
| 问题合理性（Reasonableness） | **98.0%** | 0.493 |
| 标签一致性（Agreement with GPT-4） | **88.7%** | 0.472 |
| 参考答案正确性（Correctness） | **87.3%** | 0.439 |

---

## 4. 实验结果

### 4.1 弱点识别（ISR）

#### 指令遵循任务

| 模型 | Format（格式约束） | General（通用） | **Overall** |
|------|----------------------|---------------|-----------|
| Llama2-7b-Chat | 55.3 | 37.8 | **43.3** |
| Mistral-7b-Instruct | 52.9 | 32.5 | **38.2** |
| GPT-3.5-turbo | 35.1 | 21.7 | **25.5** |
| Claude-3-Sonnet | 29.3 | 12.8 | **19.2** |

> **关键发现**：越弱的模型 ISR 越高（Llama2 达 43.3% 而 Claude 仅 19.2%），说明 AutoDetect 成功为不同能力水平的模型找到了与其匹配的弱点难度。

#### 数学任务

| 模型 | Geometry（几何） | Analysis（分析） | **Overall** |
|------|----------------|----------------|-----------|
| Llama2-7b-Chat | 89.8 | 93.3 | **88.8** |
| GPT-3.5-turbo | 56.3 | 35.6 | **50.2** |
| Llama3-70b-Instruct | 41.9 | 30.0 | **38.7** |

> **关键发现**：即使是 GPT-3.5-turbo 和 Claude-3-Sonnet 这类强模型，在数学任务上的 ISR 也超过 **40%**，暴露了传统基准（如 GSM8k 上已达很高准确率）未能发现的深层弱点。

#### 代码任务

| 模型 | Data Structure（数据结构） | Math & Algorithm（数算） | **Overall** |
|------|--------|--------|-----------|
| Llama2-7b-Chat | 83.3 | 81.1 | **74.8** |
| Claude-3-Sonnet | 37.3 | 32.0 | **29.9** |
| GLM-4-Air | 32.2 | 45.6 | **28.7** |

### 4.2 模型增强效果

![](AutoDetect_fig4_improve.png)

> **Figure 4**：使用 AutoDetect 发现的 Llama2-7B-Chat 自身弱点数据，比使用 GPT-3.5-turbo 的弱点数据进行训练，提升效果显著更大。这证明了**针对性弱点检测数据的优越性**。

| 模型 | IFEval-p | IFEval-i | GSM8k | MATH | HumanEval |
|------|----------|----------|-------|------|-----------|
| Llama2-7b-Chat | 32.3→**42.5** | 46.2→**54.7** | 18.9→**25.9** | 2.5→**4.7** | 13.4→**18.7** |
| Llama2-13b-Chat | 34.3→**43.3** | 45.8→**54.3** | 26.9→**33.7** | 3.9→**6.0** | 17.7→**24.4** |
| Llama2-70b-Chat | 44.2→**51.8** | 54.3→**63.5** | 51.9→**65.0** | 6.5→**12.6** | 31.7→**36.6** |
| Mistral-7b-Instruct | 51.2→**54.3** | 61.6→**64.7** | 42.9→**54.8** | 4.5→**12.6** | 32.9→**40.9** |

> **关键发现**：即使对于 Llama3-8B-Instruct（已很强：IFEval-p 70.1%），微调后仍有 2.5-5.5 个百分点的提升，说明 AutoDetect 发现的弱点是**真正的非饱和弱项**。

### 4.3 迭代搜索效果

![](AutoDetect_fig3_score_change.png)

> **Figure 3**：三项任务在迭代搜索过程中平均分数的下降趋势。得分随迭代轮次持续下降，说明 Questioner 成功生成越来越难的问题，有效暴露模型潜藏弱点。

#### 迭代分数变化（Llama2-13b-Chat / 指令遵循）

| 轮次 | IFEval-p | IFEval-i |
|------|----------|----------|
| Iter 0（原始） | 34.3 | 45.8 |
| Iter 1 | **43.3** (+9.0) | **54.3** (+8.5) |
| Iter 2 | **45.4** (+2.1) | **57.0** (+2.7) |
| Iter 3 | **47.1** (+1.7) | **58.2** (+1.2) |

> **关键发现**：三轮迭代可逐轮提升模型，每轮仍有非平凡的增益，展示了框架的**可扩展性**。

### 4.4 与基线对比

| 方法 | 指令遵循 ISR↑ | IFEval-p↑ | 数学 ISR↑ | GSM8k↑ | 代码 ISR↑ | HumanEval↑ | BLEU-4↓ |
|------|-----------|----------|---------|--------|----------|-----------|--------|
| Self-Instruct | 20.4 | 35.7 | 71.5 | 21.5 | 38.7 | 14.6 | 0.66 |
| OPRO | 72.9 | 34.8 | 93.2 | 21.3 | 95.1 | 14.0 | 0.48 |
| PAIR | 62.3 | 37.2 | 95.2 | 24.6 | 83.3 | 15.2 | 0.62 |
| **Ours** | **56.8** | **42.5** | **96.1** | **25.9** | **92.4** | **18.7** | **0.42** |

> **解读**：
> - **Self-Instruct**：ISR 低且多样性不足，无针对性弱点发现能力
> - **OPRO / PAIR**：ISR 很高，但过度集中于特定弱点反复挖掘，导致问题分布失衡，训练数据多样性差
> - **AutoDetect**：在保持高 ISR 的同时，各类别分布均衡，BLEU-4（与已有问题的重复度）最低，意味着生成问题多样性最佳

---

## 5. 发现的典型 LLM 弱点

![](AutoDetect_fig5_badcase.png)

> **Figure 5**：AutoDetect 揭示的三类典型弱点。
> - (A) **子类别能力不均衡**：GPT-3.5 在三角函数反解上犯了周期性的低级错误；Claude-3-Sonnet 在简单三角形求角时出错；Mistral-Large 计算矩形+半圆周长时遗漏π
> - (B) **复杂任务强但简单任务弱**：Mistral-7B 在"from os import *"时会直接 `import os` 而非导入所有函数；Llama2-70B 和 Claude-3-Sonnet 在基础数据结构操作（初始化 set 时去重）上出错
> - (C) **多约束/多步推理失败**：Llama2-13b 未遵守禁用词约束；Mistral-7b 未正确按标点符号结束列表项；GPT-3.5 在 Fibonacci 序列求第 10 项时出错

### 5.2 AutoDetect 框架的优势

![](AutoDetect_fig6_goodcase.png)

> **Figure 6**：AutoDetect 生成的创造性考题示例。
> - (A) **创造性挑战**：用 JSON 格式讲述龟兔赛跑故事、要求全随机大小写格式回答美国第一任总统
> - (B) **组合约束**：同时要求格式、长度、特定词语的多重约束；用一组罕见词汇写十四行诗并保证每个词至少出现一次；用三个"In fact..."句子解释黑洞

---

## 6. 局限性与未来方向

1. **框架本身受限于 Agent 能力**：当目标模型的性能与 GPT-4（作为 Agent）相当甚至更强时，弱点发现将变得非常困难
2. **自评估偏差**：在自进化场景中，模型评估自身输出的质量时容易过度自信（self-evaluation bias），这是一个尚待研究的开放问题
3. **生成问题质量偶有瑕疵**：尽管人工评估显示 98% 的问题合理，仍有极少数不合理的数学题生成
4. **未覆盖多模态**：当前仅针对纯文本任务
5. **成本较高**：虽然比人工更便宜，但每个目标模型需要调用大量 API（GPT-4 评分 + Agent 生成）

---

## 7. 个人思考

**方法的优雅与创新之处**：

1. **教育体系类比非常精妙**：将 weaknesses discovery 框定为"teacher assessing a student"，三个 Agent 分别对应出考卷的老师、改卷的老师和发现学生知识薄弱点、出针对性练习的老师。这种有机的整体架构远比"随机生成问题测试模型"更具系统性。

2. **动态 Taxonomy 是关键设计**。如果预先设定静态分类，测试会被已有的知识框架局限。而 Assessor → Examiner 的反馈回路让框架能够捕捉到**连设计者也未曾预料到的新弱点类型**——例如 Llama2-7B 的"幽默场景生成"和 GPT-3.5 的"诗歌场景"就是模型自身暴露的个性化盲点。

3. **迭代搜索的渐进式探针**，类比于安全领域的自适应对抗攻击，但用于通用能力评估。随着模型性能提升，逐步加压，确保始终踩在模型的"痛点"上寻找弱点。

4. **"自数据优于他数据"的发现**（Figure 4）非常有实践价值——这意味着如果企业要改进自己的模型，最好的做法是用 AutoDetect 针对该模型自身跑一遍弱点发现，而不是简单借用其他模型的测试数据。

**与当前研究的关联**：
- AutoDetect 可以视为**动态、个性化的基准测试**，与静态 MMLU、GSM8k 形成互补
- Agent 协作模式（Examiner-Questioner-Assessor）可推广到其他评估场景，如多模态能力、长上下文理解评估
- 发现的"复杂任务强但简单任务弱"的现象与当前很多"LLM 反直觉失败"的研究（如 reversal curse、功能性问题）相吻合

**潜在改进方向**：
- 引入**难度自适应机制**（DKT / Elo-based item selection）替代固定 10 步迭代，节省 API 成本
- Assessor 可以尝试更复杂的因果推断方法（如反事实推理）而非简单的``低分导致新弱点''推测
- 可以尝试将 AutoDetect 发现的弱点与缩写词频 / 训练数据分析结合，从机理上解释"为什么模型在这里出错"

---

## 8. 关键引用

```bibtex
@inproceedings{cheng2024autodetect,
  title={AutoDetect: Towards a Unified Framework for Automated Weakness Detection in Large Language Models},
  author={Cheng, Jiale and Lu, Yida and Gu, Xiaotao and Ke, Pei and Liu, Xiao and Dong, Yuxiao and Wang, Hongning and Tang, Jie and Huang, Minlie},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2024}
}
```
