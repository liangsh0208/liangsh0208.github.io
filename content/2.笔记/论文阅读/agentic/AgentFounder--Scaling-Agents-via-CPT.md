---
created: 2026-06-09
paper: https://arxiv.org/abs/2509.13310
authors: Liangcai Su*, Zhen Zhang*, Guangyu Li*, Zhuo Chen*, Chenxi Wang, Maojia Song, Xinyu Wang (project leader), Kuan Li, Jialong Wu, Xuanzhong Chen, Zile Qiao, Zhongwang Zhang, Huifeng Yin, Shihao Cai, Runnan Fang, Zhengwei Tao, Wenbiao Yin, Chenxiong Qian, Yong Jiang, Pengjun Xie, Fei Huang, Jingren Zhou (Tongyi Lab, Alibaba Group)
tags:
  - AgenticRL
  - DataSynthesis
  - ContinualPretraining
  - Agent
  - DeepResearch
---

# AgentFounder: Scaling Agents via Continual Pre-training

## 一句话总结
通过在预训练和微调之间引入 **Agentic Continual Pre-training (Agentic CPT)** 阶段，使用无需监督信号的 FAS（一阶动作合成）和有监督信号的 HAS（高阶动作合成）两种数据合成方法，将通用基础模型预对齐为 agentic 基础模型，从而在 10 个基准上实现 SOTA，其中 AgentFounder-30B 在 BrowseComp-en 上达到 39.9%、HLE 上达到 31.5%。

---

![](AgentFounder_fig1_performance.png)
> **Figure 1**: AgentFounder-30B 与 SOTA 深度研究智能体的性能对比。在 10 个基准测试上超越所有开源模型，并在多个基准上超过或逼近商业模型（如 OpenAI Deep Research、OpenAI-o3）。

## 1. 研究背景与动机

### 1.1 Agentic 任务的数据挑战

现代大语言模型（LLMs）已经演化为能够自主调用工具、执行多步推理的 agentic 系统。然而，当前主流的训练范式（预训练 → SFT/RL 微调）存在一个根本性问题：

> 通用基础模型缺乏 agentic 归纳偏置（agentic inductive biases），迫使后训练阶段必须**同时学习多样化的 agentic 行为**并使其**对齐到专家演示**。这造成了根本性的优化张力（fundamental optimization tensions）。

换言之：模型在一个阶段内既要「学会怎么用工具」又要「学会对齐人类偏好」，两者相互冲突，导致最终性能始终落后于商业系统（如 OpenAI Deep Research 在 BrowseComp 上达 51.5，而最好的开源模型 DeepSeek-V3.1 仅为 30.0）。

### 1.2 现有数据合成方法的不足

现有的 agent 训练数据合成方法主要面临三类不足：

1. **依赖商业 API 成本高**：完整的 agent 轨迹生成需要调用搜索引擎、访问网页等，大规模数据生产代价高昂。
2. **数据覆盖范围有限**：基于确定性监督信号的方法容易将模型锁入复制特定行为模式，而非培养灵活决策能力。
3. **轨迹利用率低**：SFT 和 RL 中的拒采样（rejection sampling）导致大量低质量轨迹被完全丢弃，造成信号浪费。

本文的关键洞察是：**在预训练阶段就嵌入 agentic 推理和工具使用能力**，让后训练阶段专注于对齐，从根本上缓解上述张力。

---

## 2. 方法

### 2.1 核心思想

![](AgentFounder_fig2_pipeline.jpg)
> **Figure 2**: Agentic Training Pipeline。在标准「预训练 → 后训练（SFT/RL）」两阶段范式中插入 Agentic CPT 作为中间对齐层。基于 Qwen3 系列基础模型，Stage 1 处理约 200B tokens（32K context），Stage 2 处理约 100B tokens（128K context）。

AgentFounder 的核心方法可以概括为：

**AgentFounder = 通用基础模型 + Agentic CPT（FAS + HAS）+ 下游 SFT/RL**

其训练管线包含三个阶段：
1. **通用预训练**：在通用语料上训练基础模型（如 Qwen3-30B-A3B-Base）。
2. **Agentic CPT**：两阶段持续预训练（200B+100B tokens），使模型预对齐 agentic 行为。
3. **后训练（SFT/RL）**：在标准的 agent trajectories 上进行监督微调和强化学习对齐。

#### 预训练阶段的损失函数

标准的 next-token prediction 目标函数：

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_{t+1} | x_1, x_2, \ldots, x_t)
$$

| 符号 | 含义 |
|:-----|:-----|
| $\mathcal{L}$ | 交叉熵损失函数（Cross-Entropy Loss） |
| $T$ | 序列总长度 |
| $x_t$ | 序列中第 $t$ 个 token |
| $P(x_{t+1} \| \ldots)$ | 给定前 $t$ 个 token 后，模型预测下一个 token 的条件概率 |
| 整体含义 | 最大化模型正确预测下一个 token 的概率，即标准的自回归语言建模目标 |

AgentFounder 在 CPT 阶段**不改变损失函数**，而是通过精心构建的 agentic 语料，让模型在 next-token prediction 的过程中**隐式习得** agentic 能力。

### 2.2 FAS (First-order Action Synthesis) 方法

FAS 的核心特点是 **零监督信号**，即不依赖任何真实工具调用或人类标注的完整轨迹，仅利用已有数据源进行合成。包含三个子部分：

#### 2.2.1 上下文构建：Knowledge-to-Question Transformation

FAS 的第一步是将静态知识转化为动态 agent 训练上下文，通过两阶段变换：

**Phase 1: Entity-Anchored Open-World Knowledge Memory（实体锚定的开放世界知识记忆）**

将非结构化文本转换为以实体为索引键的开放世界记忆结构，相比传统知识图谱不关心实体间关系，而是**增强每个实体对应的声明性知识陈述密度**。例如：

> 原始文本："The number of tourist arrivals in France increased from 3,793 thousand in May 2025 to 4,222 thousand in June."
> 
> 知识重构：`("France", "Tourist arrivals in France reached 4,222 thousand in June 2025.")`

这样形成的「活记忆」（living memory）会随着搜索结果和网页访问结果持续更新。

**Phase 2: Multi-Style Question Synthesis（多风格问题合成）**

利用上述知识记忆，采样实体簇及关联知识，合成多样化问题，覆盖：
- **事实检索（Factual retrieval）**
- **数值计算（Numerical computation）**
- **多跳推理（Multi-hop reasoning）**
- **综合任务（Synthesis tasks）**

![](AgentFounder_fig3_entity_knowledge.jpg)
> **Figure 3**: 基于可扩展信息源的多风格问答生成。左侧为动态数据源（文档、知识图谱），中间为实体锚定的开放世界记忆（Entity-Anchored Open-World Memory），右侧为多风格 QA（Single Entity / Multi-Entities）。

#### 2.2.2 Planning Action Synthesis（规划动作合成）

核心洞察：**第一步推理的质量与最终任务完成率存在强正相关**。

基于此，FAS 为每个问题 $Q$ 生成 $K$ 个不同的**问题分析**（problem analyses）和对应的**第一步动作预测**（tool invocations or direct answers）。

具体做法：
1. 给定问题 $Q$，调用 LLM 生成 $K$ 个不同视角的问题分解。
2. 每个分解对应一个第一步动作预测。
3. 不实际执行工具调用，因此训练数据生产成本为零。

关键改进——**问题级多样性扩展**：不针对单一问题生成 $K$ 个迭代，而是针对**$K$ 个不同风格（style）但共享同一知识记忆**的问题生成推理-动作数据。这能更全面地覆盖训练上下文。

**知识对齐验证的拒采样（Reject Sampling）：**

生成的推理和动作不一定准确。利用 QA 构建阶段的可访问知识，采用 **LLM-as-Judge** 策略：
- 评估当前推理和动作获取目标知识的概率。
- 拒绝采样有效过滤了大部分低质量数据。
- 初始生成正确/错误各占 50%，过滤后过滤出 43.5% 的低质量样本，保留样本的正确率从 50% 提升到 82%。

#### 2.2.3 Reasoning Action Synthesis（推理动作合成）

在 agent 工作流中，一个重要场景是：综合各类工具调用/用户交互获取到的信息，生成最终答案或报告。这属于**逻辑推理**而非数值计算。

FAS 的两步推理数据合成方案：

1. **Step 1**: 要求 LLM 将问题 $Q$ 分解为多个子问题，利用内部知识生成合理推测和初步答案 $A_1$。
2. **Step 2**: 给定问题 $Q$ 和映射后的必需知识，要求模型精修答案 $A_1$，纠正逻辑错误，生成最终答案 $A_2$。

**关键设计**：两个阶段都禁止调用外部工具。这样设计的动机是：当模型同时获得问题和必要知识时，它会以**真实推理的中间节点**方式利用知识，而非机械模拟思考过程。

### 2.3 HAS (Higher-order Action Synthesis) 方法

HAS 的核心特点是**利用监督信号**（来自 post-training 的真实轨迹反馈），解决轨迹利用率低的问题。

#### 2.3.1 问题定义

标准设定：
- 问题 $Q$
- Agent 轨迹 $T = \{(S_1, R_1), \ldots, (S_K, R_K)\}$，其中 $S_k$ 是第 $k$ 步的推理和工具调用，$R_k$ 是工具/环境的响应
- 二值判断 $J \in \{0, 1\}$，0 表示失败，1 表示成功

#### 2.3.2 Pathway from Trajectory Imitation to Decision-Based Reasoning

传统方法（SFT/RL）中的问题在于：
- 依赖轨迹级延迟反馈（trajectory-level delayed feedback）评估质量。
- 成功/失败的粗粒度评价导致大量学习信号被浪费。
- 直接利用不确定的步骤级奖励信号又容易导致训练崩溃。

核心洞察：虽然不是每个子决策都一定有正确标签，但**每个步骤都有一个高质量的上下文**（原始问题 + 之前步骤 + 真实反馈），定义了一个具有广泛可行推理-动作选项的推理状态。因此，每个步骤本质上是一个隐藏决策过程。

如果将目标从「轨迹模仿」转变为「逐步决策」，就可以充分利用轨迹中的每一步。

#### 2.3.3 HAS 的两大组件

**(1) Step-level Scaling（步骤级扩展）**

对于任意步骤 $S_k$，其条件上下文为：

$$
C_k = (Q, S_1, R_1, \ldots, S_{k-1}, R_{k-1})
$$

对于上下文 $C_k$，使用 LLM 生成 $N$ 个替代的「思考与调用」候选：

$$
A_k = \{S_k^{(1)}, \ldots, S_k^{(N)}\}
$$

将原始步骤 $S_k^{(0)} \equiv S_k$ 与这些候选合并，得到 $N+1$ 个可行步骤，随机打乱形成序列 $\bar{A}_k$，同时记录原始步骤在序列中的位置 $n_k$。

这样，每条轨迹被扩展为具有 **$(N+1) \times K$ 个潜在推理-动作** 的决策空间。

**(2) Contrastive Decision-Action Synthesis（对比决策-动作合成）**

将扩展了选项的轨迹转化为**渐进的决策过程**。

- 从问题 $Q$ 出发，对于每一步 $k$，显式模拟一个**多选项选择和决策**过程。
- 枚举 $\bar{A}_k$ 中的每个选项，插入局部动作决策语句："I will choose option $n_k$"（我将选择选项 $n_k$）。
- 紧接着对应的真实响应 $R_k$。
- 在最后，追加判断文本："My decision is {Correct/Incorrect}"（对应 $J$）。

完整的合成训练样本 = 问题 + 每一步的选择决策过程 + 最终判断文本。

这种模式的好处是：
- 避开了直接使用不确定步骤级奖励的风险。
- 使模型通过学习多样化的推理路径进行决策。
- 防止对特定轨迹模式的过拟合。
- 将以往利用率低的轨迹数据转化为丰富的训练信号。

![](AgentFounder_fig5_has_trajectory.png)
> **Figure 5**: 高阶动作合成（HAS）与原始轨迹的对比。上方为原始轨迹（$S_1 \to R_1 \to S_2 \to \ldots$），下方为 HAS 扩展后的多决策轨迹——在每一步产生多个候选方案（"I have 3 solutions"），经过选择后接续真实响应，并在最后给出判断（"My decision is {Correct}"）。

### 2.4 两阶段训练策略

为高效吸收 FAS 和 HAS 两类合成数据，AgentFounder 提出**渐进的两阶段训练策略**：

| 阶段 | 数据 | Context Length | Tokens | 目标 |
|:-----|:-----|:-------------|:-------|:-----|
| Agentic CPT Stage 1 | FAS 数据 + 短 HAS 数据 | 32K | ~200B | 初步习得 agentic 行为，包括工具调用模式和简单推理链 |
| Agentic CPT Stage 2 | 高质量 HAS 数据 | 128K | ~100B | 发展对复杂动作空间和长程规划策略的深入理解 |

关键设计动机：
- **Stage 1 负责广度**：在较短的 context 窗口内，让模型广泛接触多样化的 agentic 情境。
- **Stage 2 负责深度**：在更长的 context 中，让模型学习复杂的、多步的决策序列。
- **成本考量**：直接用 128K context 从 0 开始训练的计算成本会高得不可接受。

---

## 3. 实验结果

### 3.1 实验设置

**数据**：
- CPT 语料：高质量 web-crawled 数据（经事实准确性过滤）、历史工具调用记录（搜索结果和网页内容）、离线 Wikipedia 数据、以及先前 post-training 迭代中混合质量的丢弃轨迹。

**评估工具**：
- Search、Visit（网页内容提取）、Google Scholar、Python Interpreter（代码执行）、File Parser（文档处理）。

**后训练变体**：
| 变体 | 描述 |
|:-----|:-----|
| SFT-A | 先在通用对话数据上训练，再在 React-style agent trajectories 上训练 |
| SFT-B | 在每一阶段都混合通用对话和 React-style trajectories |
| SFT-C | 通用对话 SFT + React with summarized reasoning trajectories |

**基线分类**：
- 通用 LLM + 工具（Qwen3, DeepSeek-R1, Claude-4-Sonnet）
- 商业深度研究 Agent（Kimi-Researcher, OpenAI-o3, OpenAI Deep Research, Grok, Perplexity, Gemini）
- 开源深度研究 Agent（WebThinker, ASearcher, WebSailer, WebShaper, GLM-4.5, DeepSeek-V3.1 等 12 个）

**Benchmarks**：共 10 个
- **通用网页搜索**：BrowseComp-en, BrowseComp-zh, GAIA, Xbench-DeepSearch, WebWalkerQA
- **场景化搜索**：HLE, DeepResearch Bench, Frames, SEAL-0, AcademicBrowse

### 3.2 主实验结果

#### RQ1: AgentFounder vs. SOTA

**通用网页搜索基准（Table 1）**

| 模型类型 | BrowseComp-en | BrowseComp-zh | GAIA | Xbench-DeepSearch | WebWalkerQA |
|:---------|:-------------|:-------------|:-----|:-----------------|:------------|
| OpenAI Deep Research | 51.5 | - | 70.5 | 66.0 | - |
| DeepSeek-V3.1 | 30.0 | 49.2 | 66.0 | 71.0 | 61.2 |
| GLM-4.5 | 26.4 | 37.5 | 66.0 | 70.0 | 65.6 |
| **AgentFounder-30B** | **39.9** | **43.3** | **72.8** | **73.0** | **71.9** |

**场景化搜索基准（Table 2）**

| 模型 | HLE (Pass@1) | DeepResearch Bench | Frames (Pass@1) | SEAL-0 (Pass@1) | AcademicBrowse (Pass@1) |
|:-----|:-------------|:-------------------|:----------------|:----------------|:------------------------|
| OpenAI-o3 | 20.2 | - | 84.0 | - | - |
| Gemini Deep Research | 26.9 | 49.7 | - | - | - |
| DeepSeek-V3.1 | 29.8 | 35.4 | 83.7 | 42.6 | 65.0 |
| **AgentFounder-30B** | **31.5** | **47.9** | **89.6** | **43.9** | **75.3** |

**关键发现**：
1. 在 BrowseComp-en 上超越最好开源模型 DeepSeek-V3.1 达 **10.0 个百分点**，逼近 OpenAI 闭源模型。
2. **GAIA 达 72.8%——目前所有模型中单 agent 最高准确率**。
3. **HLE 达 31.5%——首个超过 30 分阈值的开源模型**。
4. Frames 达 89.6%，全面超越所有模型。
5. AcademicBrowse 达 75.3%，大幅领先所有开源和闭源模型。

#### RQ2: AgentFounder Base Model 的后训练适应性

| Base Model | SFT Data | BrowseComp-en | BrowseComp-zh | GAIA | HLE | 平均增益 |
|:-----------|:---------|:--------------|:--------------|:-----|:----|:--------|
| Qwen3-30B-A3B-Base | SFT-A | 26.9 | 29.8 | 67.0 | 23.5 | - |
| AgentFounder-30B-Base | SFT-A | 31.4 | 35.6 | 72.8 | 30.4 | **+5.75%** |
| Qwen3-30B-A3B-Base | SFT-B | 28.6 | 35.6 | 71.8 | 27.0 | - |
| AgentFounder-30B-Base | SFT-B | 39.9 | 43.3 | 72.8 | 31.5 | **+6.13%** |
| Qwen3-30B-A3B-Base | SFT-C | 24.5 | 36.7 | 68.9 | 27.9 | - |
| AgentFounder-30B-Base | SFT-C | 38.8 | 44.3 | 71.8 | 28.9 | **+6.45%** |

**结论**：
1. AgentFounder Base 在所有后训练配置下**一致且显著地优于** Qwen3 Base。
2. 后训练数据的选择仍然关键——不同 SFT 配置间可产生 8.5% 差距。
3. **信息检索类任务（BrowseComp）从 Agentic CPT 中获益更多**，而知识密集类任务（HLE）需要更强的知识理解能力作为补充。

#### RQ5: Scaling Law 探索

![](AgentFounder_fig6_scaling_law.png)
> **Figure 6**: Agentic 能力的 Scaling Law 探索。(a) 模型规模：从 1B 到 30B-A3B 参数的平均准确率变化。(b) 数据规模：从 0B 到 315B tokens 训练数据量的 Pass@3 平均性能。双轴均为对数坐标。

**模型规模 Scaling**：
- 1B → 20.4%，4B → 32.7%，30B → 48.9%
- AgentFounder-30B（48.9%）超过更大的基线 DeepSeek-V3.1（43.0%）和 Kimi-K2（29.6%）
- **Superior scaling efficiency**：相同的模型容量，通过 Agentic CPT 能更有效利用

**数据规模 Scaling**（固定 128K context，两阶段训练）：
- 0B → 54.2%，15B → 58.0%（+3.8%），50B → 61.1%，210B → 62.1%，315B → 62.2%（总增益 +8.0%）
- **对数 scaling law 成立**：训练数据量和性能呈对数关系，大部分提升来自前 15B tokens
- **Stage 2 在长 context 上一致提升**：50B→65B（+1.8%），210B→315B（+1.0%）

### 3.3 消融实验

#### RQ3: 两阶段训练策略

| 策略 | BrowseComp-en P@1 | BrowseComp-en P@3 | BrowseComp-zh P@1 | BrowseComp-zh P@3 | GAIA P@1 | GAIA P@3 |
|:-----|:-----------------|:-----------------|:-----------------|:-----------------|:--------|:--------|
| Stage 1 Only | 31.4 | 49.9 | 34.3 | 50.5 | 69.9 | 81.6 |
| Stage 1 & 2 | **35.5** | **52.0** | **37.2** | **58.5** | **72.8** | **82.5** |
| Relative Δ | **+4.1** | **+2.1** | **+2.9** | **+8.0** | **+2.9** | **+0.9** |

结论：两阶段训练配置相比单阶段，在 Pass@1 上平均提升 **3.3%**，Pass@3 上平均提升 **3.7%**。Stage 2 学习的完整长上下文 agent 数据是必要的，而非简单截断序列。

#### RQ4: 数据类型贡献（50B tokens，单阶段）

| 数据 | BrowseComp-en (P@1/P@3) | BrowseComp-zh (P@1/P@3) | GAIA (P@1/P@3) |
|:-----|:------------------------|:------------------------|:---------------|
| Non-CPT (baseline) | 26.9 / 38.0 | 29.8 / 45.3 | 67.0 / 75.7 |
| FAS | 31.4 / 49.9 | 37.0 / 54.3 | 72.8 / 80.6 |
| FAS + HAS | 31.4 / 50.1 | **40.1** / 54.7 | 69.9 / **82.5** |

结论：
1. **FAS 数据效果明显**：纯 FAS 即可带来显著提升，Cross-zh 上 Pass@1 的提升达 9.0 个百分点。
2. **HAS 提供互补增益**：FAS+HAS 混合数据在多个指标上一致带来正向增益，BrowseComp-zh Pass@1 从 37.0 提升到 40.1，GAIA Pass@3 从 80.6 提升到 82.5。
3. FAS 建立性能基线，HAS 进一步提升上限。

#### 综合训练效率分析

![](AgentFounder_fig7_training_loss.png)
> **Figure 7**: 在相同下游 SFT 语料上的训练损失演进对比。AgentFounder 变体均显著优于基线（最终 loss：AgentFounder-30B(315B) 为 0.7953 vs. 基线 0.8656）。右下角放大图显示最后 100 步的差异。

关键观察：
1. **AgentFounder 大幅降低 SFT 损失**：所有 AgentFounder 变体均显著优于基线。
2. **CPT 数据量与 loss 单调递减**：从纯 FAS 到 315B tokens，损失逐步降低。
3. **FAS+HAS 混合优于纯 FAS**：验证了将后训练中的监督信号重组为 CPT 格式的价值。

#### 工具调用分析

![](AgentFounder_fig8_tool_calls.png)
> **Figure 8**: 不同基准上的工具调用分布对比。左图：WebWalker vs GAIA；右图：HLE vs BrowseComp-en。

观察发现：
1. **复杂研究任务呈现密集型工具使用**：BrowseComp-en 和 HLE 呈现长尾分布，表明模型在深入研究中需要大量工具调用。
2. **结构化任务采用保守型工具使用**：WebWalker 在低调用次数处尖锐峰值（快速导航）；GAIA-text 紧凑分布（定义良好的问题）。
3. 模型能根据任务复杂度**校准工具使用策略**——开放研究任务中密集探索，结构化问题中定向调用。

#### 通用工具使用能力

| 模型 | ACEBench |
|:-----|:---------|
| Qwen3-30B-A3B | 67.2 |
| **AgentFounder-30B** | **70.0** |

结论：AgentFounder-30B 不仅在深度研究任务上表现出色，还**提升了通用工具使用能力**（+2.8 个百分点），证明 Agentic CPT 框架可推广到更广泛的 agentic 场景。

---

## 4. 局限性与未来方向

1. **BrowseComp-zh 与 BrowseComp-en 的差距**：中文数据在训练语料中占比较低，且底层搜索工具（Google Search）可能在中国语境下存在偏差。
2. **知识密集型任务收益有限**：HLE 等任务需要信息检索和知识理解双重能力，单纯增强 agentic base model 还不够。
3. **两阶段计算开销**：Stage 2 使用 128K context 训练的成本仍然较高。
4. **数据污染风险**：论文未对训练数据与测试基准的重叠情况进行彻底分析。

未来可以探索的方向：
- 将 Agentic CPT 应用到更多模型架构（如纯 Dense 模型）。
- 研究 Agentic CPT 与更先进的 RL 算法（如 GRPO）的结合。
- 探索多模态 Agentic CPT，将视觉/网页渲染信息纳入 CPT 训练。
- 进一步优化数据过滤策略，提高从高噪声轨迹中提取有效信号的效率。

---

## 5. 个人思考

1. **「预对齐」是核心创新**：本文最有启发性的点是「把 agentic 能力内化为基础模型的 inductive bias」，这和传统「先学语言再学 agent」的范式形成了根本性的区分。类比来看，就像人类在上学之前已经通过日常探索学会了使用工具的基本逻辑，正式教育只需要对齐专业规范和深度知识。

2. **FAS 的无监督信号设计非常精巧**：将静态知识通过 entity-anchored memory + multi-style QA 重构为 agentic 训练上下文，完全不依赖外部 API 调用，这个设计在成本和可扩展性上具有巨大优势。拒采样从 50%→82% 的提升也证明了知识对齐验证的有效性。

3. **HAS 的 contrastive decision-action 视角**：传统的轨迹数据合成是「模仿一条路径」，而 HAS 将其转为「每一步的决策分支」，这是一种从 CV 领域的对比学习思想迁移到 agent 训练的巧妙尝试。$(N+1) \times K$ 的扩展因子让数据的利用率得到质的飞跃。

4. **Scaling law 的验证意义**：logarithmic scaling 的发现（前 15B tokens 贡献最大增益）对于 agent data 的工程实践具有很好的指导意义——不必盲目追求 TB 级数据，合理的数据配比和过滤更重要。

5. **与当前研究方向的关联**：本文的方法论和 AgenticRL 的 trajectory-based 训练形成了很好的互补——AgentFounder 解决「基础模型预对齐」问题，而 AgenticRL 解决「从预对齐模型进一步优化策略」的问题。两者结合可能产生更强大的 agent 系统。

6. **和 WebShaper 的对比**：WebShaper 也是一种数据合成方法，但它更聚焦于通过 knowledge projection 构建困难问题，而 AgentFounder 关注的是「大规模、无需 API 的 agentic 数据合成」，两者在 data-centric agent training 的方向上各有侧重。

---

## 6. 关键引用

```bibtex
@article{su2025scaling,
  title={Scaling Agents via Continual Pre-training},
  author={Su, Liangcai and Zhang, Zhen and Li, Guangyu and Chen, Zhuo and Wang, Chenxi and Song, Maojia and Wang, Xinyu and Li, Kuan and Wu, Jialong and Chen, Xuanzhong and Qiao, Zile and Zhang, Zhongwang and Yin, Huifeng and Cai, Shihao and Fang, Runnan and Tao, Zhengwei and Yin, Wenbiao and Qian, Chenxiong and Jiang, Yong and Xie, Pengjun and Huang, Fei and Zhou, Jingren},
  journal={arXiv preprint arXiv:2509.13310},
  year={2025}
}
```

---

## 参考链接
- 项目博客：https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/
- GitHub：https://github.com/Alibaba-NLP/DeepResearch
