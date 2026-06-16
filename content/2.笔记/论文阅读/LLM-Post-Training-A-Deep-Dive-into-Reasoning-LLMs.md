---
created: 2026-06-09
paper: https://arxiv.org/abs/2502.21321
code: https://github.com/mbzuai-oryx/Awesome-LLM-Post-training
authors: Komal Kumar*, Tajamul Ashraf*, Omkar Thawakar, Rao Muhammad Anwer, Hisham Cholakkal, Mubarak Shah, Ming-Hsuan Yang, Phillip H.S. Torr, Fahad Shahbaz Khan, Salman Khan (MBZUAI + UCF + UC Merced / Google DeepMind + Oxford)
tags:
  - LLM
  - post-training
  - survey
  - reinforcement-learning
  - SFT
  - test-time-scaling
  - reasoning
---

# LLM Post-Training: A Deep Dive into Reasoning Large Language Models

## 一句话总结

这是一篇**全面系统综述 LLM 后训练方法**的论文，将后训练拆解为三大支柱——**SFT（监督微调）、RL（强化学习）、Test-Time Scaling（推理时扩展）**，系统梳理了从奖励建模（显式/隐式/过程/结果/自适应）到策略优化（PPO/GRPO/DPO/TRPO/RLAIF/ORPO/OREO）再到推理时方法（CoT/ToT/GoT/MCTS/Self-Consistency）的完整技术图谱，并讨论了灾难性遗忘、奖励黑客、推理成本等关键挑战。

![](LLMPT_x1.png)

> **Figure 1**: LLM 后训练技术分类总览（Radial Taxonomy）。外圈为具体模型（GPT-4/Claude/DeepSeek-R1/Qwen/Llama 等），内圈为后训练技术分类：Fine-Tuning（全模型/参数高效/知识蒸馏）、Reinforcement Learning（策略优化/奖励建模）、Test-Time Scaling（搜索/解码/推理）。

---

## 1. 研究背景与动机

### 1.1 后训练的定义与价值

LLM 的生命周期通常分为：
1. **Pre-training**（预训练）：在大规模无标注文本上学习语言表示
2. **Post-training**（后训练）：在预训练基础上通过多种技术进一步优化

后训练的核心目标是：
- **任务特化**：将通用模型适配到特定任务
- **推理增强**：提升多步推理和逻辑一致性
- **对齐优化**：使输出符合人类意图和伦理标准
- **事实准确性**：减少幻觉，提高知识可靠性

> **核心洞察**：预训练提供**广泛的语言基础**，后训练提供**精细的行为优化**。没有后训练，模型只是"语言统计器"；有了后训练，模型才成为"有用的助手"。

### 1.2 三大后训练支柱

| 支柱 | 核心方法 | 目标 | 风险 |
|------|---------|------|------|
| **SFT** | 指令微调、CoT 微调、对话微调、领域特化 | 任务适配 | 过拟合、计算成本高、泛化下降 |
| **RL** | PPO、GRPO、DPO、RLHF、RLAIF | 偏好对齐 | 奖励黑客、灾难性遗忘、不稳定 |
| **Test-Time Scaling** | CoT、ToT、MCTS、Self-Consistency | 推理增强 | 推理成本激增、延迟增加 |

---

## 2. Preliminaries: LLM 的数学基础

### 2.1 最大似然估计（MLE）

$$
\mathcal{L}_{\text{MLE}} = -\sum_{t=1}^{T} \log P_{\theta}(y_t \mid y_{<t}, X)
$$

**局限**：token-wise 训练确保流畅度，但推理时错误无法被纠正，导致**级联错误**（cascading errors）。

### 2.2 RL 形式化

将自回归生成建模为 **MDP**：
- **状态** $s_t$：已生成的 token 序列
- **动作** $a_t$：下一个 token
- **奖励** $R(s_t, a_t)$：评估输出质量
- **目标**：

$$
J(\pi_{\theta}) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)\right]
$$

其中 $\gamma$ 为折扣因子，越高越重视长期回报。

### 2.3 策略梯度

$$
\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau}\left[\sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(x_t \mid x_{1:t-1}) \cdot A(s_t, a_t)\right]
$$

| 函数 | 定义 | 含义 |
|------|------|------|
| **Value** $V(s)$ | $\mathbb{E}[\text{future return} \mid s]$ | 状态的价值评估 |
| **Q-function** $Q(s,a)$ | $\mathbb{E}[\text{future return} \mid s, a]$ | 动作的价值评估 |
| **Advantage** $A(s,a)$ | $Q(s,a) - V(s)$ | 动作相对好坏 |

---

## 3. 早期的 RL 方法（Pre-Transformer Era）

| 方法 | 核心思想 | 公式 | 局限 |
|------|---------|------|------|
| **REINFORCE** | 蒙特卡洛策略梯度，使用 return $G$ 作为梯度方向 | $\theta \leftarrow \theta + \alpha (G - b) \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)$ | 高方差，需要基线 $b$ 减方差 |
| **MIXER** | 课程学习：从 MLE 逐渐过渡到 RL | $\mathcal{L} = \lambda(t) \mathcal{L}_{\text{MLE}} + (1-\lambda(t)) \mathcal{L}_{\text{RL}}$ | $\lambda(t)$ 需要精心设计 |
| **SCST** | 自批评序列训练：用 greedy decode 作为基线 | $\nabla_{\theta} J \approx (r(y^s) - r(\hat{y})) \nabla_{\theta} \log \pi_{\theta}(y^s)$ | 仅适用于序列生成 |
| **MRT** | 最小风险训练：对所有候选计算期望损失 | $\mathcal{L}_{\text{MRT}} = \sum_{y \in \mathcal{Y}} p_{\theta}(y \mid x) \Delta(y, y^*)$ | 候选空间爆炸 |
| **A2C/A3C** | Actor-Critic：同时训练策略和价值函数 | Actor: $\theta \leftarrow \theta + \alpha A(s_t,a_t) \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)$; Critic: $\phi \leftarrow \phi - \beta \nabla_{\phi}(V_{\phi}(s_t) - G_t)^2$ | 需要同时训练两个网络 |

---

## 4. RL 增强的 LLM：Reinforced LLMs

![](LLMPT_x2.png)

> **Figure 2**: LLM 后训练方法全景图。左侧为推理时推理方法（Tree of Thoughts、CoT、Self-feedback 等），中间为后训练分类（SFT → 奖励模型训练 → 策略优化），右侧为具体 RL 算法（RLHF → DPO → TRPO → GRPO 等）。

### 4.1 奖励建模（Reward Modeling）

奖励模型的训练基于**偏好数据**——对同一问题 $x$ 的两个回答 $(y_j, y_k)$，标注者偏好 $y_j \succ y_k$。

**Bradley-Terry 模型（成对偏好）**：

$$
P(y_j \succ y_k \mid x; \theta) = \frac{\exp(R_{\theta}(x, y_j))}{\exp(R_{\theta}(x, y_j)) + \exp(R_{\theta}(x, y_k))}
$$

**Plackett-Luce 模型（排序偏好）**：

$$
P(y_{j_1}, \ldots, y_{j_m} \mid x; \theta) = \prod_{\ell=1}^{m} \frac{\exp(R_{\theta}(x, y_{j_{\ell}}))}{\sum_{k=\ell}^{m} \exp(R_{\theta}(x, y_{j_k}))}
$$

#### 奖励建模五类

| 类型 | 定义 | 优势 | 局限 |
|------|------|------|------|
| **显式奖励** (Explicit) | 直接数值信号（人工标注/规则） | 精确可控 | 成本高、难以规模化 |
| **隐式奖励** (Implicit) | 从交互指标推断（点赞、点击率、停留时间） | 可规模化 | 易被利用（exploitation）、信号噪声大 |
| **结果奖励** (Outcome/ORM) | 仅评估最终结果（如答案是否正确） | 简单易实现 | **信用分配问题**：长序列中难定位错误步骤 |
| **过程奖励** (Process/PRM) | 评估中间推理步骤 | 对数学推导、法律论证、代码调试极有价值 | 标注成本高、需要逐步评估 |
| **自适应奖励** (Adaptive) | 奖励模型与策略模型协同进化 | 抗奖励黑客、抗漂移 | 训练不稳定、计算成本高 |

> **核心对比**：ORM 关注"答对了没"，PRM 关注"每一步走得对不对"。DeepSeek-R1 的成功很大程度上归功于 PRM 的引入。

### 4.2 策略优化方法

#### 4.2.1 RLHF（Reinforcement Learning from Human Feedback）

**三阶段 pipeline**：
1. **SFT**：在高质量标注数据上监督微调
2. **奖励模型训练**：在人类偏好对 $(y_w, y_l)$ 上训练 $R_{\theta}$
3. **PPO 优化**：用奖励模型输出作为 reward，PPO 优化策略

**局限**：
- 需要大量人工标注偏好数据
- 奖励模型可能过拟合到训练偏好，泛化到新任务差
- **奖励黑客**：策略学会"欺骗"奖励模型而非真正改进

#### 4.2.2 RLAIF（RL from AI Feedback）

用**AI 模型替代人类标注者**进行偏好标注。降低了标注成本，但引入了 AI 自身的偏见。

#### 4.2.3 PPO（Proximal Policy Optimization）

核心创新：引入 **clipped surrogate objective** 限制策略更新幅度：

$$
r_t(\theta) = \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{ref}}}(a_t \mid s_t)}
$$

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]
$$

**在 LLM 中的应用**：
- 使用预训练模型作为参考策略 $\pi_{\theta_{\text{ref}}}$
- 添加 KL 散度惩罚防止策略偏离太远
- 价值网络估计 Advantage function

#### 4.2.4 TRPO（Trust Region Policy Optimization）

通过**KL 散度约束**确保策略更新在安全区域内：

$$
\max_{\theta} \mathbb{E}_{s \sim \rho_{\theta_{old}}, a \sim \pi_{\theta_{old}}} \left[\frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{old}}(a \mid s)} A^{\pi_{\theta_{old}}}(s, a)\right]
$$

$$
\text{s.t.} \quad \mathbb{E}_{s \sim \rho_{\theta_{old}}} \left[\text{KL}(\pi_{\theta_{old}}(\cdot \mid s) \| \pi_{\theta}(\cdot \mid s))\right] \leq \delta
$$

**PPO vs TRPO**：PPO 用 clipping 近似 TRPO 的约束，实现更简单高效。

#### 4.2.5 DPO（Direct Preference Optimization）

**核心突破**：直接从偏好数据优化策略，**无需显式训练奖励模型**！

将 RLHF 的偏好目标重写为对数似然比损失：

$$
\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma\left(\beta \log \frac{\pi_{\theta}(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_{\theta}(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]
$$

**优势**：
- 省去奖励模型训练，降低复杂度
- 训练更稳定（无 PPO 的策略-价值网络协调问题）
- 开源模型偏好（LLaMA 3、Qwen2 等均使用 DPO）

**局限**：偏好数据质量敏感，对噪声鲁棒性不如 PPO。

#### 4.2.6 GRPO（Group Relative Policy Optimization）

**核心创新**：消除 Critic 网络，用**组内相对基线**代替 Advantage 估计。

流程：
1. 对同一问题 $x$，采样一组回答 $G = \{y_1, \ldots, y_G\}$
2. 计算每个回答的奖励 $r_i = R(x, y_i)$
3. 组内相对优势：$A_i = \frac{r_i - \text{mean}(G)}{\text{std}(G)}$
4. 用 PPO 的目标优化

**DeepSeek-R1 的关键技术**：GRPO 使得大规模 RL 训练在无需价值网络的情况下可行，大幅降低了训练成本。

#### 4.2.7 ORPO（Odds Ratio Preference Optimization）

直接从成对偏好优化策略，使用**赔率比（odds ratio）**：

$$
P_{\phi}(y_j \succ y_k \mid x) = \sigma\left(\ln \frac{\pi_{\phi}(y_j \mid x)}{\pi_{\phi}(y_k \mid x)}\right)
$$

损失函数：

$$
\mathcal{L}_{\text{ORPO}}(\phi) = -\sum_{(x, y_j \succ y_k) \in \mathcal{D}} \log\left(\frac{\pi_{\phi}(y_j \mid x)}{\pi_{\phi}(y_j \mid x) + \pi_{\phi}(y_k \mid x)}\right)
$$

⚠️ **局限**：对多奖励信号组合不够灵活。

#### 4.2.8 OREO（Offline Reasoning Optimization）

针对推理任务的离线优化方法，无需在线交互即可优化策略。

#### 4.2.9 策略优化方法对比总结

| 方法 | 需要奖励模型 | 需要价值网络 | 在线/离线 | 主要用途 | 代表模型 |
|------|------------|------------|----------|---------|---------|
| **PPO** | ✅ | ✅ | 在线 | 通用 RLHF | GPT-4, InstructGPT |
| **TRPO** | ✅ | ✅ | 在线 | 安全策略更新 | 早期工作 |
| **DPO** | ❌ | ❌ | 离线 | 开源模型偏好对齐 | LLaMA 3, Qwen2 |
| **GRPO** | ✅ | ❌ | 在线 | 大规模推理 RL | DeepSeek-R1 |
| **ORPO** | ❌ | ❌ | 离线 | 轻量级偏好优化 | 小型实验 |
| **RLAIF** | ✅ (AI) | ✅ | 在线 | 降低人工标注成本 | Claude, Starling |

### 4.3 纯 RL 的 LLM 精炼（四阶段 Pipeline）

| 阶段 | 名称 | 目标 | 技术 |
|------|------|------|------|
| 1 | **Cold-Start RL** | 从零开始优化策略 | 无 SFT 热启动，直接用 RL |
| 2 | **Rejection Sampling + FT** | 过滤高质量输出做 SFT | 采样 N 个回答，选 top-k 做 SFT |
| 3 | **Reasoning-Oriented RL** | 增强推理能力 | 推理特定奖励，长 CoT 奖励 |
| 4 | **Distillation** | 压缩到更小的模型 | 知识蒸馏到小模型 |

> **DeepSeek-R1 的训练流程**：Cold-Start SFT → GRPO RL → Rejection Sampling → 第二轮 RL → Distillation。

---

## 5. 监督微调（SFT）

### 5.1 SFT 类型

| 类型 | 描述 | 目标数据 |
|------|------|---------|
| **指令微调** (Instruction FT) | 高质量人工编写示例 | $(指令, 回答)$ 对 |
| **对话微调** (Dialogue FT) | 多轮对话数据 | 多轮 $(用户, 助手)$ 对 |
| **CoT 推理微调** | 思维链推理数据 | $(问题, 推理过程, 答案)$ |
| **领域特化** | 医学/法律/编码等 | 领域专用数据集 |
| **蒸馏微调** | 从大模型学习 | 教师模型生成的高质量回答 |
| **偏好对齐 SFT** | 偏好数据 + SFT | 偏好排序 + SFT 结合 |
| **高效微调** | LoRA / Adapter / Prompt Tuning | 冻结大部分参数 |

### 5.2 高效微调方法

| 方法 | 参数占比 | 核心思想 |
|------|---------|---------|
| **LoRA** | ~0.1-1% | 低秩分解：$W = W_0 + BA$，训练 $B \in \mathbb{R}^{d \times r}$ 和 $A \in \mathbb{R}^{r \times d}$ |
| **Adapter** | ~3-5% | 在 Transformer 层间插入小型 bottleneck 网络 |
| **Prompt Tuning** | <0.01% | 训练软提示（soft prompt）嵌入而非模型参数 |
| **Prefix Tuning** | <0.1% | 在 attention key/value 前添加可训练前缀 |
| **IA3** | ~0.1% | 学习缩放向量，逐元素缩放 hidden states |

---

## 6. 测试时扩展（Test-Time Scaling, TTS）

TTS 核心思想：**不修改模型参数，通过增加推理时的计算来提升性能**。

### 6.1 搜索方法

| 方法 | 机制 | 公式/伪代码 |
|------|------|------------|
| **Best-of-N** | 生成 N 个候选，选奖励最高的 | $y^* = \arg\max_{\{y_i\}_{i=1}^N} R(x, y_i)$ |
| **Beam Search** | 保持 top-k 部分序列，逐步扩展 | 维护 beam width 个候选，每步保留 top-k |
| **MCTS** | 蒙特卡洛树搜索：选择→扩展→模拟→反向传播 | UCB: $\frac{Q}{N} + c\sqrt{\frac{\ln N_{\text{parent}}}{N}}$ |

### 6.2 解码/推理方法

| 方法 | 核心思想 | 适用场景 |
|------|---------|---------|
| **CoT** (Chain-of-Thought) | "Let's think step by step"，分解长推理 | 数学、逻辑、常识推理 |
| **ToT** (Tree-of-Thoughts) | 维护推理树，支持回溯和分支 | 需要探索的创造性任务 |
| **GoT** (Graph-of-Thoughts) | 推广 ToT 到任意图结构 | 复杂依赖关系的推理 |
| **Self-Consistency** | 多条 CoT 路径，多数投票 | 答案空间有限的多选题 |
| **Self-Improvement** | 生成→批评→修改→重新生成 | 可迭代的生成任务 |
| **Chain-of-Action-Thought** | 推理与执行交替 | 工具使用、具身智能 |

### 6.3 计算最优扩展

核心问题：给定固定推理预算，如何分配计算？
- 采样 **N 个短回答** vs **1 个长回答**
- **DeepMind 发现**：对于简单问题，多采样更有效；对于复杂问题，长 CoT 更有效。

### 6.4 Scaling Law：Pre-training vs Test-Time

| 维度 | Pre-training Scaling | Test-Time Scaling |
|------|---------------------|-------------------|
| **投资对象** | 模型参数 | 推理计算 |
| **边际收益** | 随参数增加递减 | 随计算增加递减 |
| **优势** | 一次性投资，永久受益 | 按需付费，灵活可调 |
| **趋势** | GPT-4 级别后收益递减明显 | o1/o3 证明仍有巨大空间 |

> **OpenAI o1/o3 的启示**：测试时 scaling 可能是通往 AGI 的第三条路径（前两条：预训练 scaling、后训练 scaling）。

---

## 7. Benchmarks 与评估

### 7.1 推理 Benchmark

| 领域 | Benchmark | 评估指标 |
|------|-----------|---------|
| **数学推理** | GSM8K, MATH, AIME, AMC | 准确率 |
| **代码生成** | HumanEval, MBPP, CodeElo, LiveCodeBench | pass@k, 功能正确性 |
| **逻辑推理** | StrategyQA, BBH (Big-Bench Hard) | 准确率 |
| **科学推理** | GPQA Diamond, JEEBench, MMLU-STEM | 准确率 |
| **通用知识** | MMLU, ARC, TruthfulQA | 准确率 / 宏平均 |
| **工具使用** | APIBench, ToolBench | 任务完成率 |

### 7.2 对齐评估

| 维度 | 指标 | 说明 |
|------|------|------|
| **Helpfulness** | 有用性评分 | 回答是否满足用户需求 |
| **Harmlessness** | 无害性评分 | 是否产生有害内容 |
| **Honesty** | 诚实性评分 | 是否避免幻觉 |
| **Stylistic coherence** | 风格一致性 | 输出风格是否符合预期 |

---

## 8. 核心挑战与未来方向

### 8.1 关键挑战

| 挑战 | 描述 | 现状 |
|------|------|------|
| **灾难性遗忘** (Catastrophic Forgetting) | 后训练导致预训练知识丢失 | 用 KL 散度约束缓解，但无法完全避免 |
| **奖励黑客** (Reward Hacking) | 策略学会欺骗奖励模型 | PRM、自适应奖励模型部分缓解 |
| **推理成本** | CoT/TTS 使推理时间/成本倍增 | Compute-Optimal Scaling 提供部分解决方案 |
| **主观奖励** | 人类偏好不一致、文化差异 | RLAIF + 多元标注者群体 |
| **延迟奖励** | 长序列的信用分配困难 | PRM 分步评估 |

### 8.2 未来方向

1. **更高效的 RL 算法**：GRPO 只是起点，需要更低方差、更稳定的策略优化方法
2. **自适应奖励模型**：动态调整奖励函数以对抗奖励黑客
3. **多模态后训练**：将后训练方法扩展到视觉、音频、具身智能
4. **可扩展的监督**：减少对人工标注的依赖，转向 AI 反馈（RLAIF）和自动验证
5. **推理与理解的统一**：当前 CoT 主要是文本推理，需要与感知、行动统一

---

## 9. 主要模型实现一览

| 模型 | RL 方法 | 微调方式 | 架构 | TTS 支持 |
|------|---------|---------|------|---------|
| **DeepSeek-R1** | GRPO, DPO | SFT + RL | MoE (240B-A22B) | ✅ |
| **GPT-4.5** | RLHF, PPO, RBRM | SFT + RL | MoE | ✅ |
| **LLaMA 3** | DPO | SFT | Single Model | ❌ |
| **Qwen2** | DPO | SFT | Single/MoE | ✅ |
| **Gemini** | RLHF | SFT | Single Model | ❌ |
| **O3** | RL through CoT | SFT + RL | Single Model | ✅ |
| **Claude 3.5** | RLHF, RLAIF | SFT + RL | Single Model | ❌ |
| **Grok 3** | RLHF | SFT + RL | Single Model | ✅ |
| **Kimi K1.5** | RLHF | SFT + RL | Multi-modal | ✅ |

---

## 10. 个人思考

### 10.1 与 OpenThoughts 的互补视角

[OpenThoughts](OpenThoughts--Data-Recipes-for-Reasoning-Models.md) 告诉我们**如何构建高质量的推理 SFT 数据**（数据配方），而本文告诉我们**SFT 之后还能做什么**（RL + TTS）。

两者构成完整的推理模型训练流程：

$$
\text{Pre-training} \rightarrow \underbrace{\text{SFT (OpenThoughts recipe)}}_{\text{数据质量决定上限}} \rightarrow \underbrace{\text{RL (GRPO/DPO)}}_{\text{对齐与精炼}} \rightarrow \underbrace{\text{TTS (CoT/MCTS)}}_{\text{推理时扩展}}
$$

### 10.2 GRPO 的革命性

GRPO 的组内相对基线设计是一个非常优雅的简化——它消除了价值网络（critic），这是 RL 训练中最不稳定的部分之一。这使得：
- 训练成本大幅降低（无需训练第二个大模型）
- 超参数更少（没有 critic 的学习率、架构选择）
- 更容易扩展到大规模（DeepSeek-R1 的成功证明）

**与 DPO 的对比**：DPO 也消除了奖励模型，但它是离线方法，无法利用在线交互；GRPO 保留了在线 RL 的探索能力，同时消除了 critic。

### 10.3 测试时 Scaling 的范式意义

OpenAI o1/o3 展示了测试时 scaling 的巨大潜力。这代表了一种新思路：

> **与其让模型在训练时"记住"所有知识，不如让它在推理时"思考"更久。**

这与人类智能更接近——我们不会把所有知识储存在大脑里（因为记不住），而是学会**如何思考**和**如何查找知识**。

### 10.4 本文的实用价值

作为一篇综述，本文最大的价值是**结构化梳理**了后训练技术的完整图谱。对于从业者：
- **模型选型**：根据 TTS 需求选择模型（DeepSeek-R1、O3 支持 TTS；LLaMA 3、Gemini 不支持）
- **技术选型**：资源有限选 DPO（无需奖励模型）；需要最强性能选 GRPO（DeepSeek-R1 路线）
- **评估设计**：不仅要评估最终答案（ORM），还要评估推理过程（PRM）

---

## 11. 关键引用

```bibtex
@article{kumar2025llm,
  title={LLM Post-Training: A Deep Dive into Reasoning Large Language Models},
  author={Kumar, Komal and Ashraf, Tajamul and Thawakar, Omkar and Anwer, Rao Muhammad and Cholakkal, Hisham and Shah, Mubarak and Yang, Ming-Hsuan and Torr, Phillip HS and Khan, Fahad Shahbaz and Khan, Salman},
  journal={arXiv preprint arXiv:2502.21321},
  year={2025}
}
```

---

**Awesome List**: https://github.com/mbzuai-oryx/Awesome-LLM-Post-training

**相关论文**：
- [OpenThoughts](OpenThoughts--Data-Recipes-for-Reasoning-Models.md) — SFT 数据配方的工程实践
- [LLMRF](LLMRF--Large-Language-Model-Reasoning-Failures.md) — LLM 推理失败的系统诊断
- [SkillVerse](SkillVerse--Assessing-and-Enhancing-LLMs-with-Tree-Evaluation.md) — 推理能力的树形评估
- [EvalTree](EvalTree--Profiling%20Language%20Model%20Weaknesses%20via%20Hierarchical%20Capability%20Trees.md) — 推理弱点的层次化定位
