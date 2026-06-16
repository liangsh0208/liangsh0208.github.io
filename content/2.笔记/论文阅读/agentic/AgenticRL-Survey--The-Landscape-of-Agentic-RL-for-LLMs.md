---
created: 2026-06-09
paper: https://arxiv.org/abs/2509.02547
authors: Guibin Zhang, Hejia Geng, Xiaohang Yu, Zhenfei Yin, Zaibin Zhang, Zelin Tan, Heng Zhou, Zhongzhi Li, Xiangyuan Xue, Yijiang Li, Yifan Zhou, Yang Chen, Chen Zhang, Yutao Fan, Zihu Wang, Songtao Huang, Francisco Piedrahita-Velez, Yue Liao, Hongru Wang, Mengyue Yang, Heng Ji, Jun Wang, Shuicheng Yan, Philip Torr, Lei Bai
published: Transactions on Machine Learning Research
tags:
  - AgenticRL
  - Survey
  - LLM
  - ReinforcementLearning
  - Agent
---

# The Landscape of Agentic Reinforcement Learning for LLMs: A Survey

## 一句话总结

本文首次系统性地提出了 **Agentic Reinforcement Learning（Agentic RL）** 的概念框架，通过 **双重视角分类体系（Capability + Task）** 对 500 余篇相关文献进行全景式综述，阐明了 RL 如何将 LLM 从被动序列生成器转化为动态环境中自主决策的智能体。

---

![](AgenticRL_fig1_paradigm.png)

> **Figure 1**: LLM RL 到 Agentic RL 的范式迁移。红色表示传统 LLM RL 的特征，青色表示 Agentic RL 所需的特征，紫色表示现有 Agentic RL 实现。

---

## 1. 研究背景与动机

### 1.1 传统 LLM RL 的局限

传统 LLM RL（如 RLHF、PPO、DPO）将 LLM 视为**被动序列生成器**：
- 输入：单条 prompt（$T=1$）
- 输出：纯文本
- 反馈：单一标量奖励 $r(a)$
- 目标：最大化期望奖励 $\mathbb{E}[r(a)]$

然而，现实世界中的智能体需要：
- **多步交互**：与动态环境持续交互
- **结构化动作**：不仅输出文本，还需调用工具、操作 GUI、执行代码等
- **时序信度分配**：稀疏/延迟奖励下的长期规划
- **部分可观测性**：环境状态不完全可见

### 1.2 Agentic RL 的定义

本文将 **Agentic RL** 定义为：
> "将 LLM 从被动序列生成器重构为嵌入复杂动态世界中的自主决策智能体的强化学习范式。"

核心转变体现在 MDP/POMDP 的五个维度：

| 维度 | 传统 LLM RL | Agentic RL |
|------|------------|-----------|
| **状态** | 单条 prompt，$T=1$ | 多步、部分观测 |
| **动作空间** | 纯文本 | 文本 $\cup$ 结构化动作（工具调用、代码执行等） |
| **转移函数** | 确定性 | 动态、随机性 |
| **奖励** | 单一标量 $r(a)$ | 分步奖励，稀疏/稠密混合 |
| **目标** | $\mathbb{E}[r(a)]$ | $\mathbb{E}[\sum_t \gamma^t R(s_t, a_t)]$ |

### 1.3 研究空白与本文贡献

现有综述存在明显缺口：
- **RLHF 综述**：聚焦对齐，不涉及 Agent 能力
- **LLM Agent 综述**：多为应用层面的排列组合，缺乏根植于 RL 原理的系统性分类

本文核心贡献：
1. 首次提出 **Agentic RL** 正式定义与数学框架
2. 构建 **双重视角分类体系**（Capability + Task）
3. 系统梳理 **RL 算法家族**（20+ 种方法对比，见 Table 2）
4. 覆盖 **环境模拟器与开发框架**
5. 前瞻性讨论 **开放性挑战与未来方向**

---

## 2. 方法/框架

### 2.1 数学框架

Agentic RL 基于 **POMDP**（部分可观测马尔可夫决策过程）建模：

- **观测空间** $\mathcal{O}$：环境返回的部分观测（文本、图像、结构化数据等）
- **动作空间** $\mathcal{A}$：LLM 智能体的动作，包括纯文本和结构化动作（工具调用、代码执行、API 请求等）
- **转移函数** $P(o_{t+1} \mid s_t, a_t)$：环境动态，通常是随机且未知的
- **奖励函数** $R(s_t, a_t)$：分步奖励，可为稀疏或稠密
- **目标**：最大化累积折扣奖励 $\mathbb{E}\left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t)\right]$

### 2.2 RL 算法家族

本文将 Agentic RL 算法分为三大族（详见 Table 2）：

**A. Online Policy Gradient（在线策略梯度）**
- REINFORCE：蒙特卡洛梯度估计
- PPO： clipped surrogate objective，稳定策略更新
- TRPO： KL 散度约束的自然策略梯度
- GAE：广义优势估计减少方差

**B. Offline/Preference-based RL（离线/偏好 RL）**
- DPO：直接偏好优化，绕过奖励模型
- IPO：身份偏好优化
- KTO：卡尼曼-特沃斯基优化
- SLiC-HF：序列似然校准

**C. Evolution/Search-based（进化/搜索方法）**
- Rs：拒绝采样
- Best-of-N：N 次采样选最优
- Beam Search + Reward Model：束搜索 + 重排序
- MCTS + LLM：蒙特卡洛树搜索引导生成

| 特性 | REINFORCE | PPO | DPO | GRPO |
|------|-----------|-----|-----|------|
| 需要 Value Model | 否 | 是 | 否 | 否 |
| 在线采样 | 是 | 是 | 否 | 是 |
| 需 Reward Model | 可选 | 可选 | 否（直接用偏好数据） | 可选 |
| 主策略更新目标 | 策略梯度 | Clipped surrogate | Bradley-Terry 似然 | 组内相对优势 |

### 2.3 双重视角 Taxonomy

本文提出两个正交的分类轴：

**视角一：Capability（模型能力）**
聚焦 Agent 应具备的**核心认知与行动能力**：Planning、Tool Using、Memory、Self-Improvement、Reasoning、Perception 等。

**视角二：Task（应用领域）**
聚焦 Agent 落地的**具体任务域**：Search/Research、Code、Mathematics、GUI、Vision、Embodied、Multi-Agent 等。

两个视角相互交叉：同一种能力可以在多个任务中体现，同一任务也需要多种能力协同。

---

## 3. 核心内容分类

### 3.1 Capability Perspective（能力视角）

![](AgenticRL_fig2_capability.png)

> **Figure 2**: Agentic RL 能力视角全景图，展示 Planning、Tool Using、Memory、Self-Improvement、Reasoning、Perception 等核心能力及其相互关系。

#### 3.1.1 Planning（规划）

规划是 Agent 在动态环境中制定行动序列的能力。

**外部引导（External Guide）**
- **RAP**（Reasoning via Planning）：将推理建模为马尔可夫决策过程，利用 MCTS 进行规划
- **LATS**（Language Agent Tree Search）：将 LLM 与 MCTS 结合，通过树搜索探索推理路径
- **Learning When to Plan**：自适应决定何时需要显式规划
- **MAPF-DT**：多智能体路径规划的决策树方法

**内部驱动（Internal Driver）**
- **ETO**（Explore to Optimize）：通过探索优化规划能力
- **VOYAGER**：Minecraft 中的终身学习 Agent，通过代码技能库实现规划
- **DSP**：通过结构化 prompting 实现规划
- **RLTR**：RL for Tool Reasoning，工具推理的强化学习
- **AdaPlan**：自适应规划器
- **Planner-R1**：基于 DeepSeek-R1 的推理规划方法

**展望**：直觉与深思熟虑的融合（synthesis of deliberation and intuition）。

#### 3.1.2 Tool Using（工具使用）

工具使用是 Agent 调用外部资源扩展自身能力的关键。

![](AgenticRL_fig3_tooluse_timeline.png)

> **Figure 3**: Agentic 工具使用能力的发展时间线，展示了从 Toolformer 到 Search-o1 等代表性工作的演进轨迹。

**三阶段演进**：

**阶段 1：ReAct 风格（推理+行动）**
- **Toolformer**：训练 LLM 决定何时以及如何调用 API
- **FireAct**：ReAct 模式的精炼，集成多种工具
- **AgentTuning**：构建 Agent 轨迹数据并进行指令微调
- **Agent-FLAN**：将 Agent 任务转化为自然语言格式进行学习

**阶段 2：工具集成 RL（Tool-integrated RL）**
- **ToolRL**：为工具调用设计的 RL 框架
- **OTC-PO**：工具调用偏好优化
- **ReTool**：通过 RL 优化工具选择
- **AutoTIR**：自动化工具交互推理
- **VTool-R1**：视觉工具推理的 RL 方法
- **DeepEyes**：深度视觉工具使用
- **Pixel-Reasoner**：像素级推理与工具调用
- **Agentic Reasoning**：Agent 推理框架
- **ARTIST**：工具学习的 RL 框架
- **ToRL**：工具导向的 RL
- **WebSailor / WebDancer**：Web 环境中的工具使用 Agent
- **Search-o1 / ReSearch**：搜索增强的推理框架
- **GiGPO**：群体智能工具搜索
- **SpaRL**：空间推理与工具使用 RL

**展望**：长程工具交互推理（long-horizon TIR）中的时序信度分配问题。

#### 3.1.3 Memory（记忆）

记忆使 Agent 能够利用历史经验指导当前决策。

| 类别 | 代表性工作 | 机制描述 |
|------|-----------|---------|
| **RAG 风格** | Memory-R1, Mem-$\alpha$, Memory-as-action | 检索增强型记忆，将历史信息作为外部知识库检索 |
| **Token 级** | MemAgent, MEM1, ReSum, Context Folding | 在上下文中直接编码记忆，通过 token 级操作管理 |
| **结构化** | （新兴方向） | 图结构、数据库等结构化记忆表示 |

#### 3.1.4 Self-Improvement（自我改进）

Agent 通过反思和自我修正实现能力提升。

- **Verbal Self-Correction**：通过语言形式的自我纠错
- **Internalizing Self-Correction**：将纠错能力内化为模型参数
- **Iterative Self-Training**：通过迭代自举训练持续提升

**展望**：反思能力的元进化（Meta Evolution of Reflection Ability）。

#### 3.1.5 Reasoning（推理）

- **Fast Reasoning**：直觉式、高效的快速推理
- **Slow Reasoning**：深思熟虑、结构化的慢速推理（如 CoT、ToT）

**展望**：将慢速推理整合到 Agentic 推理中。

#### 3.1.6 Perception（感知）

从被动视觉认知到主动感知：
- **Grounding-driven**：基于视觉定位的主动感知
- **Tool-driven**：工具驱动的感知（如调用视觉 API）
- **Generation-driven**：生成式感知
- **Audio Perception**：音频感知（新兴方向）

---

### 3.2 Task Perspective（任务视角）

![](AgenticRL_fig7_domain_tree.png)

> **Figure 7**: Agentic RL 任务视角分类树，涵盖 Search、Code、Mathematics、GUI、Vision、Embodied、Multi-Agent 等主要任务领域。

#### 3.2.1 Search & Research Agent（搜索与研究 Agent）

**开源方向**：
- 外部互联网搜索：OpenScholar、STORM、WebAgent
- 内部知识搜索：基于私有文档库的 RAG Agent

**闭源方向**：
- 工业级 Agent：OpenAI Deep Research、Google Deep Research
- **OpenAI Deep Research 案例研究**：结合多步搜索、信息整合与报告生成，代表了当前搜索 Agent 的最高水平

#### 3.2.2 Code Agent（代码 Agent）

![](AgenticRL_fig5_code_benchmarks.png)

> **Figure 5**: 代码 Agent 评测基准概览，涵盖代码生成、代码修复、软件工程等多种任务类型。

**三大方向**：
1. **代码生成**：
   - 结果奖励 RL：PASS@K 作为奖励信号
   - 过程奖励 RL：对代码生成过程的中间步骤赋予奖励

2. **迭代代码优化**：
   - 通过编译器反馈、测试用例结果进行多轮迭代改进

3. **自动化软件工程（ASE）**：
   - SWE-bench、GitHub Issue 解决
   - Code World Models（新兴）：构建代码执行的世界模型

#### 3.2.3 Mathematical Agent（数学 Agent）

![](AgenticRL_fig6_math_benchmarks.png)

> **Figure 6**: 数学推理评测基准概览，涵盖 Informal Reasoning（GSM8K、MATH 等）和 Formal Reasoning（Lean、Isabelle 等）。

**两大范式**：

| 范式 | 奖励类型 | 代表性工作 |
|------|---------|-----------|
| **Informal Reasoning** | 结果奖励（答案正确性） | GSM8K、MATH 上的 RL 训练 |
| | 过程奖励（每步正确性） | PRM800K、Math-Shepherd |
| **Formal Reasoning** | 结果奖励（证明验证通过） | HTPS、DeepSeek-Prover |
| | 过程奖励（策略状态值） | ReProver、LeanDojo |
| | 混合奖励 | 结合形式验证与过程监督 |

#### 3.2.4 GUI Agent（GUI Agent）

**三个阶段**：
1. **无 RL 方法**：纯 VLM + 提示词、静态数据 SFT
2. **静态环境中的 RL**：在预录制的 GUI 交互数据上训练
3. **交互式环境中的 RL**：在真实或模拟 GUI 环境中在线交互学习

#### 3.2.5 Vision Agents（视觉 Agent）

- **Image Agents**：视觉问答、图像编辑、视觉定位
- **Video Agents**：视频理解、视频生成、时序推理
- **3D Agents**：3D 场景理解、视觉导航

#### 3.2.6 Embodied Agents（具身智能）

- **VLA Navigation**：视觉-语言-动作导航（如 RT-2、OpenVLA）
- **VLA Manipulation**：机器人操作（如 RT-X、Octo）
- **案例研究：Voyager**：Minecraft 中的终身学习 Agent，通过"技能库"（skill library）实现开放式目标完成

#### 3.2.7 Multi-Agent Systems（多智能体系统）

**四个层次**：
1. **无 RL 演化**：基于规则/提示词的多 Agent 协作
2. **非参数协调模块的 RL**：学习通信协议、调度策略
3. **部分 Agent 策略的 RL**：只训练部分 Agent 的策略
4. **端到端多 Agent RL**：所有 Agent 策略同时通过 RL 训练

---

## 4. 关键图表

### 4.1 环境交互循环

![](AgenticRL_fig4_environment.png)

> **Figure 4**: Agent-环境交互的 RL 循环。Agent 基于核心能力（Planning、Tool Using、Memory 等）生成动作，环境返回观测与奖励，形成闭环。

### 4.2 RL 算法对比表（Table 2）

本文对比了 20+ 种 RL 算法变体，核心维度包括：
- 是否需要 Value Model
- 在线/离线训练
- 是否需要 Reward Model
- 策略更新目标函数形式
- 适用场景

### 4.3 记忆方法对比表（Table 3）

| 类别 | 方法 | 记忆位置 | 优点 | 局限 |
|------|------|---------|------|------|
| RAG 风格 | Memory-R1 | 外部向量库 | 扩展性强、可解释 | 检索质量依赖嵌入 |
| Token 级 | MEM1 | 上下文窗口 | 端到端训练 | 受限于上下文长度 |
| 结构化 | 图/数据库 | 外部结构化存储 | 关系推理强 | 需要额外的结构化学习 |

---

## 5. 开放挑战与未来方向

### 5.1 可信赖性（Trustworthiness）
- **安全性（Security）**：Agent 在开放环境中可能被攻击、诱导执行恶意操作
- **幻觉（Hallucination）**：Agent 在多步交互中累积和放大幻觉
- **谄媚（Sycophancy）**：Agent 为追求奖励而迎合用户错误观点

### 5.2 Agentic 训练规模化（Scaling up Agentic Training）
- **计算资源**：Agentic 训练需要大量的环境交互，计算成本巨大
- **模型规模**：更大的模型是否意味着更强的 Agentic 能力？ scaling law 尚不明确
- **数据规模**：高质量交互数据的获取与标注成本
- **效率优化**：样本效率、时间效率的提升空间

### 5.3 Agentic 环境规模化（Scaling up Agentic Environments）
- 从封闭、静态环境到开放、动态环境的过渡
- 环境多样性与泛化能力的平衡
- 真实世界部署的安全边界

### 5.4 RL 在 LLM 中的机制性辩论（Mechanistic Debate）

**案例研究：数学推理**
- **观点 A**：RL 主要通过"模式匹配"和"记忆训练数据"提升性能
- **观点 B**：RL 真正教会了模型"推理"和"泛化"
- **本文立场**：当前证据尚不充分，需要更严格的因果分析和可解释性研究

### 5.5 架构模式（Architectural Patterns）
- **Guardrails**：安全护栏机制
- **Human-in-the-loop**：人在回路中的有效交互设计
- **Hierarchical Orchestration**：分层编排架构
- **Inter-Agent Communication**：Agent 间通信协议

### 5.6 更广泛的社会影响
- **双重用途风险**：Agent 能力可被用于恶意目的
- **环境可持续性**：Agentic 训练的巨大能耗
- **劳动力市场**：Agent 自动化对就业的影响
- **偏见放大**：Agent 在开放环境中可能放大训练数据偏见
- **评测污染**：基准数据泄漏导致的性能虚高

---

## 6. 环境与框架

### 6.1 环境模拟器

| 类别 | 代表性环境 | 说明 |
|------|-----------|------|
| **Web 环境** | WebShop, WebArena, Mind2Web | 网页交互模拟 |
| **GUI 环境** | OSWorld, Windows Agent Arena | 操作系统级交互 |
| **代码环境** | SWE-bench, InterCode | 代码执行与软件工程 |
| **科学/研究** | ScienceWorld, MATH, PubMed | 特定领域任务 |
| **游戏/通用** | Minecraft (Voyager), ALFWorld | 开放式环境 |

### 6.2 RL 框架

- **Agentic RL 专用框架**：如用于 Agent 训练的专用库
- **RLHF/LLM 微调框架**：TRL, OpenRLHF, veRL, OpenRL
- **通用 RL 框架**：RLlib, Stable-Baselines3, Tianshou

---

## 7. 个人思考

### 7.1 分类框架的启发

本文的 **Capability + Task 双重视角** 是非常优雅的分类方式。相比传统的按应用领域排列，这种分类让读者既能理解"一个 Agent 需要哪些能力"，也能看到"某个任务需要组合哪些能力"。这种正交分类特别适合快速定位相关工作。

### 7.2 当前的热点与缺口

**热点方向**：
- **工具使用 + 搜索**：Search-o1、Deep Research 代表了当前最高水平的应用
- **代码 Agent + RL**：SWE-bench 上的 RL 训练是工程落地的关键
- **多 Agent 协作**：从简单提示词编排走向 RL 优化的通信协调

**缺口与机会**：
- **长程时序信度分配**：当前工具使用和规划在长序列上仍面临梯度消失/爆炸
- **统一评价指标**：不同任务域缺乏统一的 Agentic RL 评价框架
- **与世界模型的结合**：代码世界模型、物理世界模型与 LLM Agent 的深度融合尚在早期

### 7.3 方法论反思

本文提出的 **POMDP 框架** 是一个很好的数学基础，但在实际应用中：
- 环境的转移函数 $P(o_{t+1} \mid s_t, a_t)$ 通常是**未知且不可微**的，这限制了基于模型的 RL 方法
- 奖励设计（Reward Engineering）仍然是高度**启发式**的，缺乏系统性原则
- **样本效率**是最大瓶颈：LLM 每次交互成本高昂，导致在线 RL 的探索受限

### 7.4 与自己研究的关联

对于从事 LLM Post-Training 的研究者，本文提供了全面的 Agentic RL 算法地图。特别值得关注：
- **GRPO** 等无需 Value Model 的在线 RL 方法，降低了训练门槛
- **过程奖励模型（PRM）** 在数学和代码任务上的成功经验，可以迁移到其他需要中间步骤验证的任务
- **Memory 的结构化表示** 是提升长程任务能力的关键，值得深入研究

---

## 8. 关键引用

```bibtex
@article{zhang2025landscape,
  title={The Landscape of Agentic Reinforcement Learning for LLMs: A Survey},
  author={Zhang, Guibin and Geng, Hejia and Yu, Xiaohang and Yin, Zhenfei and Zhang, Zaibin and Tan, Zelin and Zhou, Heng and Li, Zhongzhi and Xue, Xiangyuan and Li, Yijiang and Zhou, Yifan and Chen, Yang and Zhang, Chen and Fan, Yutao and Wang, Zihu and Huang, Songtao and Piedrahita-Velez, Francisco and Liao, Yue and Wang, Hongru and Yang, Mengyue and Ji, Heng and Wang, Jun and Yan, Shuicheng and Torr, Philip and Bai, Lei},
  journal={Transactions on Machine Learning Research},
  year={2025},
  url={https://arxiv.org/abs/2509.02547}
}
```

---

## 附录：文中涉及关键方法速查

| 缩写 | 全称 | 类别 |
|------|------|------|
| RAP | Reasoning via Planning | 规划 |
| LATS | Language Agent Tree Search | 规划 |
| ETO | Explore to Optimize | 规划 |
| VOYAGER | Voyager (Minecraft Agent) | 规划/具身 |
| Toolformer | Toolformer | 工具使用 |
| ReAct | Reasoning + Acting | 工具使用 |
| DPO | Direct Preference Optimization | RL 算法 |
| PPO | Proximal Policy Optimization | RL 算法 |
| GRPO | Group Relative Policy Optimization | RL 算法 |
| PRM | Process Reward Model | 奖励建模 |
| ORM | Outcome Reward Model | 奖励建模 |
| MCTS | Monte Carlo Tree Search | 搜索 |
| VLA | Vision-Language-Action | 具身智能 |
| SWE-bench | Software Engineering Benchmark | 代码评测 |
| GSM8K / MATH | 数学推理基准 | 数学评测 |
