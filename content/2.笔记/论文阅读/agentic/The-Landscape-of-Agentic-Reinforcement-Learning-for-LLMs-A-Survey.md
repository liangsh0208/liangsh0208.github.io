---
created: 2026-06-09
paper: https://arxiv.org/abs/2509.02547
code: https://github.com/...  # 未提供
tags:
  - agent
  - reinforcement-learning
  - LLM
  - survey
  - POMDP
  - tool-use
  - planning
  - multi-agent
---

# The Landscape of Agentic Reinforcement Learning for LLMs: A Survey

## 一句话总结

本文是一篇**系统性综述 Agentic RL（智能体强化学习）的论文**，系统对比了传统的 **PBRFT（偏好反馈强化微调）**与** Agentic RL 的核心区别——从单步退化的 MDP 到时间扩展的 POMDP**；提出了围绕**六大智能体能力**（规划、工具使用、记忆、自我改进、推理、感知）和**八大任务领域**（搜索、代码、数学、GUI、视觉、具身、多智能体、其他）的双重分类法；整合 500+ 文献，并汇编了开源环境、benchmark 和框架的实用指南。

![](AgenticRL_fig1_paradigm.png)

> **Figure 1**: Agentic RL 的范式总览。左侧为 Agentic LLM 的六大核心能力（Reasoning, Planning, Memory, Perception, Tool Use, Self-Improve）；中间为 Agent 与环境的双向交互（Action <-> Reward）；右侧为环境类型（Web, Vision, Code, Robot, GUI, Collaboration, Math, Market, Lab, Drive 等）。底部标注了主要 RL 算法层（DPO, PPO, VAPO, KTO, SimPO, IPO 等）。

---

### 核心范式对比：PBRFT vs Agentic RL

![](AgenticRL_fig2_paradigm.png)

> **Figure 2**: PBRFT（偏好回馈强化微调）与 Agentic RL 的范式对比。左下为 PBRFT 的退化 MDP（单步决策，确定性转移，纯文本动作）；右侧和右上为 Agentic RL 的完整 POMDP（多步决策，动态转移，文本+动作混合空间，部分可观测，时代扩展）。中间标注了代表性模型和算法。上图还标注了 PBRFT 的特征（自然语言建模、单步推理、逐 token）与 Agentic RL 的特征（长期目标、多步规划、工具调用）的对比。

---

## 1. 研究背景

### 1.1 两条互补的研究线

| 方向 | 目标 | 代表方法 |
|------|------|---------|
| **RL for training LLMs** | 用 RL 训练 LLM 的能力 | PPO, GRPO, Actor-Critic — 用于指令遵循、伦理对齐、代码生成 |
| **LLMs for RL** | 用 LLM 增强 RL | LLM as Planner, Reward Designer, Goal Generator, Information Processor |

### 1.2 LLM-based Agents 的定义

> LLM-based agents 是一种新兴范式，其中 LLM 作为自主或半自主的决策实体，具备推理、规划和执行行动的能力，以追求复杂目标。

### 1.3 PBRFT vs Agentic RL：形式化对比

| 维度 | **PBRFT**（传统 LLM RL） | **Agentic RL** |
|------|--------------------------|----------------|
| **状态空间** | $\mathcal{S}_{\text{trad}} = \{\text{prompt}\}$ 单次静态提示 | $\mathcal{S}_{\text{agent}}$；观测 $o_t = \mathcal{O}(s_t)$；视界 $T > 1$ |
| **动作空间** | $\mathcal{A}_{\text{text}}$ 纯文本序列 | $\mathcal{A}_{\text{agent}} = \mathcal{A}_{\text{text}} \cup \mathcal{A}_{\text{action}}$ 文本 + 工具动作 |
| **转移** | 确定性 $P(s_1 \mid s_0, a) = 1$ | 动态 $P(s_{t+1} \mid s_t, a_t)$ |
| **奖励** | 单一标量 $r(a)$ | $R(s_t, a_t) = \begin{cases} r_{\text{task}} & \text{任务完成} \cr r_{\text{sub}}(s_t, a_t) & \text{步骤级进度} \cr 0 & \text{否则} \end{cases}$ |
| **目标** | $\mathbb{E}_{a \sim \pi_\theta}[r(a)]$（最大化单次回答质量） | $\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \gamma^t R(s_t, a_t)\right]$（最大化长期回报） |

**核心洞察**：PBRFT 训练的是**更好的回答生成器**，Agentic RL 训练的是**能在动态环境中持续决策的智能体**。

### 1.4 RL 算法公式速览

**REINFORCE**：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s_0}\left[\frac{1}{N}\sum_{i=1}^N\left(\mathcal{R}(s_0,a^{(i)})-b(s_0)\right)\nabla_\theta\log\pi_\theta(a^{(i)}|s_0)\right]
$$

**PPO**（剪切目标）：

$$
L_{PPO}(\theta) = \frac{1}{N}\sum_{i=1}^N \min\left(\underbrace{\frac{\pi_\theta(a_t^{(i)}|s_t)}{\pi_{\theta_{old}}(a_t^{(i)}|s_t)}}_{r_t(\theta)} A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)
$$

其中 $A(s_t, a_t) = \mathcal{R}(s_t, a_t) - V(s_t)$

**DPO**（直接偏好优化）：

$$
L_{DPO} = -\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]
$$

**GRPO**（组相对策略优化）：

$$
\hat{A}(s_t, a_t) = \frac{R(s_t, a_t) - \text{mean}(R(s_t^{(g)}, a_t^{(g)})}{\text{std}(R(s_t^{(g)}, a_t^{(g)})}
$$

---

## 2. 双重分类法

### 2.1 分类法一：能力视角（Model Capability）

| 能力 | 子方向 | 代表性工作 |
|------|--------|-----------|
| **Planning** | 外部引导（RAP, LATS）<br>内部驱动（ETO, VOYAGER, DSP, RLTR, AdaPlan, Planner-R1） | RAP, LATS, MAPF-DT, ETO, VOYAGER, AdaPlan, PilotRL |
| **Tool Use** | ReAct → Tool-integrated RL → Long-horizon TIR | ReAct, AgentTuning, FireAct, ToolRL, OTC-PO, ReTool, VTool-R1, DeepEyes, ARTIST, ToRL |

### 工具使用的发展阶段

![](AgenticRL_fig3_tooluse_timeline.png)

> **Figure 3**: Agentic 工具使用的发展历程。从早期的 ReAct-style Tool Calling（ReAct, AgentTuning, FireAct），到中期的 Tool-integrated RL（ToolRL, ToOL, Agentic Reasoning, ARTIST, DeepEyes, Pixel-Reasoner, OSWORLD），再到最新的 Long-horizon TIR（OpenAI o3, DeepResearch, Spark, Kimi-K2）。时间轴展示了从简单工具调用到复杂、多步骤、长程工具使用推理的演进。
| **Memory** | RAG-style Memory<br>Token-level Memory<br>Structured Memory | MemoryBank, MemGPT, HippoRAG, MemAgent, MEM1, MemoryLLM, IMM |
| **Self-Improvement** | Verbal Self-correction<br>Internalizing Self-correction<br>Iterative Self-training | — |
| **Reasoning** | Fast Reasoning（直觉推理）<br>Slow Reasoning（审慎推理） | 快思维链 vs 深思维树 |
| **Perception** | Text → Image/Video → Audio | 从被动感知到主动视觉认知 |

### 2.2 分类法二：任务视角（Task Application）

| 任务领域 | 关键方法 | 典型环境 |
|----------|---------|---------|
| **Search & Research** | 开源：Web search RL, Internal knowledge RL<br>闭源：OpenAI DeepResearch | Web, 互联网 |
| **Code Agent** | Outcome.reward RL, Process.reward RL, Code World Models | SWE-Bench, GitHub, IDE |
| **Math Agent** | Informal mathematical reasoning, Formal mathematical reasoning (Hybrid reward RL) | Lean, Isabelle, Coq |
| **GUI Agent** | RL-free, Static GUI RL, Interactive GUI RL | OSWorld, MobileEnv |
| **Vision Agent** | Image tasks, Video tasks, 3D vision | Embodied AI 环境 |
| **Embodied Agent** | VLA Navigation, VLA Manipulation, Voyager | Habitat, RoboGen |
| **Multi-Agent** | Non-parametric coordination RL, Selected policy RL, End-to-end MARL | Multi-Agent 合作环境 |
| **Other Tasks** | TextGame, Table, Time Series, Social | 各领域专用 benchmark |

---

## 3. Agentic RL 的核心算法层

### 3.1 算法全景

| 算法 | 变体/相关 | 用途 | 特点 |
|------|----------|------|------|
| **PPO** | — | 通用 RLHF | 剪切目标，稳定性高 |
| **DPO** | IPO, KTO, SimPO, VAPO | 偏好对齐 | 无需奖励模型，开源首选 |
| **GRPO** | Step-GRPO, rStar2-GRPO | 推理 RL | 无价值网络，组内基线 |
| **GHPO** | — | 分组层次偏好优化 | 分层优势估计 |
| **GPPO** | — | 广义 PPO | 更灵活的代理目标 |
| **REINFORCE++** | REINFORCE+ | 策略梯度 | REINFORCE 基线增强 |
| **DSPO** | — | Direct RPO | 无需参考模型 |
| **VPO** | — | Vanilla Policy Optimization | 简化 PPO |

### 3.2 商业系统

OpenAI DeepResearch, Kimi K2, Kimi K2.5, Qwen QwQ-32B, Zhipu GLM Z1, Microsoft rStar2-Agent, Meituan LongCat

---

## 4. 环境模拟器与框架

### 4.1 环境模拟器分类

| 类别 | 环境示例 |
|------|---------|
| **Web** | WebShop, Mind2Web, BrowserGym, WebArena, VisualWebArena, TTS |
| **GUI** | OSWorld, AndroidEnv, MobileEnv |
| **Coding & SWE** | SWE-bench, ProjectEuler, LiveCodeBench, DS-1000, MLE-bench |
| **科学** | ScienceWorld, PhotoshopGPT, Bioinformatics Environment |
| **游戏** | Minecraft, ALFWorld, TextWorld, MiniHack, NetHack |
| **通用** | GAIA, AgentBench, AgentBoard, ToolBench |

### 4.2 RL 框架

- **Agentic RL 框架**：rllm-rl, EasyR1, Agentic RL Toolkit
- **RLHF & LLM 微调框架**：trl, OpenRLHF, LLaMA-Factory
- **通用 RL 框架**：RLLib, Tianshou

---

## 5. 开放挑战与未来方向

### 5.1 挑战

| 挑战 | 描述 |
|------|------|
| **Trustworthiness** | 安全性（Security）、幻觉（Hallucination）、谄媚（Sycophancy） |
| **Scaling Training** | 计算、模型规模、数据规模、训练效率 |
| **Scaling Environments** | 环境复杂度、交互延迟、真实世界部署困难 |
| **Mechanistic Debate** | RL 在 LLM 中的机制理解（案例：数学推理仍是黑箱） |
| **Deployment** | 护栏与安全模式、人在环路验证、层次编排、智能体间通信协议 |
| **Broader Impact** | 双重用途风险、环境可持续性、劳动力影响、偏见放大、评估污染 |

### 5.2 未来方向

1. **Deliberation + Intuition 的融合**：内部化结构化搜索过程（非外部工具）
2. **Meta Evolution of Reflection**：自我改进能力的进化学习
3. **跨模态统一**：文本、图像、视频、音频、具身感知的统一 Agentic 框架
4. **Real-World Deployment**：从模拟环境到真实世界的安全部署

---

## 6. 局限性与思考

### 6.1 本文局限性
1. **Out of scope**：RL for human value alignment（如 RLHF 对齐）不在本文讨论范围内
2. **传统 RL 不覆盖**：非 LLM-based 的传统 RL 算法略过
3. **纯静态 Benchmark 训练**：不讨论用 RL 提升 LLM 在静态 benchmark 上的性能
4. **500+ 文献综述**：广度有余，深度取决于各子领域

### 6.2 个人思考

**与 LLM Post-Training 综述的互补性**：[LLM Post-Training 综述](LLMPT--LLM-Post-Training-A-Deep-Dive-into-Reasoning-LLMs.md) 聚焦 LLM 后训练技术（SFT/RL/TTS），本文进一步将 LLM "智能体化"——不仅训练更好的语言模型，而是让 LLM 在动态环境中自主决策。

**Agentic RL 的独特价值**：通过将 LLM 嵌入 POMDP，模型从"被动回答者"变为"主动探索者"。**工具使用**在此框架下获得了形式化基础——工具调用是动作空间 $\mathcal{A}_{\text{action}}$ 的一部分，而非简单的 prompt engineering。

**最有前景的方向**：规划能力的外化 vs 内化。当前主流工作（RAP, LATS）将规划作为外部模块，而未来方向是**模型内部化搜索过程**——这本质上是在 LLM 中实现 System 2 thinking（慢思考）。

---

## 7. 关键引用

```bibtex
@article{zhang2026agentic,
  title={The Landscape of Agentic Reinforcement Learning for LLMs: A Survey},
  author={Zhang, Guibin and Geng, Hejia and Yu, Xiaohang and Yin, Zhenfei and Zhang, Zaibin and Tan, Zelin and Zhou, Heng and Li, Zhongzhi and Xue, Xiangyuan and Li, Yijiang and Zhou, Yifan and Chen, Yang and Zhang, Chen and Fan, Yutao and Wang, Zihu and Huang, Songtao and Piedrahita-Velez, Francisco and Liao, Yue and Wang, Hongru and Yang, Mengyue and Ji, Heng and Wang, Jun and Yan, Shuicheng and Torr, Philip and Bai, Lei},
  journal={arXiv preprint arXiv:2509.02547},
  year={2026}
}
```

---

**相关论文**：
- [LLM Post-Training 综述](LLMPT--LLM-Post-Training-A-Deep-Dive-into-Reasoning-LLMs.md) — LLM 后训练技术全面梳理
- [DeepSeek-R1 / OpenThinker3](OpenThoughts--Data-Recipes-for-Reasoning-Models.md) — 推理模型训练的工程实践
- [OpenThoughts](OpenThoughts--Data-Recipes-for-Reasoning-Models.md) — SFT 数据配方
