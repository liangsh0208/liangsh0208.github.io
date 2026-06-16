---
created: 2026-06-09
tags:
  - AgenticRL
  - Survey
  - PaperMap
  - 总结
---

# Agentic RL 论文全景与知识地图

> 本文档是对 [syhya 博客](https://syhya.github.io/zh/posts/2025-09-30-agentic-rl/) 涉及的 20 篇论文的系统性梳理，按 **"评估 → 数据 → 奖励 → 算法 → 框架 → 案例"** 的完整 pipeline 结构组织，浓缩 9 篇核心论文笔记的精华。

---

## 0. 一张图看懂 Agentic RL 全景

**定义**：Agentic RL = 让 LLM 通过与外部环境（搜索引擎、代码解释器、浏览器、数据库等）的交互，以强化学习的方式自主学习何时调用工具、如何调用工具、以及如何整合多轮交互结果来完成复杂任务。

与传统 LLM-RL（如 RLHF、 reasoning RL）的核心区别：

| 维度 | 传统 LLM RL | Agentic RL |
|------|------------|-----------|
| 环境 | 静态文本（人类偏好、答案正确性） | 动态外部系统（搜索、代码、浏览器） |
| 动作空间 | 生成 token | 生成 token + 工具调用决策 |
| 观测 | 用户 prompt | 工具返回结果（外部知识） |
| 训练挑战 | 奖励稀疏、值估计困难 | 环境延迟、观测噪声、训练崩溃 |
| 关键创新 | PPO, GRPO, DPO | 检索掩码、训推分离、Agentic CPT |

---

## 1. 评估：如何衡量 Agent 能力？

### 1.1 评估设计原则

一篇未在 blog 中单独开笔记但引用频繁的报告是 **Wei, 2024 ("Successful language model evals")**，提出好评估应具备 7 大特质：

> Validity（有效性）→ Reliability（可靠性）→ Generalization（泛化性）→ Coverage（覆盖度）→ Efficiency（效率）→ Human-Interpretability（可解释性）→ Robustness（鲁棒性）

反面教材包括：
- **HELM**（Liang et al., 2022）：指标过多，难以指导实际决策
- **BIG-Bench**（Srivastava et al., 2022）：运行复杂，评估成本高
- **GLUE / SuperGLUE**（Wang et al., 2018/2019）：易过拟合，已接近饱和

另一关键概念来自同一作者的 **"Asymmetry of verification"**：验证的不对称性（Verifier's Law）—— 验证答案比生成答案容易得多。这是 Agentic RL 中**结果奖励（outcome reward）**可行的理论基础。

### 1.2 Agentic 核心基准

| 基准 | 论文 | 评估能力 | 难度 |
|------|------|---------|------|
| **BrowseComp** | Wei et al., 2025 | 网页浏览与信息综合 | 🔴 高 |
| **SWE-bench** | Jimenez et al., 2023 | 真实 GitHub Issue 修复 | 🔴 高 |
| **SWE-bench Verified** | OpenAI, 2024 | 人工验证的高质量子集 | 🔴 高 |
| **GAIA** | Mialon et al., 2023 | 通用 Agent 能力 | 🟡 中高 |
| **HLE** | Phan et al., 2025 | 专家级知识问答 | 🔴 极高 |

---

## 2. 数据合成：Agent 训练的燃料

数据是 Agentic RL 的最大瓶颈之一——高质量 agent 轨迹难以获取、昂贵且覆盖有限。

### 2.1 AgentFounder：无监督 + 有监督的混合合成

**论文**: [AgentFounder: Scaling Agents via Continual Pre-training](AgentFounder--Scaling-Agents-via-CPT.md) | arXiv:2509.13310

**核心贡献**：在预训练与后训练之间插入 **Agentic Continual Pre-training (Agentic CPT)** 阶段

| 方法 | 监督信号 | 核心思想 | 成本 |
|------|---------|---------|------|
| **FAS** (First-order) | 无 | Entity-Anchored Memory → Multi-Style QA | 零 API 成本 |
| **HAS** (Higher-order) | 有（轨迹反馈） | Step-level Scaling + Contrastive Decision-Action | 利用废弃轨迹 |

**关键数据**：
- FAS 拒采样过滤后正确率从 50% → 82%
- 两阶段训练（32K → 128K context），200B + 100B tokens
- AgentFounder-30B 在 10 个基准上取得 SOTA

### 2.2 WebShaper：形式化驱动的知识投影

**论文**: [WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization](WebShaper--Agentic-Data-Synthesizing.md) | arXiv:2507.15061

**核心思想**：用**集合论**形式化信息搜索任务 → 以**知识投影 (Knowledge Projection)** 为基本单元 → Agentic Expander 逐层扩展

| 组件 | 功能 |
|------|------|
| KP (Knowledge Projection) | 知识树的交/并运算，生成困难问题 |
| Agentic Expander | 三种扩展范式（随机/顺序/逐层） |
| 逐层扩展策略 | 系统控制难度递进 |

---

## 3. 奖励设计：如何给 Agent 打分？

博客中明确提到奖励设计是 Agentic RL 的关键课题。当前主流方向：

### 3.1 结果奖励（Outcome Reward）—— 最简单也最 scalable

| 论文 | 奖励形式 | 设计 |
|------|---------|------|
| **Search-R1** | Exact Match | $r = \text{EM}(a_{pred}, a_{gold})$，无任何格式奖励 |
| **ReTool** | Outcome-based | AIME 答案正确性 + 代码执行结果 |
| **DeepSeekMath/GRPO** | Rule-based | 数学答案的自动判题 |

**优势**：无需训练 Reward Model，避免 reward hacking，易于扩展
**劣势**：信用分配（credit assignment）在长轨迹中困难

### 3.2 生成式奖励（LLM-as-a-Judge）

**论文**: Zheng et al., 2023 — Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

- 用 GPT-4 等强模型作为评判者
- 适合开放域、主观性强的任务
- 成本较高，且存在位置偏差、长度偏差等问题

---

## 4. 优化算法：从 PPO 到 GRPO 到 GSPO

### 4.1 三阶段演进

```
PPO (2017) 
  → GRPO (2024, DeepSeekMath)  去除 Critic，组相对优势
  → GSPO (2025, Qwen Team)     修正 token 级噪声，序列级优化
```

### 4.2 PPO：基线算法

**论文**: Schulman et al., 2017 — Proximal Policy Optimization Algorithms

- Actor-Critic 架构，需 Value Model
- 优点：训练稳定，理论基础扎实
- 缺点：显存开销大（= 2 × Policy 模型），值估计在长序列上不可靠

### 4.3 GRPO：去 Critic 的革命

**论文**: [DeepSeekMath / GRPO](GRPO--DeepSeekMath.md) | arXiv:2402.03300

**核心公式**：

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o^{(i)}|}\sum_{t=1}^{|o^{(i)}|}\min\left(\frac{\pi_\theta}{\pi_{\text{old}}}\hat{A}_t^{(i)}, \text{clip}(\cdot)\hat{A}_t^{(i)}\right) - \beta\mathbb{D}_{\text{KL}}[\pi_\theta\|\pi_{\text{ref}}]\right]$$

| 改进点 | PPO | GRPO |
|--------|-----|------|
| Value Model | ✅ 需要（与 Policy 等大） | ❌ 不需要 |
| Baseline | Critic 估计 | 组内均值 |
| 优势估计 | GAE | 组相对优势 $(r_i - \bar{r}) / \sigma(r)$ |
| 显存占用 | ~4× Policy | ~2.5× Policy |

**关键结果**：DeepSeekMath-7B 在 MATH 上达 51.7%，接近 GPT-4 水平。

### 4.4 GSPO：修正 GRPO 的根本缺陷

**论文**: [Group Sequence Policy Optimization (GSPO)](GSPO--Group-Sequence-Policy-Optimization.md) | arXiv:2507.18071

**核心洞察**：GRPO 在 token 级别计算重要性比率 $\frac{\pi_\theta(y_t)}{\pi_{\text{old}}(y_t)}$，但单个 token 的重要性采样是高方差、高噪声的，尤其在长序列上会导致**灾难性崩溃**。

**GSPO 修正**：

$$
\text{GSPO-Clip: } \quad r^{(i)}(\theta) = \text{clip}\left(\frac{\pi_\theta(y^{(i)})}{\pi_{\text{old}}(y^{(i)})}, 1-\epsilon, 1+\epsilon\right)$$

| 改进点 | GRPO | GSPO |
|--------|------|------|
| 裁剪粒度 | Token 级 | **序列级** |
| 优势归一化 | 组内 z-score | **组内秩（Ranking）** |
| MoE 兼容性 | 专家激活不稳定 | **稳定** |
| 训练崩溃 | 常见 | 显著减少 |

---

## 5. 系统框架：让 Agentic RL 跑得起来

### 5.1 HybridFlow / veRL：统一 RLHF 框架

**论文**: [HybridFlow](HybridFlow--A-Flexible-and-Efficient-RLHF-Framework.md) | arXiv:2409.19256

**核心设计**：
- **混合编程范式**：单控制器（Ray）+ 多控制器（Megatron-LM + vLLM）
- **3D-HybridEngine**：零冗余权重重分片（ZeRO-like），支持 FSDP/TP/PP/DP 自由组合
- **统一调度**：支持 PPO, ReMax, Safe-RLHF 等多种算法

**性能**：最高 **20.57 倍**吞吐提升（相比 DeepSpeedChat）。

### 5.2 StreamRL：训推分离的第一性原理

**论文**: [StreamRL](StreamRL--Scalable-RL-for-LLMs.md) | arXiv:2504.15930

**核心洞察**：RL 的生成-训练两阶段负载特征截然不同——生成是 memory-bound，训练是 compute-bound，强制共置导致资源浪费。

**StreamRL 方案**：
- **SGS (Streaming Generation Service)**：部署在推理优化硬件（如 TPUs）
- **Trainer**：部署在训练优化硬件（如 GPUs）
- **流式生成重叠**：消除 pipeline 气泡
- **偏度感知调度**：消除长尾延迟
- **弹性扩缩容**：动态调整实例数

**性能**：最高 **2.66×** 吞吐提升，**1.33×** 成本优化。

---

## 6. 案例研究：工具调用的 RL 实践

### 6.1 Search-R1：教会模型使用搜索引擎

**论文**: [Search-R1](Search-R1--Training-LLMs-to-Leverage-Search-Engines.md) | arXiv:2503.09516

**场景**：多轮实时检索问答

**关键技术**：
- **检索 Token 掩码**：$I(y_t) = 0$ for retrieved tokens，屏蔽外部内容参与策略梯度
- **特殊 Token 控制**：`<search>`, `<information>`, `<answer>`
- **极简奖励**：Exact Match，无格式奖励

**结果**：Qwen2.5-7B 较 RAG 提升 **41.7%**，较纯推理 RL（R1）提升 **56%**

**核心洞察**：RL 可以教会 Base 模型从零开始学会搜索——Search-R1-base > Search-R1-instruct

### 6.2 ReTool：策略性代码解释器调用

**论文**: [ReTool](ReTool--RL-for-Strategic-Tool-Use.md) | arXiv:2504.11536

**场景**：数学推理中的代码解释器（Code Interpreter）调用

**动机**：纯文本推理在数值计算、符号操作上存在累积误差，代码解释器可以提供精确计算。

**关键发现**：
- 400 step 训练达到 **67% AIME 准确率**
- 响应长度缩短 **40%**（用代码替代冗长文本推理）
- 模型自主学会「顿悟时刻」（aha moment）——自我纠错

| 行为 | 纯文本 RL | ReTool |
|------|----------|--------|
| 计算精度 | 低（语言模型算术） | 高（Python 执行） |
| 响应长度 | 长 | 缩短 40% |
| 可验证性 | 低 | 高（代码可执行） |

---

## 7. 论文关系图谱

### 7.1 按主题的关联网络

```
                                    ┌─→ [PPO] ─┐
[AgenticRL Survey] ── 算法框架 ─────┼─→ [GRPO] ──┼── 理论演进 ──→ [GSPO]
                │                  └─→ [DPO] ───┘          (修正 token 级噪声)
                │
                ├─ 系统支持 ──→ [HybridFlow] ── 统一 RLHF 框架
                │             [StreamRL]   ── 训推分离
                │
                ├─ 数据引擎 ──→ [AgentFounder] ── FAS/HAS 合成 + Agentic CPT
                │             [WebShaper]    ── 形式化知识投影
                │
                └─ 案例分析 ──→ [Search-R1] ── 搜索工具
                              [ReTool]   ── 代码解释器
```

### 7.2 按演进时间线的技术脉络

| 时间 | 里程碑 | 意义 |
|------|--------|------|
| 2017 | PPO | RL 算法的稳定基线 |
| 2022 | HELM / BIG-Bench | 评估体系化（反面教材） |
| 2023 | LLM-as-a-Judge | 生成式奖励方法 |
| 2024 | GRPO (DeepSeekMath) | 去除 Critic，让 RL 可 scale |
| 2024 | HybridFlow / veRL | 统一系统框架 |
| 2025 | Search-R1 / ReTool | 工具调用的 RL 实践验证 |
| 2025 | GSPO | 修正 GRPO，解决 MoE 不稳定 |
| 2025 | StreamRL | 训推分离的系统优化 |
| 2025 | AgentFounder / WebShaper | 数据合成新范式 |
| 2025 | AgenticRL Survey | 全景综述与分类框架 |

### 7.3 核心技术矛盾与演进方向

```
优化张力 1: 复杂 agentic 行为 vs. 人类偏好对齐
           ↓ 解法: AgentFounder 提出 Agentic CPT「预对齐」

优化张力 2: RL 训练稳定性 vs. 计算效率
           ↓ 解法: GRPO 去除 Critic → GSPO 序列级优化

优化张力 3: 生成吞吐 vs. 训练吞吐
           ↓ 解法: HybridFlow 统一调度 → StreamRL 训推分离

优化张力 4: 数据规模 vs. 数据质量 vs. API 成本
           ↓ 解法: FAS 零成本合成 → HAS 轨迹再利用 → Knowledge Projection
```

---

## 8. 核心洞察汇总

### 8.1 方法论层面的共识

1. **极简奖励可扩展**：Search-R1、ReTool、GRPO 均使用简单的结果奖励，证明 Agentic RL 不需要复杂的 reward modeling
2. **Base 模型经 RL 可获得 agentic 能力**：Search-R1-base > Search-R1-instruct，说明 RL 可以替代部分 SFT
3. **预对齐优于后对齐**：AgentFounder 的 Agentic CPT 从根本上改变了训练 pipeline 的结构
4. **系统与算法必须协同设计**：GSPO 修正算法 → StreamRL 优化系统 → 两者结合才能 scale

### 8.2 尚未解决的挑战

| 挑战 | 当前状态 | 潜在方向 |
|------|---------|---------|
| **过程奖励设计** | 几乎空白 | 轻量级 PRM、隐式信用分配 |
| **多工具协同** | 单工具为主 | 工具切换策略、工具依赖图 |
| **在线部署延迟** | 搜索引擎调用慢 | 缓存、异步检索、预测性搜索 |
| **检索器联合优化** | 固定检索器 | REINFORCE 更新检索器参数 |
| **多模态 Agent** | 文本为主 | 视觉感知、网页渲染理解 |

### 8.3 对实际工作的启示

- **数据优先**：AgentFounder 和 WebShaper 证明，在 algorithmic 创新之前，高质量 agentic 数据本身就是巨大壁垒
- **稳定训练是关键**：GSPO 的发现（token 级噪声导致崩溃）提示，在落地 RL 训练时必须关注重要性采样的粒度
- **系统不是 afterthought**：StreamRL 从第一性原理出发，训推分离的收益远超渐进式工程优化
- **少即是多**：极简的奖励和掩码设计往往比复杂的端到端方案更有效

---

## 9. 论文速查表

| # | 论文 | arXiv | 核心贡献 | 笔记 |
|---|------|-------|---------|------|
| 1 | The Landscape of Agentic RL for LLMs: A Survey | 2509.02547 | Capability + Task 双视角分类框架 | [笔记](AgenticRL-Survey--The-Landscape-of-Agentic-RL-for-LLMs.md) |
| 2 | DeepSeekMath / GRPO | 2402.03300 | 去除 Critic，组相对优势 | [笔记](GRPO--DeepSeekMath.md) |
| 3 | Group Sequence Policy Optimization (GSPO) | 2507.18071 | 序列级优化，MoE 稳定 | [笔记](GSPO--Group-Sequence-Policy-Optimization.md) |
| 4 | HybridFlow / veRL | 2409.19256 | 统一 RLHF 框架，20× 吞吐 | [笔记](HybridFlow--A-Flexible-and-Efficient-RLHF-Framework.md) |
| 5 | StreamRL | 2504.15930 | 训推分离，2.66× 吞吐 | [笔记](StreamRL--Scalable-RL-for-LLMs.md) |
| 6 | Search-R1 | 2503.09516 | 检索 Token 掩码 + 搜索 RL | [笔记](Search-R1--Training-LLMs-to-Leverage-Search-Engines.md) |
| 7 | ReTool | 2504.11536 | 代码解释器的策略性 RL 调用 | [笔记](ReTool--RL-for-Strategic-Tool-Use.md) |
| 8 | AgentFounder | 2509.13310 | Agentic CPT，FAS/HAS 数据合成 | [笔记](AgentFounder--Scaling-Agents-via-CPT.md) |
| 9 | WebShaper | 2507.15061 | 形式化知识投影数据合成 | [笔记](WebShaper--Agentic-Data-Synthesizing.md) |

---


