---
created: 2026-06-09
paper: https://arxiv.org/abs/2507.15061
authors: Zhengwei Tao et al. (Tongyi Lab, Alibaba Group & Peking University & Southeast University & Zhongguancun Academy)
tags:
  - AgenticRL
  - DataSynthesis
  - InformationSeeking
  - WebAgent
---

# WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization

## 一句话总结
通过集合论建立信息搜索（IS）任务的形式化框架，以**知识投影（KP）**为基本单元，配合**Agentic Expander**和**逐层扩展策略**，系统合成高质量训练数据，在 GAIA 和 WebWalkerQA 上达到开源 IS Agent 的 SOTA。

![](WebShaper_fig1_intro.png)

> **Figure 1**: GAIA benchmark 上各 Deep Research 模型/系统的性能对比。WebShaper（7B/32B/72B）在开源方法中取得最佳性能，72B 模型达到 60.19，接近 OpenAI DR（67.4）。带 $*$ 的表示使用相同的两个浏览工具 API。

---

## 1. 研究背景与动机

### 1.1 信息搜索（IS）Agent 的数据瓶颈

LLM-powered Agent 的一个核心能力是**信息搜索（Information-Seeking, IS）**，即在开放网络环境中通过多轮检索、浏览和推理来回答复杂问题。这一能力不仅是学术前沿，也是 OpenAI Deep Research、Gemini、Perplexity 等商业产品的核心。

当前 IS Agent 的典型开发范式为：

1. **构建任务轨迹**：收集问答对和推理路径；
2. **SFT**：通过监督微调获得基础能力；
3. **RL**：通过在线策略强化学习泛化决策策略。

> 整个开发流程的根基是**高质量的训练数据**。

然而，IS 任务本身极其复杂，导致高质量数据集**稀缺且难以通过众包构建**。因此，如何通过精心设计的自动化流程合成训练数据，成为 IS Agent 开发的关键问题。

### 1.2 现有方法的两个核心缺陷

现有数据合成方法通常采用**信息驱动（information-driven）范式**：

1. 先从网络收集信息并组织成结构化格式；
2. 再基于这些信息让 LLM 生成自然语言问题。

这种方法面临两个关键局限：

| 局限 | 说明 |
|------|------|
| **信息-推理结构不一致** | LLM 可能无法完全理解信息结构，导致生成的问题推理结构混乱或答案错误 |
| **信息检索冗余限制多样性** | 无序的信息检索会收集大量同质化的信息结构，限制了数据多样性 |

### 1.3 WebShaper 的范式转换

WebShaper 提出**形式化驱动（formalization-driven）**的数据合成新范式：

![](WebShaper_fig2_formalization.png)

> **Figure 2**: 信息搜索任务的集合论形式化示例。紫色框表示一个知识投影（KP），即一组实体。示例：" founded in 1966 且为东德足球队的球队中，2004-05 赛季效力的、出生于 90 年代的球员是谁？" 目标 $T$ 通过 KP 的交（Intersection）和 $R$-并（$R$-Union）运算组合而成。

相比信息驱动方法，形式化驱动框架带来三个优势：

1. **更广泛的覆盖**：系统探索任务形式化空间，不受预检索内容的约束；
2. **控制性强**：通过形式化参数精确指定推理结构和复杂度；
3. **结构与答案一致性**：形式化表示天然可解释、可验证，减少信息-推理不一致、问题-答案不一致。

---

## 2. 方法

### 2.1 核心思想

WebShaper 将 IS 任务视为一个统一的问题空间，通过**集合论（set theory）**进行形式化。任务的基本单元是**知识投影（Knowledge Projection, KP）**，通过 KP 的组合运算来构造任意复杂度的 IS 任务。

### 2.2 Information-Seeking Formalization

#### 2.2.1 基本定义：知识投影 (KP)

设 $\mathcal{E}$ 为实体的全集（如球员、球队、年份），$R \subseteq \mathcal{E} \times \mathcal{E}$ 为某种关系下的实体对子空间。

**知识投影（KP）**定义为：

$$
R(V) = \{u \mid \exists v \in V,\ (u, v) \in R \text{ 或 } (v, u) \in R\} \tag{1}
$$

| 符号 | 含义 |
|------|------|
| $\mathcal{E}$ | 实体全集（如球员、球队、年份等） |
| $R$ | 关系子空间（如 `bornIn`、`playAt`） |
| $V \subseteq \mathcal{E}$ | 实体子集 |
| $R(V)$ | 在关系 $R$ 下与 $V$ 中实体相连的所有实体集合 |

> 直观理解：$R(V)$ 就是"与 $V$ 中实体具有关系 $R$ 的所有实体"构成的集合。例如，若 $R$ 为 `bornIn`，则 $R(\{\text{90s}\})$ 即所有出生于 90 年代的人。

**KP 是 IS 任务的基本操作单元**，所有 IS 任务都可以通过 KP 的两种运算组合而成。

#### 2.2.2 两种 KP 运算

**运算一：$R$-并（$R$-Union） $\rcup$**

当目标条件具有不确定性时（例如只知道年份范围 2000-2010 而非确切年份），需要按关系 $R$ 进行集合的并运算：

$$
R(V) = R(S_1) \rcup R(S_2) \rcup \cdots \rcup R(S_m) \tag{2}
$$

| 符号 | 含义 |
|------|------|
| $S_1, S_2, \ldots, S_m$ | 实体集合 |
| $\rcup$ | $R$-并运算：同一关系 $R$ 下的并集 |

> 直观理解：球员在 2000-2010 年间效力过的球队，等于在 2000 年效力过的球队 $\cup$ 2001 年效力过的球队 $\cup$ ... $\cup$ 2010 年效力过的球队。

**运算二：交（Intersection）$\cap$**

当目标需要同时满足多个条件时（如"既效力某队又出生于 90 年代"），使用不同关系 KP 的交集：

$$
R(V) = R_1(S_1) \cap R_2(S_2) \cap \cdots \cap R_n(S_n) \tag{3}
$$

| 符号 | 含义 |
|------|------|
| $R_1, R_2, \ldots, R_n$ | 不同的关系 |
| $\cap$ | 集合交集 |

> 直观理解：2000 年效力的球员 与 90 年代出生的球员的交集，即"2000 年效力且出生于 90 年代的球员"。

#### 2.2.3 IS 任务的一般形式

基于上述两种运算，目标集合 $T$ 的一般表达为：

$$
T = \bigcap_{i=1}^{p} \left(R_i(S_{i,1}) \rcup R_i(S_{i,2}) \rcup \cdots \rcup R_i(S_{i,t_i})\right) \tag{4}
$$

其中 $S_{i,j} \subset \mathcal{E}$ 为实体集合。

更一般地，$T$ 可通过递归替换实现层级推理：

$$
T = R_1(T_1) \cap R_2(T_2) \cap \ldots \cap R_k(T_k) \tag{5}
$$

一个 IS 任务最终被形式化为：

$$
q(T) \triangleq ?T \tag{6}
$$

即：找出目标集合 $T$ 包含哪些实体。

#### 2.2.4 简化性质：$R$-并的分配律（Proposition 1）

> **Proposition 1**: 对于同一关系 $R$，$R$-并满足分配律：
> $$
> R(S_1) \cup R(S_2) = R(S_1 \cup S_2)
> $$

**证明直觉**：
- 左 $\subseteq$ 右：若 $x \in R(S_1) \cup R(S_2)$，则 $x$ 与 $S_1$ 或 $S_2$ 中某实体有关系 $R$，因而也与 $S_1 \cup S_2$ 中某实体有关系 $R$。
- 右 $\subseteq$ 左：若 $x \in R(S_1 \cup S_2)$，则存在 $y \in S_1 \cup S_2$，使得 $x$ 与 $y$ 有关系 $R$；$y$ 要么在 $S_1$ 中要么在 $S_2$ 中，因此 $x$ 属于 $R(S_1)$ 或 $R(S_2)$。

> 这一性质使得 $R$-并可以用一个合并后的常量集 $S_1 \cup S_2$ 来表示（例如将 $\{1990\} \cup \{1991\} \cup \cdots \cup \{1999\}$ 简写为 `{90s}`），极大简化了后续表示。

### 2.3 知识投影 (Knowledge Projection) 控制

#### 2.3.1 KP 表示法

为了让 Expander Agent 理解形式化结构，WebShaper 设计了 **KP 表示法**：

- **常量（Constant）**：明确由元素定义的实体子集，如 $\{\text{90s}\}$、$\{2004, 2005\}$
- **变量（Variable）**：元素未显式给出的实体子集，作为符号占位符

一个 KP $R(S)$ 用**三元组 $[X, r, S]$** 表示：

| 元素 | 含义 |
|------|------|
| $X$ | 变量（结果侧实体） |
| $r$ | 关系名称 |
| $S$ | 变量或常量（条件侧实体） |

变量前缀为 `V@`，常量前缀为 `@C`。例如 $R_{\text{bornIn}}(\{\text{90s}\})$ 表示为 `[@V, bornIn, 90s]`。

交运算 $\cap$ 自然表现为三元组列表：`[[X, r1, S1], [X, r2, S2], ...]`。

递归通过变量代入展开。例如 $R^1(R^2(S))$ 表示为 `[[V@X, r1, V@Y], [V@Y, r2, S]]`。

示例问题（Figure 2）的完整 KP 表示：

```
q(T) ≜ ?T s.t. [[V@T, playIn, V@X],
                [V@T, playAt, C@2004_05],
                [V@T, bornIn, C@90s],
                [V@X, foundIn, C@1966],
                [V@X, isA, C@East German football team]]
```

#### 2.3.2 逐层扩展策略 (Layer-wise Expansion)

种子问题的扩展策略是关键。WebShaper 将 KP 表示转换为**图**：节点是变量/常量，边是关系。

![](WebShaper_fig3_expansion.png)

> **Figure 3**: 不同扩展范式对比。(a) 随机结构：随机添加常量，导致常-常连接（冗余）；(b) 顺序结构：线性推理链，导致常量直接连目标（推理捷径）；(c) 逐层结构：按层遍历叶常量，将其替换为变量和新的子问题。

现有方法的问题：

| 问题 | 随机结构 | 顺序结构 |
|------|----------|----------|
| **冗余（Redundancy）** | 常量直接与其他常量相连，增加"没有扩展推理"的子句（如"Dynamo Berlin 是一家位于柏林的足球俱乐部"） | - |
| **推理捷径（Reasoning Shortcut）** | - | 常量过于靠近甚至直接连接目标，Agent 可能跳过深层推理直接猜答案 |

**逐层扩展（Layer-wise Expansion）**的解决方案：

1. 按**层遍历**图，找到所有**叶常量**（leaf constants）；
2. 对每个叶常量，Expander 将其转化为一个新的子问题（答案为该常量）；
3. 将子问题与当前问题合并，用变量替换原叶常量；
4. 保证扩展后答案不变，$q^{n+1}(T)$ 与 $q^n(T)$ 的答案始终是同一个 $T$。

> 这样得到的结构不存在冗余（没有常量直连常量）也不存在推理捷径（没有常量过于靠近目标）。

### 2.4 数据合成 Pipeline

#### 2.4.1 种子问题构建

1. 离线构建 Wikipedia 数据库并保留超链接；
2. 执行**随机游走**，遍历文章间的链接；
3. 聚合游走经过的文章内容，用 LLM 生成问答对；
4. 通过 WebDancer 框架 + QwQ 模型进行 5 次 rollout 验证，只保留至少有一次正确回答的问题；
5. 最终构建 **18K 种子问题** $q^1(T)$。

#### 2.4.2 Agentic Expander

Expander 是一个**具备工具的自主 Agent**，基于 ReAct 框架（Thought-Action-Observation 循环），一次扩展一步。

配备三种工具：

| 工具 | 作用 | 关键性 |
|------|------|--------|
| **Search** | 对常量进行 Google 检索，获取相关 URL 和摘要 | 信息收集 |
| **Summarize** | 访问多个 URL 并总结内容，获得 $R$-并的合并常量集 | **$R$-并的核心** |
| **Validate** | 验证子问题：(1) 与常量类型一致；(2) 不能过于简单（可被 LLM 直接回答） | 质量控制 |

#### 2.4.3 轨迹构建

扩展后的问题通过另一个基于 QwQ 的 ReAct Agent 完成轨迹收集：

- 每个问题执行 **5 次 rollout**；
- 配备 Search 和 Visit 两种工具；
- 过滤条件：**正确性**（答案正确）+ **质量**（无幻觉、无严重重复）；
- 最终获得 **5,000 条高质量轨迹** 用于 SFT 和 RL。

#### 2.4.4 Agent 训练

**SFT 阶段**：轨迹为 $\mathcal{T} = (\tau_1, \alpha_1, o_1, \ldots, \tau_n, \alpha_n, o_n)$，仅对 Action 部分计算损失，Observation 部分被 mask：

$$
\mathcal{L} = -\frac{1}{\sum_{i=1}^{|\mathcal{T}|} \mathbb{I}[x_i \in o]} \sum_{i=1}^{|\mathcal{T}|} \mathbb{I}[x_i \in o] \cdot \log \pi_{\theta}(x_i \mid x_{<i}) \tag{7}
$$

| 符号 | 含义 |
|------|------|
| $\mathcal{T}$ | 整条轨迹（token 序列） |
| $\tau_i$ | Thought（自由文本推理） |
| $\alpha_i$ | Action（结构化工具调用） |
| $o_i$ | Observation（环境返回） |
| $\mathbb{I}[x_i \in o]$ | 指示函数：仅当 $x_i$ 属于 Observation 时为 1（mask） |
| $\pi_\theta$ | 待训练的策略模型 |

> 注意：原始论文公式编号为 (8)，为保持一致性，这里与论文原文保持一致。

**RL 阶段**：使用 **GRPO**（Group Relative Policy Optimization）：

$$
\begin{aligned}
\mathcal{J}_{\mathrm{GRPO}}(\theta) &= \mathbb{E}_{(q,a)\sim \mathcal{D},\{y_i\}_{i=1}^{G}\sim \pi_{\theta_{\mathrm{old}}}(\cdot \mid \text{context})} \\
& \quad \left[\frac{1}{\sum_{i=1}^{G}|y_i|} \sum_{i=1}^{G} \sum_{t=1}^{|y_i|} \min\left( r_{i,t}(\theta)\hat{A}_{i,t},\ \text{clip}\left(r_{i,t}(\theta), 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}\right)\hat{A}_{i,t} \right) \right] \\
r_{i,j}(\theta) &= \frac{\pi_{\theta}\left(o_i \mid q_i,\,o_{i,<t}\right)}{\pi_{\theta_{\mathrm{old}}}\left(o_i \mid q_i,\,o_{i,<t}\right)}, \quad \hat{A}_{i,j} = \frac{R_i - \mathrm{mean}\left(\{R_i\}\right)}{\mathrm{std}\left(\{R_i\}\right)}
\end{aligned} \tag{8}
$$

| 符号 | 含义 |
|------|------|
| $(q, a)$ | 问答对 |
| $G$ | 每组 rollout 数量（论文使用 8） |
| $y_i$ | 第 $i$ 条 rollout 输出 |
| $r_{i,t}(\theta)$ | 重要性采样比率（新旧策略之比） |
| $\varepsilon_{\text{low}}, \varepsilon_{\text{high}}$ | Clipping 范围（GRPO 的超参数） |
| $\hat{A}_{i,t}$ | 第 $i$ 条 rollout 在 $t$ 步的优势估计（归一化奖励） |
| $R_i$ | 第 $i$ 条 rollout 的最终奖励 |
| context | 包含模型历史输出和工具响应的完整上下文 |

---

## 3. 实验结果

### 3.1 实验设置

**评测基准**：
- **GAIA** (Mialon et al., 2023)：通用 AI 助手基准，Level 1/2/3 难度递增
- **WebWalkerQA** (Wu et al., 2025b)：Web 浏览与信息搜索基准，分 Easy/Medium/Hard

**评测方式**：LLM-as-Judge，使用 `Pass@1` 指标

**基线方法**：
- 框架：Search-o1, WebDancer, WebThinker, SimpleDeepResearch, WebSailor
- 数据集基线：WebWalkerQA, E2HQA, MHQA

### 3.2 主实验结果

| Backbone | Framework | GAIA L1 | GAIA L2 | GAIA L3 | GAIA Avg | WW Easy | WW Med | WW Hard | WW Avg |
|----------|-----------|---------|---------|---------|----------|---------|--------|---------|--------|
| Qwen-2.5-72B | Base | 20.5 | 13.5 | 0.0 | 14.6 | 9.4 | 7.1 | 3.3 | 6.3 |
| GPT-4o | Base | 23.1 | 15.4 | 8.3 | 17.5 | 6.7 | 6.0 | 4.2 | 5.5 |
| QwQ-32B | Base | 30.8 | 15.4 | 25.0 | 22.3 | 7.5 | 2.1 | 4.6 | 4.3 |
| OpenAI DR | -- | 74.3 | 69.1 | 47.6 | **67.4** | -- | -- | -- | -- |
| Qwen-2.5-32B | WebShaper | **61.5** | **53.8** | **16.6** | **52.4** | **58.1** | **51.4** | **47.0** | **51.4** |
| QwQ-32B | WebShaper | **69.2** | **50.0** | **16.6** | **53.3** | **55.8** | **49.2** | **45.4** | **49.7** |
| Qwen-2.5-72B | WebShaper | **69.2** | **63.4** | **16.6** | **60.1** | **56.2** | **52.1** | **49.5** | **52.2** |

> **Table 1**: GAIA 和 WebWalkerQA 主实验结果。黑体为对应设置下的最优值。蓝色为所有开源方法中的最高值。WebShaper 在开源 IS Agent 中取得 SOTA，72B 在 GAIA 上达 60.19，是唯一超过 60 分的开源方法，已接近 OpenAI DR（67.4）。

**关键发现**：
- 在全部骨干模型上，WebShaper 均取得最佳性能，展现出**强泛化性**；
- 是唯一开源突破 60 分的模型（GAIA），证实高质量 IS 数据能深度激发 Deep Research Agent 能力。

### 3.3 消融实验

#### 3.3.1 数据集对比（SFT 阶段）

| Backbone | Dataset | GAIA Avg |
|----------|---------|----------|
| Qwen-2.5-32B | WebWalkerQA | 32.0 |
| Qwen-2.5-32B | E2HQA | 39.8 |
| Qwen-2.5-32B | MHQA | 35.9 |
| **Qwen-2.5-32B** | **WebShaper** | **43.6** |
| Qwen-2.5-72B | WebWalkerQA | 38.8 |
| Qwen-2.5-72B | E2HQA | 44.6 |
| Qwen-2.5-72B | MHQA | 43.6 |
| **Qwen-2.5-72B** | **WebShaper** | **45.6** |
| QwQ-32B | WebWalkerQA | 45.6 |
| QwQ-32B | E2HQA | 45.6 |
| QwQ-32B | MHQA | 41.7 |
| **QwQ-32B** | **WebShaper** | **53.3** |

> **Table 2**: 不同数据集 SFT 结果对比。WebShaper 在所有骨干模型上均显著优于基线数据集，验证形式化驱动数据合成的优越性。

#### 3.3.2 RL 刺激效果

![](WebShaper_fig4_rl_gaia.png)

> **Figure 4(a)**: GAIA 上 SFT vs RL 对比。32B 模型 +7.8 分，72B 模型 +13.5 分。

![](WebShaper_fig4b_rl_ww.png)

> **Figure 4(b)**: WebWalkerQA 上 SFT vs RL 对比。32B +7.7 分，72B +6.8 分。

#### 3.3.3 形式化消融 (FL vs NL)

![](WebShaper_fig5_dis_fl.png)

> **Figure 5(a)**: 形式化语言 (FL) vs 自然语言 (NL) 消融。FL 在所有骨干模型上均优于 NL，说明形式化表示能减少合成中的错误传播，提升问答一致性。

#### 3.3.4 逐层扩展策略消融 (L vs S)

![](WebShaper_fig5b_dis_lw.png)

> **Figure 5(b)**: 逐层结构 (L) vs 顺序结构 (S) 消融。逐层扩展在所有骨干模型上均优于顺序扩展，验证其在减少冗余和推理捷径方面的有效性。

#### 3.3.5 Tool Call 分析

| 工具 | 发现 |
|------|------|
| Search | WebShaper 任务的 Search 次数 >3 的比例是 E2HQA/MHQA 的 3-4 倍，说明其更能处理需要迭代精化的信息密集型查询 |
| Visit | WebShaper 在超过 10 步 visit 仍保持高比例，而竞品数据集急剧下降，体现更强的导航智能 |
| Total | 总工具调用 >3 的比例翻倍，且非零比例可持续到 30 次调用，证明其在复杂组合推理上的可扩展性 |

![](WebShaper_fig6_search.png)

> **Figure 6(a)**: Search 次数分布对比。

![](WebShaper_fig6b_visit.png)

> **Figure 6(b)**: Visit 次数分布对比。

![](WebShaper_fig6_total_tools.png)

> **Figure 6(c)**: 总工具调用次数分布对比。WebShaper 的长尾分布显著优于 E2HQA、MHQA 和种子任务，说明其合成数据涵盖更复杂的推理链。

---

## 4. 局限性与未来方向

### 4.1 局限性

1. **计算成本较高**：平均每例合成需要约 20 次 LLM completion、6 次搜索、6 次访问、约 7 分钟端到端时间；
2. **仅覆盖事实性问答**：当前形式化聚焦于事实搜索，未涉及主观判断或敏感话题；
3. **RL 在 QwQ-32B 上增益有限**：可能与模型特性或训练设置有关；
4. **规模尚未充分探索**：当前合成 5,000 条轨迹，更大规模的合成潜力待验证。

### 4.2 未来方向

1. **扩展形式化到其他任务类型**：如 browsing, web navigation, multimodal IS；
2. **更大规模合成与课程学习（curriculum learning）**：从简单到复杂逐步构建更丰富的训练数据；
3. **与其他领域形式化的融合**：数学证明（Lean 4）、知识图谱 QA（FOL）等已有成熟形式化领域可交叉借鉴；
4. **强化学习任务设计的自动化**：形式化驱动的方法天然适配 RL 的任务复杂度控制。

---

## 5. 个人思考

**形式化驱动的范式转换非常关键。** 当前信息搜索领域的主流是"先收集信息，再生成问题"，其本质是让 LLM 做"信息到问题"的映射。这不可避免地面临结构不匹配的问题——信息结构和推理结构不总是一致的。WebShaper 的洞察在于：先定义好问题的形式结构，再根据结构去检索和验证信息。这一从"reactive"到"proactive"的转变，不仅提升了数据质量，更让整个合成过程具备了**可控性**。

**知识投影（KP）的设计优雅。** 仅通过一个基本操作（KP）和两个运算（$R$-并和交），就能表示各种复杂的信息搜索任务。尤其是分配律（Proposition 1）的发现，使得 $R$-并可以被合并为单个常量集，极大简化了表示和扩展。这体现了"简单形式 + 组合能力 = 表达复杂"的设计哲学。

**逐层扩展策略是工程上的精妙之处。** 随机扩展导致冗余，顺序扩展导致推理捷径，而逐层扩展恰好避免了这两个问题。这不是形式化本身的功劳，而是将形式化映射到图结构后，利用图的层级拓扑性质进行有控制的扩展。这种对"问题结构的几何理解"很有启发性。

**对 Agentic RL 研究的启示：** 形式化驱动的方法天然适合为 RL 提供可控的任务分布。论文中 RL 带来了 +7~13 分的增益，且 WebShaper 的数据在长工具调用链（30+ steps）上保持稳定表现，说明合成数据的质量直接决定了 RL 的上限。这对构建更复杂的 Agentic RL 环境（如 OpenAI Operator 或 Deep Research Agent）有重要参考价值。

> 一个延伸思考：KP 的形式化能否从"信息搜索"泛化到更广泛的 Agent 能力（如工具使用、代码生成、多模态交互）？如果能建立一套跨领域的"任务形式化元语言"，或许能实现真正统一的 Agent 能力训练框架。

---

## 6. 关键引用

```bibtex
@inproceedings{tao2026webshaper,
  title={WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization},
  author={Tao, Zhengwei and Wu, Jialong and Yin, Wenbiao and Zhang, Junkai and Li, Baixuan and Shen, Haiyang and Li, Kuan and Zhang, Liwen and Wang, Xinyu and Jiang, Yong and Xie, Pengjun and Huang, Fei and Zhou, Jingren and Zhang, Wentao},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

**相关论文**：
- WebDancer (Wu et al., 2025a): 简单到复杂的 E2HQA 数据合成
- WebWalkerQA (Wu et al., 2025b): Web 浏览与多源问答基准
- WebThinker (Li et al., 2025c): Deep Research 能力赋予推理模型
- GRPO (Shao et al., 2024): Group Relative Policy Optimization
- GAIA (Mialon et al., 2023): 通用 AI 助手基准
