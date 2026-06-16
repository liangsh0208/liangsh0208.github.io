---
created: 2026-06-09
paper: https://arxiv.org/abs/2606.02113
code: https://github.com/RenBing-Sumeru/Awesome-LLM-Reasoning-Data
authors: Yaoming Li, Guangxiang Zhao, Qilong Shi, Lin Sun, Xiangzheng Zhang, Tong Yang (北京大学, 清华大学, Qiyuan Tech)
published: 2026-06-01
tags:
  - 论文阅读
  - 模型评估
  - 推理数据
  - 后训练
  - 数据工程
  - 综述
---

# A Primer in Post-Training Reasoning Data: What We Know About How It Works

## 一句话总结
本文是第一篇系统综述后训练（Post-Training）推理数据的入门文章，通过分析150+篇公开论文和系统报告，围绕「数据对象—数据质量—数据构建—收益归因」四维框架，揭示了推理数据领域的7大反直觉陷阱，提出了以验证器（Verifier）为核心的数据分类体系，并为未来推理数据的发布与归因提供了可操作化的报告清单。

![](PTPRD_fig1_overview.png)

> **Figure 1**: 超越 prompt–response 对。一个推理数据项不仅包含问题和答案，还打包了问题/状态、模型行为、评判反馈和归因元数据。图中预览了本文围绕的四个核心问题：存在什么数据对象、什么使数据有用、数据如何构建、以及收益如何归因。

---

## 1. 研究背景与动机

### 1.1 问题定义

大语言模型（LLM）的训练通常分为两个阶段：预训练（Pre-training）和后训练（Post-training）。随着 o1-style test-time scaling 范式和思维模型（Thinking Models）的兴起，后训练在提升模型推理能力方面越来越重要。在后训练管线中，**推理数据的质量和构建方式对模型性能的影响往往超过训练算法或优化策略本身**。然而，尽管数据处于核心地位，此前却缺乏一篇系统性地聚焦于后训练推理数据的综述。

本文填补了这一空白，是第一篇专门针对后训练推理数据的综述（primer），综合了 **150 余篇** 数据集论文、强化学习配方（RL recipes）、奖励模型研究、基准测试和前沿系统报告。

### 1.2 现有方法的不足与反直觉教训

作者在全文开篇就用一张表（Table 1）总结了该领域反复出现的 **7 大反直觉陷阱**，每个陷阱都揭示了推理数据领域一个常见的错误假设：

| 陷阱 | 错误假设 | 真相 | 建议报告字段 |
|------|---------|------|-------------|
| Long CoT = good reasoning | 长推理链 = 好推理 | 长 trace 可能只是在合理化答案、复制教师风格或隐藏真实原因；质量需要有效性和 grounding | trace source, process labels, execution context, localization tests, robustness checks |
| Harder data = better data | 更难的数据 = 更好的数据 | 难度是 base-relative 的；同一道题对不同基座模型难度不同 | base model, sampling protocol, verifier, rollout count, temperature, pass-rate band, estimation date |
| More data = better coverage | 更多数据 = 更好覆盖 | 覆盖是 recipe 而非 count；source mixture、filters、generators、leakage、lineage 决定了继承什么 | source mixture, generator, teacher, filtering rule, split, decontamination status, lineage risks |
| Clean successful transcript = ideal | 干净的成功记录 = 理想数据 | 对于智能体，清理的成功 trace 会抹除 failures、retries、recoveries、state diffs——这些是信用分配的证据 | replayable episodes with states, actions, observations, failures, retries, terminal predicates, scaffold metadata |
| Self-play removes curation | 自博弈消除了人工策选 | Self-play 只是把策选转移到 anchor（答案、解释器、验证器、多数投票、档案、role split） | anchor, selection rule, verifier, admissibility rule, replay policy, failure modes |
| The optimizer explains the gain | 优化器解释了全部收益 | 优化器可见性 ≠ 因果隔离；同样的 RL scaffold 可以隐藏不同的 prompt pools、trace writers、substrates | prompt source, trace author, search substrate, reward channel, scaffold, budget |
| Scaling means the model got better | 规模化 = 模型变好了 | 更高分数可能来自移动 ceiling、提高效率或改变测量表面——这些是不同的 claim | unique data, reuse rate, training compute, inference budget, search topology, verifier refresh, evaluation protocol |

![](PTPRD_table1_counterintuitive_lessons.png)

> **Table 1**: 推理数据归因的反直觉教训。表中提炼了后训练推理数据工作中反复出现的陷阱、它们为何失效、每个教训对应正文展开章节，以及未来报告应包含的字段以使收益可被归因。

> 💡 **核心洞察**：推理数据的成功不能仅凭最终 benchmark 分数判断——**必须拆开看是数据基板、验证器、搜索拓扑还是优化器出了力**。

---

## 2. 七大反直觉陷阱的深度拆解

论文的核心贡献之一是提炼出 7 个在后训练推理数据领域反复出现的认知陷阱。每个陷阱都对应正文中的详细论证，以下逐一拆解其**错误直觉的表面合理性**、**反例与证据**、**深层机制**，以及**对当前主流工作的映射**。

---

### 2.1 陷阱一：长推理链 ≠ 好推理

> **误区的表面合理性**：CoT（Chain-of-Thought）的提出者（Wei et al., 2023）展示了推理链能显著提升复杂任务的准确率。直觉上，一条更长、更详细的推理链意味着模型「想得更深」。

#### 反例与证据

- **Lanham et al. (2023)** 与 **Turpin et al. (2023)** 发现：模型生成的推理链可能是**事后合理化（post-hoc rationalization）**——模型已经通过某种捷径（如表面模式匹配、记忆召回或偏置启发式）得出了答案，然后再用看似合理的逻辑来解释这个结果。Trace 越长，合理化能力越强，但不一定代表推理更忠实。
- **Faithfulness 的缺失**：可见的 trace 只是 **suggestive evidence**，不是 **proof of mechanism**。一条 trace 完全可能是「正确的答案 + 错误的推导」或「正确的推导 + 无关的中间步骤」。

#### 深层机制：过程监督也有盲区

论文进一步指出，人们常把希望寄托在 **Process Supervision**（过程监督，即对每一步进行验证）上，但这类方法内部也存在关键分歧：

| 过程监督方法 | 信号来源 | 局限性 |
|-------------|---------|--------|
| **PRM800K** (Lightman et al., 2023) | 人工标注的逐步标签 | 人工成本极高，且人类标注者也可能遗漏中间错误 |
| **Math-Shepherd** (Wang et al., 2024) | Rollout value（蒙特卡洛估计每一步的期望回报） | Rollout 估计的方差大，对罕见错误步不敏感 |
| **OmegaPRM** (Luo et al., 2024) | 寻找 first error 的精炼过程估计 | 假设「只有一个 first error」，对多错误步或连续性错误的定位不准确 |

更重要的是，**ProcessBench / PRMBench** 的评估表明：PRM 本身必须在 **localization（定位精度）、soundness（可靠性）、sensitivity（敏感性）、robustness（鲁棒性）** 四个维度上接受测试——否则过程监督只是把一个黑箱拆解成了多个小黑箱。

#### 核心洞察与教训

> 📌 **Trace quality 的正确定义**：不是「长度」或「格式正确性」，而是 **validity（每一步在逻辑/语义上是否成立）** 和 **grounding（推理是否锚定在问题本身，而非偏置、记忆或教师风格）**。数据发布时应报告 trace source、process labels、execution context、localization tests、robustness checks——让读者能够独立判断 trace 的质量，而非仅凭长度判断。

---

### 2.2 陷阱二：更难的数据 ≠ 更好的数据

> **误区的表面合理性**：教育心理学中有「最近发展区」（Zone of Proximal Development）的概念，似乎难题就是帮助学习者突破舒适区的最佳工具。

#### 反例与证据

论文提出一个三元关系：

$$
\text{Difficulty} = f(\text{item}, \text{model}, \text{sampling})
$$

- 同一道题对 **基座模型 A** 可能「太难」（unreachable），模型几乎无法产生正确回答，梯度信号极为微弱，训练效果趋近于零；
- 对 **基座模型 B** 可能「刚好合适」（useful），处于学习最适区，能产生有效梯度；
- 对 **基座模型 C** 可能已经「饱和」（saturated），模型早已掌握，再训练也无增益。

**Skywork-OR1** (He et al., 2025a) 的实践中，研究人员**针对模型变体估计难度**，而非对题目本身打难度标签。**DAPO** (Yu et al., 2025) 则更直接：干脆**移除准确率极端（0 或 1）的 prompt group**——太容易的没有梯度，太难的同样没有有效梯度，两者都要剔除。

#### 深层机制：「中等难度」这个标签掩盖了什么

论文尖锐地指出：当我们说一组数据是「medium difficulty」时，这个标签背后可能隐藏了至少六个未报告的变量：

```
"medium difficulty" 的隐藏条件：
├── base model（哪个基座模型？）
├── prompt format（什么格式的 prompt？）
├── rollout count（多少次采样？）
├── temperature（采样温度？）
├── verifier（用什么验证器判定对错？）
└── estimation date（何时估计的难度？模型已更新否？）
```

这意味着：一篇论文声称「我们的数据难度适中」，在缺乏上述上下文时，这个声明几乎不可复现、不可比较。

#### 核心洞察与教训

> 📌 **useful difficulty 的可操作化定义**：不是题目本身的绝对难度，而是 **对特定基座模型、在特定采样配置下、处于「非 trivial（不总能做对）」且「非 unreachable（不总能做错）」之间的 band**。发布数据时应报告 base model、sampling protocol、verifier、rollout count、temperature、pass-rate band、estimation date——否则「难度」只是一个无法归因的黑箱标签。

---

### 2.3 陷阱三：更多数据 ≠ 更好覆盖

> **误区的表面合理性**：大数定律似乎暗示，数据量越大，对真实分布的逼近越好。在预训练时代，「更多数据 = 更好」几乎是一条铁律。

#### 反例与证据

论文用三个层次的证据击碎了这个直觉：

**层次一：覆盖是 recipe，不是 count**
- **OpenThoughts** (Guha et al., 2025) 通过系统消融证明：source、mixture、filter、generator、teacher 的每一个选择都会改变最终学习效果，而这些因素与数据量无关。

**层次二：结构胜过规模**
- **Structure Trumps Size** (Xu et al., 2025b) 明确表明：**数据的结构（如多样性、难度分布、领域配比）本身可能比传统意义上的「干净」或「规模」更能决定最终效果**。

**层次三：合成数据的跨代污染**
- 合成数据飞轮使数据廉价，但跨代传播的往往不是「更多样的问题分布」，而是：
  - **trace style**（教师模型的写作风格，如过度使用「让我们一步步来」）
  - **decoding policy**（如总是先生成某类中间步骤）
  - **filter bias**（如过滤规则无意中偏爱某种表达）
  - **teacher preference**（教师模型偏好某种解法而非最优解法）
- **Shumailov et al. (2024)** 著名的「模型崩溃（model collapse）」研究表明：当合成数据在 generations 之间循环时，模型会逐渐丢失尾部信息（tail events）。
- **Green et al. (2025)** 和 **Han et al. (2025)** 进一步揭示了 hidden teacher trait transfer 和 search contamination 的风险。

#### 深层机制：什么叫做真正的「覆盖」

| 表面的覆盖指标 | 实际的覆盖问题 |
|---------------|--------------|
| 语料规模大（10M+ 题目） | 可能来自同一 generator，内部结构高度同质化 |
| 领域标签多（数学/代码/医疗） | 每个领域内的 source mixture 和 filtering rule 可能严重偏斜 |
| 通过 contamination audit（去污染） | 去污染只处理了训练-测试重叠，未处理 generator-教师-验证器之间的隐式重叠 |
| 声称「多样性高」 | 可能仅在 prompt surface 上多样化，而 solution path 的结构仍然单一 |

#### 核心洞察与教训

> 📌 **覆盖的归因清单**：发布推理数据时，「我们发布了 X 百万道题」是一句信息量极低的陈述。真正需要报告的是：**source mixture（来源配比）、generator（生成器）、teacher（教师模型）、filtering rule（过滤规则）、split（训练/验证/测试划分逻辑）、decontamination status（去污染方法）、lineage risks（谱系风险）**。覆盖必须被审计，而不能被计数。

---

### 2.4 陷阱四：干净的成功记录 ≠ 理想数据

> **误区的表面合理性**：监督学习的基本范式是「向成功者学习」。如果模型看到足够多「正确解决问题的方式」，它自然会内化这些模式。

#### 反例与证据

论文在 **agentic reasoning** 场景下给出了最有力的反驳：

- Qin et al. (2023)、Pan et al. (2025)、Xie et al. (2024) 的研究表明：在agent环境中，**overly cleaned success trajectories 会精确地移除 credit assignment 所需的 failures、retries 和 state changes**。
- 具体来说，一条被「清理过」的成功记录抹除了：
  - **Failed actions**（哪些工具调用失败了）
  - **Retries**（重试策略是什么）
  - **Recoveries**（如何从错误状态中恢复）
  - **State diffs**（状态如何变化）
  - **Hidden predicates**（为什么在某个节点判定为失败/成功）

在工具使用/环境交互场景中（第6页），论文进一步强调：successful transcripts alone erase failed tool calls、state diffs 和 hidden predicates——而这些分支恰恰是 **credit assignment 最可见的地方**。

#### 深层机制：信用分配（Credit Assignment）

强化学习的核心问题之一就是 credit assignment：在一个长轨迹中，究竟是哪个动作（或哪组动作）最终导致了成功？

| 数据发布策略 | Credit assignment 可见性 | 训练效果 |
|-------------|----------------------|---------|
| **仅发布成功路径** | ❌ 失败分支被完全隐藏 | 模型无法学习「什么不该做」和「如何从错误恢复」 |
| **发布成功+失败路径** | ⚠️ 失败路径存在但无标签 | 模型知道有失败，但不知道失败原因 |
| **发布完整 replayable episodes** | ✅ 状态、动作、观察、失败、重试全可见 | 模型可以学习完整的决策边界和恢复策略 |

#### 核心洞察与教训

> 📌 **数据的纪录片原则**：对于 agent 推理数据，理想的数据发布不是「成功高光集锦」，而是「推理过程的完整纪录片」。应发布 **replayable episodes**，包含完整的 states、actions、observations、failures、retries、terminal predicates 和 scaffold metadata。**环境验证不能归约为最终成功**——那些失败的分支正是训练中最富信息量的部分。

---

### 2.5 陷阱五：自博弈 ≠ 消除策选

> **误区的表面合理性**：Self-play（如 AlphaZero、STaR）在围棋、国际象棋等领域证明了自我对弈可以产生超越人类监督的数据质量。直觉上，模型生成问题→自我验证→自我训练，这个闭环不需要人类策选。

#### 反例与证据

论文的论断简洁而有力：

> **Self-play does not eliminate curation; it moves curation into the anchor.**

锚点（anchor）是那个「将生成的行为转化为可训练反馈」的机制。不同 self-play 系统的锚点截然不同：

| 系统 | 锚点机制 | 策选转移到了哪里 |
|------|---------|----------------|
| **STaR** | 锚定到外部答案（external answer） | 策选转移到了「哪些外部答案被视为正确」 |
| **R-Zero** | 分离 Challenger 和 Solver 角色 | 策选转移到了「角色定义和对战规则」 |
| **Absolute Zero** | 使用 Python interpreter 作为锚点 | 策选转移到了「interpreter 的输出判定」和「可执行性检查」 |
| **TTRL** | Test-time majority vote 转为 reward | 策选转移到了「多数投票的阈值」和「样本量」 |
| **AlphaEvolve / 多智能体** | 功能分布在不同角色上 | 策选转移到了「角色分配」和「跨角色协调规则」 |

这些系统之间的关键差异**不在于「有多少个智能体」，而在于「什么使得一条轨迹被接纳（admissible）」**：答案可得性、可执行反馈、多数投票选择、角色介导的挑战，或基于档案的评估。

#### 深层机制：Anchor 定义了 Support Bounds

锚点同时定义了两件事：
1. **Support bounds（支持边界）**：什么样的生成行为会被纳入训练集？
2. **Label reliability（标签可靠性）**：这些行为的标签有多可靠？

一个常被忽视的后果是：**on-policy RLVR（可验证奖励强化学习）可能会保守地重新加权那些已经被基座策略支持的轨迹**，从而系统性地回避基座模型不擅长的领域——表面上「自动生成」，实际上在悄悄 narrow 覆盖范围。

#### 核心洞察与教训

> 📌 **策选的位移而非消失**：任何 claiming「无需人工策选」的 self-play 系统都值得追问：**策选转移到了哪个 anchor？** 数据发布时应报告 anchor、selection rule、verifier、admissibility rule、replay policy、failure modes——而不仅仅是「我们生成了 X 万条 self-play 数据」。

---

### 2.6 陷阱六：优化器可见性 ≠ 因果隔离

> **误区的表面合理性**：如果论文明确报告了使用 GRPO、DPO 或 PPO，并展示了消融实验，读者自然倾向于相信「是优化器带来了增益」。

#### 反例与证据

论文提出了一个极强的论断：

> **But optimizer visibility is not causal isolation.**

在 **DeepSeek-R1、Kimi K1.5、Qwen3、Magistral、Phi-4-reasoning、Llama-Nemotron** 等一系列主流推理系统的实践中，反复出现同一模式：**即使优化器名称相同（如都标榜使用 GRPO），其实际行为和效果也可能截然不同**，因为：

| 系统 | 名义优化器 | 实际改变的关键变量 |
|------|-----------|-----------------|
| **DAPO** | RL-style | Filtering 策略和 loss geometry（损失几何形状） |
| **MiniMax-M1** | RL-style | Importance-weight control（重要性权重控制） |
| **Qwen3-Coder-style** | RL-style | 将训练 signal 移入长视野工具交互 |

这意味着：**优化器只是 scaffold 的最后一层**；真正的因果因素隐藏在上游的 prompt pools、trace writers、search substrates、verifiers 和 budgets 中。两篇论文都说「我们用了 GRPO」，但它们的数据基板、验证器、搜索空间可能完全不同，导致 GRPO 在这两个系统中的「实际身份」也不同。

#### 深层机制：Attribution Ledger 的设计思想

Table 2（Attribution Ledger）的核心设计逻辑是：

```
每个上游构建层：
├── 暴露一个可见字段（visible field）
├── 隐藏一个不同的混淆因子（hidden confound）
└── 决定需要什么元数据来解释最终增益
```

| 构建层 | 可见字段 | 隐藏混淆因子 | 需要的元数据 |
|--------|---------|-------------|------------|
| Prompt sourcing | 用了什么 benchmark | 问题支持度和通过率 band | Source, filter, pass-rate band |
| Trace writing | 推理链作者是谁 | 继承的推理风格和隐式偏见 | Author, format, grounding method |
| Search substrate | 搜索空间的定义 | 探索能力和可回放性 | Rollout policy, branch pruning, replay buffer |
| Self-play anchor | 锚点类型 | 策选重入的位置和规则 | Anchor type, selection rule, admissibility |
| Reward/verifier | 验证器种类 | 什么被计为成功 | Verifier version, failure modes, gaming risks |
| Frontier pipeline | 整体系统框架 | 报告的收敛可能只是表面 | Full stack disclosure |

#### 核心洞察与教训

> 📌 **优化器只是冰山一角**：一篇论文如果只报告了「我们用了 XX 优化器」，那它解释的只是增益的 **1/6**（如果按 Table 2 的六个层次来算）。完整的归因链必须包括：prompt source、trace author、search substrate、reward channel、scaffold、budget。**「我们用了 GRPO」是一句远不足以概括实验设计的陈述。**

---

### 2.7 陷阱七：规模化 ≠ 模型变好了

> **误区的表面合理性**：Scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022) 在预训练时代建立了「规模越大越好」的强大经验规律。直觉上，后训练的 scaling 也遵循同样的叙事。

#### 反例与证据

论文明确指出：benchmark 增益有三种完全不同的来源，却被混为一谈：

| 增益来源 | 机制 | 是否代表「模型能力真的变强了」 |
|---------|------|---------------------------|
| **移动 Ceiling** | 数据基板质量提升、验证器升级、上下文更丰富 → 原本 unreachable 的问题变得可解 | ⚠️ 部分代表，但可能是「题目变易」而非「模型变强」 |
| **提升效率** | 优化器设计、采样策略、课程安排改进 → 以更少计算更快接近同一 ceiling | ❌ 不代表能力边界扩大，只代表训练效率提升 |
| **改变测量表面** | 搜索拓扑、推理预算、评估协议变化 → 「同一能力」被不同地衡量 | ❌ 不代表能力变化，只代表评分标准变化 |

**Hochlehnert et al. (2025)** 发现：分数本身会随 seed 和 budget 漂移（score drift）。**Yue et al. (2025)** 发现：搜索 agent 可以在评估期间 retrieve benchmark-adjacent answers（检索与基准测试相近的答案，而非真正推理）。**Tan et al. (2026)** 的 scaling 分析进一步表明：model size、compute 和 data reuse 之间存在复杂的交互，不能简单归因于单一变量。

#### 深层机制：Khatri-Tan 分解

论文引用 Khatri et al. (2025) 的方程作为「阅读设备」而非「普适定律」：

$$
R(C) = R_0 + \frac{A - R_0}{1 + (C_{\text{mid}}/C)^B}
$$

这个方程的核心启示是：
- **$A$（可达天花板）** 和 **$B$（接近效率）** 是两个独立维度。
- 一篇论文报告「分数从 60% 提升到 75%」时，如果没有披露完整上下文，读者无法判断：
  - 是 $A$ 增加了（天花板被推动，可能是数据基板/验证器升级）？
  - 还是 $B$ 增加了（接近效率提高，可能是优化器/采样改进）？
  - 还是测量方式变了（$R$ 的定义或被测问题集变了）？

#### 核心洞察与教训

> 📌 **Scaling claim 的三重检验**：当看到一篇论文报告「规模化带来了提升」时，必须追问三个问题：
> 1. **Ceiling mover 还是 efficiency mover？** 这次干预改变的是可达上限，还是接近上限的效率？
> 2. **数据基板是否也变了？** 发布日期不是中性时间戳——不同的 prompt pool、新的验证器、变更的推理预算都意味着不同的数据基板。
> 3. **评估协议是否一致？** 搜索拓扑、verifier refresh、evaluation protocol 的变化使得前后分数可能不可比。
>
> 没有披露 unique data、reuse rate、training compute、inference budget、search topology、verifier refresh、evaluation protocol 的 scaling 报告，本质上是不可归因的。

---

## 3. 方法：以验证器为核心的分类框架

### 2.1 核心思想：Verifier-Anchored Taxonomy

传统的推理语料库通常按领域（domain）命名（如数学、代码、医疗）。本文提出，**学习信号的定义取决于「什么可以被检查」**，因此应按 **verification contract（验证契约）** 而非领域来分类数据。基本单元不再是 prompt–response pair，而是 **verifier-bearing sample（携带验证器的样本）**。

本文将推理数据分为三大类验证契约：

| 验证契约类型 | 说明 | 典型场景 |
|-------------|------|---------|
| **Programmatic verification**（程序化验证） | 答案可被规范化、代码可执行、证明状态可验证 | 数学、代码、Lean 形式化证明 |
| **Environmental verification**（环境验证） | 交互过程本身产生可验证的 state transition | 工具使用、网页/软件/操作系统任务、多智能体环境 |
| **Judgment-required verification**（需要判断的验证） | 不存在确定性验证器，可复用单元是可审计的判断记录 | 医疗、事实性、安全性、rubric-reward 评估 |

![](PTPRD_fig2_3_taxonomy_supervision.png)

> **Figure 2**: 以验证器锚定的分类法。验证契约（而非领域）定义了推理样本的原生信号、可训练对象、失败面和所需审计字段。

> **Figure 3**: 监督信号在何处进入轨迹。同一任务在不同后训练数据中会呈现不同形态：反馈可以 targeting 最终答案、中间推理步骤、状态-动作转换，或整个 episode。反馈点的移动会改变可训练对象，即使底层任务相同。

### 2.2 三个横切维度（Three Cross-Cutting Axes）

验证契约说明了「什么可以被检查」，但还有三个横切的维度解释了「信号进入训练后如何表现」：

**1. Supervision Granularity（监督粒度）**

同一任务在不同数据中会呈现为截然不同的训练信号：
- **Answer-level supervision**：仅标注最终答案正确性（最粗粒度）
- **Step-level supervision**：标注中间步骤（如 PRM800K、Math-Shepherd）
- **State/action-level supervision**：标注状态-动作转换（如工具反馈、transition reward）
- **Full-episode supervision**：标注整个 episode 的 terminal success/failure/return（最细粒度）

粒度的移动将 **capability**（能力）与 **mechanism**（机制）解耦：粗粒度显示「问题是否被解决」，细粒度显示「解决步骤、rollout 和执行过程」。

**2. Behaviour Bounding（行为边界）**

行为边界数据定义了模型何时应该回答、拒绝、弃权、提供安全补全，或暴露推理链。核心问题是：弃权（abstention）、冗长（verbosity）或暴露的推理链并不一定意味着可靠性。数据发布时应分离：intent（意图）、risk（风险）、allowed content（允许内容）、response type（响应类型）、trace visibility（推理链可见性）和 mismatch（不匹配）。

**3. Cross-Generational Lineage（跨代谱系）**

合成数据飞轮（synthetic flywheels）使数据变得廉价，但真正传播的是 trace style、decoding policy、filter 或 teacher preference，而非问题分布本身。规模化可以与 narrowing、hidden trait transfer、leakage 或 search contamination 共存。因此谱系应作为 sample-level 元数据来报告。

### 2.3 质量四维矩阵

正确性（correctness）、难度（difficulty）、trace quality（推理链质量）和覆盖/泄露（coverage/leakage）这四个质量声明各自需要不同的验证器、基座模型、轨迹证据和谱系证据。没有单一字段可以授权所有质量声明。

| 质量声明 | 关联视角 | 什么不够用 | 最小支持字段 |
|---------|---------|-----------|------------|
| Correctness | verifier-relative | 仅答案字符串 | verifier, extractor, invalid-output policy |
| Trace quality | trajectory-relative | 推理链长度 | process labels, execution lineage |
| Difficulty | base-relative | 问题来源 | base model, sampling protocol |
| Coverage/leakage | lineage-relative | 语料规模 | generator, filter, teacher, contamination lineage |

![](PTPRD_fig4_5_quality_traps.png)

> **Figure 4**: 质量支持矩阵。没有单一字段可以授权所有质量声明：正确性、难度、trace quality 和覆盖各自需要不同的验证器、基座模型、轨迹证据和谱系证据。

> **Figure 5**: 两个常见质量陷阱：长推理链 ≠ 好推理，难题 ≠ 有用题。Trace quality 取决于有效性和 grounding；useful difficulty 位于「对给定基座模型总能做对」和「总能做错」之间的 band 中。

---

## 3. 数据构建过程

### 3.1 推理数据从何而来？

本文将数据构建拆解为 6 个层次，每个层次都引入了不同的隐藏变量：

| 构建层次 | 决定的问题 | 隐藏变量 |
|---------|-----------|---------|
| 1. Prompt sourcing（提示来源） | 问题/状态从何而来 | benchmark、mined task、synthetic prompt、usage log |
| 2. Trace writing（推理链撰写） | 推理链由谁写、什么风格 | human / teacher LLM / self-distilled / tool-grounded / process-labeled |
| 3. Search substrate（搜索基板） | 在什么样的轨迹空间中搜索 | rollout、tool call、env interaction、branch pruning、replay buffer |
| 4. Self-play / closed-loop anchor（自博弈/闭环锚定） | 什么行为被视为成功并被保留 | external answer / interpreter / verifier / majority vote / archive / role split |
| 5. Reward / verifier layer（奖励/验证器层） | 什么被计为成功 | formal checker / process verifier / learned reward model / rubric judge / closed-loop selector |
| 6. RL scaffold / optimizer（RL脚手架/优化器） | 如何用梯度更新策略 | DPO / PPO / GRPO / REINFORCE / curriculum |

![](PTPRD_appdx_figA2_construction.png)

> **Figure A2** (Appendix): 推理数据构建栈。推理数据配方在下游优化器应用之前积累了隐藏变量。Prompt 来源决定问题支持；trace 撰写决定继承的推理风格；搜索基板保留或抹除失败分支；锚点决定策选重入的位置；验证器决定什么被计为成功。

### 3.2 构建作为归因账簿（Attribution Ledger）

每个构建层次暴露了不同的归因把手（attribution handle），可以追溯最终增益的来源。Table 2 将各层次与所需审计字段对应：

| 构建层 | 归因把手 | Q T E V L |
|--------|---------|-----------|
| Prompt sourcing | problem support / pass-rate band | ✓ − − △ △ |
| Trace writing | inherited reasoning style | △ ✓ − △ ✓ |
| Search substrate | exploration and replayability | △ △ △ − △ |
| Self-play anchor | where curation re-enters | △ △ △ △ ✓ |
| Reward/verifier | what counts as success | △ △ △ ✓ ✓ |
| Frontier pipeline | where reports appear to converge | ✓ ✓ △ ✓ ✓ |

*Q = prompt/source; T = trace or teacher; E = environment/substrate; V = verifier/reward; L = lineage*

✓ 标记主要可训练字段，△ 标记通常缺失或非主要字段。

### 3.3 验证器种类与失败面

奖励通道（Reward channel）本身就是一个数据对象：formal checker、process verifier、learned reward model、rubric judge、closed-loop selector 等各自暴露了不同的使用场景、失败模式和审计字段。

![](PTPRD_fig6_verifier_families.png)

> **Figure 6**: 验证器族与失败面。奖励通道是一个数据对象：formal checkers、process verifiers、learned reward models、rubric judges 和 implicit selectors 各自暴露不同的使用场景、失败模式和审计字段。

**关键洞察**：
- 主密钥攻击（master-key attacks）、spurious rewards、GSM-Symbolic perturbations 表明奖励信号可以很广但也很脆弱
- LLM judges 可能奖励 trigger tokens
- Math accuracy 可能在表面变化下崩溃
- Rule verifiers 可能 false-reject
- Learned verifiers 可能被 hack

---

## 4. 理论分析：Scaling — 数据基板的渐近线与优化器的效率

### 4.1 渐近线与效率分解

近期的规模化研究使推理后训练看起来更像一个「什么改变了的账簿」而非单一规律。作者引用 Khatri et al. (2025) 和 Tan et al. (2026) 的方程作为阅读设备：

$$
R(C) = R_0 + \frac{A - R_0}{1 + (C_{\text{mid}}/C)^B}
$$

$$
\log L(N, C) = E(N) - k(N) \log C
$$

| 符号 | 含义 |
|------|------|
| $R(C)$ | 在计算预算 $C$ 下可达到的性能 |
| $R_0$ | 初始性能 |
| $A$ | 可达天花板（reachable ceiling） |
| $B$ 或 $k(N)$ | 接近效率（approach efficiency） |
| $C_{\text{mid}}$ | 达到半数提升所需计算量 |
| $L(N, C)$ | Loss，与模型规模 $N$ 和计算量 $C$ 相关 |
| $E(N)$ | 模型规模相关的效率项 |

**核心观点**：benchmark 增益本身不能自解释——它可能来自：
1. **移动 ceiling**（可达上限提高）
2. **提升效率**（以更少计算更快接近上限）
3. **改变测量表面**（评估方式本身变化）

![](PTPRD_table2_fig7_scaling.png)

> **Figure 7**: 渐近线-效率分解。一次 scaling 结果可以改变数据基板使什么变得可达、训练方法多高效地接近前沿、或两者兼具。

> **Table 3**: Scaling 归因账簿。✓ 标记旋钮最直接影响的维度，△ 标记条件效应。$A$ = 可达天花板；$B/k$ = 接近效率。搜索拓扑、推理预算和评估协议改变的是测量能力，应与训练计算量区分开来。

### 4.2 Ceiling Movers vs. Efficiency Movers

| 类型 | 改变什么 | 具体因素 |
|------|---------|---------|
| **Ceiling movers**（天花板推动者） | 改变数据基板使什么变得可达 | data substrate（质量、多样性、新鲜度）、verifier quality、support coverage、context、architecture、search topology |
| **Efficiency movers**（效率推动者） | 改变策略接近前沿的效率 | loss design、sampling、rollout budget、curriculum、precision、warm-start distillation |

两者区分并非绝对：某些干预（如 verifier refresh 或 environment redesign）可能同时影响两者。

### 4.3 小池 vs. 大池覆盖

![](PTPRD_fig8_table3_pool_coverage.png)

> **Figure 8**: 小池 vs. 大池覆盖。小的策选池在重复采样基座模型已能处理的能力带时可以有效；大池在有用梯度位于尾部或必须覆盖多个基座、验证器、领域时更重要。

核心洞见：scaling 不是参数量的简单函数，而是 **base prior 下的参数计数、uniqueness budget、teacher lineage 和 stage order** 的综合结果。

### 4.4 发布时间是 Scaling 对象

Figure A6 将发布时间视为 scaling 对象的一部分：一个发布日期不是中性的时间戳，而是版本化的数据制品标记——它可能标记不同的 prompt pool、新的验证器、变更的推理预算或新的污染审计。

---

## 5. 局限性与未来方向

### 5.1 本文局限性

1. **证据受限**：受限于后训练推理数据领域的公开证据。Closed pipelines、proprietary data mixtures 和未记录的发布实践必然缺失。
2. **报告不完整**：许多公开报告缺少 lineage cards、verifier versions、compute and inference budgets、contamination audits。
3. **问题驱动非元分析**：综合是问题驱动的，而非正式 meta-analysis。纳入的工作标准是「至少暴露了一个推理数据组件」（数据对象、验证器、trace source、reward channel、environment、scaling rule 或 release metadata）。
4. **不独立验证**：作者不独立重新运行训练配方、审计污染或验证每个 verifier。
5. **框架待扩展**：分类法需要更新以适应多模态、多语言、agentic 和 co-evolving verifier-generator 设置。

### 5.2 未来方向

本文的核心未解决问题是 **attribution（归因）**：当一个模型改进时，是其数据基板、验证器、基座模型、谱系、优化器、脚手架还是推理预算变了？本文的目的是帮助社区从「报告增益」转向「使增益可检查、可比较、可测试」。

具体建议：
- 发布推理数据时应报告完整的 attribution ledger（Table 2/3 所示字段）
- 区分 ceiling movers 和 efficiency movers
- 将 agent 轨迹作为可 replayable episodes 发布，而非仅清理后的成功记录
- 将 self-play 的 anchor、selection rule、replay policy 作为数据对象报告
- 将发布日期视为版本化的 scaling 对象

---

## 6. 个人思考

### 6.1 方法的优雅之处

1. **Verifier-Anchored Taxonomy 是一个极具洞察力的框架**：跳出「按领域分类」的惯性思维，从「什么可以被检查」出发，揭示了推理数据的本质——学习信号取决于验证能力，而非问题所属领域。这在哲学上将「可验证性」置于「知识领域」之上。

2. **反直觉教训表的价值**：Table 1 不仅是总结，更是一个操作化的「偏见清单」。每个陷阱都对应了社区中仍在广泛传播的迷思（如「数据越多越好」「CoT 越长越好」），而作者不仅纠正了误解，还附上了「未来应报告字段」——这对实际研究者和数据发布者极具指导意义。

3. **Attribution Ledger 的概念**：将数据构建过程视为一个「账簿」，每个层次都引入了一个「归因把手」。这种思维方式将模糊的「数据质量」问题转化为可审计的工程实践。

### 6.2 局限与未尽之处

1. **缺少定量分析**：作为 primer，本文以定性的框架构建和概念澄清为主，缺少对 150+ 篇论文的定量荟萃分析（如元回归）。

2. **实践指导可更具体**：虽然提出了很多「应报告字段」，但对于如何实际操作地构建高质量推理数据集（如具体的过滤策略、难度校准方法），具体可操作的指南仍然有限。

3. **与当前 SOTA 的时效性**：论文引用大量 2024-2025 年的工作，但推理数据领域变化极快（如 DeepSeek-R1、Kimi k1.5、OpenAI o3 等系统的不断迭代），部分结论可能需要随新证据更新。

### 6.3 启发

- **数据工程 > 算法工程**：本文强化了一个核心信念——后训练中，**数据的构造方式比训练算法本身更重要**。这在当前「堆算力刷 benchmark」的风气下是一个必要的纠偏。
- **可审计性即科学性**：作者反复强调 attribution、audit fields、lineage——这实际上是在呼吁推理数据领域建立类似实验科学的「可复现性标准」。
- **失败面的重要性**：文中多处强调「 verifier 也是 failure surface」（验证器也会失败），这提醒我们：在追求更好验证器的同时，也要记录验证器的 failure modes、attack surfaces 和 refresh cadences。

---

## 7. 关键引用

```bibtex
@article{li2025primer,
  title={A Primer in Post-Training Reasoning Data: What We Know About How It Works},
  author={Li, Yaoming and Zhao, Guangxiang and Shi, Qilong and Sun, Lin and Zhang, Xiangzheng and Yang, Tong},
  journal={arXiv preprint arXiv:2606.02113},
  year={2025}
}
```

---

## 8. 速查索引：数据集、系统与关键概念

### 8.1 代表性推理数据集

| 数据集 | 类型 | 验证契约 | 核心特点 |
|--------|------|---------|---------|
| **DeepMath-103K** | 数学 | Programmatic | 大规模数学数据集，extractions 和 rule verification operationalized |
| **DAPO** | 数学/代码 | Programmatic | Open-source LLM RL system，开源推理训练系统 |
| **PRM800K** | 数学 | Step-level / Process | 逐步标注的数学推理过程监督数据 |
| **Math-Shepherd** | 数学 | Step-level / Process | 估计 rollout value 的 PRM 数据 |
| **OmegaPRM** | 数学 | Step-level / Process | 寻找 first error 的过程监督数据 |
| **OpenThoughts** | 通用推理 | Judgment-required / Mixed | 数据构造可消融，source/mixture/filter/generator/teacher 明确 |
| **Skywork-OR1** | 数学 | Difficulty model-aware | 对难度进行 base-aware 估计 |
| **LIMO / s1** | 数学 | Few-shot elicitation | 小规模策选数据可激发强行为 |
| **GSM-Symbolic** | 数学 | Programmatic perturbation | 用于验证reward hacking和验证器鲁棒性 |

### 8.2 代表性系统与框架

| 系统/框架 | 类别 | 核心机制 |
|----------|------|---------|
| **STaR** | Self-play | 将 rollout anchor 到外部答案 |
| **R-Zero** | Self-play | 分离 Challenger 和 Solver 角色 |
| **Absolute Zero** | Self-play | 使用 Python interpreter 作为 anchor |
| **TTRL** | Self-play | 将 test-time majority vote 转为 reward |
| **AlphaEvolve** | Self-play / Evolution | 多智能体变体，将功能分布在不同角色上 |
| **ProcessBench / PRMBench** | Verifier evaluation | 测试 PRM 的 localization、soundness、sensitivity、robustness |
| **CoVeR / DeepSeekMath-V2** | Co-evolution | 共同演化的验证器与生成器 |
| **AlphaEvolvolve** | Search substrate | discovery through auditable counter-pressure |
| **MiniMax-M1** | RL | 改变 importance-weight control |
| **Qwen3-Coder-style** | RL / Agent | 将 signal 移入长视野工具交互 |

### 8.3 六种训练用途（Training Uses）

推理数据的后训练使用方式不是一个单一操作，而是一个谱系。根据论文 Figure 1 下方的训练用途横条：

| 训练方式 | 核心思想 | 对数据的要求 |
|---------|---------|-------------|
| **SFT**（监督微调） | 在高质量输入-输出对上直接微调 | 需要 curated、filtered 的 prompt-response 对或 traces |
| **Distillation**（蒸馏） | 从强教师模型传输知识到学生 | 需要 teacher traces，关注 inherited reasoning style |
| **Preference Learning**（偏好学习） | 学习比较和排序 | 需要 paired comparisons 或 ranked outputs |
| **Process Supervision（PRM）** | 对中间步骤进行奖励建模 | 需要 step-level labels，first-error localization |
| **RLVR**（可验证奖励的RL） | 使用程序化/环境验证器进行RL | 需要可验证的环境/答案，reward channel 明确 |
| **Agent Training** | 在交互环境中训练智能体 | 需要 replayable episodes，包含 states/actions/observations/failures |

> 🔗 延伸阅读：论文项目仓库整理了完整的 Awesome-LLM-Reasoning-Data 资源列表 → `https://github.com/RenBing-Sumeru/Awesome-LLM-Reasoning-Data`

---

## 附录：补充图表与速查参考

### A.1 智能体轨迹审计字段（Table A3）

对于环境推理数据（agent reasoning data），prompt–answer 数据集过于贫乏——可训练对象是轨迹（trajectory）。一个仅包含最终成功路径的发布常常不够：它可能隐藏失败的 tool calls、retries、state diffs、无效动作、scaffold 干预或终止谓词。

以下审计字段使得 agent 轨迹可 replayable 且可审计：

| 审计字段                     | 应发布的内容                                                                      | 为何重要                      |
| ------------------------ | --------------------------------------------------------------------------- | ------------------------- |
| **Task state**           | 初始状态、文件、UI 状态、仓库快照、环境版本                                                     | 决定 episode 是否可以重置和重放      |
| **Goal and constraints** | 用户目标、隐藏测试、策略约束、允许/禁止动作                                                      | 区分任务成功与 scaffold 合规       |
| **Action schema**        | 工具/API schema、动作语法、参数格式、超时与权限规则                                             | 定义模型可用的动作空间               |
| **Observations**         | 工具输出、浏览器/应用观察、终端日志、截图、错误信息                                                  | 暴露 episode 期间实际可用的反馈      |
| **State diffs**          | 文件变更、patches、数据库变更、UI 转换、环境 deltas                                          | 使信用分配和恢复行为可被检查            |
| **Failures and retries** | 无效动作、失败调用、被拒绝的 patches、恢复尝试、回溯                                              | 保存通常被清理后的 SFT trace 抹除的分支 |
| **Terminal predicate**   | 成功条件、单元测试、grader、judge、环境谓词                                                 | 定义什么被计为完成或奖励              |
| **Scaffold metadata**    | planner、工具包装器、记忆、prompt template、agent loop、停止规则                            | 防止 scaffold 变更被误认为是模型变更   |
| **Budget and sampling**  | rollout 次数、token 预算、时间限制、temperature、pass@k 或 selector                      | 使跨系统的推理时间计算可比较            |
| **Lineage and split**    | generator、teacher、verifier version、filtering rule、split、contamination audit | 支持跨代和跨发布的归因               |

> 💡 **核心建议**：环境验证不能归约为最终成功。如果发布只包含成功路径，它就抹除了 tool misuse、recovery、exploration 和 credit assignment 变得可见的分支——而这些分支对于后训练恰恰是最富信息量的部分。

![](PTPRD_appdx_tableA3_A4.png)

> **Table A3 & Table A4**: 智能体轨迹审计字段与来源放置指南。上表（A3）列出了使 agent 轨迹可回放和可审计的10个关键字段；下表（A4）说明模型报告、数据发布、过程监督、环境基准、验证器研究和规模化研究应分别服务于不同的归因声明，不可互换使用。

### A.2 来源放置指南（Table A4）

本文根据每种来源暴露的证据类型来放置引用。以下为速查参考：

| 来源角色 | 贡献什么 | 典型实例 | 放置规则 |
|---------|---------|---------|---------|
| **Model reports** | 公开后训练脚手架、配方描述、推理预算 | DeepSeek-R1、Kimi K1.5、Qwen3-style reports | 用于可见的 pipeline 声明，**不能单独用于数据因果归因** |
| **Reasoning-data releases** | prompt source、answer format、verifier、filter、split、contamination audit | DeepMath-103K、DAPO、OpenThoughts | 用于数据对象和构建声明 |
| **Process-supervision work** | step labels、PRM training、rollout values、first-error localization | PRM800K、Math-Shepherd、OmegaPRM | 用于 trace quality 和监督粒度声明 |
| **Agent / environment benchmarks** | state、action、observation、terminal predicate、replay 或 task scaffold | SWE-Gym、AppWorld、OSWorld-style tasks | 用于环境验证和轨迹审计声明 |
| **Verifier and judge studies** | verifier failure modes、judge bias、reward hacking、false rejection | verifier robustness、master-key、spurious reward 研究 | 用于正确性和审计风险声明 |
| **Scaling studies** | compute、data reuse、asymptote、approach efficiency、inference-time budget | RL scaling 和后训练 scaling 分析 | 用于 scaling 归因声明 |

> 💡 **关键原则**：不同来源**不应被当作同一声明的可互换证据**。模型报告、数据发布、验证器审计和 scaling 研究应分别服务于不同的归因目的。

### A.3 补充图示

![](PTPRD_appdx_figA1_tableA1.png)

> **Figure A1 & Table A1**: 补充分类视角与纳入标准。验证契约说明「什么可以被检查」，而粒度、行为边界和跨代谱系三个横切维度解释了「信号进入训练后如何表现」。它们应与程序化、环境和需判断验证契约正交报告。

![](PTPRD_appdx_figA3_A4_A5_A6.png)

> **Figure A3–A6**: Trace 撰写者分类（A3）、Self-play 策选重定位（A4）、Ceiling vs. Efficiency movers 详解（A5）、以及发布时间作为 scaling 对象（A6）。

对于环境推理数据（agent reasoning data），prompt–answer 数据集过于贫乏——可训练对象是轨迹（trajectory）。一个仅包含最终成功路径的发布常常不够：它可能隐藏失败的 tool calls、retries、state diffs、无效动作、scaffold 干预或终止谓词。

以下审计字段使得 agent 轨迹可 replayable 且可审计：

| 审计字段 | 应发布的内容 | 为何重要 |
|---------|-------------|---------|
| **Task state** | 初始状态、文件、UI 状态、仓库快照、环境版本 | 决定 episode 是否可以重置和重放 |
| **Goal and constraints** | 用户目标、隐藏测试、策略约束、允许/禁止动作 | 区分任务成功与 scaffold 合规 |
| **Action schema** | 工具/API schema、动作语法、参数格式、超时与权限规则 | 定义模型可用的动作空间 |
| **Observations** | 工具输出、浏览器/应用观察、终端日志、截图、错误信息 | 暴露 episode 期间实际可用的反馈 |
| **State diffs** | 文件变更、patches、数据库变更、UI 转换、环境 deltas | 使信用分配和恢复行为可被检查 |
| **Failures and retries** | 无效动作、失败调用、被拒绝的 patches、恢复尝试、回溯 | 保存通常被清理后的 SFT trace 抹除的分支 |
| **Terminal predicate** | 成功条件、单元测试、grader、judge、环境谓词 | 定义什么被计为完成或奖励 |
| **Scaffold metadata** | planner、工具包装器、记忆、prompt template、agent loop、停止规则 | 防止 scaffold 变更被误认为是模型变更 |
| **Budget and sampling** | rollout 次数、token 预算、时间限制、temperature、pass@k 或 selector | 使跨系统的推理时间计算可比较 |
| **Lineage and split** | generator、teacher、verifier version、filtering rule、split、contamination audit | 支持跨代和跨发布的归因 |

> 💡 **核心建议**：环境验证不能归约为最终成功。如果发布只包含成功路径，它就抹除了 tool misuse、recovery、exploration 和 credit assignment 变得可见的分支——而这些分支对于后训练恰恰是最富信息量的部分。

### A.2 来源放置指南（Table A4）

本文根据每种来源暴露的证据类型来放置引用。以下为速查参考：

| 来源角色 | 贡献什么 | 典型实例 | 放置规则 |
|---------|---------|---------|---------|
| **Model reports** | 公开后训练脚手架、配方描述、推理预算 | DeepSeek-R1、Kimi K1.5、Qwen3-style reports | 用于可见的 pipeline 声明，**不能单独用于数据因果归因** |
| **Reasoning-data releases** | prompt source、answer format、verifier、filter、split、contamination audit | DeepMath-103K、DAPO、OpenThoughts | 用于数据对象和构建声明 |
| **Process-supervision work** | step labels、PRM training、rollout values、first-error localization | PRM800K、Math-Shepherd、OmegaPRM | 用于 trace quality 和监督粒度声明 |
| **Agent / environment benchmarks** | state、action、observation、terminal predicate、replay 或 task scaffold | SWE-Gym、AppWorld、OSWorld-style tasks | 用于环境验证和轨迹审计声明 |
| **Verifier and judge studies** | verifier failure modes、judge bias、reward hacking、false rejection | verifier robustness、master-key、spurious reward 研究 | 用于正确性和审计风险声明 |
| **Scaling studies** | compute、data reuse、asymptote、approach efficiency、inference-time budget | RL scaling 和后训练 scaling 分析 | 用于 scaling 归因声明 |

> 💡 **关键原则**：不同来源**不应被当作同一声明的可互换证据**。模型报告、数据发布、验证器审计和 scaling 研究应分别服务于不同的归因目的。

### A.3 补充图示

![](PTPRD_appdx_figA1_tableA1.png)

> **Figure A1 & Table A1**: 补充分类视角与纳入标准。验证契约说明「什么可以被检查」，而粒度、行为边界和跨代谱系三个横切维度解释了「信号进入训练后如何表现」。它们应与程序化、环境和需判断验证契约正交报告。

![](PTPRD_appdx_figA3_A4_A5_A6.png)

> **Figure A3–A6**: Trace 撰写者分类（A3）、Self-play 策选重定位（A4）、Ceiling vs. Efficiency movers 详解（A5）、以及发布时间作为 scaling 对象（A6）。
