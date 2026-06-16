---
created: 2026-06-10
published: 2025
paper: https://arxiv.org/abs/2505.08775
code: https://github.com/openai/simple-evals
authors: Rahul K. Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Quiñonero-Candela, Foivos Tsimpourlas, Michael Sharman, Meghan Shah, Andrea Vallone, Alex Beutel, Johannes Heidecke, Karan Singhal (OpenAI)
tags:
  - Benchmark
  - LLM-Evaluation
  - Healthcare
  - Rubric-Evaluation
  - OpenAI
  - Multi-turn-Conversation
---

# HealthBench: Evaluating Large Language Models Towards Improved Human Health

## 一句话总结

OpenAI 联合 262 名医生构建了 HealthBench —— 一个包含 5,000 组多轮对话、48,562 条对话专属评分细则（rubric）的开源医疗 benchmark；在两年时间尺度上，前沿模型得分从 GPT-3.5 的 16% 提升到 o3 的 60%，但当前最强模型在 Hard 子集上也仅达 32%，仍有巨大提升空间。

![](HealthBench_fig1_infographic.jpg)

> **Figure 1**: HealthBench 的整体框架示意。每一组评测示例包含一段多轮对话和由医生撰写的、针对该对话的评分细则；模型回复被逐条对照细则打分，最终汇总得出一个 [0,1] 区间的分数。

---

## 1. 研究背景与动机

### 1.1 问题定义

AI 在医疗健康领域的应用日益广泛，包括扩展健康信息获取渠道、辅助临床决策、帮助个人做出更优健康选择等。然而，**如何可靠地衡量大语言模型（LLM）在真实医疗场景中的表现**，仍然是一个悬而未决的核心问题。

### 1.2 现有方法的三大不足

作者系统性地指出了当前医疗评测基准的三重缺陷：

| 维度 | 追问 | 现有 benchmark 的问题 |
|------|------|----------------------|
| **Meaningful** | 分数是否反映真实世界的影响？ | 多数 benchmark 是选择题或短答题，无法捕捉真实人机交互的复杂性 |
| **Trustworthy** | 分数是否忠实于医生的专业判断？ | 缺少大规模、跨专科、跨地域的医生参与验证 |
| **Unsaturated** |  benchmark 是否仍有提升空间以驱动进步？ | 很多 medical benchmark 已被刷到接近天花板 |

> 相比 MedQA、PubMedQA 等经典 benchmark 以 multiple-choice question（MCQ）为主，HealthBench 的核心创新在于 **open-ended、conversation-level 的 rubric 评估**，让模型在真实对话场景中被医生认可的标准所衡量。

### 1.3 健康场景的特殊性

医疗场景不同于通用问答，其独特性体现在：
- **安全性要求极高**：错误信息可能直接危害患者健康
- **上下文依赖性强**：同一症状在不同患者背景（年龄、既往史、地域医疗条件）下处理方式截然不同
- **沟通角色多变**：患者、家属、医生、护士对回复深度和专业术语的需求各不相同
- **不确定性常态化**：模型需要学会表达不确定性，而非强行给出确定性答案

---

## 2. 方法：Rubric Evaluation 详解

### 2.1 核心思想

HealthBench 采用 **rubric evaluation**（评分细则评估）范式：

1. **输入**：一段多轮对话（conversation），以用户最后一条消息作为待回复的 prompt
2. **输出**：模型对该消息的回复
3. **打分**：将回复与一组 **对话专属**（conversation-specific）的 rubric criteria 逐条对照，满足则得满分，不满足则得 0 分
4. **聚合**：取所有 criteria 得分的均值，再 clip 到 [0,1] 区间，作为该 example 的最终得分

### 2.2 评分机制

每条 rubric criterion 的结构：

| 字段 | 说明 |
|------|------|
| 描述 | 该 criterion 要求模型回复具备的某种属性 |
| 分值 | 介于 -10 到 10 之间的非零整数；负分用于惩罚不良行为（如给出危险建议） |
| 判定 | 满足 → 得全部分值；不满足 → 得 0 分（无中间分） |

**总分计算公式**：

$$
\text{Score} = \text{clip}\left( \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\text{criterion}_i \text{ met}] \cdot p_i, \; 0, \; 1 \right)
$$

其中 $N$ 为 criteria 数量，$p_i$ 为第 $i$ 条 criterion 的分值，$\mathbf{1}[\cdot]$ 为示性函数，clip 到 [0,1] 区间。

> 与经典 MCQ（非对即错、每题等权重）不同，HealthBench 的 rubric 允许 **不同对话有不同的评分侧重点**，并且负分机制能显式惩罚危险回复。

### 2.3 关键设计选择

**为什么是 conversation-level rubric，而非 global rubric？**
- 同一主题下的不同对话，所需信息、回复风格和深度差异巨大
- 例如"紧急转诊"场景，有些需要快速识别危险信号，有些则需询问关键补充信息
- 对话专属 rubric 能更精确地刻画"这条回复是否好"

**为什么是 binary scoring，而非 Likert scale？**
- 降低医生标注的主观差异
- 简化模型自动评分器的学习难度
- 在 meta-evaluation 中证明 binary 判定与医生判断的一致性足够高

---

## 3. 数据构建

### 3.1 医生团队

![](HealthBench_fig3_map.png)

> **Figure 3**: HealthBench 医生团队在全球 60 个国家有执业经验，覆盖 26 个医学专科，通晓 49 种语言。

| 指标 | 数据 |
|------|------|
| 参与医生 | 262 名 |
| 招募周期 | 11 个月 |
| 执业国家 | 60 个 |
| 医学专科 | 26 个 |
| 流利语言 | 49 种 |
| 执业角色 | 50% 独立执业/主治医师，17% 专科培训医师（fellow），23% 高年住院医，10% 低年住院医 |
| 筛选率 | 1,021 人表达兴趣 → 262 人入选（26%）→ 31 人被后续筛除 |

### 3.2 对话来源

为获得真实且多样的人机对话，数据来自三条管道：

1. **合成生成（Synthetic）**：通过定制的语言模型程序流水线生成多样化对话风格
2. **医生 red-teaming**：让医生主动挑战 LLM，暴露其在医疗场景中的弱点
3. **HealthSearchQA 改编**：将 Google 发布的健康搜索问答数据转化为对话形式

**过滤标准**：必须满足真实性（realistic）、与身体健康相关、内容完整。

### 3.3 数据集统计

| 统计量 | 字符数 | 对话轮数 | Criteria 数 |
|--------|--------|----------|-------------|
| min | 4 | 1 | 2 |
| median | 281 | 1 | 11 |
| mean | 667.6 | 2.6 | 11.5 |
| max | 9,853 | 19 | 48 |

- 总计 **5,000 组对话**
- 总计 **48,562 条** 对话专属 unique rubric criteria
- 中位数对话仅 1 轮（单轮问答为主），但 95% 分位达 7 轮

### 3.4 两个变体

**HealthBench Consensus**
- 从所有 criteria 中筛选出 **34 条**经医生共识验证的核心准则（出现 8,053 次）
- 保留至少包含 1 条共识准则的 3,671 组对话
- 用于聚焦最重要、最少争议的评估维度

**HealthBench Hard**
- 精选 **1,000 组**当前前沿模型（截至论文发表时）表现最差的困难对话
- 当前最高分仅 **32%**，作为未来模型能力的"压力测试"

---

## 4. Themes 与 Axes：多维评估体系

### 4.1 Themes（健康情境主题）

| Theme | 数量 | 占比 | 核心挑战 |
|-------|------|------|---------|
| Global health（全球健康） | 1,097 | 21.9% | 适应不同国家/地区的医疗条件和文化 |
| Responding under uncertainty（不确定性应对） | 1,071 | 21.4% | 识别不确定性并以恰当语言回应 |
| Expertise-tailored communication（专业匹配沟通） | 919 | 18.4% | 根据用户身份（患者/医生）调整专业深度 |
| Context seeking（上下文追问） | 594 | 11.9% | 在信息缺失时主动追问最相关的补充信息 |
| Emergency referrals（紧急转诊） | 482 | 9.6% | 识别紧急情况并引导就医 |
| Health data tasks（健康数据处理） | 477 | 9.5% | 安全准确地完成结构化临床数据处理 |
| Response depth（回复深度适配） | 360 | 7.2% | 根据用户需求和任务复杂度调整回复深度 |

### 4.2 Axes（行为维度）

| Axis | 数量 | 占比 | 定义 |
|------|------|------|------|
| Completeness（完整性） | 22,285 | 39% | 回复是否包含所有关键信息 |
| Accuracy（准确性） | 18,888 | 33% | 信息是否事实正确、符合医学共识 |
| Context awareness（上下文意识） | 8,991 | 16% | 是否恰当回应上下文线索、必要时追问澄清 |
| Communication quality（沟通质量） | 4,522 | 8% | 结构清晰、简洁、用词匹配用户专业水平 |
| Instruction following（指令遵循） | 2,551 | 4% | 遵循用户指令的同时不牺牲安全性 |

> 值得注意的是，**Completeness（39%）** 和 **Accuracy（33%）** 合计占据了 72% 的 criteria，反映出医疗场景对"给出完整且正确的信息"的极度重视。而 Instruction following 仅占 4%，是因为医疗安全往往要求**不盲目遵循**可能有害的指令。

---

## 5. 实验结果

### 5.1 模型性能总览

![](HealthBench_fig4_frontier.png)

> **Figure 4**: OpenAI 模型在 HealthBench 上的性能演进。近两年从 GPT-3.5 Turbo（16%）到 GPT-4o（32%）为稳健初期进步，最近几个月从 o1 到 o3 跃升至 60%，加速趋势明显。

**主要模型得分**（HealthBench 完整版）：

| 模型 | 得分 |
|------|------|
| o3 | **60%** |
| o4-mini | ~55% |
| GPT-4.1 | ~48% |
| o1 | ~42% |
| Claude 3.7 Sonnet (extended thinking) | ~40% |
| Gemini 2.5 Pro | ~38% |
| Grok 3 | ~37% |
| GPT-4o (Aug 2024) | 32% |
| Llama 4 Maverick | ~30% |
| GPT-3.5 Turbo | 16% |

### 5.2 分 Theme 表现

![](HealthBench_fig5_theme.png)

> **Figure 5**: 不同模型在各 Theme 上的得分。紧急转诊（Emergency referrals）和专业匹配沟通（Expertise-tailored communication）普遍得分最高；上下文追问（Context-seeking）、健康数据处理（Health data tasks）和全球健康（Global health）是各模型的共同短板。

**关键观察**：
- **Emergency referrals** 和 **Expertise-tailored communication** 表现最好 —— 这类任务目标相对明确
- **Context-seeking**、**Health data tasks**、**Global health** 显著落后 —— 这类任务需要模型推断缺失信息、适应地域差异性，或完成结构化数据处理，难度更高

### 5.3 分 Axis 表现

![](HealthBench_fig6_axis.png)

> **Figure 6**: 不同模型在各行为维度（Axis）上的得分。完整性（Completeness）和上下文意识（Context awareness）是普遍短板；o3 在 Completeness 上相较前代模型有显著提升。

**关键观察**：
- **Completeness** 和 **Context awareness** 的得分明显低于 Accuracy、Communication quality 和 Instruction following
- o3 在 **Completeness** 上大幅领先 —— 说明 test-time compute / reasoning 对"给出更完整的回答"有直接帮助
- 有趣的是，Claude、GPT-4o 和 o1 的 **Communication quality** 得分高于 o3，但 o3 在 Completeness 上远超它们，体现了不同模型的 trade-off

### 5.4 性能-成本前沿

![](HealthBench_fig2_cost.png)

> **Figure 2**: OpenAI 模型在 HealthBench 上的得分与推理成本对比。GPT-4.1 nano 得分超越 GPT-4o（2024年8月版），成本仅为后者的 1/25。

**核心发现**：
- **GPT-4.1 nano**（小模型）得分 **> GPT-4o**，且推理成本低 **25 倍**
- 这说明通过模型架构和训练策略的优化，即使不增大模型规模也能在医疗场景获得显著进步
- o3 / o4-mini 系列在低/中/高 reasoning 级别下呈现成本-性能正相关，test-time compute 扩展有效

### 5.5 可靠性：Worst-at-k

![](HealthBench_fig7_worst.png)

> **Figure 7**: 各模型 worst-at-k 性能（k 次采样的最差得分）。o3 的 worst-at-16 得分为 GPT-4o 的两倍以上，说明更强大的模型不仅平均表现好，最差表现也更稳定。

- o3 的 worst-at-16 得分为 GPT-4o（2024年8月版）的两倍以上
- 但 o3 整体得分从 60% 降至 worst-at-16 时下降了约三分之一，说明即使是当前最强模型，在多轮采样中仍有不可忽略的失败率

### 5.6 得分分布与天花板

![](HealthBench_fig8_dist.png)

> **Figure 8**: 各模型在 HealthBench 上的 example-level 得分分布。

- **没有"所有模型都已解决"的问题** —— 每个 example 至少有一个前沿模型失败，说明 benchmark 远未饱和
- **也没有"所有模型都失败"的问题** —— 大多数问题至少被部分模型部分解决，说明该 benchmark 有合理的梯度
- 这验证了 HealthBench 的 **unsaturated** 属性，能持续区分模型能力提升

### 5.7 回复长度与性能

| 模型 | 与回复长度的 Pearson r |
|------|----------------------|
| GPT-4o | -0.053（几乎无关） |
| o3 | +0.123（轻度正相关） |

Win rate 对比（o3 vs GPT-4.1）：
- 不控制长度：o3 胜率 72.9%
- 控制长度（相差 10% 以内）：o3 胜率降至 63.7%
- GPT-4.1 vs o1：控制长度后胜率从 61.0% 反转为 65.2%

> 结论：近期模型在 HealthBench 上的进步**部分**可归因于回复更详尽，但 verbosity 不能完全解释性能提升。

### 5.8 HealthBench Consensus

![](HealthBench_fig9_consensus.png)

> **Figure 9**: 模型在 34 条共识准则上的错误率。从 GPT-3.5 到 GPT-4.1，错误率下降了 4 倍以上。

- 模型在 Context-seeking（上下文追问）、Responding under uncertainty（不确定性回应）和 Response depth（深度适配）上的错误率仍然较高
- 说明这些需要"主动行为"而非"被动知识"的能力是当前模型的核心瓶颈

### 5.9 HealthBench Hard

![](HealthBench_fig10_hard.png)

> **Figure 10**: 各模型在 HealthBench Hard（1,000 组困难对话）上的分 Axis 性能。o3 在 Hard 子集上仅得 32%，与其在完整版上的 60% 形成鲜明对比。

- 即使是 o3，在 Hard 子集上的得分也几乎腰斩（60% → 32%）
- 这说明 HealthBench Hard 有效地筛选出了当前模型尚未掌握的困难场景
- Hard 子集尤其暴露了 Completeness 和 Context awareness 的极限

---

## 6. 医生写作响应：人与 AI 的对比

### 6.1 实验设计

将医生分为三组，要求其为 HealthBench 对话撰写回复：

| 组别 | 条件 |
|------|------|
| De novo | 无 AI 辅助，但允许使用互联网 |
| Sep 2024 参考 | 提供 GPT-4o 和 o1-preview 的回复作为参考 |
| Apr 2025 参考 | 提供 GPT-4.1 和 o3 的回复作为参考 |

### 6.2 核心结果

![](HealthBench_fig11_human.png)

> **Figure 11**: 医生撰写的回复与 AI 参考回复的 HealthBench 得分对比。

| AI 参考版本 | 医生改进率 | 医生退步率 |
|------------|-----------|-----------|
| Sep 2024 | 56.2% | 39.8% |
| Apr 2025 | 46.8% | 47.7% |

**关键发现**：
- **Sep 2024 参考**：医生在看到 GPT-4o/o1-preview 回复后能显著改进自己的回复，尤其在 Completeness 方面最受益
- **Apr 2025 参考**：医生**无法**进一步改进 AI 的回复（改进率 ≈ 退步率）
- **De novo 医生回复**：得分较低，部分原因是医生倾向于写更简短的回答，而 HealthBench 的 rubric 往往奖励详尽回复

> 这意味着：到 2025 年4月，前沿 AI 模型在 HealthBench 上的回复质量**已经超越了独立工作的医生**（至少在 rubric 衡量的维度上），AI 作为辅助工具的价值正在发生变化。

---

## 7. 评分可信性：模型评分 vs 医生评分

### 7.1 Meta-Evaluation 设计

核心问题：**用模型（如 GPT-4.1）作为评分器，其判定是否与医生一致？**

- 构建 60,896 组 meta-examples，每条共识 criterion 平均有 1,791 组元评估数据
- 计算 Macro F1：

$$
\text{MF1} = 0.5 \times (\text{F1}_{\text{pos}} + \text{F1}_{\text{neg}})
$$

其中正类 F1 衡量"criterion 被满足"的判定一致性，负类 F1 衡量"criterion 未被满足"的判定一致性。

### 7.2 Meta-Evaluation 结果

![](HealthBench_fig12_meta.png)

> **Figure 12**: 医生评分者（灰色点，每位医生一个点）与模型评分器（GPT-4.1，蓝色线）在各 Theme 上的 MF1 一致性。模型评分器的得分与专家群体的中位数相当。

| Theme | 医生 MF1 均值 | 模型 MF1 | 模型百分位 |
|-------|-------------|---------|-----------|
| Global health | 0.648 | 0.706 | 73.5% |
| Context seeking | 0.646 | 0.706 | 88.2% |
| Emergency referrals | 0.647 | 0.662 | 70.0% |
| Responding under uncertainty | 0.640 | 0.679 | 68.3% |

**不同模型作为评分器的表现**：

| 评分器模型 | MF1 |
|-----------|-----|
| GPT-4.1 | **0.709** |
| o4-mini | 0.692 |
| o3 | 0.681 |
| GPT-4.1 mini | 0.661 |
| GPT-4.1 nano | 0.580 |

> GPT-4.1 作为评分器在多个主题上达到了医生评分者群体的中上水平（68-88 百分位），验证了 model-based grading 的可信度。同时，评分器模型本身的能力也会影响 grading 质量（nano 与 GPT-4.1 差距明显）。

### 7.3 运行稳定性

16 次独立运行的标准差仅约 **0.002**，说明 HealthBench 评测结果高度稳定。

---

## 8. 讨论与未来方向

### 8.1 数据质量的挑战

- 医生之间对同一回复的判定一致性约为 **55%-75%** —— 这个范围既反映了评分的固有主观性，也反映了医疗实践中真实存在的观点差异
- "个体的 criteria 不够全面，但在 aggregate 层面覆盖了足够多的评估维度"
- 医生撰写对话回复并非其日常工作，可能对 baseline 得分有一定影响

### 8.2 局限性

1. **未衡量健康结局（health outcomes）**：HealthBench 衡量的是回复质量，而非最终对患者健康结果的实际影响
2. **rubric 完备性**：单个对话的 criteria 不可能穷尽所有重要的评估角度
3. **医生撰写回复的非日常性**：如前述，医生在测试中写回复可能与实际临床沟通有差异
4. **基准线天花板尚不明确**：虽然示例层面的分数分布显示未饱和，但最终天花板仍未知

### 8.3 未来方向

- **真实世界研究**：在特定医疗工作流中部署模型，同时衡量回复质量和实际健康结局
- **多模态扩展**：当前 HealthBench 仅覆盖文本对话，未来可扩展至包含医学影像、化验单等多模态输入
- **实时迭代**：持续扩展对话库和 rubric，确保 benchmark 随模型能力提升而不断进化

---

## 9. 个人思考

### 9.1 方法论的优雅之处

HealthBench 的 rubric evaluation 范式值得在更多垂直领域借鉴：

1. **对话专属 rubric** 比 global rubric 更精确，但也带来了标注成本远高于传统 MCQ 的问题。5,000 对话 × 平均 11 条 criteria = 约 55,000 条人工标注，这需要巨大的资源投入。在资源有限时，HealthBench Consensus 的 34 条核心 criteria + 出现 8,053 次的设计是一种很好的折中。

2. **负分机制** 设计精妙。传统 benchmark 只能区分"对/错"，但医疗场景中"给出正确但危险的建议"比"给出无关回答"更糟。负分让评测器能显式惩罚风险行为。

3. **multi-turn** 设计贴近真实使用场景。medical queries 往往不是一次性问答，而是需要多轮追问和澄清的对话。

### 9.2 对 LLM 评估的启示

- **从 benchmark 到生态**：HealthBench 不只是一个数据集，它同时定义了 themes、axes、consensus criteria、hard subset、meta-evaluation 标准，形成了一个完整的评估生态
- **成本-性能前沿** 的披露极具价值。GPT-4.1 nano 在 1/25 成本下超越 GPT-4o，意味着医疗 AI 的部署门槛正在快速降低
- **Hard subset** 的构建方法（筛选当前模型最难的 example）比随机抽样更有效地区分前沿模型

### 9.3 谨慎看待的方面

- HealthBench 的 rubric 由 OpenAI 联合医生设计，且模型评分器也使用 OpenAI 模型，虽然 meta-evaluation 验证了评分器的可信度，但**利益相关方**的存在仍需要第三方独立验证
- 医生群体虽多元化，但 compensation 机制可能引入参与偏差（愿意为 pay 参与评测的医生，其背景和观点是否能代表全体医生？）
- "AI 回复已经超越独立医生"的结论需要小心解读：这是 rubric 层面的得分，不等于临床实际价值。医生的简短回复在实际临床场景中可能是高效且足够的

### 9.4 与当前研究的关联

HealthBench 代表了 LLM 评估从 **"知识测试"**（如 MedQA 考医学知识）到 **"能力测试"**（在 realistic conversations 中考察 accuracy + completeness + communication + context awareness + safety）的重要转变。这与当前 Agentic AI 和 long-context evaluation 的研究趋势高度一致。

---

## 10. 关键引用

```bibtex
@article{arora2025healthbench,
  title={HealthBench: Evaluating Large Language Models Towards Improved Human Health},
  author={Arora, Rahul K. and Wei, Jason and Hicks, Rebecca Soskin and Bowman, Preston and Qui{~n}onero-Candela, Joaquin and Tsimpourlas, Foivos and Sharman, Michael and Shah, Meghan and Vallone, Andrea and Beutel, Alex and Heidecke, Johannes and Singhal, Karan},
  journal={arXiv preprint arXiv:2505.08775},
  year={2025},
  url={https://arxiv.org/abs/2505.08775},
  code={https://github.com/openai/simple-evals}
}
```
