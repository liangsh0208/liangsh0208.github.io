---
created: 2026-06-10
published: 2025-08-16
paper: https://arxiv.org/abs/2508.11894
authors: Ao Li, Bin Yan, Bingfeng Cai, Chenxi Li, Cunzhong Zhao, Fugen Yao, Gaoqiang Liu, Guanjun Jiang, Jian Xu, Liang Dong, Liansheng Sun, Rongshen Zhang, Xiaolei Gui, Xin Liu, Xin Shang, Yao Wu, Yu Cao, Zhenxin Ma, Zhuang Jia (Quark Medical Team, Alibaba Group)
tags:
  - 医疗大模型
  - 领域微调
  - 强化学习
  - GRPO
  - RAG
  - QuarkMed
  - 阿里巴巴
---

# QuarkMed Medical Foundation Model Technical Report

## 一句话总结
QuarkMed 是阿里巴巴 Quark 医疗团队推出的 32B 参数医疗基础模型，通过多层医疗数据策展、能力驱动的指令微调和双阶段强化学习（可验证奖励 + 多维对齐），在中文医学执照考试（正高职称）达到 **51.70%** 准确率（知识增强后 **67.7%**），显著超越 DeepSeek-R1 的 38.70%，同时在 PubMedQA、DiagnosisArena 等公开基准上取得 SOTA。

---

## 1. 研究背景与动机

### 1.1 问题定义
医学领域语言具有**高度专业化词汇、复杂临床概念和细微语义**三大特征。通用 LLM 在该领域存在明显 **knowledge gap**，导致回答"不满意甚至不安全"。医疗场景对模型的要求不仅是"知识渊博且准确"，还必须"高度可靠且可定制"。

### 1.2 现有方法的不足
- **数据层面**：医疗数据质量参差不齐，缺乏系统性的多层策展管线，导致知识注入效率低。
- **训练层面**：传统 SFT 容易受低质量合成数据污染；RL 在医疗领域面临"高质量数据需求大、奖励函数难以精确设计"的挑战。
- **推理层面**：通用 LLM 在医疗问诊中易产生细微幻觉（subtle hallucinations），尤其是涉及药品用法、检验推荐等关键决策时。

### 1.3 QuarkMed 的核心思路
构建一套"**数据策展 → 能力驱动微调 → 双阶段强化学习 → 医学 RAG**"的端到端管线，让 32B 规模的模型在医疗垂直场景中匹敌甚至超越更大规模的通用模型。

---

## 2. 数据策展管线（Data Curation Pipeline）

QuarkMed 的数据层是整个系统的基石。团队在通用 LLM 之外，系统性地构建了三类医疗数据，并通过多层质量增强机制确保数据的可用性、权威性和隐私合规。

### 2.1 Medical Materials（医学材料）：~1T Tokens

**数据来源**：通过网页爬取和采购获取，涵盖教科书、临床指南、共识声明、学术文献、药品说明书、医学百科全书、临床路径。

**知识覆盖检测体系**（Knowledge Coverage Detection）：
- 基于医学专家手工构建的框架，对照 Bloom's Taxonomy 将知识点分为三类：
  - **Factual**（事实性知识）
  - **Conceptual**（概念性知识）
  - **Procedural**（程序性/推理性知识）
- 使用 Quark Search 查询的 query-CoT 和内部医学知识图谱构建测试集
- 通过**迭代评估与补充**（iterative evaluation and supplementation），最终达到：
  | 知识类别 | 覆盖率 |
  |---------|--------|
  | Factual | **> 90%** |
  | Conceptual | **84%** |
  | Procedural (QCOT) | **75%** |
- 这一覆盖率的递进结构（事实 → 概念 → 推理）与模型训练的不同阶段对齐。

**图像材料质量增强**：
- 初始方案：OCR + 布局分析模型提取图像材料文本
- **改进方案**：训练基于 **Qwen2.5-VL** 的细粒度内容结构化模型（fine-grained content structuring model）
- 效果：pre-training corpus 质量相比原始 OCR 方法提升 **> 30%**
- 平均数据可用率（data usability rate）达到 **> 90%**，结构化良好的图像（如教科书）接近上限

**权威性分级（Authoritativeness Labeling）**：
- 基于循证医学（evidence-based medicine）原则，按材料类型、来源、影响因子分为五级：**A-E**
- 高权威数据（A/B 级）在库中的分布：
  - 临床指南：**> 40%**
  - 学术文献：**26%**
  - 书籍：**5%**
- 权威性分级用于不同阶段的训练数据过滤和 RAG 检索的优先级排序

**知识合成（Knowledge Synthesis）**：
- 针对原始材料中概念性知识覆盖不足的问题，在疾病、症状、药物、手术、检验等关键子领域进行系统性合成
- **具体案例**：对同通用名药物，合并来自不同厂商的说明书，结合百科全书和药理教材信息，为每个通用药物名称合成一份 comprehensive insert，用于通用知识增强

### 2.2 Medical Knowledge（医学知识）

结构化和非结构化医学知识的处理策略不同。

**非结构化数据处理**：
- 数据来源分类与规模（Table 1）：

| 主类别 | 子类别 | 近似规模 |
|-------|--------|---------|
| Web-based Resources | Q&A 平台、文章、百科条目 | 数千万 |
| Professional Materials | 临床指南、出版物、药品说明书、医学标准、医学考试 | 数百万 |
| Knowledge Bases | 标准医学术语集、医学本体、词典 | 数千万 |
| Medical Scenario Data | 在线问诊对话、患者病历记录 | 数千万 |
| Other Supporting Data | 法律法规、医学 AI 数据库、临床试验数据库、医患沟通数据 | 数百万 |

- 这些非结构化数据根据具体方法（持续预训练、IFT、SFT、RL）的数据选择流程，在不同训练阶段使用

**结构化数据 → 自然语言转换（Knowledge Transformation）**：

由于 LLM 无法直接使用结构化数据（如知识图谱的 SPO 三元组），QuarkMed 设计了一个三阶段转换流程：

1. **训练翻译模型**：构建 SPO 三元组与自然语言描述的平行语料，训练 seq2seq 模型学习映射
2. **三元组提取与回译（Back-Translation）**：从非结构化医学文本中提取三元组，再回译为自然语言，检验一致性和正确性
3. **质量过滤**：三层过滤机制：
   - 语义一致性检查（semantic consistency）
   - 语法与流畅度评估（grammatical and fluency evaluation）
   - 领域相关性过滤（domain relevance filtering）

**知识注入效果验证**：
- 使用单样本探测（single-shot methods）+ 引导性提示词（"leading text" prompt engineering），如："Based on the medical knowledge you have learned, what is the most likely diagnosis for..."
- 注入前后对比：
  | 指标 | 注入前 | 注入后 |
  |------|--------|--------|
  | 知识探测准确率 | **39.00%** | **60.57%** |
- 提升在 concept-based 任务上尤为显著，说明模型形成了更好的医学领域结构化理解

### 2.3 Medical Records（医疗记录）

**数据来源**（均为公开或去标识化数据，不披露确切数据集名称和体积）：
- **公开在线问诊对话**：覆盖症状叙述、医生问诊、初步鉴别诊断、分诊建议，捕捉日常用语和 pragmatic decision cues
- **公开 EHR 数据集**（去标识化）：
  - 门诊 EHR：主诉（chief complaint）、现病史（HPI）、评估/计划（A/P）、处方
  - 住院 EHR：入院记录、病程记录、手术记录、出院小结、实验室检查、影像报告

**隐私与质量控制管线**：
1. **保守 PHI 移除管线**（conservative PHI-removal pipeline）：去除受保护的健康信息
2. **文本归一化与分段**：将非结构化文本切分为连贯的临床文档
3. **质量 enforcing**：
   - 自动判别模型（automatic discriminator models）
   - 医生主导的抽样审核（physician-led spot audits）
4. **用途**：持续预训练（学习临床文档的结构和词汇） + SFT（增强复杂场景推理）

---

## 3. 训练管线详解（Training Pipeline）

QuarkMed 的训练分为四个阶段：**IFT → SFT → Stage 1 RL（可验证奖励）→ Stage 2 RL（一般对齐）**。各阶段之间并非独立，而是通过数据流和模型检查点紧密衔接。

![](QuarkMed_fig1_overview.png)

> **Figure 1**: 能力驱动框架。四大能力维度：Comprehension（理解）、Generation（生成）、Knowledge Application（知识应用）、Analysis & Reasoning（分析与推理）。Problem-Driven 闭环（Figure 2）在此基础上定向增强模型的薄弱环节。

### 3.1 IFT（Instruction Fine-Tuning）：对齐与基础能力构建

**目标**：将通用预训练模型从"文本补全引擎"转变为"能理解并执行医学指令的助手"。

**任务体系设计**：
- **双轨策略**：Ability-Driven（能力驱动，基础覆盖） + Problem-Driven（问题导向，弱点修复）
- **112 个核心 IFT 任务**，超过 **400,000 高质量样本**
- 四大能力维度（详见 Figure 1）：

| 维度 | 具体内容 | 示例任务 |
|------|---------|---------|
| **Comprehension** | 信息提取、文本分类、语义相似度 | 医学实体识别、关键信息抽取 |
| **Generation** | 逻辑连贯性、简洁性、流畅性 | Sentence Ordering（句子排序）、Hyponym/Hypernym 辨别 |
| **Knowledge Application** | 术语适配（临床 vs. 通俗）、人群禁忌知识 | 儿科/老年用药禁忌、药物相互作用 |
| **Analysis & Reasoning** | 多步推理 | 单位换算、数值比较、多轮诊断对话、临床笔记推理 |

**Problem-Driven 数据增强循环**（Figure 2）：

![](QuarkMed_fig2_data_pipeline.png)

> **Figure 2**: 问题驱动的数据增强闭环。① Weakness Identification：通过评估发现模型能力缺陷（如事实幻觉、输出不稳定）；② Task Design：针对性设计任务（如 Factual Consistency Judgment）；③ Sample Augmentation：生成对抗样本（如 synonymous instruction pairs、RAG noise samples）；④ Continuous Evaluation：持续评估并反馈到下一轮。

- **Counterfactual Robustness**：训练模型识别并拒绝基于错误前提的问题（"Factual Consistency Judgment" 任务）
- **Output Stability**：生成"同义指令-相同输出"对，确保语义等价查询得到一致响应
- **RAG Noise Resistance**：构建含相关+无关检索片段的噪声样本，训练模型准确识别、引用并综合最相关信息，忽略干扰

**任务构造三大原则**：
1. **Task Atomicity**：每个任务只针对单一、明确定义的目标，便于精确追踪能力变化和归因
2. **Instruction Generalization**：为每个 IFT 任务开发独特的 prompt templates，让模型学习底层的指令遵循行为，而非记忆表层模式
3. **Task Decomposition**：复杂多步任务分解为可处理的子任务。例如 RAG 场景中先训练"Relevance Extraction"子任务，再训练最终答案生成

**样本构造策略**：
| 样本类型 | 方法 | 质量控制 |
|---------|------|---------|
| 高质量基础样本 | 多模型采样 → 交叉验证投票 → 领域专家手工校验 | 金标准参考 |
| 复杂/对抗样本 | Self-Instruct：few-shot exmplars 引导大模型生成多样化 prompts 和 responses | 人工审核 |
| 安全对齐样本 | 专用"Red-Teaming"模型生成对抗 prompts | 符合安全协议 [14, 2] |

**训练策略**：

1. **课程学习（Curriculum Learning）**：
   - 从简单原子任务（如信息提取）向复杂复合任务（如多轮诊断对话）递进
   - 提升收敛速度和最终性能

2. **任务采样比例优化（Bayesian Optimization）**：
   - 核心问题：112 个任务的最优采样比分布
   - 方法：
     - 为每个核心能力建立**自动化评估套件**
     - 使用**高斯过程回归（Gaussian Process Regression, GPR）**建模"组间采样比」与「加权整体性能得分"的关系
     - 通过**填充准则（infill criterion）**高效探索高维搜索空间，寻找近最优比例分布
   - *注：论文未披露 GPR 的核函数、采集函数类型或优化迭代次数*

### 3.2 SFT（Supervised Fine-Tuning）：高质量医学对话数据构造

**查询选择策略**：
- **混合来源**：合成数据（系统性生成，覆盖受控场景） + 真实在线查询（捕捉用户查询的多样性和真实语境）
- 合成数据重点培养的五大能力（Table 2）：分散信息综合、矛盾信息处理、事实不一致识别、时效性、权威性引用

| 能力类别 | 具体能力 | 目的 |
|---------|---------|------|
| Summarization & Induction | Dispersed Information Synthesis | 将分散信息整合为连贯答案 |
| Disturbance Resistance | Contradictory Information | 从冲突来源中识别并使用正确信息 |
| Error Resistance | Factual Inconsistency | 识别并处理错误或不合逻辑的查询 |
| Fundamental Capabilities | Timeliness | 提供最及时的信息 |
| Fundamental Capabilities | Authoritativeness | 优先引用权威来源 |

**四阶段数据策展管线**（Figure 3）：

![](QuarkMed_fig3_training_framework.png)

> **Figure 3**: SFT 高质量合成数据的四阶段策展流程。① Grounded Knowledge Generation：利用 Quark Medical Search 检索参考材料（专业文献、医学 Q&A 论坛、信息性笔记）；② Candidate Answer Sampling (Best-of-N)：通过自有的医学质量模型和奖励模型进行多候选采样并筛选最优；③ Human-Expert Verification：医学专家团队按结构化格式审核每个候选答案；④ Rule-Based Data Annotation：实时服务自动验证格式和正确性。

**各阶段详解**：

**阶段 1：Medical Knowledge Grounded Generation**
- 采样匿名真实在线查询 + 合成查询
- 利用 **Quark Medical Search** 检索参考材料：
  - 专业医学文献
  - 医学 Q&A 论坛
  - 信息性笔记（informational notes）
  - 及时的全网内容补充
- 目标：创建反映真实世界复杂度、具有足够难度、覆盖医学知识/临床实践/医学应用全谱的 SFT 数据集
- 所有原始数据源（专有医学文本和患者记录）经过严格隐私保护

**阶段 2：Candidate Answer Sampling (Best-of-N)**
- 对每个 prompt，生成多个针对不同数据需求的候选答案
- 使用**自有的医学质量模型**和**内部奖励模型**
- 采用 **Best-of-N** 策略选择最优响应作为下一阶段候选

**阶段 3：Human-Expert Data Verification**
- 组建医学专家审核团队
- 审核员遵循**结构化格式**：参考材料 → 初始问题 → 模型最优答案 → 其他主流模型响应 → 最终答案可能需要的关键点总结
- 确保每个响应符合安全、准确、有用的严格标准

**阶段 4：Rule-Based Data Annotation**
- 实时服务自动验证格式和正确性
- 进一步提升标注数据的质量和一致性

### 3.3 Stage 1 RL：大规模医学可验证强化学习

**目标**：聚焦医学核心推理任务（疾病诊断、合理用药、检验/影像选择、医学考试），系统性提升模型的医学推理能力。

#### 3.3.1 Cold Start：SFT 初始化

**为什么需要 Cold Start**：加速 RL 收敛、节省计算资源，确保模型在进入 RL 前具备基础推理能力和格式遵循能力。

- **数据集**：从目标推理任务中精选 **700+ 高质量标注样本**
- **关键约束**：SFT **仅 2 个 epoch**
  - 目的：防止模型熵（entropy）降得过低，保留探索空间
  - 设计理念：让模型"学会推理"而非"记住答案"

#### 3.3.2 高质量 RL 训练数据管线

围绕三大核心原则设计：

| 原则 | 实现方式 | 具体细节 |
|------|---------|---------|
| **Diversity**（多样性） | 标签分层采样 | 从 EHR 和医学考试题等异质数据源出发，按标签进行分层采样，确保训练数据类别平衡 |
| **Difficulty**（难度） | 动态模型感知过滤器 | 持续筛选高准确率样本 → 分析推理复杂度 → 合成更复杂问题。该过滤器为每个模型架构（Llama vs. Qwen）和规模（8B vs. 32B）定制 |
| **Accuracy**（准确性） | 模型辅助 + 专家验证 | 高性能模型生成初始响应 → 标记差异 → 医学专家审核纠正 |

**数据质量和难度对 RL 的影响**（消融实验，Figure 5）：

![](QuarkMed_fig5_rl_data_impact.png)

> **Figure 5**: 数据优化对疾病诊断 RL 性能的影响。横轴为训练步数，纵轴为准确率。三条曲线：base（基础数据）、base+Accuracy（按准确率筛选）、base+Accuracy+Difficulty（按准确率和难度双重筛选）。双重筛选策略在 200 步时达到最高准确率（~0.56），两者均对 RL 性能有正向贡献。

#### 3.3.3 混合奖励模型设计（Hybrid Reward Model / Verifier）

这是 Stage 1 RL 的核心。设计哲学：**规则优先（rule-first），模型补充**。

**Why rule-first**：
- 医疗领域需要客观、稳定的奖励信号
- 防止模型基于偏好（preference-based）的 reward hacking
- 标准化输出格式（如诊断用 ICD 编码），支持自动化评分

**Why model-augmented**：
- 纯规则匹配过于脆弱（brittle）
- 需要处理同义词、层级关系（hierarchies）、不完整标签

**混合策略 = Rules + Model-based component**：
- 相比纯规则方法，疾病诊断性能提升 **3 个百分点**
- Verifier 经过迭代优化，减少潜在偏差

**各任务的 Verifier 组成**（Table 3）：

| 任务类型 | 数据量 | Verifier 组成 | 关键指标 |
|---------|--------|-------------|---------|
| 疾病诊断 | 16,000 | Rules (ICD Matching) + Model | Accuracy, Recall |
| 合理用药 | 10,000 | Rules (JSON Extraction) + Model | Drug Entity Accuracy |
| 检验/影像选择 | 16,000 | Rules (Keyword Matching) | Accuracy of Recommendations |
| 医学考试题 | 27,000 | Rules (Exact Answer Match) | Answer Accuracy |

**ICD Matching 规则引擎的 Model 组件细节**：
- 处理同义词映射（如"心梗"→"心肌梗死"）
- 处理层级关系（如分类到父类而非精确子类时的部分匹配）
- 处理不完整标签（如模型输出部分正确但遗漏并发症编码）

*注：论文未披露 Model component 的具体架构、训练数据或权重配比。*

#### 3.3.4 RL 实施与优化

**训练框架**：**VeRL**（HybridFlow / 灵活高效的 RLHF 框架）

**算法选择**：**GRPO（Group Relative Policy Optimization）**

**GRPO 的核心机制**（论文描述）：
- 在样本组内归一化 Advantage 函数（normalizes the advantage function within groups of samples）
- 天然支持多任务训练（naturally supports multi-task training）
- 提升训练稳定性（improves stability）

*注：论文未提供 GRPO 的完整数学公式。GRPO 的核心差异在于不单独训练 Critic 模型，而是通过组内样本的相对奖励来估计 Advantage，相比 PPO 减少了模型数量。典型的 GRPO 目标可描述为：*

$$L_{GRPO}(\pi_\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \left( \min\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} \hat{A}_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i \right) - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref}) \right) \right]$$

其中组内 Advantage 估计为：
$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}$$

*上述公式基于 DeepSeekMath/DeepSeek-R1 论文中公开的 GRPO 形式，QuarkMed 论文未直接给出。*

**效率与稳定性优化**：

| 优化手段 | 具体实现 | 效果 |
|---------|---------|------|
| **动态重采样（Dynamic Resampling）** | 每 epoch 开始时移除模型已掌握的简单样本 | 训练效率提升 **~20%** |
| **增加 Rollout 数量** | Policy model: **32**<br>Verifier: **8** | 提升策略模型探索稳定性和验证器评分准确性 |

*注：论文未披露 temperature、PPL clipping、batch size、step count、KL 系数等具体超参数。*

**Stage 1 性能结果**（Table 4）：

| 模型 | 初级职称 | 中级职称 | 副高职称 | 正高职称 | 诊断 Top-1 Acc. | 诊断 List Score |
|------|---------|---------|---------|---------|----------------|---------------|
| DeepSeek-R1 | 0.814 | 0.826 | 0.723 | 0.387 | 0.75 | 1.46 |
| Quark Stage1 | 0.822 | 0.772 | 0.683 | 0.524 | **0.86** | **3.32** |

### 3.4 Stage 2 RL：一般强化学习对齐

**目标**：通过 RL Align 模型行为与人类偏好和价值观，综合评估输出的 Honesty、Helpfulness 和 Content Compliance。

![](QuarkMed_fig6_rl_method.png)

> **Figure 6**: Stage 2 通用 RL 训练架构总览。Query Pool 中的查询经过 QuarkMed 生成回答后，分为两条并行路径：① General Tasks（无 Ground Truth）由 Helpful Reward Model 和 Honesty Reward Model 评分；② Reasoning Tasks（有 Ground Truth）由 Generative Verifier 和 Rule-based Verifier 评分。两类奖励经过 Advantage Estimation 后驱动 Policy Update，同时通过 Dynamic Difficulty Filtering 形成循环，持续优化 Query Pool 的难度分布。

#### 3.4.1 Honesty Reward（诚实性奖励）

**设计挑战**：医学事实错误的人工标注成本极高（costly manual annotation for factual errors）。

**迭代优化闭环**（Figure 4/6）：

![696](QuarkMed_fig4_benchmark_results.png)

> **Figure 4/6**: Honesty Reward 模型的迭代优化工作流。① 在手动校准的 SFT 样本上训练**生成式 CoT 奖励模型**（Generative CoT Reward Model），输出 CoT 推理链 + 评分；② 用该模型对多个候选响应评分，生成**高质量偏好对**（preference pairs）；③ 结合过滤后的偏好对与人工标注样本，使用 **Bradley-Terry (BT) 模型**训练 Reward Model (RM)；④ **闭环迭代**：新 RM 评估更多候选，错误/模糊评分的样本反馈给人工标注员重标注，持续提升两个模型。

**BT 模型**：经典的成对比较概率模型，定义偏好概率为：
$$P(\text{response}_i \succ \text{response}_j) = \sigma(r_\theta(q, o_i) - r_\theta(q, o_j))$$
其中 $\sigma$ 为 sigmoid 函数，$r_\theta$ 为奖励模型。BT 模型将成对偏好转化为可优化的奖励标量。

#### 3.4.2 Helpfulness Reward（有帮助性奖励）

- **评估标准**：与人类直觉紧密对齐，标注相对直接
- **数据来源**：从在线日志中采样数万个用户 prompts，使用多个模型生成多样化候选响应
- **标注方式**：标注员对多样化输出提供偏好标签
- **对抗 reward hacking 机制**：建立了**持续反馈循环**，在 RL 训练中重新标注新样本，迭代更新 RM，增强鲁棒性

#### 3.4.3 Consistency Reward（一致性奖励）

- **目标**：解决模型输出中"推理过程"与"最终摘要"不一致的问题
- **数据管线**：多阶段数据迭代管道：
  1. 大模型自动标注（auto-labeling）
  2. 模型校准（model-based label calibration）
  3. 迭代式主动学习（iterative active learning）
- **效果**：经过该管线精炼的样本，一致性奖励得分 **> 80**

#### 3.4.4 General Verifier for Content Compliance（内容合规通用验证器）

- **本质**：一个**指令遵循模型**（instruction-following model）
- **工作机制**：提供明确的评估原则（evaluation principles），基于原则对响应评分
- **优势**：
  - **灵活性**：修改指导原则即可快速适应新标准
  - **抗 hacking**：可快速添加新的验证规则应对传统 RM 难以覆盖的风格问题
- **应用场景**：严格样式和格式要求（如健康笔记、结构化报告）

#### 3.4.5 Stage 2 训练数据构造

| 数据类型 | 数量 | 来源/说明 |
|---------|------|----------|
| Reasoning-intensive data | ~60,000 prompts | 与早期阶段隔离，确保持续增益 |
| General-purpose data | ~20,000 prompts | 在线日志，通过多样性采样选择 |
| 数据比例 | **3:1** (推理 : 通用) | 使用 SFT 阶段模型进行多次 rollout，优先选择奖励得分多样性高的 prompts |

#### 3.4.6 算法对比：DPO vs GRPO

| 对比项 | DPO | GRPO |
|-------|-----|------|
| 偏好对构造 | 每个 prompt 的最高/最低奖励响应构成一对 | 每个 prompt 采样 **8 个候选响应** |
| KL 正则化 | 内建 | KL divergence penalty coefficient = **0.01** |
| 整体得分 | 3.49 | **3.84** |
| Honesty | 2.16 | **2.40** |
| Harmless | 2.72 | **2.88** |

**结论**：GRPO 在 Overall 和 Honesty 等关键维度上显著优于 DPO，被选为核心 RL 算法。

*注：论文未披露 DPO 和 GRPO 的具体学习率、batch size、训练步数等超参数。*

---

## 4. 医学 RAG 集成

QuarkMed 的 RAG 系统不是简单的"检索+生成"，而是作为**主要可靠性层（primary reliability layer）**深度集成。

**核心设计**：基于**权威排序语料库**的稠密检索（Dense Retrieval over authority-ranked corpora）。

**实际生产中的观测效果**：
1. **事实精度显著提升**：细微幻觉（如过时的治疗方案推荐）大幅减少，尤其对低频实体和最新更新的禁忌症
2. **校准改善（Improved Calibration）**：当检索到的段落相互矛盾时，模型更可能限定不确定性或提出替代假设
3. **考试鲁棒性增强**：多步临床案例题中，检索片段常能消除参数记忆混淆的近义干扰项

**投资优先级**：索引新鲜度（index freshness）、权威性评分（authority scoring）、冗余剪枝（redundancy pruning）、抗噪声提示包装（noise-resilient prompt packaging）。

---

## 5. 实验结果

### 5.1 评测方法论

| 配置项 | 设置 |
|-------|------|
| 推理 temperature | **0.6**（所有基准测试统一） |
| MMLU 验证 | 测试集无答案时使用验证集 |
| 大规模数据集采样 | RareBench 等超大数据集均匀采样，每测试集最多 **1,000 样本** |
| 标准化评分 | 选择题要求 JSON 格式输出；开放式题用 DeepSeek-V3-0324 做标准化后处理和答案评分 |

### 5.2 公开基准表现（Table 6）

| 数据集 | QuarkMed | 最佳对比 | QuarkMed 排名 |
|-------|---------|---------|--------------|
| MedQA (USMLE) | **86.02%** | DeepSeek-R1: 90.02% | 第 2 |
| MedMCQA | **75.50%** | DeepSeek-R1: 79.87% | 前列 |
| PubMedQA | **79.00%** | — | **SOTA** |
| CMExam | **88.60%** | Qwen3-235B: 90.10% | 第 2 |
| AfriMed-QA | **74.40%** | — | **SOTA** |
| MedXpertQA | **28.68%** | Gemini-2.5-pro: 46.42% | 中等 |
| DiagnosisArena | **61.90%** | DeepSeek-R1: 60.65% | **SOTA** |
| RareBench | **52.90%** | DeepSeek-R1: 57.56% | 中等 |
| MedBullets | **77.27%** | DeepSeek-R1: 82.06% | 前列 |
| MMLU(Med) | **88.37%** | Gemini-2.5-pro: 90.18% | 前列 |
| **平均** | **71.36%** | — | 同规模SOTA |

### 5.3 私有考试基准 CPQExam（Table 7）

**CPQExam**：基于中国卫生健康专业技术资格考试（医师执业资格与晋升考试）的私有数据集，重点考察病例分析和实际应用能力。

| 职称级别 | QuarkMed | 知识增强后 | 次优模型 | 差距 |
|---------|---------|-----------|---------|------|
| 初级职称 | 81.50% | **83.3%** | DeepSeek-R1: 81.42% | +1.9% |
| 中级职称 | 75.08% | **85.4%**\* | DeepSeek-R1: 82.58% | +2.8% |
| 副高职称 | 66.67% | **75.3%**\* | DeepSeek-R1: 72.33% | +3.0% |
| 正高职称 | **51.70%** | **67.7%**\* | DeepSeek-R1: **38.70%** | **+29.0%** |

\* 预测阶段启用知识增强（Knowledge Augmentation，即 RAG）

**按题型分解**（QuarkMed 知识增强后）：

| 题型 | QuarkMed 知识增强 | 次优 |
|------|------------------|------|
| 单选题 (Multiple-Choice) | **91.80%** | DeepSeek-R1: 88.14% |
| 多选题 (Multiple-Response) | **76.40%** | DeepSeek-R1: 32.89% |
| 共用题干 (Shared Stem) | **81.30%** | DeepSeek-R1: 85.31% |
| 病例分析 (Case Analysis) | **58.50%** | DeepSeek-R1: 48.75% |

**关键发现**：
- **正高职称考试**：QuarkMed 在不使用 RAG 时已大幅领先（51.70% vs 38.70%），使用 RAG 后差距扩大到 **29 个百分点**。这说明在超高难度临床综合推理任务上，领域专用管线（数据 + RL + RAG）具有**不可替代性**。
- **多选题（Multiple-Response）**：QuarkMed 知识增强后 76.40% 对比 DeepSeek-R1 仅 32.89%，差距超过 **43 个百分点**。多选题要求同时识别多个正确答案，对模型的全面性和精确性要求极高。

---

## 6. 局限性与讨论

### 6.1 RL 的局限与启示（论文 5.2 节的深度分析）

**RL 的优势**（论文坦诚总结的正面经验）：
1. 混合规则+模型验证器的奖励塑形（reward shaping）显著提升了格式保真度，并减少了纯模型偏好信号的 reward gaming
2. **组内归一化（GRPO）**在异构奖励尺度下稳定了多任务优化
3. **课程式难度重现（curriculum-style difficulty resurfacing）**防止了模型过早收敛到浅层启发式

**RL 的局限**：

| 局限 | 具体表现 | 根因 |
|------|---------|------|
| **可验证性偏差（Verifiability Bias）** | 性能增益集中于离散可验证端点（ICD 编码、结构化选项）。细微咨询、纵向管理规划、生活方式调整等梯度/上下文依赖任务优化不足 | 奖励信号只能精确建模离散正确性 |
| **奖励覆盖缺口（Reward Coverage Gaps）** | 时间推理、因果论证连贯性、不确定性表达未充分建模 | 验证器不完备 |
| **过拟合风险（Overfitting Risk）** | 与确定性格式/验证器模式紧密耦合，当模式变化时（新版编码、指南重构）行为易脆化 | 过度依赖规则化端点 |
| **稀疏/延迟反馈（Sparse/Delayed Feedback）** | 多轮对话质量、对抗探测安全性、累积患者中心化效用缺乏密集可靠的自动信号 | 长程交互的奖励难以自动构造 |
| **对齐权衡（Alignment Trade-offs）** | 最大化可验证推理有时降低风格共情或简洁性 | 多目标未显式调和 |

**未来 RL 改进方向**（论文直接提出）：
1. **半可验证复合奖励**：融合概率化事实性估计器 + 话语/因果连贯性评分
2. **主动不确定性引导**：奖励校准的放弃（deferral）或来源引用
3. **分层 RL**：将战略性临床框架与战术性实体选择分离
4. **模拟/合成患者状态转移**：用于时间信用分配的 RL
5. **持续验证器刷新管线**：与不断演化的指南对齐

### 6.2 RAG 的关键作用

论文明确指出：尽管对参数知识进行了大量投入，日常医疗辅助和考试答题中最强、最可靠的性能仍然依赖 RAG。RAG 不是辅助增强，而是**主要可靠性层**。在高风险事实、指南时效性或新兴主题查询上，仅依赖参数记忆会达到瓶颈。

### 6.3 其他局限
- 医学知识的动态演进性质：确保实时更新和解决不同来源间冲突仍是持续挑战
- 当前仅聚焦文本数据，缺乏医学影像解读能力
- 实时个性化（在隐私保护下适配个体健康上下文）尚未实现

---

## 7. 个人思考

### 7.1 数据管线的工程成熟度

QuarkMed 最令我印象深刻的是其数据层的**工程系统性**。这不是简单的"收集数据 → 清洗 → 训练"，而是一个多层、多阶段、有明确质量指标的复杂系统：

- **覆盖率检测**：用 Bloom's Taxonomy 将知识分为 factual/conceptual/procedural，通过内部测试集量化追踪，而非凭感觉判断"数据够不够"
- **权威性分级**：A-E 五级体系直接用于训练和 RAG 时的优先级排序，形成了"数据权威性 → 模型可靠性"的传导链路
- **OCR → Qwen2.5-VL**：用领域内的视觉大模型替代通用 OCR，30% 的质量提升是非常可观的工程收益
- **知识合成**：合并多厂商药物说明书 + 药理教材，这种"人工知识工程 + 自动合成"的 hybrid 模式，可能是垂直领域模型数据准备的标准范式

### 7.2 双阶段 RL 的工程智慧

Stage 1（可验证奖励）+ Stage 2（偏好对齐）的分工极其清晰：

- **Stage 1 解决"硬任务"**：ICD 编码、JSON 提取、答案精确匹配。这些任务的奖励函数可以精确编程，RL 的效果可预期。
- **Stage 2 解决"软质量"**：Honesty、Helpfulness、Consistency。这些维度必须依赖人工偏好和迭代式奖励模型优化。

这种分层避免了单阶段 RL 中 reward signal 的冲突——如果同时用"答案正确性"和"回答友好度"训练一个奖励模型，模型可能会为了友好而牺牲正确性，或者反之。分开训练、分阶段注入，是一种务实的工程选择。

### 7.3 最让人震惊的数据点

**正高职称考试的 67.7% vs 51.7% vs DeepSeek-R1 的 38.7%**：

中国医学正高职称考试是中国医生晋升体系中的最高级别考核，涉及复杂的临床综合判断、罕见病鉴别、多系统疾病交叉诊断。这个考试不仅考察知识记忆，更考察"临床思维"。

QuarkMed 的 32B 规模模型在不使用 RAG 的情况下（51.7%）就已经大幅超越 DeepSeek-R1（671B？）的 38.7%，使用 RAG 后更是达到 67.7%。这个差距说明，**领域专用训练管线在超高难度专业任务上的 ROI 极高**——不是单纯靠模型规模就能弥补的。

### 7.4 RAG 的临界点效应

CPQExam 的数据特别有意思：初级职称考试有无 RAG 差距很小（81.5% → 83.3%），但正高职称考试差距巨大（51.7% → 67.7%）。

这说明：**基础的、高频的医学知识模型可以通过预训练内化；但高阶的、低频的、更新的、跨领域的临床知识必须靠 RAG 检索**。RAG 带来的增益不是线性均匀的，而是在高难度任务上呈现"临界点效应"——一旦超过某个复杂度阈值，参数记忆的盲区就需要外部检索来填补。

### 7.5 论文的诚实性

这篇技术报告的一个优点是相当坦诚。它没有声称在所有基准上全面第一，而是明确列出劣势领域（如 MedXpertQA 28.68% vs Gemini-2.5-pro 46.42%）。同时，论文对 RL 局限性的分析（5.2 节）非常深入，没有回避可验证性偏差、奖励覆盖缺口、过拟合风险等问题，并提出了具体的未来改进方向。这种诚实性增强了报告的技术可信度。

### 7.6 与相关工作的关联

- 与 [[OPRD--On-Policy Representation Distillation]]：OPRD 聚焦 on-policy 蒸馏的效率，QuarkMed 更关注垂直领域系统工程的完整性。两者在"如何高效训练专用模型"上有互补性。
- 与 [[ScaleRL--The-Art-of-Scaling-Reinforcement-Learning-Compute-for-LLMs]]：ScaleRL 讨论 RL 的 scaling law，QuarkMed 验证了 RL 在医疗垂直领域的显著增益。但医疗领域的数据规模受限，其 scaling 曲线可能与通用领域不同。

---

## 8. 关键引用

```bibtex
@article{li2025quarkmed,
  title={QuarkMed Medical Foundation Model Technical Report},
  author={Li, Ao and Yan, Bin and Cai, Bingfeng and Li, Chenxi and Zhao, Cunzhong and Yao, Fugen and Liu, Gaoqiang and Jiang, Guanjun and Xu, Jian and Dong, Liang and Sun, Liansheng and Zhang, Rongshen and Gui, Xiaolei and Liu, Xin and Shang, Xin and Wu, Yao and Cao, Yu and Ma, Zhenxin and Jia, Zhuang},
  journal={arXiv preprint arXiv:2508.11894},
  year={2025}
}
```

---

## 附录：论文未披露的关键超参数

| 参数类别 | 缺失信息 | 说明 |
|---------|---------|------|
| Base Model | 具体架构和系列 | 推断为 Qwen-based，32B 规模 |
| Pre-training | LR、Batch size、Context length、Optimizer | 论文未涉及预训练阶段细节 |
| IFT Bayesian Opt. | GPR 核函数、采集函数、迭代次数 | 仅描述方法论 |
| SFT | 过滤阈值（长度/格式/去污染）、训练 epoch | 未披露 |
| Stage 1 RL | Temperature、PPL clipping、KL coef、Global batch size、总 step 数 | 仅披露 rollout 数和动态重采样 |
| Stage 2 RL | LR、Batch size、训练步数 | 仅披露 KL penalty = 0.01 |
| Hybrid Reward | Rules vs Model 的权重配比 | 仅描述策略，未披露权重 |
| ICD Matching | 部分匹配阈值、层级深度、ICD 版本 | 未披露 |

> **标注原则**：本笔记中所有未在论文原文中找到对应描述的公式或参数，均以 *斜体注* 形式标注，避免与原文事实混淆。



![773](Pasted%20image%2020260610141445.png)


