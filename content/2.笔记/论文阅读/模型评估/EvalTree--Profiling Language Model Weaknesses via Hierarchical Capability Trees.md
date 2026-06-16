---
created: 2026-06-09
paper: https://arxiv.org/abs/2503.08893
code: https://github.com/ZhiyuanZeng/EvalTree
authors: Zhiyuan Zeng, Yizhong Wang, Hannaneh Hajishirzi, Pang Wei Koh (University of Washington, Allen Institute for AI)
published: 2025-03-11
tags:
  - LLM评估
  - 弱点分析
  - 层次化能力树
  - 自动诊断
  - 数据收集
---

# EvalTree: Profiling Language Model Weaknesses via Hierarchical Capability Trees

## 一句话总结

<font color="#ff0000">提出 EvalTree 方法，通过自动构建层次化能力树（Capability Tree），从 benchmark 评测结果中精确提取 LLM 的细粒度弱点描述，在精度和覆盖率上显著优于基线方法，且其弱点指导的数据收集能带来 2.5 倍于通用指导的性能提升</font>。

![](evaltree_fig1_overview.png)

> **Figure 1**: EvalTree 方法总览。左侧为输入（benchmark 实例及模型表现），中间为自动构建的层次化能力树，右侧为提取出的弱点描述（weakness profile）。树的每个节点代表一种能力，叶子节点关联具体实例。

---

## 1. 研究背景与动机

### 1.1 问题定义

当前 LLM 评估将复杂的表现压缩为单一聚合分数（如 MATH 准确率 75%），这掩盖了模型在不同能力维度上的表现差异。例如 GPT-4o mini 在 MATH 上，"排列组合"准确率 75.1%，但"用三角原理解决几何关系"仅 49.1%。

**核心问题**：如何从模型在 benchmark 每道题的对错数据中，自动生成一组自然语言描述的弱点（weakness profile），帮助开发者精确了解模型的薄弱环节？

### 1.2 现有方法的不足

| 方法类型 | 代表工作 | 局限 |
|---------|---------|------|
| 单层分类 | QualEval, Moayeri et al. | 粒度固定，无法兼顾精确与全面 |
| 浅层分类体系 | HELM, Bai et al. | 依赖人工预定义，扩展性差 |
| 递归聚类 | Wang et al. 2023, Zhong et al. 2024 | 未形式化"弱点分析"问题，缺少定量评估 |
| 对比分析 | TextDiff | 成本高（~20× EvalTree），且只对比正确/错误集合 |

本文**首次形式化定义了 weakness profiling 问题**，并提出了定量评估框架。

---

## 2. 方法

### 2.1 核心思想

EvalTree 的核心洞察：**模型的能力具有天然的层次结构**（如"数学" → "几何" → "三角函数求解"），通过构建这棵树并结合模型表现数据，可以在合适的粒度上定位弱点——既不会太泛（"数学不好"），也不会太细（"这道题不会"）。

![](evaltree_fig2_pipeline.png)

> **Figure 2**: 能力树构建四阶段流程。Stage 1 对每个实例标注所需能力，Stage 2 生成能力描述的 embedding，Stage 3 通过递归聚类构建树结构，Stage 4 自底向上为每个节点生成描述。

### 2.2 能力树构建（四阶段流水线）

**Stage 1: 能力标注（Capability Annotation）**

用 LLM（默认 gpt-4o-mini）为每道 benchmark 题目标注所需能力：

```
输入: benchmark 实例 x
输出: 自然语言能力描述 c(x)（动名词短语）
约束: 不提及具体题目内容，只描述所需的通用能力
```

**Stage 2: 能力嵌入（Capability Embedding）**

使用 text-embedding-3-small 生成能力描述的向量表示：

$$\mathbf{e}_i = \text{Embed}(\text{prefix} + c(x_i))$$

其中 prefix = "The model has the following skill or capability: "

**Stage 3: 递归聚类构建树（Recursive Clustering）**

从根节点（所有实例）开始递归：
1. 对当前节点的能力 embedding 集合，尝试 K-Means 聚类（K = 2, 3, ..., 10）
2. 选择 Silhouette Score 最高的聚类方案
3. 若所有方案的 Silhouette Score ≤ 0，则当前实例直接作为叶子节点
4. 否则每个聚类成为一个子节点，递归继续

**Stage 4: 能力描述（Capability Description）**

自底向上为非叶节点生成描述：
- 叶子节点：直接使用 Stage 1 的标注
- 非叶节点：LLM 将所有子节点描述归纳为一个上层描述

### 2.3 从能力树中提取弱点

给定阈值 $\tau$（用户可调的严格度），通过统计检验提取弱点节点：

**统计检验**：对每个节点进行二项分布检验

$$H_0: p \geq \tau \quad \text{vs} \quad H_1: p < \tau$$

其中 $p$ 为该节点关联实例的正确率，样本量为关联实例数 $n$，成功次数为正确解答数。

**节点提取条件**：节点 $v$ 被提取为弱点当且仅当：
1. $v$ 通过统计检验（即表现显著低于 $\tau$）
2. $v$ 的所有（有足够样本的）直接子节点也通过检验

条件 (2) 的意义：如果某个子节点表现良好，说明弱点不在该子能力上，需要继续深入其他子树寻找更精确的弱点定位。

**停止条件**：
- 实例数低于阈值（默认 5）
- 节点已被提取

**关键性质**：提取的节点互不重叠（同一实例不会被多个弱点覆盖）。

### 2.4 评估框架（三项定量评估）

**Assessment 1: Low-Performance Identification（精度评估）**

$$\text{Score}_1 = \frac{1}{|W|} \sum_{w_i \in W} F(A(w_i))$$

其中 $F(S) = \frac{\sum_{x \in S} f_x}{|S|}$ 为集合 $S$ 上的平均性能，$A(w_i)$ 为与弱点 $w_i$ 关联的实例集合。数值越低说明识别的弱点越"真"。

**Assessment 2: Ground-Truth Weakness（召回评估）**

人工构造已知弱点的合成数据，用 F1 衡量方法能否恢复预设弱点。

**Assessment 3: Extrinsic Assessment（外部实用性评估）**

用识别出的弱点指导训练数据收集，看模型训练后的性能提升。

---

## 3. 实验结果

### 3.1 实验设置

| 要素 | 配置 |
|------|------|
| Benchmarks | MATH, WildChat10K, DS-1000, MMLU |
| 评估模型 | Llama 3.1 8B Instruct, DART-Math-Llama3-8B, DeepSeek-Coder-Base 6.7B |
| 基线方法 | TextDiff, QualEval, Random |
| LLM 标注 | gpt-4o-mini (temperature=0) |
| Embedding | text-embedding-3-small |
| 聚类 | sklearn KMeans, 最大 10 clusters |

### 3.2 精度评估（Low-Performance Identification）

![](evaltree_fig3_low_perf.png)

> **Figure 3**: 精度评估结果。横轴为最小弱点数 M'（或最小关联实例数 N'），纵轴为平均性能（越低越好）。EvalTree 在所有设置下一致性地取得最低曲线，表明其识别的弱点最为精确。

核心发现：
- EvalTree 在 MATH 和 WildChat10K 上均**一致性地低于所有基线**
- 优势在弱点数较少时（M'=1~5）尤为明显，说明 EvalTree 能精准定位最严重的弱点
- TextDiff 虽有"使用相同关联实例判定实现"的不公平优势，仍不如 EvalTree

### 3.3 Ground-Truth 评估

![](evaltree_fig4_ground_truth.png)

> **Figure 4**: Ground-truth 弱点恢复 F1。EvalTree 在大多数 profile size 下的 F1 超过其他方法能达到的最高值。

人工设计了 10 个不同粒度的 ground-truth 弱点，合成具有这些弱点的评测数据。结果显示 EvalTree 的 F1 在多数 profile size 下超越基线的最佳值。

### 3.4 弱点指导的数据收集（Extrinsic Assessment）

![](evaltree_fig5_extrinsic.png)

> **Figure 5**: 训练数据收集实验。EvalTree 指导的数据收集带来的性能提升约为通用能力指导的 2.5 倍，甚至略超直接从 profiling set 采样（后者有数据泄漏的不公平优势）。

| 策略 | MATH 提升 | DS-1000 提升 |
|------|-----------|-------------|
| Initial LM | 48.70 | 29.20 |
| Generic guidance | +1.0 | +2.5 |
| TextDiff-guided | +2.5 | +5.0 |
| QualEval-guided | +2.8 | +4.2 |
| **EvalTree-guided** | **+3.72** | **+7.70** |
| Profiling set sampling (unfair) | +3.5 | +7.2 |

EvalTree 指导的训练数据收集甚至略优于直接从 profiling set 采样（后者等于数据泄漏）。

### 3.5 成本对比

![](evaltree_fig6_cost.png)

> **Figure 6**: LLM 调用成本对比。识别 20 个弱点时，TextDiff 的成本约为 EvalTree 的 20 倍，QualEval 约为 8 倍。EvalTree 的成本与弱点数量无关（因为树只需构建一次）。

关键优势：EvalTree 的 LLM 调用成本是**常数级**（与弱点数无关），因为能力树只需构建一次，之后可通过调节 $\tau$ 免费获取不同粒度的弱点。

### 3.6 阈值 τ 的分析

![](evaltree_fig7_threshold.png)

> **Figure 7**: 阈值 τ 对提取结果的影响。随着 weakness τ 降低，弱点节点上的性能确实降低；随着 strength τ 升高，优势节点上的性能确实升高。τ 是有效的严格度控制参数。

---

## 4. 进一步应用

### 4.1 发现 Chatbot Arena 的评估缺陷

![](evaltree_fig8_chatbot_arena.png)

> **Figure 8**: Chatbot Arena 能力树分析。在涉及有毒请求的节点上，较弱模型（Zephyr-7B-β, Alpaca 13B）的 Elo 排名竟高于 GPT-4 和 Claude 2.1——因为人类评审偏好提供有毒内容的回复而非对齐后的拒绝回复。

这揭示了 Chatbot Arena 的一个系统性偏差：在安全相关能力维度上，"更好"的对齐模型反而得分更低。

### 4.2 交互式界面

提供了 Web UI（zhiyuan-zeng.github.io/EvalTree）供用户探索能力树，支持交互式地浏览不同粒度的模型能力分布。

---

## 5. 与 SkillVerse 的深度对比

[SkillVerse](SkillVerse--Assessing-and-Enhancing-LLMs-with-Tree-Evaluation.md)（Tian et al., 2026, UCLA + Google）与 EvalTree 是同期出现的两棵"能力树"。与 EvalTree 侧重"工程师视角的系统化弱点定位"不同，SkillVerse 更偏向"产品经理视角的开放域诊断与推理增强"。以下从九个维度进行系统对比（完整镜像对比见 [SkillVerse 笔记](SkillVerse--Assessing-and-Enhancing-LLMs-with-Tree-Evaluation.md) 的 §6）。

### 5.1 核心定位

| 维度 | **EvalTree** | **SkillVerse** |
|------|-------------|----------------|
| **核心目标** | 从标准 benchmark 对错结果中提取自然语言弱点，支持**弱点分析 + 数据收集** | 从开放域对话数据中提取能力画像，支持**诊断 + 增强 + 预测** |
| **问题形式化** | **首次严格形式化** weakness profiling，建立三项定量评估 | 未严格形式化，强调实用性和灵活性 |
| **适用场景** | 标准 benchmark 评测后分析（MATH, MMLU, DS-1000） | 真实世界部署（ChatbotArena 开放对话） |

### 5.2 输入数据与评价来源

| 维度 | **EvalTree** | **SkillVerse** |
|------|-------------|----------------|
| **输入** | Benchmark 实例 + binary 对/错标记 | 开放域 `prompt, response` 对 |
| **评价来源** | 标准答案判断 | LLM-as-Judge 自由格式 critique |
| **Failure 类型** | Hard failure（题做错了） | Hard + Soft failure（格式错、逻辑错等） |
| **LLM 依赖度** | 仅用于标注/描述 | 核心 pipeline（Judge + critique） |

> EvalTree 输入更简单可靠（无需 LLM Judge），但只能发现"硬错误"；SkillVerse 输入更灵活，能发现模型"答了但答得不对"的软错误，但引入了 Judge 的系统性偏见。

### 5.3 树的构建方式

| 维度 | **EvalTree** | **SkillVerse** |
|------|-------------|----------------|
| **语义表示** | 能力标注（动名词短语）→ 加 prefix embedding | 原子判断（3组件）→ 仅 Object 做 embedding |
| **聚类方向** | **Top-down Recursive K-Means**（递归分裂） | **Bottom-up Agglomerative**（凝聚合并） |
| **停止/切分策略** | Silhouette Score ≤ 0 停止分裂；K 从 2~10 自动选 | Elbow method 确定两层切分；支持任意水平切割 |
| **树类型** | n-ary capability tree（最多 10 子节点） | Dendrogram（支持多叉合并） |

> EvalTree 的递归分裂产生更规整、可控的树结构；SkillVerse 的凝聚合并更灵活，但粒度控制更依赖主观选择。

### 5.4 弱点识别机制

| 维度 | **EvalTree** | **SkillVerse** |
|------|-------------|----------------|
| **识别方法** | 二项分布统计检验（$p < \tau$）+ "所有子节点也低于 τ"的精确化约束 | 计算 cluster 成功率，人工/自动比较 |
| **节点重叠** | **无重叠**（实例只属于一个弱点节点） | 有重叠可能（需 anchoring 合并） |
| **粒度精确度** | 更细（可达具体知识点，如"用三角原理解几何"） | 较粗（cluster 级别，如"写 SQL 查询"） |
| **严格度控制** | 可调阈值 τ（统计显著性） | 剪枝阈值 T（成功率阈值） |

> EvalTree 的"所有子节点也低于 τ"是一个精妙的约束：如果某子节点表现好，说明弱点不在该子能力上，算法必须继续深入。这使得弱点定位更精确。SkillVerse 缺乏这种机制。

### 5.5 多模型对比

| 维度 | **EvalTree** | **SkillVerse** |
|------|-------------|----------------|
| **多模型支持** | 间接（分别建树后人工比较） | **原生支持**（Anchoring 机制：质心相似 + 重叠度双重条件合并 cluster） |
| **Inverse Scaling** | 未报道 | **主动发现**（比较同家族大小模型） |

> SkillVerse 的 anchoring 是独特优势：不同模型的错误分布不同导致树结构不同，需要显式合并才能横向对比。EvalTree 在这方面尚未提出解决方案。

### 5.6 下游应用

| 维度 | **EvalTree** | **SkillVerse** |
|------|-------------|----------------|
| **核心应用** | **训练时**：Weakness-guided 数据收集 → **2.5x 于通用策略** | **推理时**：Tree-Search C-ICL → **+25%**；Auto-Discovery → **55% 准确率** |
| **应用哲学** | 告诉开发者"哪里缺数据" | 告诉部署者"用什么样的示范更好" + "未来可能在哪里翻车" |
| **成本特性** | **常数级 O(1)**（树建一次，任意调 τ） | 与数据量/模型数正相关 |

> 两篇论文的应用天然互补：EvalTree 做训练前的问题定位 + 数据治理；SkillVerse 做推理时的 prompt 优化 + 部署前的风险预测。

### 5.7 评估方式

| 维度 | **EvalTree** | **SkillVerse** |
|------|-------------|----------------|
| **定量框架** | **三项严格评估**：精度、召回（Ground-Truth F1）、外部实用性 | 聚类准确性、锚定 PR、ICL 性能提升、预测成功率 |
| **评估严格度** | **高**（同行可复用的形式化基准） | 中等（实用性驱动） |
| **基线对比** | TextDiff, QualEval, Random | 相似度-only、从原则学习、无信息预测 |

> EvalTree 在方法论严谨性上更有优势：它定义了一个可定量比较的研究问题并建立了复用协议。SkillVerse 的评估更偏向"效果验证"。

### 5.8 互补性与选择建议

| 你的场景 | 推荐方法 |
|----------|----------|
| 标准 benchmark 事后分析 | **EvalTree** —— benchmark 有标准答案，输入天然对齐 |
| 开放域对话部署诊断 | **SkillVerse** —— 必须依赖 LLM Judge；需发现 soft failure |
| 推理时 prompt 优化/ICL 增强 | **SkillVerse** —— Tree-Search C-ICL 可直接用 |
| 训练数据精准补采/数据治理 | **EvalTree** —— Weakness-guided 收集已验证 2.5x 提升 |
| 模型发布前 QA 测试设计 | **SkillVerse** —— Auto-Discovery 主动识别潜在翻车点 |
| 跨模型/跨版本的持续监控 | **两者结合** —— EvalTree 做 benchmark 常态追踪 + SkillVerse 做开放域投产诊断 |

### 5.9 根本差异总结

| 设计选择 | **EvalTree** | **SkillVerse** |
|----------|-------------|----------------|
| **从什么建树？** | 从**实例所需能力**（这道题需要什么知识） | 从**判断结果**（模型做了什么/没做什么） |
| **树的叶子是什么？** | Benchmark 的一道具体题目 | 原子判断的 Object（一个任务描述） |
| **树的方向** | Top-down（能力到题目分裂） | Bottom-up（判断到技能归纳） |
| **弱点如何定义？** | 统计检验**自动提取**，人只需调 τ | 用户通过切割粒度**间接定义** |
| **与改进的连接** | **训练时**：指导数据配比 | **推理时**：选择示范/预测风险 |

> **一句话总结**：EvalTree 从"题目需要什么"出发自上而下分裂知识树，用于**系统化弱点定位和数据治理**；SkillVerse 从"模型输出了什么"出发自下而上归纳技能树，用于**实时诊断和推理增强**。二者不是竞争关系，而是**互补的诊断工具链**。

---

## 6. 局限性与未来方向

1. **树结构优化**：当前的递归 K-Means 可能不是最优分层方案，可探索其他层次聚类（如 SkillVerse 的 agglomerative 聚类）
2. **粒度可控性**：目前粒度由聚类自然形成，未来可让用户指定期望粒度
3. **模型相关结构**：当前树结构与被评估模型无关，可探索模型特定的能力层次
4. **多模态扩展**：将方法推广到视觉、音频等非文本模态
5. **能力树比较**：开发量化比较不同能力树的方法
6. **训练数据配比**：利用能力树指导预训练/微调数据的混合比例

---

## 7. 个人思考

1. **方法的优雅之处**：将"弱点分析"形式化为可定量评估的问题是一个重要贡献。此前的工作大多停留在定性展示，缺乏系统对比框架。三项定量评估（精度、召回、外部实用性）覆盖了不同角度。

2. **实用价值**：EvalTree 的常数成本特性使其特别适合持续评估场景——构建一次能力树后，可以反复调节阈值探索不同严格度的弱点，无需额外 LLM 调用。

3. **与模型改进的闭环**：弱点指导的数据收集实验证明了这不仅是"好看的分析"，而是可以直接指导模型改进的实用工具。2.5× 的提升幅度值得关注。

4. **潜在局限**：
   - 依赖 LLM 自身做能力标注——如果 LLM 对某类能力的理解有偏差，会传递到树结构中
   - Silhouette Score 选择聚类数可能导致某些维度被过度/不足分割
   - 二项检验假设实例间独立，但 benchmark 中相似题目可能违反此假设

5. **对评估实践的启示**：Chatbot Arena 的发现提醒我们，单一 Elo 分数可能掩盖严重的评估偏差。层次化分析应成为评估报告的标配。

---

## 8. 关键引用

```bibtex
@article{zeng2025evaltree,
  title={EvalTree: Profiling Language Model Weaknesses via Hierarchical Capability Trees},
  author={Zeng, Zhiyuan and Wang, Yizhong and Hajishirzi, Hannaneh and Koh, Pang Wei},
  journal={arXiv preprint arXiv:2503.08893},
  year={2025}
}
```
