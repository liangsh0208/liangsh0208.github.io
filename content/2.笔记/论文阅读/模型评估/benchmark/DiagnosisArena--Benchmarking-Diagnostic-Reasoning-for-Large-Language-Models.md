---
created: 2026-06-10
published: 2025-05-20
paper: https://arxiv.org/abs/2505.14107
code: https://github.com/SPIRAL-MED/DiagnosisArena
authors: Yakun Zhu, Zhongzhen Huang, Linjie Mu, Yutong Huang, Wei Nie, Shaoting Zhang, Pengfei Liu, Xiaofan Zhang (SJTU, SII, SPIRAL Lab, GAIR, Shanghai Chest Hospital)
tags:
  - LLM-Benchmark
  - Medical-AI
  - Diagnostic-Reasoning
  - Clinical-Case
  - Model-Evaluation
---
- **基于casereport 筛选构建难度病历，构造评测集。 1113.**

# DiagnosisArena--Benchmarking Diagnostic Reasoning for Large Language Models

## 一句话总结
DiagnosisArena 是一个面向大语言模型的临床诊断推理基准，从 10 种顶级医学期刊收集了 1,113 例高难度真实病例（跨越 28 个专科），实验揭示即使是 o3-mini、o1、DeepSeek-R1 等最先进推理模型，其诊断准确率也分别仅为 45.82%、31.09% 和 17.79%，暴露了当前 LLM 在真实临床诊断推理中的严重瓶颈。

![](DiagnosisArena_fig1_performance.png)

> **Figure 1**: 当前 SOTA 模型在 DiagnosisArena 与其他医学基准上的表现对比。现有基于考试题的基准（如 MedQA、MMLU）已趋于饱和（>90%），而 DiagnosisArena 能显著拉开模型差距，真正检验临床诊断推理能力。

---

## 1. 研究背景与动机

### 1.1 问题定义

医学诊断推理要求临床医生通过分析患者症状、病史、体格检查和诊断测试结果，从多种可能疾病中识别出特定的疾病或病症。这与传统医学考试题的知识回忆型任务有本质区别——诊断推理是信息密集型的**归纳推理**过程，需要整合多维患者数据进行复杂推断。

### 1.2 现有方法的不足

| 不足维度 | 具体表现 |
|---------|---------|
| **基准饱和** | 现有医学基准（MMLU、MedQA 等）以医学执业考试题为主，SOTA 模型准确率已超 90%，失去区分度 |
| **形式局限** | 多数基准为多项选择题（MCQ），人为限制了鉴别诊断范围，简化了推理过程 |
| **信息简化** | 现有病例基准往往使用过度简化的病例信息，与真实临床复杂度相距甚远 |
| **场景脱离** | 缺乏真实临床场景中的完整患者记录（体格检查、实验室检查、影像学等） |

作者指出：**MCQ 格式本质上为模型提供了"捷径"**，模型可以凭部分知识或表面线索排除错误选项，而不需要完成真实诊断所需的完整演绎推理。

---

## 2. DiagnosisArena 基准构建

### 2.1 核心思想

DiagnosisArena 的设计目标是在真实临床复杂度下，**严格评估 LLM 的诊断推理上限**。为此采用四个关键策略：
1. **真实来源**：从顶级医学期刊的真实病例报告取材，确保临床真实性
2. **信息完整**：保留病例信息、体格检查、诊断测试等完整患者记录
3. **难度筛选**：多轮迭代过滤排除简单病例，确保挑战性
4. **质量验证**：AI 与专家协同验证，保证诊断信息的充分性和准确性

### 2.2 构建流程详解

![](DiagnosisArena_fig2_pipeline.png)

> **Figure 2**: DiagnosisArena 构建流程总览。(a) 四阶段构建管线：数据收集 → 数据结构化 → 迭代过滤 → 专家-AI 协同验证；(b) 数据源自 10 种顶级医学期刊；(c) 覆盖 28 个医学专科；(d) 病例具有明确定义的分段结构和信息密集的临床内容。

构建流程包含四个阶段：

**1. 数据收集**
- <font color="#ff0000">从 10 种顶级医学期刊（Lancet、NEJM、JAMA、Annals of Internal Medicine、Cell 等）收集 4,175 例病例报告</font>
- 选择病例报告的原因是：通常呈现具有研究价值的疑难病例，同时提供全面的诊断信息

**2. 数据结构化**
- 手动过滤治疗方案和预后信息（按章节标题规则过滤）
- 使用 Claude-3.5-sonnet 提取诊断相关内容
- 每份病例重组为四个段落：**病例信息**、**体格检查**、**诊断测试**、**最终诊断**
- 前三段构成临床呈现信息，最后一段作为 ground truth

**3. 迭代过滤**
- **第一轮（简单性过滤）**：使用 Baichuan-M1、DeepSeek-V3、GPT-4o 各采样 2 次，任一模型答对即排除
- **第二轮（合理性审查）**：AI 专家（多个前沿模型）评估每个病例是否包含充分且明确的诊断线索，全票通过才保留
- 过滤后保留 1,783 例

**4. 专家-AI 协同验证**
- DeepSeek-R1 进行 8 次采样投票，无法达成相关诊断共识的病例排除
- 执业医师人工审查，信息缺失或诊断模糊的病例排除
- 最终保留 **1,113 例**，覆盖 **28 个医学专科**

| 期刊 | 最终入选数 |
|------|-----------|
| JAMA | 488 |
| Cell | 254 |
| Annals of Internal Medicine | 192 |
| NEJM | 87 |
| Lancet | 26 |
| 其他 | 70 |

### 2.3 与其他基准的对比

| 基准 | 样本量 | 平均长度 | 题型 | 体格检查 | 诊断测试 | 临床场景 | 数据来源 |
|------|--------|---------|------|---------|---------|---------|---------|
| MedQA-USMLE | 1,273 | 215.46 | MCQ | ✔/✘ | ✔ | 执业考试 |
| CMB-Clin | 74 | 792.55 | 开放式 | ✔/✔ | ✔ | 医院 |
| RareArena | 72,661 | 310.36 | 开放式 | ✔/✔ | ✔ | PubMed |
| MedXpertQA | 2,450 | 257.43 | MCQ | ✔/✔ | ✔ | 考题 |
| **DiagnosisArena** | **1,113** | **545.02** | **开放式&MCQ** | **✔/✔** | **✔** | **顶级期刊** |

DiagnosisArena 的关键差异化：
- **最长的平均文本长度**（545 token），信息密度更高
- 同时支持开放式诊断和 MCQ 两种评估形式，可直接比较形式差异
- 文章来源的临床真实性和疑难病例比例更高

---

## 3. 实验设置

### 3.1 评估方法

**开放式评估（Open-Ended）**
- 使用 GPT-4o 作为评判模型，将模型输出与 ground truth 的关系分为三类：
  - **identical**：诊断结果与 ground truth 一致（计为正确）
  - **relevant**：相关但不完全一致
  - **irrelevant**：不相关
- 每个病例生成 5 个候选诊断结果，按置信度降序排列
- 计算 **Top-k 准确率**：正确答案出现在前 k 个预测中的比例

**MCQ 评估（DiagnosisArena-MCQ）**
- 从 o1 和 DeepSeek-R1 的部分正确结果中提取干扰项
- 构建四选一多项选择题
- 规则化提取答案并计算准确率

### 3.2 评估模型

涵盖专有模型、开源模型和医疗领域模型：

| 类别 | 代表模型 |
|------|---------|
| 推理型模型 | o3-mini, o1, DeepSeek-R1, QwQ-32B, Gemini 2.5 Pro, Qwen3-Thinking |
| 通用模型 | GPT-4o, Claude-3.5-Sonnet, Qwen2.5-Max, DeepSeek-V3 |
| 医疗专项模型 | Baichuan-M1 |

---

## 4. 实验结果

### 4.1 主实验结果

![](DiagnosisArena_fig3_results.png)

> **Figure 3**: (a) 各模型在 DiagnosisArena 上的 Top-k 表现。o3-mini 以 45.82% 领先，但即使如此 DiagnosisArena 对所有现有模型仍是巨大挑战；(b) MCQ 版本的 DiagnosisArena 上模型表现大幅提升，o1 达到 61.90%，说明 MCQ 显著降低了任务难度。

**开放式诊断 Top-1 准确率**：

| 模型 | Top-1 准确率 |
|------|------------|
| o3-mini | **45.82%** |
| o1 | 31.09% |
| QwQ-32B | 25.69% |
| DeepSeek-R1 | 17.79% |
| Gemini 2.5 Pro | ~17% |
| Qwen3-Thinking | ~15% |
| GPT-4o | ~15% |
| Claude-3.5-Sonnet | < 20% |
| Qwen2.5-Max | < 20% |
| DeepSeek-V3 | ~4% |
| Baichuan-M1 | < 10% |

**核心发现**：

1. **即便是顶级推理模型也表现不佳**：o3-mini 仅 45.82%，o1 仅 31.09%，DeepSeek-R1 仅 17.79%。这揭示了当前 LLM 在专业级临床诊断中的**严重泛化瓶颈**。

2. **推理能力带来显著优势**：
   - Claude-3.5-Sonnet 和 Qwen2.5-Max 等强模型准确率低于 20%
   - 推理增强模型 QwQ-32B（32B 参数）达到 25.69%
   - DeepSeek-R1 相对其基座 DeepSeek-V3 提升 **13.66%**，验证了推理训练的关键作用

3. **MCQ 严重低估真实难度**：
   - o1 在 MCQ 上跃升至 61.90%（提升约 30%）
   - Baichuan-M1-14B 从 <10% 跃升至 58.31%
   - MCQ 的预定义选项缩小了问题空间，模型可通过部分知识排除错误选项，**这不等于完成了真实的诊断推理**

### 4.2 数据泄漏检测

- 收集了 2022–2024 年 690 篇期刊文章进行预实验
- 主流模型（o3-mini, DeepSeek-R1, GPT-4o, Claude-3.5-Sonnet）在三年间的准确率保持稳定，仅有微小波动
- 在 DiagnosisArena 全量数据上进行年代分析，未发现显著的时间趋势异常
- **结论**：数据泄漏极其罕见，评估结果稳健

### 4.3 案例分析

![](DiagnosisArena_fig5_casestudy.png)

> **Figure 5**: DiagnosisArena 中一个 AMVT（Accessory Mitral Valve Tissue）病例的案例研究。除 o3-mini 外的模型均给出错误诊断，DeepSeek-R1 过度依赖常见病推理路径，忽略了支持罕见诊断的间接证据。

**案例：Accessory Mitral Valve Tissue (AMVT)**

**病例核心线索**：
- 左心室流出道（LVOT）发现高活动性异常结构
- 该结构在心动周期中表现出显著运动模式
- 影像显示该结构与二尖瓣/主动脉瓣解剖关系不明确
- 偶然发现二尖瓣环分离

**模型失败分析（以 DeepSeek-R1 为例）**：
1. **诊断范围偏差**：R1 的诊断推理集中在肿瘤、栓子、纤维层状肿瘤等 LVOT 相关病变，完全没有考虑二尖瓣相关结构问题
2. **影像特征误判**：R1 识别出活动性结构并推测为乳头状纤维层状肿瘤或 Lambl 赘生物，但未考虑 AMVT 可能表现为类似的丝状或活动性结构
3. **症状关联不足**：将心悸、头晕、气短等症状与心脏肿瘤、血栓等疾病关联，但未充分认识到 AMVT 可能导致轻中度 LVOT 梗阻或间歇性心律失常

**根本原因**：当前 SOTA 推理模型**尚未真正适应医学场景的复杂推理需求**。模型倾向于优先选择常见疾病的推理路径，而非基于可用线索进行推断——本质上仍依赖知识复现而非深度推理。

---

## 5. 局限性与未来方向

1. **评估范围**：基准聚焦诊断层面，治疗方案制定和预后预测尚未涵盖（因需要长期临床试验验证）
2. **评价指标**：依赖 GPT-4o 作为评判模型，虽然设定明确的三分类标准，但复杂的医学诊断边界仍可能存在主观性
3. **数据规模**：相比部分大规模基准（如 RareArena 的 7 万+），1,113 例的规模更适合高难度评估而非大规模训练
4. **专科分布**：28 个专科分布不均（如 JAMA 来源占绝对多数），可能影响跨专科评估的公平性

**未来方向**：
- 扩展基准覆盖治疗决策和预后评估
- 探索更细粒度的诊断过程评估（如鉴别诊断路径、检查选择合理性）
- 结合多模态数据（影像、病理切片）构建更完整的临床评估体系
- 将 DiagnosisArena 作为医学推理模型训练的数据源

---

## 6. 个人思考

### 方法的亮点
1. **真实性优先**：从顶级期刊病例报告取材的策略非常明智——这些病例天然具有高信息密度和诊断难度，避免了人为构造病例时的"过度简化"倾向
2. **过滤管线严谨**：三层过滤（简单性过滤 → AI 合理性审查 → 专家-AI 协同验证）确保保留的每个病例都真正具有推理挑战性
3. **MCQ 对比实验设计精妙**：通过同一病例的两种形式对比，以量化数据证明了 MCQ 作为医学 LLM 评估形式的根本缺陷

### 对当前 LLM 医学能力的启示
- **"考试高分 ≠ 临床胜任"**：模型在执业考试上的优异表现，不等于其已具备临床级诊断推理能力。真实诊断要求的是"从噪声中提炼信号"的归纳推理，而非"从选项中选择最佳"的知识检索
- **推理模型的价值与局限**：DeepSeek-R1 相比 V3 的 13.66% 提升验证了推理训练的方向是正确的，但 17.79% 的绝对水平说明医学推理仍有巨大提升空间

### 关联阅读
- [[MediEval--Patient-Contextual-Medical-Reasoning-Benchmark]] — 同样关注真实临床场景的医学推理评估
- [[Huatuogpt-o1]] — 医学场景中的 o1-like 推理

---

## 7. 关键引用

```bibtex
@article{zhu2025diagnosisarena,
  title={DiagnosisArena: Benchmarking Diagnostic Reasoning for Large Language Models},
  author={Zhu, Yakun and Huang, Zhongzhen and Mu, Linjie and Huang, Yutong and Nie, Wei and Zhang, Shaoting and Liu, Pengfei and Zhang, Xiaofan},
  journal={arXiv preprint arXiv:2505.14107},
  year={2025}
}
```
