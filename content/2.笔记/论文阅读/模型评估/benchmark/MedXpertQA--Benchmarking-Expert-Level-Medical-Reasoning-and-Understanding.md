---
created: 2026-06-10
published: 2025
paper: https://arxiv.org/abs/2501.18362
code: https://github.com/TsinghuaC3I/MedXpertQA
authors: Yuxin Zuo, Shang Qu, Yifei Li, Zhangren Chen, Xuekai Zhu, Ermo Hua, Kaiyan Zhang, Ning Ding, Bowen Zhou (Tsinghua University, Shanghai AI Lab)
tags:
  - Medical-AI
  - Benchmark
  - Multimodal
  - Reasoning
  - ICML2025
---

# MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding

## 一句话总结

MedXpertQA 是一个包含 4,460 道高难度问题的专家级医学推理基准，覆盖 17 个医学专科和 11 个人体系统，包含文本（Text）和多模态（MM）两个子集，并特别设计了面向 o1-like 推理模型的 Reasoning 子集。评测表明，当前最先进的模型也在该基准上表现有限，唯独 o1 在多模态子集上超越了人类预执业医师水平。

![](MedXpertQA_fig1_performance.png)

> **Figure 1**: MedXpertQA Text 与其他医疗基准的模型性能对比。即使是 o1-preview，在 MedQA-USMLE 上已达 96%，但在 MedXpertQA Text 上 o1 也仅 44.67%，说明 MedXpertQA 成功突破了现有基准的难度瓶颈。

---

## 1. 研究背景与动机

### 1.1 问题定义

如何构建一个既能覆盖真实临床场景、又具有足够挑战性以区分前沿模型的医疗 AI 评测基准？

### 1.2 现有方法的不足

**文本医疗基准的饱和**
- MedQA-USMLE 上 o1-preview 已达 **96%**，MMLU Medical Genetics 达 **99%**
- MedQA、PubMedQA、MedMCQA、MMLU 等缺乏对专科场景的覆盖（如家庭医学、成瘾医学）

**多模态医疗基准的失真**
- 传统基准（VQA-Rad、Path-VQA、Slake、PMC-VQA、OmniMedVQA）存在三大问题：
  1. **范围有限、难度不足**：仅评估基础视觉感知，忽略诊断决策和治疗规划所需的专家级推理
  2. **缺乏真实性与临床相关性**：基于图像标题自动生成简单 QA，与真实临床场景严重偏离
  3. **图像类型单一**：无法覆盖真实诊断中遇到的多样化视觉信息
- MMMU (H&M) 系列虽引入考题，但属于大学水平，非医学专科设计，且缺乏详细临床信息

---

## 2. 方法

### 2.1 核心思想

MedXpertQA 的构造遵循"**权威来源 → 多维过滤 → 数据增强 → 专家审核**"的严格流程，目标是构建一个：
- **高难度**：能挑战当前最先进模型
- **广覆盖**：涵盖专科评估和真实临床场景
- **低泄露**：通过重写和扩充降低数据污染风险
- **可解释**：区分为 Reasoning 和 Understanding 两个子集

![](MedXpertQA_fig2_overview.png)

> **Figure 2**: MedXpertQA 总览。左侧展示其多样化的数据来源（USMLE、COMLEX、17 个专科委员会考试、NEJM Image Challenge 等）、图像类型（放射影像、病理切片、生命体征、文档图表等）以及问题属性。右侧对比了 MedXpertQA MM 的复杂推理题与传统基准（VQA-Rad）的简单识图题，体现了真实临床决策所需的复杂多步推理能力。

### 2.2 基准构造流程（含公式）

#### Step 1: AI Expert Filtering（AI 专家过滤）

使用 8 个模型分基础和高级两类，每题进行 14 次采样投票：
- 若任一模型 4 次采样全对 → 题目太简单，剔除
- 若所有模型全错 → 保留（足够困难）

#### Step 2: Human Expert Filtering（人类专家过滤）

基于 **Brier Score** 的自适应阈值进行分层过滤：

$$
B = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y_i})^2
$$

| 符号 | 含义 |
|------|------|
| $N$ | 选项数量 |
| $y_i$ | 选项 $i$ 的标签（正确答案为 1，其余为 0） |
| $\hat{y_i}$ | 选择选项 $i$ 的人类考生比例 |

Brier Score 越低 → 人类整体预测越准 → 题目越简单。相比准确率，Brier Score 利用**所有选项的响应分布**更精确地度量难度。

随后，根据医学专家标注的先验难度（3 级），设定自适应阈值（第 3 百分位到第 25 百分位），实现**高难度题的过采样**。

#### Step 3: Similarity Filtering（相似性过滤）

使用 MedCPT-Query-Encoder 获取问题嵌入，计算两两余弦相似度，用 **IQR（四分位距）** 检测异常高相似度离群点对，剔除较易的那道。这有效防止了模板化问题导致的 benchmark overfitting。

#### Step 4: Question & Option Augmentation（数据增强）

为减轻数据泄露风险并提升难度：
- **Question 重写**：用 gpt-4o 和 claude-3.5-sonnet 对题目进行同义改写和结构调整，保持事实准确
- **Option 扩充与提纯**：
  - 剔除低质量干扰项（人类选择率极低的选项）
  - Text 子集扩充至 **10 个选项**，MM 子集保持 **5 个选项**（因图像选项难以扩充）

#### Step 5: Expert Review（专家审核）

持有医师执照的医学专家审核每道题，重点检查：
- 题目：信息缺失、事实错误、多余内容
- 选项：有效性、上下文合理性

### 2.3 Reasoning vs Understanding 子集标注

使用 gpt-4o 标注每道题的类型：
- **Reasoning**：需要复杂多步推理（如鉴别诊断、综合多源信息）
- **Understanding**：主要考察医学知识或图像感知，推理负担轻

对随机抽取的 10% 题目（Text 490 题 + MM 400 题）进行人工复核，错误率仅 **4.3%**。

### 2.4 关键设计选择

| 设计选择 | 理由 |
|---------|------|
| Brier Score 而非准确率 | 利用全选项分布，更精确反映题目难度 |
| 自适应阈值 | 匹配人类专家先验评级，保留不同难度层次的挑战题 |
| Text 10 选项 / MM 5 选项 | Text 可扩充；MM 选项常依赖图像，无法随意生成 |
| 引入专科委员会考题 | 模拟真实临床专科场景，填补现有基准空白 |
| 重写+扩充双重增强 | 既防数据泄露，又提升区分度 |

---

## 3. 数据分布与对比

### 3.1 医学覆盖度

- **17 个医学专科**：来自美国医学专业委员会（ABMS）的 17/25 个成员委员会考试
- **11 个人体系统**：全覆盖（骨骼、神经、心血管、呼吸、消化、内分泌、生殖、泌尿、皮肤、肌肉、淋巴）
- **3 大任务类别**：Diagnosis（50.54%）、Treatment（26.83%）、Basic Medicine（22.63%）

### 3.2 与现有基准对比

| 基准 | 规模 | 图像数 | 图像/题比 | 均长 | 图像类型 | 标注 | 临床场景 | 专科 |
|------|------|--------|-----------|------|----------|------|----------|------|
| VQA-Rad | 451 | 204 | 0.45 | 14.61 | 1 | 自动 | ✗ | ✗ |
| OmniMedVQA | 127,995 | 118,010 | 0.92 | 42.40 | 4 | 自动 | ✗ | ✗ |
| MMMU-Pro (H&M) | 346 | 431 | 1.25 | 107.08 | 7 | 专家 | ✗ | ✗ |
| **MedXpertQA MM** | **2,000** | **2,852** | **1.43** | **149.35** | **10** | **专家** | **✓** | **✓** |

| 基准 | 规模 | 均长 | 临床场景 | 专科 |
|------|------|------|----------|------|
| MedQA-USMLE | 1,273 | 215.46 | ✓ | ✗ |
| MMLU-Pro (Med.) | 586 | 166.63 | ✗ | ✗ |
| **MedXpertQA Text** | **2,450** | **257.37** | **✓** | **✓** |

MedXpertQA MM 拥有最高的平均问题长度（149.35 tokens）和最多的图像类型（10 种），是多模态医疗基准中首个融入真实临床场景和专科评估的。

![](MedXpertQA_fig3_diversity.png)

> **Figure 3**: MedXpertQA 的属性分布，展示了其在问题主题、人体系统和任务子类上的多样性与全面性。例如，骨骼系统占比最高（19.8%），而在诊断任务中，病因诊断（32.81%）和鉴别诊断（18.99%）占主导。

---

## 4. 实验结果

### 4.1 实验设置

- **评测方式**：Zero-shot CoT prompting，greedy decoding
- **模型范围**：18 个前沿模型（含 LMMs 和 LLMs），覆盖 proprietary 和开源模型
- **成本限制**：o1 和 o3-mini 仅在 10% 子集上评估（seed=42）

### 4.2 主实验结果

#### LMMs 性能（Table 4）

| 模型 | Text R | Text U | Text Avg | MM R | MM U | MM Avg | 总 Avg |
|------|--------|--------|----------|------|------|--------|--------|
| Expert (Pre-Licensed) | 41.74 | 45.44 | 42.60 | 45.76 | 44.97 | 45.53 | 43.92 |
| **o1** ‡ | **46.24** | 39.66 | **44.67** | **52.78** | **65.45** | **56.28** | **49.89** |
| GPT-4o | 30.63 | 29.54 | 30.37 | 40.73 | 48.19 | 42.80 | 35.96 |
| Gemini-2.0-Flash | 20.53 | 20.71 | 20.57 | 35.48 | 41.70 | 37.20 | 28.04 |
| Qwen2.5-VL-72B | 17.89 | 18.17 | 17.96 | 29.53 | 31.05 | 29.95 | 23.35 |

#### LLMs 性能（Table 5，仅 Text）

| 模型 | Reasoning | Understanding | Avg |
|------|-----------|---------------|-----|
| Expert (Pre-Licensed) | 41.74 | 45.44 | 42.60 |
| **o1** ‡ | **46.24** | 39.66 | **44.67** |
| DeepSeek-R1 | 37.88 | 37.35 | 37.76 |
| o3-mini ‡ | 37.63 | 36.21 | 37.30 |
| DeepSeek-V3 | 23.91 | 24.96 | 24.16 |

**关键发现**：
1. **o1 是目前唯一在 MM 上超越人类的模型**（56.28% vs 45.53%），但在 Text 上的优势并不显著
2. **vanilla 模型与人类的差距巨大**：GPT-4o（最佳 vanilla LMM）在 Text 上仅 30.37%，远低于人类 42.60%
3. **推理难度显著高于理解**：vanilla LLMs/LMMs 在 Reasoning 子集上一致弱于 Understanding，但 o1-like 模型大幅缩小该差距
4. **MedXpertQA Text 的难度远超现有基准**：Qwen2.5-72B 在 MMLU Medical 上达 89.62%，在 MedXpertQA Text 上仅 18.90%

### 4.3 消融实验与分析

#### 数据泄露风险验证（Table 6）

| 阶段 | 困惑度 ↑ | Rouge-L ↓ | Edit Distance ↓ |
|------|----------|-----------|-----------------|
| 增强前 | 1.03E+218 | 0.1893 | 0.2691 |
| 增强后 | 1.35E+247 | 0.1664 | 0.2416 |

增强后困惑度显著上升（模型对答案更"没把握"），说明数据重写和扩充有效降低了泄露风险。

#### 推理时扩展的影响（Inference-Time Scaling）

![](MedXpertQA_fig4_inference.png)

> **Figure 4**: 推理时扩展对医疗推理的影响。比较了三组模型对（DeepSeek-R1 vs V3、QwQ vs Qwen2.5-32B、QVQ vs Qwen2-VL），在 Text 和 MM 子集上的 Reasoning / Understanding 表现。**关键结论**：推理时扩展对 Reasoning 子集的提升尤为显著（如 DeepSeek-R1 比 V3 在 Text Reasoning 上提升约 14 个百分点），且无需额外医疗训练数据即可改善复杂医疗推理能力。

#### 人体系统细粒度分析

![](MedXpertQA_fig5_system.png)

> **Figure 5**: GPT-4o 在不同人体系统上的准确率（Accuracy）和正确题集中该系统题目的占比变化（Pct. Change）。**黄色柱表示该系统在正确题集中的占比减去其在全集中的占比**。GPT-4o 在 **Integumentary（皮肤）** 系统上表现最强（正确题集中占比明显上升），而在 **Cardiovascular（心血管）** 上表现最弱（正确题集中占比下降），揭示了模型在具体医学领域的差异化短板。

#### 错误类型分析

![](MedXpertQA_fig6_error.png)

> **Figure 6**: GPT-4o、Claude-3.5-Sonnet、Gemini-1.5-Pro 在 MedXpertQA 上的错误类型分布。所有模型在 Text 和 MM 上均以 **Reasoning Process Error（推理过程错误）** 为主，印证了基准在医疗推理难度上的设计有效性。MM 子集上 **Perceptual Error（感知错误）** 也占显著比例，说明医学图像理解仍是核心瓶颈。

---

## 5. 局限性与未来方向

1. **地域偏向**：题目基于美国医学体系（USMLE、专科委员会考试），其他国家和地区的医学实践差异未充分考虑
2. **静态评测**：当前为单轮多选问答，未涵盖真实临床中多轮交互、动态信息获取的决策流程
3. **临床就绪性 ≠ 基准高分**：作者在 Impact Statement 中强调，benchmark 性能不等于可临床应用，需额外保障（不确定性估计、人工监督、工作流集成）
4. **输入完整性敏感**：未来可评估模型在信息缺失场景下的鲁棒性
5. **模型幻觉与安全性**：医疗 AI 的部署仍需关注算法偏见、数据隐私和过度依赖自动系统的风险

---

## 6. 个人思考

**方法的优雅之处**：
- **Brier Score + 自适应阈值** 的难度过滤机制非常有巧思，相比简单的准确率过滤更能保留"有区分度"的难题
- **Reasoning / Understanding 的自动标注 + 人工复核** 以极低成本（4.3% 错误率）实现了细粒度的能力拆解，这种做法可推广到其他领域基准的构建
- **重写 + 扩充** 的双重增强兼顾了泄露防护和难度提升，而非简单删除泄露题

**启发与关联**：
- 论文明确展示了推理时扩展（test-time scaling / RL-based reasoning）在医疗领域的有效性，这为后续医疗专用推理模型（如 Huatuogpt-o1、MedReason 等）提供了有力的评测支撑和优化方向
- 对 Integumentary vs Cardiovascular 的细粒度分析很有临床价值，提示未来医疗 AI 的改进不应只看平均分，而应关注专科能力短板

**潜在改进空间**：
- 如果能把 Reasoning 子集进一步细分为"诊断推理"、"治疗决策推理"、"影像-文本联合推理"等维度，将提供更精准的能力画像
- 多模态子集的图像类型虽丰富，但若引入更复杂的交互场景（如"患者连续随访 + 影像对比"），会更贴近临床实际

---

## 7. 关键引用

```bibtex
@article{zuo2025medxpertqa,
  title={MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding},
  author={Zuo, Yuxin and Qu, Shang and Li, Yifei and Chen, Zhangren and Zhu, Xuekai and Hua, Ermo and Zhang, Kaiyan and Ding, Ning and Zhou, Bowen},
  journal={arXiv preprint arXiv:2501.18362},
  year={2025}
}
```
