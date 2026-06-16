---
created: 2026-06-09
paper: https://www.alphaxiv.org/abs/2512.20822
code: https://github.com/ZhanQu945/MediEval
authors: Zhan Qu, Michael Färber (TU Dresden and ScaDS.AI, Germany)
published: 2025-12-23
tags:
  - Medical-AI
  - LLM-Evaluation
  - EHR
  - DPO
  - Clinical-Reasoning
  - Benchmark
---

# MediEval: A Unified Medical Benchmark for Patient-Contextual and Knowledge-Grounded Reasoning in LLMs

## 一句话总结
MediEval 是一个将 MIMIC-IV 真实电子病历与 UMLS/SNOMED CT/RxNorm 生物医学本体知识库相结合的医学推理基准，通过四象限框架联合评估 LLM 的医学知识正确性与患者语境一致性，并提出 CoRFu（非对称惩罚 DPO）方法，在 Llama-3.1-8B 上提升 +16.4 macro-F1 并彻底消除真相反转错误。

![](MediEval_figure1_overview.png)

> **Figure 1**: MediEval 整体框架概览。对每个入院记录，系统同时利用结构化 EHR 表（诊断、手术、用药）和非结构化出院小结。首先提取临床相关章节形成患者语境；并行地将结构化医学代码通过 UMLS 归一化为 CUI，构建语义本体图，识别临床关联（含多跳推理）。基于这些关系构造候选陈述，并在四象限框架下评估：Q1(真-支持)、Q2(真-不支持)、Q3(假-支持)、Q4(假-不支持)。蓝色标记为从真实患者示例中提取的实体或陈述。

---

## 1. 研究背景与动机

### 1.1 问题定义
医学场景要求 LLM 同时满足两个条件：(i) 医学知识忠实度（factual correctness）和 (ii) 个体患者记录的应用一致性（contextual grounding）。现有评估方法存在关键缺口：
- **知识型评估**（如 MedMCQA）只测试孤立医学知识的正确性，不连接真实患者上下文；
- **EHR 型评估**（如表格数据理解）只测试患者级推理能力，不验证医学正确性。

### 1.2 现有方法的不足
临床场景中的错误不是简单的性能下降，而是可能直接转化为患者伤害的风险（patient harm）。当前模型"被训练去孤立地回忆事实，而非将其与多样化的患者信息整合"。典型失败案例：模型知道二甲双胍在严重肾损伤时禁用，但面对嘈杂且异构的患者记录时却无法正确应用这一知识。

---

## 2. 方法

### 2.1 核心思想：四象限框架

MediEval 将每个医学陈述映射到由两个维度交叉定义的四个象限：

| 象限 | 名称 | 定义 | 构造方式 |
|:---|:---|:---|:---|
| **Q1** | True–Supported | 患者语境中直接证实的正确事实 | 从真实 EHR 中的 (h, r, t) 三元组直接生成 |
| **Q2** | True–Unsupported | 医学正确但患者记录不支持 | 将本体图中的实体替换为同类实体（如同属一个父概念），产生正确但无患者支持的陈述 |
| **Q3** | False–Supported | 假关系但实体均在患者记录中 | 重组不同真实事实的实体，产生虚假关系，考验模型抵抗表面合理的错误关联 |
| **Q4** | False–Unsupported | 既假又不支持 | 引入语义 distant 的干扰实体，产生明显错误的陈述 |

### 2.2 数据构造流程（4 步）

**Step 1: 语境提取**
从 MIMIC-IV 出院小结中提取诊断、手术、治疗、病史和住院经过等临床相关章节，形成患者语境 $C_a$。

**Step 2: 语义归一化**
将 ICD-9/10 诊断代码和 NDC 药物代码通过 UMLS 映射为统一的概念唯一标识符（CUI）：

$$f: z \mapsto \text{CUI}(z), \quad U_a = \{f(z) \mid z \in S_a\}$$

其中 $z$ 为原始代码，$U_a$ 为入院 $a$ 的归一化 CUI 集合。

**Step 3: 本体图构建与关系提取**
从归一化概念构建语义图 $G=(V,E)$，节点 $V$ 为 CUI，边 $E$ 来自 UMLS Metathesaurus（如 *treated_by*, *has_associated_procedure*, *is_a*）。允许多跳推理（最多 3 跳）：

$$R(h,t) = \{r \mid (h,r,t) \in G, \, d(h,t) \leq 3\}$$

例如：*Lisinopril* $(C0023861) \xrightarrow{\text{is_a}}$ *ACE Inhibitors* $(C0003028) \xrightarrow{\text{treats}}$ *Hypertension* $(C0020538)$

**Step 4: 四象限陈述构造**
对每个事实 $(h,r,t)$，生成覆盖四个象限的样本集：

$$\mathcal{Y}_a(h,r,t) = \{y^{Q1}, y^{Q2}, y^{Q3}, y^{Q4}\}$$

### 2.3 评估指标

**基础指标**：Accuracy、Macro-P、Macro-R、Macro-F1、per-quadrant F1。

**安全性关键指标**（论文核心创新）：

**幻觉支持率（HSR）** —— 真但不支持陈述（Q2）被误分为真且支持（Q1）的比例：

$$\text{HSR} = \frac{\sum_i \mathbb{I}(q_i = Q2 \land \hat{q}_i = Q1)}{\sum_i \mathbb{I}(q_i = Q2)}$$

**真相反转率（TIR）** —— 假但表面支持陈述（Q3）被误分为真且支持（Q1）的比例：

$$\text{TIR} = \frac{\sum_i \mathbb{I}(q_i = Q3 \land \hat{q}_i = Q1)}{\sum_i \mathbb{I}(q_i = Q3)}$$

> HSR 高 = 模型"幻觉证据"，将正确医学知识当作患者记录中的事实；TIR 高 = 模型"真相反转"，将错误陈述提升为看似有效的证据。两者均为临床安全关键风险。

### 2.4 CoRFu：反事实风险感知微调

基于 DPO（Direct Preference Optimization），引入非对称惩罚项。

**DPO 偏好边界**（标准项）：

$$S(c; y_w, y_l) = \beta \log\left[\frac{\pi_\theta(y_w|c) \, \pi_{\text{ref}}(y_l|c)}{\pi_{\text{ref}}(y_w|c) \, \pi_\theta(y_l|c)}\right]$$

**CoRFu 损失函数**：

$$\mathcal{L}_{\text{CoRFu}} = -\mathbb{E}\big[\log\sigma(S)\big] + \lambda \cdot \mathbb{E}\big[\mathbb{I}(S < 0) \cdot S^2\big]$$

| 符号 | 含义 |
|:---|:---|
| $\pi_\theta$ | 当前策略模型 |
| $\pi_{\text{ref}}$ | 参考模型（SFT 后模型） |
| $y_w$ | 偏好响应（Q1 真-支持） |
| $y_l$ | 非偏好响应（Q2/Q3/Q4） |
| $\beta$ | DPO 温度系数（固定 0.1） |
| $\lambda$ | 非对称惩罚系数（最终配置 0.5） |
| $\mathbb{I}(S < 0) \cdot S^2$ | 仅当模型将非偏好响应排在偏好响应之前时激活（$S < 0$），且惩罚力度随置信度二次增长 |

> **设计直觉**：标准 DPO 鼓励模型偏好 $y_w$ 超过 $y_l$，但对模型"高置信度地将安全关键错误排在正确响应之前"的情况惩罚不足。CoRFu 通过 $S^2$ 项让"越自信的错误受到越大的惩罚"，与临床安全要求对齐。

**三种训练机制**：
- **Pairwise**：Q1 vs. Q2 / Q1 vs. Q3 / Q1 vs. Q4 分别训练；
- **Mixed**：Q1 同时对抗 Q2+Q3+Q4 的混合；
- **Curriculum**：按 Q1→Q2→Q1→Q3→Q1→Q4 顺序分阶段训练。

### 2.5 四象限示例

| Q1: True–Supported | Q2: True–Unsupported |
|:---|:---|
| ![](MediEval_figure2_example_q1.png) | ![](MediEval_figure2_example_q2.png) |
| 陈述：*GERD may be treated by omeprazole* | 陈述：*GERD may be treated by aluminum hydroxide* |
| 真实且由患者记录支持（确实在用 omeprazole） | 医学正确但患者未用 aluninum hydroxide 治疗 GERD |

| Q3: False–Supported | Q4: False–Unsupported |
|:---|:---|
| ![](MediEval_figure2_example_q3.png) | ![](MediEval_figure2_example_q4.png) |
| 陈述：*GERD may be treated by atenolol* | 陈述：*GERD may be treated by insulin* |
| GERD 和 atenolol 都出现在记录中，但 atenolol 用于高血压非 GERD | 既假又不支持：胰岛素不用于 GERD，且未出现在记录中 |

---

## 3. 实验结果

### 3.1 实验设置
- **数据**：MIMIC-IV v3.1，2,015 例独立入院记录，37,144 样本，80/10/10 划分，每患者只保留一次入院防止泄漏；
- **模型**：15 个 LLM（GPT、LLaMA、Mistral、Qwen、Meditron、Med42、ClinicalCamel）；
- **训练**：LoRA adapters，3 epochs 监督微调 + 1 epoch CoRFu；
- **CoRFu 超参**：$\beta=0.1$，$\lambda=0.5$，$r=16$，$\alpha=32$。

### 3.2 主实验结果

| 模型 | Acc. | Macro-F1 | F1-Q1 | F1-Q2 | F1-Q3 | F1-Q4 | HSR | TIR |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Llama-3.3-70B-Instruct | 73.9 | **70.7** | 86.7 | 70.0 | 65.9 | 60.0 | 21.2 | 21.1 |
| Mixtral-8x7B | 65.4 | 63.8 | 51.4 | 68.3 | 69.3 | 66.4 | **20.5** | **15.3** |
| Llama-3.1-8B-Instruct (base SFT) | 67.8 | 61.5 | 83.6 | 64.0 | 38.2 | 60.0 | 28.2 | 21.1 |
| **Llama-3.1-8B + CoRFu (Q1 vs. Q2)** | **76.8** | **77.9** | 73.2 | **76.6** | **78.9** | **79.9** | **18.2** | **0.0** |
| Qwen3-8B (base SFT) | 63.7 | 59.9 | 70.0 | 58.6 | 50.0 | 60.8 | 28.2 | 31.1 |
| **Qwen3-8B + CoRFu (Q1 vs. Q2)** | **70.7** | **71.0** | 65.1 | 66.7 | 78.0 | 74.3 | 21.8 | **0.0** |

**关键发现**：
1. **准确率不等于安全性**：Llama-3.3-70B 的 macro-F1 最高（70.7%），但 HSR/TIR 并非最优；Mixtral-8x7B 安全指标最佳（HSR 20.5%, TIR 15.3%）但 F1 非最高。
2. **CoRFu 大幅超越基础模型**：Llama-3.1-8B + CoRFu (Q1 vs. Q2) 在 F1 上比 SFT 基础模型提升 +16.4 点，并彻底消除 TIR（0.0%）。
3. **Q1 vs. Q2 配对最强**：在所有 CoRFu 变体中，Q1 对抗 Q2（真实但无支持的困境）提供了最强的整体增益。

### 3.3 消融实验

![](MediEval_metrics_vs_lambda_vertical.png)

> **Figure 6**: $\lambda$ 系数对 Qwen3-8B 和 Llama-3.1-8B 的 CoRFu (Q1 vs. Q2) 训练效果的影响。中等 $\lambda$ 值（0.5–1.0）在 macro-F1 最大化同时显著降低 HSR 和 TIR；$\lambda > 1.0$ 时出现过惩罚，macro-F1 下降且错误率上升。

| $\lambda$ | macro-F1 | HSR | TIR | 效果 |
|:---|:---|:---|:---|:---|
| 0 (vanilla DPO) | 中等 | 较高 | 较高 | 无额外安全约束 |
| 0.5–1.0 | **最佳** | **最低** | **最低** | 最佳平衡点 |
| > 1.0 | 下降 | 上升 | 上升 | 过惩罚 |

### 3.4 专有模型结果（附录 E，零样本）

| 模型 | Acc. | Macro-F1 | HSR | TIR |
|:---|:---|:---|:---|:---|
| GPT-5 | 45.7 | 30.4 | 19.3 | 0.0 |
| GPT-4o | 49.3 | 29.1 | 46.7 | 0.0 |

GPT 模型从不预测 Q3（导致 TIR=0），但这反映的是 miscalibration 而非真正的安全意识。零样本下专有模型仍明显落后于经过 CoRFu 微调的小模型。

---

## 4. 局限性与未来方向

1. **规模限制**：约 2k 入院记录是有意为之的设计选择，优先考虑临床合理性、质量控制和防止患者级信息泄漏。构造管道已完全自动化，可扩展到更大队列；
2. **标注一致性**：200 样本人工评估 agreement 达 97%（Cohen's $\kappa \approx 0.96$），所有分歧集中在 Q2/Q3 边界（正是基准设计要探测的微妙区分）；
3. **本体噪声**：继承了 MIMIC-IV、UMLS、SNOMED CT、RxNorm 的固有噪声和不一致性，通过本体引导的合理性检查和多跳路径约束缓解；
4. **应用场景**：目前为陈述分类任务，未来可扩展到问答、安全关键领域、时序推理、RAG 集成。

---

## 5. 个人思考

1. **四象限设计的优雅性**：相比单一正确/错误判断，将"知识正确性"和"语境支持性"正交分解为四个象限，能精确诊断模型在临床场景中的具体失败模式。这种框架不仅适用于医学，任何需要"事实+上下文"双验证的领域（如法律、金融合规）都可借鉴。

2. **安全指标比准确率更重要**：论文最有力的结论是"更高的准确率不一定转化为更安全的临床推理"。这提醒我们评价 LLM 在关键领域应用时，不能仅看 aggregate metric，必须设计针对性 safety-critical 指标。

3. **CoRFu 的设计启发**：将临床安全需求（"高置信度的错误应受更大惩罚"）直接编码为损失函数中的二次惩罚项 $S^2$，是一种优雅的偏好优化扩展。相比复杂的事后安全检查，在训练阶段就嵌入安全意识更具根本性。但 $\lambda$ 的敏感性（>1.0 即过惩罚）也提示实际调参需要谨慎。

---

## 6. 关键引用

```bibtex
@article{qu2026medieval,
  title={MediEval: A Unified Medical Benchmark for Patient-Contextual and Knowledge-Grounded Reasoning in LLMs},
  author={Qu, Zhan and F{\"a}rber, Michael},
  journal={arXiv preprint arXiv:2512.20822},
  year={2025}
}
```

---

## 7. 图片完整 URL 列表

| 图号 | 文件名 | 完整 URL |
|:---|:---|:---|
| Figure 1 | MediEval_figure1_overview.png | https://arxiv.org/html/2512.20822v2/figure1_overview.png |
| Figure 2 | MediEval_figure2_example_q2.png | https://arxiv.org/html/2512.20822v2/figure2_example_q2.png |
| Figure 3 | MediEval_figure2_example_q1.png | https://arxiv.org/html/2512.20822v2/figure2_example_q1.png |
| Figure 4 | MediEval_figure2_example_q3.png | https://arxiv.org/html/2512.20822v2/figure2_example_q3.png |
| Figure 5 | MediEval_figure2_example_q4.png | https://arxiv.org/html/2512.20822v2/figure2_example_q4.png |
| Figure 6 | MediEval_metrics_vs_lambda_vertical.png | https://arxiv.org/html/2512.20822v2/metrics_vs_lambda_vertical.png |
