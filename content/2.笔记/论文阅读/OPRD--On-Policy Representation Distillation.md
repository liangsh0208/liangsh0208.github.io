---
title: "OPRD: On-Policy Representation Distillation"
date: 2026-06-09
tags: ["知识蒸馏", "强化学习", "数学推理", "模型压缩"]
paper: "https://arxiv.org/abs/2606.06021"
code: "https://github.com/ShenzhiYang2000/OPRD"
authors: "Shenzhi Yang, Guangcheng Zhu, Bowen Song, Haobo Wang 等 (浙江大学 & 蚂蚁集团)"
---

# OPRD: On-Policy Representation Distillation

## 一句话总结

OPRD 将 on-policy 蒸馏从输出概率空间提升到隐层表示空间，通过直接对齐 student 和 teacher 的中间隐状态（绕过 LM Head），同时消除了采样方差和信息瓶颈，在数学推理 benchmark 上首次将 1.5B 学生模型训练到接近教师水平，且训练速度提升 1.44 倍、显存减少 54%。

### 核心结果一览 (Figure 1)

![](fig1_teaser_pareto.png)

> **Figure 1**: OPRD 在三个维度上严格 Pareto-dominant。每个气泡代表一种方法（同一 student/teacher 对训练 500 步）。横轴为训练时间↓，纵轴为 AIME24 Avg@16 准确率↑，气泡面积编码 actor-update 峰值显存↓。OPRD（深蓝色）同时在所有轴上优于最强基线 OPD top-16：+2.7pt 准确率、1.44× 速度提升、54% 显存节省。

---

## 1. 研究背景与动机

### 1.1 On-Policy Distillation (OPD) 回顾

标准的 on-policy 蒸馏流程：
1. **Student 生成 rollout**: 给定 prompt x，student 采样生成响应 ŷ ~ p_θ(·|x)
2. **Teacher 打分**: 对同一个 ŷ，计算 teacher 的 token 概率 q_T(y_t | x, ŷ_{<t})
3. **匹配输出分布**: 最小化 student 和 teacher 在输出空间的 KL 散度

核心优势是避免了 off-policy 的分布偏移问题——student 始终在自己的分布上学习。

### 1.2 OPD 的两个根本缺陷

#### 缺陷一：采样方差 → 信噪比(SNR)崩溃

OPD 使用单个采样 token 的 Monte Carlo 估计来近似 KL 梯度。对于 Qwen 系列 ~150K 的词表：

- 训练初期：student 与 teacher 差异大 → 梯度信号强 → 学习有效
- 训练后期：p_θ → q_T → 梯度均值趋零，**但方差不变** → SNR 崩溃
- 表现：训练曲线后期**振荡/停滞**，无法继续逼近教师

#### 缺陷二：LM Head 信息瓶颈

所有输出空间方法只能观察到经 LM Head 投影后的信号：

```
z = W_head · h  →  p = softmax(z)
```

问题在于 W_head 存在**有效零空间** N_W：
- 如果隐状态差异 Δh = h_θ - h_T 满足 W_head·Δh = c·**1**（常数向量）
- 则 softmax(z_θ) = softmax(z_T)，输出概率**完全相同**
- 即：**巨大的隐层差异可以在输出空间完全不可见**

实测表明：生产级 LLM 的 W_head 条件数 σ₁/σ_d 达 10³~10⁴，沿低奇异值方向的隐状态偏差被输出空间低估 10⁶~10⁸ 倍。

---

## 2. OPRD 方法

### 2.1 核心思想

**一个设计选择**解决两个问题：在 LM Head 之前的隐层空间直接对齐 student 和 teacher 的表示。

```
传统 OPD:  h_θ → W_head → softmax → KL(p_θ || q_T)
OPRD:      h_θ ←→ h_T  (直接 MSE，绕过 W_head)
```

### 方法架构图 (Figure 2)

![](fig2_method_overview.png)

> **Figure 2**: OPRD 方法架构。传统 OPD（左）通过 LM Head 将隐状态投影到词表空间后匹配概率分布；OPRD（右）绕过 LM Head，直接在选定的 transformer 层和 token 位置上对齐 student/teacher 的隐状态向量，使用归一化 MSE 损失。

### 2.2 OPRD 目标函数（公式详解）

设：
- L: transformer 层数
- T: 响应序列长度
- h_θ^(ℓ,t): student 第 ℓ 层第 t 个位置的隐状态
- h_T^(ℓ,t): teacher 第 ℓ 层第 t 个位置的隐状态

定义两个选择集合：
- **层选择**: L_layer ⊆ {1, ..., L}（选哪些层做蒸馏）
- **位置选择**: P(ŷ) ⊆ {1, ..., T}（选哪些 token 位置做监督）

位置掩码 m_t ∈ {0, 1} 表示位置 t 是否在 P(ŷ) 中。

**OPRD 损失函数**：

$$
\mathcal{L}_{\text{OPRD}}(\theta) = \frac{1}{|\mathcal{L}_{\text{layer}}|} \sum_{\ell \in \mathcal{L}_{\text{layer}}} \frac{\sum_{t=1}^{T} m_t \cdot \| h_\theta^{(\ell, t)} - h_T^{(\ell, t)} \|_2^2}{\sum_{t=1}^{T} m_t \cdot \| h_T^{(\ell, t)} \|_2^2}
$$

**各项解释**：

| 组件 | 作用 |
|------|------|
| 分子 ‖h_θ - h_T‖² | 逐位置的隐状态距离（MSE） |
| 分母 ‖h_T‖² | **归一化因子**，消除不同层隐状态尺度差异 |
| m_t | 位置掩码，短响应中超出有效长度的位置被 mask 掉 |
| 层求和取平均 | 所有选中层等权贡献 |

### 2.3 关键设计选择

| 维度 | 选项 | 论文选择 | 实验依据 |
|------|------|---------|---------|
| 层选择 L_layer | 最后一层 / 所有层 / 奇偶层 | **所有层** | 提供最完整的结构信号 |
| 位置选择 P(ŷ) | 所有 / first-k / **last-k** | **last-k (k=2000)** | student-teacher 分歧集中在响应末尾 |
| 组合方式 | 单独 / 与 OPD 混合 | 均可，支持 μ·L_OPRD + (1-μ)·L_OPD | μ 增大时性能单调提升 |

**为什么选 last-k?** 长 CoT 推理中，前面的思考步骤相对容易对齐，真正的分歧出现在推理链后半段——这是决定最终答案正确性的关键区域。

### 2.4 训练流程

```
对每个训练 batch:
1. Student 采样生成 rollout: ŷ ~ p_θ(·|x)         [on-policy]
2. Teacher 做一次 forward pass: 获取所有层 h_T^(ℓ,t)  [无需 LM Head]
3. Student 做一次 forward pass: 获取所有层 h_θ^(ℓ,t)  [无需 LM Head]
4. 计算 L_OPRD，反向传播更新 θ
```

相比 OPD 省去了：大词表 softmax 计算 + top-k logits 提取 + KL 计算。

---

## 3. 理论分析

### 3.1 定理 1：零采样方差

**陈述**: OPRD 的梯度 ∇_θ L_OPRD 是给定 rollout (x, ŷ) 的**确定性函数**，不引入额外的采样方差。

**直觉**: MSE 目标直接作用于连续的隐状态向量，不涉及离散采样（不像 OPD 需要采样 token 来估计 KL）。梯度的唯一随机性来源是 rollout 本身的采样，但这对 OPD 和 OPRD 相同。

**对比 OPD**: OPD 的梯度估计 = rollout 随机性 + token 采样随机性（额外方差来源）

**后果**: 当 p_θ → q_T 时：
- OPD: 信号→0，方差不变 → SNR → 0 → 训练停滞
- OPRD: 信号→0（隐状态差异消失），方差也→0 → SNR 保持稳定 → 可持续收敛

### 3.2 定理 2：LM Head 信息瓶颈

**定义有效零空间**:

$$
\mathcal{N}_W = \{ \Delta h \in \mathbb{R}^d : W_{\text{head}} \cdot \Delta h \in \text{span}\{\mathbf{1}\} \}
$$

**陈述**: 对任意输出空间损失 ℓ_out（KL、交叉熵等），若 h_θ - h_T ∈ N_W，则 ℓ_out(h_θ, h_T) = 0。但 L_OPRD > 0（只要 h_θ ≠ h_T）。

**证明思路**:
1. 若 W_head·(h_θ - h_T) = c·**1**，则 z_θ = z_T + c·**1**
2. Softmax 对加常数不变：softmax(z + c·**1**) = softmax(z)
3. 因此 p_θ = q_T → 任何输出空间 loss = 0
4. 但 ‖h_θ - h_T‖² > 0 → OPRD 仍能提供梯度信号

**实际影响**: W_head 的奇异值衰减意味着隐状态沿低奇异值方向 v_d 的偏差，在输出空间只被感知为原始大小的 (σ_d/σ_1)² 倍。对生产 LLM，这个压缩比达 10⁻⁶~10⁻⁸，即隐状态差 1000，输出空间只"看到" 0.001。

### 3.3 两个定理的联合效应

```
训练进入后期 (p_θ ≈ q_T):
├─ OPD 视角: 输出相同 → loss ≈ 0 → 梯度为噪声 → 停止学习
└─ OPRD 视角: 隐层仍有差距 → MSE > 0 → 确定性梯度 → 持续改进
```

这解释了为什么实验中 OPRD 单调上升而 OPD 平台化。

---

## 4. 实验结果

### 4.1 实验设置

| 配置项 | 具体值 |
|--------|--------|
| Teacher | JustRL-1.5B (Qwen2.5-Math-1.5B 经 RL 训练) |
| Student 初始化 | R1-distill-1.5B (Qwen2.5-Math-1.5B) |
| 硬件 | 8×A100 80GB, FSDP |
| 训练步数 | 500 optimizer steps |
| 评测方式 | Avg@16 (16次独立采样取均值), T=0.7 |
| Benchmark | AIME 2024 (30题), AIME 2025 (30题), AIMO (83题) |

### 4.2 主实验结果

| 方法 | AIME24 | AIME25 | AIMO | 与教师差距 |
|------|--------|--------|------|-----------|
| Teacher (JustRL-1.5B) | 50.8 | 35.6 | 79.5 | — |
| Student (初始) | 32.9 | 21.9 | 62.2 | -17.9 / -13.7 / -17.3 |
| OPD top-1 | 42.3 | 33.5 | 77.0 | -8.5 / -2.1 / -2.5 |
| OPD top-16 | 47.1 | 34.0 | 76.5 | -3.7 / -1.6 / -3.0 |
| **OPRD** | **49.8** | **34.6** | **79.1** | **-1.0 / -1.0 / -0.4** |

**关键发现**:
- OPRD 在所有 benchmark 上将差距缩小到 ~1 分以内
- OPD top-16 虽然传递了更多输出分布信息，但仍然平台化（SNR 崩溃预测被验证）
- 增加 top-k 的 k 值不能从根本上解决问题

### 4.3 效率对比

| 指标 | OPD top-1 | OPD top-16 | OPRD |
|------|-----------|-----------|------|
| 训练时间 (500步) | 813 min | 812 min | **563 min** |
| 相对速度 | 1.0× | 1.0× | **1.44×** |
| Actor-update 峰值显存 | 基准 | +1% | **-54%** |
| 平均响应长度 | ~7000 tok | ~7000 tok | **~5700 tok** |

效率提升原因：
- 不需要计算 150K 词表的 softmax
- 不需要存储 teacher 的完整 logits
- MSE 计算远比 KL 散度简单

### 4.4 训练动态分析

#### OPRD vs OPD top-1 训练曲线 (Figure 3a-c)

| AIME24 | AIME25 | AIMO |
|--------|--------|------|
| ![](fig3a_aime24_vs_top1.png) | ![](fig3b_aime25_vs_top1.png) | ![](fig3c_aimo_vs_top1.png) |

> **Figure 3 (上)**: OPRD vs OPD top-1 的训练曲线对比。OPRD（蓝色）在所有 benchmark 上**单调递增**直至接近教师水平；OPD top-1（橙色）在后期**振荡/平台化**，验证了 SNR 崩溃的理论预测。

#### OPRD vs OPD top-16 训练曲线 (Figure 3d-f)

| AIME24 | AIME25 | AIMO |
|--------|--------|------|
| ![](fig3d_aime24_vs_top16.png) | ![](fig3e_aime25_vs_top16.png) | ![](fig3f_aimo_vs_top16.png) |

> **Figure 3 (下)**: OPRD vs OPD top-16 的训练曲线对比。即使 OPD 使用 top-16 传递更多输出分布信息，仍然在后期平台化（~AIMO 上比 OPRD 低 2.6 分）。这证明增加输出空间信息量不能根本解决问题——信息瓶颈在 LM Head 本身。

**关键观察**:
- **OPRD**: 准确率**单调递增**，无波动
- **OPD**: 后期出现平台/振荡现象
- 当 policy gradient loss ≈ 0 时（student 输出已接近 teacher），OPD 无力继续改进，但 OPRD 的隐层信号仍在驱动学习
- 这与定理 2 的预测完全吻合

### 4.5 消融实验关键结论

**层选择**: 所有层 > 最后一层 > 奇偶层子集

**位置选择**: last-2000 > all positions > first-2000
- 解释：数学推理中，前面的问题理解部分 student 已经较好，分歧集中在推理链后段

**混合系数 μ 消融** (Figure 4):

![](fig4_ablation_mu.png)

> **Figure 4**: 混合系数 μ 的消融。μ=0 为纯 OPD，μ 增大表示 OPRD 占比增大。性能**单调递增**，证明 OPRD 信号与 OPD 信号互补且 OPRD 贡献始终为正。

#### 熵动态分析 (Figure 5)

| 纯 OPD | OPD + 1× OPRD | OPD + 10× OPRD |
|--------|---------------|----------------|
| ![](fig5a_entropy_opd.png) | ![](fig5b_entropy_opd_plus_1x_oprd.png) | ![](fig5c_entropy_opd_plus_10x_oprd.png) |

> **Figure 5**: token 熵分布随训练的演化。纯 OPD（左）在收敛后出现熵分布不规则振荡；加入 OPRD 后（中/右），熵分布更加平滑有序，student 的内部表示更稳定地向 teacher 收敛。这是隐层对齐带来"内在稳定性"的直观证据。

---

## 5. 局限性与未来方向

### 5.1 核心局限

| 局限 | 影响 | 可能的解决方案 |
|------|------|--------------|
| **同架构约束** | Teacher 和 Student 必须隐层维度一致 | 可学习投影层 / 对比学习目标 |
| **仅验证数学** | 代码、对话、Agent 场景未验证 | 不同场景可能需不同 position 策略 |
| **均匀层权重** | 未利用不同层的重要性差异 | 自适应层加权 |
| **固定 k** | last-k 的 k 是超参数 | 基于 gradient signal 自适应选择 |

### 5.2 论文提出的未来方向

1. **跨架构 OPRD**: 通过投影头将不同维度的隐状态映射到公共空间
2. **自适应层/位置选择**: 根据 student-teacher gap 或梯度信息量动态加权
3. **理解相变现象**: 训练中存在某个临界点后 OPRD 优势急剧放大
4. **扩展到其他任务**: 代码生成、Agent 交互、开放对话等

---

## 6. 个人思考

### 6.1 方法的优雅之处

OPRD 的核心 insight 非常简洁：**与其在信息被压缩后（输出空间）做蒸馏，不如在信息完整时（隐层空间）直接对齐**。这个思路对所有使用 on-policy 蒸馏的场景都有启发。

### 6.2 与 EvalTree 的关联

如果用 EvalTree 分析 OPRD 蒸馏后的模型：
- 可以揭示 student 是否在所有细粒度数学能力上均匀逼近 teacher
- 或者某些能力子树（如几何证明 vs 组合计数）仍有显著差距
- last-k position 的选择可能对不同类型推理（短推理 vs 长推理）影响不同

### 6.3 局限性反思

- 同架构约束在实际部署中可能是最大障碍（通常是大模型蒸馏到小模型）
- 只有 500 步训练 + 1.5B 模型的验证，更大规模的 scaling 行为未知
- Avg@16 的评测方式可能掩盖了单次推理的不稳定性

---

## 7. 关键引用

```bibtex
@article{yang2026oprd,
  title={OPRD: On-Policy Representation Distillation},
  author={Yang, Shenzhi and Zhu, Guangcheng and Song, Bowen and Wang, Haobo and Xia, Mingxuan and Zheng, Xing and Ma, Yingfan and Chen, Zhongqi and Wang, Weiqiang and Chen, Gang},
  journal={arXiv preprint arXiv:2606.06021},
  year={2026}
}
```
