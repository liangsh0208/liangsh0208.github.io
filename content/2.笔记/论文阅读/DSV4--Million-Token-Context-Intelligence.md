---
created: 2026-06-16
published: 2026-04-24
paper: https://huggingface.co/collections/deepseek-ai/deepseek-v4
code: https://huggingface.co/collections/deepseek-ai/deepseek-v4
authors: DeepSeek-AI
tags:
  - LLM
  - MoE
  - Long-Context
  - Attention
  - Training-Infrastructure
  - DeepSeek
---

# DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence

---

## 整体总结与创新点分析

### 一句话定位

DeepSeek-V4 不是一篇单纯的方法论文，而是一篇**架构–系统–训练–后训练四位一体的全栈式技术报告**。它在保留 DeepSeekMoE + MTP 框架的基础上，通过**混合稀疏注意力（CSA/HCA）**、**流形约束残差（mHC）**、**Muon 优化器**、**FP4 量化训练**等创新，实现了**1M token 超长上下文的高效推理**，并在开源模型中达到了 SOTA。

### 五大核心创新点

| 创新层级 | 创新点 | 核心价值 |
|---------|--------|---------|
| **架构层** | **CSA + HCA 混合注意力** | 序列维度先压缩再稀疏选择，1M 上下文下 KV Cache 降至 BF16 GQA8 基线的 **~2%**，单 token FLOPs 降至 **10%–27%** |
| **架构层** | **mHC（Manifold-Constrained Hyper-Connections）** | 将残差映射约束在双随机矩阵流形上，解决多层 Hyper-Connections 的数值不稳定问题，保留表达能力 |
| **优化层** | **Muon 优化器** | 用 Hybrid Newton-Schulz 正交化替代 AdamW，收敛更快、更稳定；10 步迭代可在 BF16 下稳定运行 |
| **系统层** | **FP4 QAT + TileLang + EP Overlap** | 训练到推理全链路优化：FP4 量化专家权重、TileLang DSL 确保 bit-identical 输出、EP 通信计算重叠加速 **1.5–1.96×** |
| **后训练层** | **两段式管线：Specialist Training → OPD 蒸馏** | 先独立培养各领域专家（SFT + GRPO + GRM），再用 reverse KL 全词表蒸馏整合到统一模型，避免 weight-merging 退化 |

### 效率突破的量化对比

| 指标 | DeepSeek-V3.2 | V4-Pro | V4-Flash |
|------|--------------|--------|----------|
| 总参数 | 671B | **1.6T** | **284B** |
| 激活参数 | 37B | **49B** | **13B** |
| 1M token 单 token FLOPs | 100% | **27%** | **10%** |
| 1M token KV Cache | 100% | **10%** | **7%** |

### 全文结构图

```
§1 Introduction          → 动机、效率瓶颈、核心创新总览
§2 Architecture          → V3 继承、mHC、CSA/HCA、Muon
§3 General Infrastructures → EP Overlap、TileLang、Deterministic Kernels、
                            FP4 QAT、Training Framework、Inference Framework
§4 Pre-Training          → 数据构造、训练设置、稳定性缓解、评测
§5 Post-Training         → 两段式后训练（Specialist → OPD）、评测与真实任务
§6 Conclusion            → 结论、局限与展望
Appendix A               → 作者名单与致谢
Appendix B               → 评测细节（内部自建评测集结果）
```

---

## 按章节深度解读

---

### §1 Introduction — 为什么要做高效百万 token 上下文？

**核心论点**：当前的 LLM 推理范式正从"参数扩展"转向 **test-time scaling**（通过更长的思维链、更多的 rollout 来提升输出质量）。但这一范式在根本上受到 **vanilla attention 二次复杂度** 的制约——上下文越长，attention 的计算量和 KV Cache 存储量线性乃至超线性增长，导致长程任务成本极高。

**论文指出三个瓶颈**：
1. **计算瓶颈**：1M token 的 dense attention FLOPs 是 4K token 的 **62500×**。
2. **内存瓶颈**：百万级 token 的 KV Cache 直接限制了 batch size 和并发能力。
3. **长程任务刚需**：agentic 工作流、跨文档分析、代码库级理解等场景都需要 ultra-long context support。

**DeepSeek-V4 的解题思路**（三个层面协同）：
- **架构层**：CSA（压缩+稀疏选择）处理中等粒度全局注意力，HCA（重度压缩）处理超长上下文，SWA（滑动窗口）保留局部细粒度依赖。
- **模型层**：mHC 增强深层残差稳定性，Muon 加速收敛。
- **系统层**：FP4 QAT、TileLang、EP Overlap 从训练到推理全链路优化。

> **图表说明**：Figure 1（封面页右侧）展示了 V4 系列与 V3.2 在不同序列长度下的**累计 KV Cache** 和**单 token FLOPs** 对比曲线。
> - 横轴为序列长度（256 → 1024K），纵轴为相对值。
> - **V4-Pro**（蓝色实线）在 1M 序列长度时，累计 KV Cache 仅为 V3.2 的 **10%**，单 token FLOPs 为 **27%**。
> - **V4-Flash**（绿色实线）更进一步，KV Cache 降至 **7%**，FLOPs 降至 **10%**。
> - 这得益于 CSA/HCA 的混合注意力设计：CSA 压缩率 $m=4$（每 4 个 token 合并为 1 个压缩KV），HCA 压缩率 $m'=128$（每 128 个 token 合并为 1 个），两者交错使用，使得长序列下的 KV 存储和计算开销被极度压缩。

> ⚠️ **如图片显示不清晰**：Figure 1 位于 PDF 第 1 页，包含左右两个子图（benchmark 性能柱状图 + FLOPs/KV Cache 曲线图）。请使用 PDF 阅读器截图补充。

---

### §2 Architecture — 架构设计的四大支柱

![](Pasted%20image%2020260616105159.png)

#### 2.1 V3 继承设计

DeepSeek-V4 保留了 V3 的核心成功要素，但做了针对性改进：

| 保留模块 | 说明 |
|---------|------|
| **DeepSeekMoE** | 细粒度 routed experts + shared experts，loss-free 负载均衡 |
| **MTP** | Multi-Token Prediction，配置与 V3 一致 |

| 改进点 | 具体变化 |
|--------|---------|
| 路由亲和度计算 | $Sigmoid(\cdot)$ → $Sqrt(Softplus(\cdot))$ |
| 负载均衡 | 增加轻微**序列级 balance loss**，防止单序列内极端不均衡 |
| 路由并行策略 | **移除 routing target node 数量限制**，重新设计并行策略 |
| 前层 FFN | 前若干层 dense FFN 替换为 **Hash routing 的 MoE 层**（按 token ID 预定义 hash 函数分配 expert） |

*斜体注：Hash routing 的具体 hash 函数形式未披露。*

#### 2.2 mHC — 为什么标准 Hyper-Connections 会崩？怎么解决？

**问题背景**：标准 Hyper-Connections（HC）通过将残差流宽度从 $d$ 扩展到 $n_{hc} \times d$，提供了一个与 hidden size 解耦的额外缩放维度。但多层堆叠时，训练常出现**数值不稳定**（梯度/信号爆炸或消失）。

**mHC 的核心思想**：将残差映射矩阵 $B_l$ 约束在**双随机矩阵流形（Birkhoff polytope）**上。

- 双随机矩阵满足：行和为 1、列和为 1、所有元素非负。
- 这一约束保证了 $||B_l||_2 \leq 1$（非扩张），且对乘法封闭，深层前向/反向传播均稳定。

**前向公式**：

$$
X_{l+1} = B_l X_l + C_l F_l(A_l X_l), \quad X_l \in \mathbb{R}^{n_{hc} \times d}
$$

其中：
- $X_l$：第 $l$ 层前的残差状态（$n_{hc}=4$ 条并行残差流，每条维度 $d$）。
- $F_l(\cdot)$：第 $l$ 层主体计算（MoE 层）。
- $A_l$：输入映射；$B_l$：残差变换矩阵（约束为双随机）；$C_l$：输出映射。

**动态参数化与约束应用**

首先，对输入 $X_l \in \mathbb{R}^{n_{hc} \times d}$ 做 flatten 和归一化：

$$
\hat{X}_l = \mathrm{RMSNorm}(\mathrm{vec}(X_l)) \in \mathbb{R}^{1 \times n_{hc}d}
$$

然后生成三组**未约束的原始参数**（原文 Eq.3–5）：

$$
\tilde{A}_l = \alpha_l^{pre} \cdot (\hat{X}_l W_l^{pre}) + S_l^{pre}
$$

$$
\tilde{B}_l = \alpha_l^{res} \cdot \mathrm{Mat}(\hat{X}_l W_l^{res}) + S_l^{res}
$$

$$
\tilde{C}_l = \alpha_l^{post} \cdot (\hat{X}_l W_l^{post})^T + S_l^{post}
$$

其中 $W_l^{pre}, W_l^{post} \in \mathbb{R}^{n_{hc}d \times n_{hc}}$，$W_l^{res} \in \mathbb{R}^{n_{hc}d \times n_{hc}^2}$；$S_l^{pre} \in \mathbb{R}^{1 \times n_{hc}}$、$S_l^{res} \in \mathbb{R}^{n_{hc} \times n_{hc}}$、$S_l^{post} \in \mathbb{R}^{n_{hc} \times 1}$ 为可学习的静态偏置；$\alpha_l^{pre}, \alpha_l^{res}, \alpha_l^{post} \in \mathbb{R}$ 为可学习的门控因子（初始化小值）。

**约束应用**（原文 Eq.6–7）：

$$
A_l = \sigma(\tilde{A}_l), \qquad C_l = 2\sigma(\tilde{C}_l)
$$

对残差映射 $\tilde{B}_l$，通过 **Sinkhorn-Knopp 迭代（20 步）** 投影到双随机流形上：

$$
M^{(0)} = \exp(\tilde{B}_l),\quad M^{(t)} = \mathcal{T}_r\!\left(\mathcal{T}_c\!\left(M^{(t-1)}\right)\right),\quad B_l = M^{(t_{max})}
$$

其中 $\mathcal{T}_r, \mathcal{T}_c$ 是行/列归一化操作。

> **Why A not B？为什么约束到双随机流形，而不是简单的 gradient clipping 或 layer normalization？**
> - Gradient clipping 是"事后补救"，不平滑且会损失梯度信息。
> - LayerNorm 只约束均值/方差，不约束残差变换矩阵的谱范数。
> - 双随机矩阵在数学上保证了谱范数 ≤ 1 且对乘法封闭，是从**结构层面**解决不稳定问题，而非从**数值层面**打补丁。

*斜体注：$\alpha_l$ 与静态偏置 $S_l$ 的具体初始化数值未披露。*

#### 2.3 混合注意力 — CSA 与 HCA 如何协同工作？

这是 DeepSeek-V4 **最核心的架构创新**，直接决定了 1M context 的可行性。

##### CSA（Compressed Sparse Attention）

**直觉**：先沿序列维度将 KV 压缩，再做稀疏选择。相当于"先粗筛，再精挑"。

**四步数据流**：

1. **生成可压缩 KV 与压缩权重**：
   - 对输入 $H \in \mathbb{R}^{n \times d}$，生成两组 KV 候选 $C^a, C^b$ 和压缩权重 $Z^a, Z^b$：

   $$
   C^a = H \cdot W^{aKV},\quad C^b = H \cdot W^{bKV}
   $$

   $$
   Z^a = H \cdot W^{aZ},\quad Z^b = H \cdot W^{bZ}
   $$

   其中 $W^{aKV}, W^{bKV}, W^{aZ}, W^{bZ} \in \mathbb{R}^{d \times c}$ 为可训练参数。

2. **Overlapped Softmax 压缩**（关键设计）：
   - 每 $m$ 个 entry 合并为 1 个，但使用**重叠窗口**（实际吸收 $2m$ 个源 token）。
   - 相邻压缩块共享边界信息，缓解硬边界信息损失。

   $$
   \big[S^a_{mi:m(i+1)-1};\, S^b_{m(i-1):mi-1}\big] = \mathrm{Softmax}_{\mathrm{row}}\!\left(\big[Z^a_{mi:m(i+1)-1} + B^a;\; Z^b_{m(i-1):mi-1} + B^b\big]\right)
   $$

   $$
   C^{\text{Comp}}_i = \sum_{j=mi}^{m(i+1)-1} S^a_j \odot C^a_j \;+\; \sum_{j=m(i-1)}^{mi-1} S^b_j \odot C^b_j
   $$

3. **Lightning Indexer（低秩稀疏选择）**：
   - 用低秩 query 与压缩 KV 块计算相似度 $I_{t,s}$，执行 Top-k 筛选，只保留 $k$ 个压缩 KV entry。

   **低秩 query 生成**（原文 Eq.13–14）：

   $$
   c_t^Q = h_t \cdot W^{DQ}
   $$

   $$
   \big[q_{t,1}^I, \dots, q_{t,n_h^I}^I\big] = q_t^I = c_t^Q \cdot W^{IUQ}
   $$

   其中 $h_t \in \mathbb{R}^d$ 为 query token 的输入 hidden state；$c_t^Q \in \mathbb{R}^{d_c}$ 为 query 的低秩压缩隐向量；$n_h^I$ 为 indexer query head 数；$W^{DQ} \in \mathbb{R}^{d \times d_c}$、$W^{IUQ} \in \mathbb{R}^{d_c \times c \cdot n_h^I}$ 分别为 down-projection 和 up-projection 矩阵。

   **Index score 计算**（原文 Eq.15–16）：

   $$
   \big[w_{t,1}^I, \dots, w_{t,n_h^I}^I\big] = w_t^I = h_t \cdot W^w
   $$

   $$
   I_{t,s} = \sum_{h=1}^{n_h^I} w_{t,h}^I \cdot \mathrm{ReLU}\!\left(q_{t,h}^I \cdot K^{\mathrm{IComp}}_s\right)
   $$

   其中 $W^w \in \mathbb{R}^{d \times n_h^I}$；$K^{\mathrm{IComp}}$ 为对 indexer key 执行同样压缩后的表征。

   **Top-k 选择**（原文 Eq.17）：

   $$
   C_t^{\mathrm{SprsComp}} = \Big\{ C_s^{\mathrm{Comp}} \;\Big|\; I_{t,s} \in \mathrm{Top\text{-}k}(I_{t,:}) \Big\}
   $$

4. **Shared KV MQA（Multi-Query Attention）**：
   - 在选中的稀疏压缩 KV 上执行 attention，采用 Shared KV 进一步降低内存。

   **Attention query 生成**（原文 Eq.18）：

   $$
   \big[q_{t,1}, \dots, q_{t,n_h}\big] = q_t = c_t^Q \cdot W^{UQ}
   $$

   **Core Attention**（原文 Eq.19）：

   $$
   o_{t,i} = \mathrm{CoreAttn}\!\left(\mathrm{query}=q_{t,i},\; \mathrm{key}=C_t^{\mathrm{SprsComp}},\; \mathrm{value}=C_t^{\mathrm{SprsComp}}\right)
   $$

   其中 $o_{t,i} \in \mathbb{R}^c$ 为第 $t$ 个 token 在第 $i$ 个 head 上的 core attention 输出。

   **Grouped Output Projection**：
   - 将 $n_h$ 个输出分成 $g$ 组，每组先投影到中间维度 $d_g$（$d_g < c \cdot n_h / g$），再投影回 $d$ 维，避免直接做 $c \cdot n_h \to d$ 的巨大矩阵乘法。

![](Pasted%20image%2020260616105610.png)
> **图表说明**：Figure 3（Page 9）展示了 CSA 的完整数据通路：
> - 左侧：Input Hidden States → Token-Level Compressor（Softmax 加权压缩）→ 生成 $C^{Comp}$。
> - 中间：Lightning Indexer（低秩 query 计算 index score）→ Top-k Selector。
> - 右侧：选中的 $C^{SprsComp}$ 进入 Shared KV Multi-Query Attention → Grouped Output Projection → Output。
> - 图中箭头清晰展示了"压缩 → 索引 → 选择 → 注意力"的流水线。

> ⚠️ **如图片显示不清晰**：Figure 3 位于 PDF 第 9 页。若 DSV4_page9.png 截图不够清晰，请直接从 PDF 第 9 页截图替换。

##### HCA（Heavily Compressed Attention）

**直觉**：CSA 还有 Top-k 选择（保留了多个压缩块），HCA 更激进——直接把所有 KV 压缩到更 coarse 的粒度，不做稀疏选择，对所有压缩 KV 做 dense attention，同时保留 Sliding Window KV 补偿局部细节。

**与 CSA 的区别**：

| 维度 | CSA | HCA |
|------|-----|-----|
| 压缩率 | $m=4$（较小） | $m'=128$（很大） |
| 稀疏选择 | 有（Top-k） | 无（dense on compressed KV） |
| 局部补偿 | 重叠窗口 | Sliding Window Attention（$n_{win}=128$）+ Attention Sink |
| 适用层 | 大部分层 | 部分层（与 CSA 交错） |
| 精度 | 较高 | 略低，但效率极高 |

**公式**（与 CSA 类似但更简单，无 Top-k 选择）：

$$
C = H \cdot W^{KV},\quad Z = H \cdot W^Z
$$

$$
S_{m'i:m'(i+1)-1} = \text{Softmax}_{\text{row}}(Z_{m'i:m'(i+1)-1} + B)
$$

$$
C^{\text{Comp}}_i = \sum_{j=m'i}^{m'(i+1)-1} S_j \odot C_j
$$

**HCA 的 CoreAttn 与 CSA 共享同样策略**（原文 Eq.24–26）：

$$
c_t^Q = h_t \cdot W^{DQ}
$$

$$
\big[q_{t,1}, \dots, q_{t,n_h}\big] = q_t = c_t^Q \cdot W^{UQ}
$$

$$
o_{t,i} = \mathrm{CoreAttn}\!\left(\mathrm{query}=q_{t,i},\; \mathrm{key}=C^{\mathrm{Comp}},\; \mathrm{value}=C^{\mathrm{Comp}}\right)
$$

注意 HCA 的 CoreAttn 没有稀疏选择，直接在压缩后的 $C^{\mathrm{Comp}}$ 上做 dense attention。

**精度补偿**：HCA 压缩率 $m'=128$ 很大，会丢失大量 token-level 细粒度信息。为此引入：
- **Sliding Window Attention（SWA）**：保留最近 $n_{win}=128$ 个 token 的未压缩 KV。
- **Attention Sink**：保留序列起始的几个 token 的 KV，防止注意力分数过度漂移。

> **图表说明**：Figure 4（Page 11）展示了 HCA 架构：
> - 与 Figure 3（CSA）相比，HCA 没有 Lightning Indexer 和 Top-k Selector。
> - 取而代之的是更粗的压缩粒度（$m' \gg m$），以及右侧的 SWA 分支（保留最近窗口的未压缩 KV）。
> - HCA 和 CSA 在模型中**交错使用**：部分层用 CSA（精度优先），部分层用 HCA（效率优先），使得整体在 1M context 下仍保持高质量。

> ⚠️ **如图片显示不清晰**：Figure 4 位于 PDF 第 11 页。若 DSV4_page11.png 不够清晰，请从 PDF 第 11 页截图替换。

> **Why A not B？为什么不全用 CSA 或全用 HCA？**
> - 全用 CSA：Top-k 选择仍有索引和选择开销，且压缩率 $m=4$ 不够极致，1M context 的 KV 仍偏大。
> - 全用 HCA：$m'=128$ 的粗粒度压缩会严重损失建模能力，尤其在需要细粒度依赖的场景。
> - **交错使用**：不同层取不同平衡点，让模型自己学习在不同抽象层次上如何分配注意力资源。

#### 2.4 Muon Optimizer — 为什么 AdamW 不够？

**问题**：AdamW 对矩阵权重的更新是**逐元素**的，没有利用权重矩阵的二维结构。对于大规模矩阵（如注意力投影、MoE 路由），这会导致收敛慢、训练不稳定。

**Muon 的核心思想**：对梯度矩阵做**正交化（Orthogonalization）**，让更新方向更接近最优，同时重缩放保证不同 shape 的矩阵更新幅度一致。

**伪代码**：

```
Require: 学习率 η, 动量 μ, 权重衰减 λ, 更新重缩放因子 γ
for each training step t:
  for each weight W ∈ ℝ^{n×m}:
    G_t = ∇_W \mathcal{L}_t(W_{t-1})  // Compute gradients
    M_t = μ M_{t-1} + G_t            // Accumulate momentum buffer
    O'_t = HybridNewtonSchulz(μ M_t + G_t)  // Nesterov trick + orthogonalization
    O_t = O'_t \cdot \sqrt{\max(n,m)} \cdot γ  // Rescale the update RMS
    W_t = W_{t-1} \cdot (1 − ηλ) − η O_t     // Perform weight decay and update
```

**Hybrid Newton-Schulz 正交化**（核心算法）：

目标是将矩阵 $M$ 近似正交化为 $UV^\top$（其 SVD 为 $M = U\Sigma V^\top$）。

迭代公式：
$$
M_k = a M_{k-1} + b (M_{k-1} M_{k-1}^\top) M_{k-1} + c (M_{k-1} M_{k-1}^\top)^2 M_{k-1}
$$

分两段共 **10 步**：
- 前 8 步：$(a,b,c) = (3.4445, -4.7750, 2.0315)$ —— **快速收敛**
- 后 2 步：$(a,b,c) = (2, -1.5, 0.5)$ —— **精确稳定到 singular value = 1**

> **Why A not B？为什么不用 eigendecomposition 做正交化？**
> - Eigendecomposition 计算成本高，且在大规模矩阵上数值稳定性差。
> - Newton-Schulz 迭代仅需矩阵乘法，可用 BF16 稳定计算，GPU 利用率极高。

**Muon vs AdamW 的模块分工**：

| 模块 | 优化器 |
|------|--------|
| Embedding、Prediction Head、mHC 静态偏置/门控、所有 RMSNorm | **AdamW**（不适合正交化） |
| 其余全部参数（Attention、MoE、mHC 矩阵等） | **Muon** |

*斜体注：Muon 相比 AdamW 的具体收敛速度倍数或训练 loss 曲线对比未披露。*

> **图表说明**：Figure 2（Page 6-7）展示了 DeepSeek-V4 整体架构图。但由于 PDF 第 6-7 页包含密集的技术图示，当前预渲染截图（DSV4_page6.png、DSV4_page7.png）可能不够清晰。建议从 PDF 原文件第 6-7 页截取高清版本替换。
> - 图中展示了 Input Embedding → Transformer Block（CSA/HCA + DeepSeekMoE + mHC）→ Prediction Head 的数据流。
> - 含 Pre/Post-Block Mixing（mHC 的残差混合）与 MTP 模块。

---

### §3 General Infrastructures — 系统层的全链路优化

#### 3.1 EP Communication-Computation Overlap

**问题**：MoE 的 Expert Parallelism 涉及大量跨 GPU 通信（Dispatch + Combine），通信延迟是主要瓶颈。

**方案**：将 MoE 层拆分为 4 个阶段——**Dispatch（通信）→ Linear-1（计算）→ Combine（通信）→ Linear-2（计算）**，并通过**流水线重叠**让通信被计算隐藏。

**关键设计**：
- 采用 **pull-based** 通信：每块 GPU 主动从远端读取，避免细粒度 push 的高通知延迟。
- 将专家划分为多个 **wave**；当前 wave 计算、下一 wave 的 token 传输、已完成 wave 的结果回传三者并发。

> **图表说明**：Figure 5（Page 15）展示了三种 EP Overlap 方案的对比：
> - **Naive**：Dispatch → Linear-1 → Combine → Linear-2 串行执行，大量空闲等待。
> - **Comet**（基线方案）：部分重叠，但仍有明显气泡。
> - **Ours**（DeepSeek-V4）：4 个阶段完全流水线化，通信几乎完全被计算隐藏。
> - 量化结果：对比强 non-fused 基线，NVIDIA GPU 一般推理负载加速 **1.50–1.73×**；RL rollout 等延迟敏感场景高达 **1.96×**。

> ⚠️ **如图片显示不清晰**：Figure 5 位于 PDF 第 15 页，包含三个子图对比。建议从 PDF 直接截取替换 DSV4_page15.png。

#### 3.2 TileLang — 为什么不用 CUDA 或 Triton 手写所有 kernel？

**定位**：一个 Domain-Specific Language（DSL），在开发生产力与运行时效率之间取平衡。

**核心设计**：
- **Host Codegen**：将 host 端逻辑下沉到 IR 级生成的 host launcher（基于 TVM-FFI），CPU 校验开销从几十/几百微秒降到 **<1 μs**。
- **Z3 SMT Solver 辅助**：集成 Z3 做形式化整数分析（layout inference、memory hazard detection、bound analysis），编译时间仅增加几秒。
- **Bit-identical 输出**：
  - 默认关闭 fast-math，精度近似仅作为显式 opt-in。
  - 提供显式 rounding mode 的 IEEE-754 兼容内建函数。
  - 通过对齐 NVCC 的代数化简规则 + layout annotation，实现与手写 CUDA **完全一致** 的输出。

**Why A not B？**
- 纯手写 CUDA：开发效率低，每次改 attention 变体都要重写 kernel。
- 纯用 Triton：编译器黑盒，难以保证 deterministic 和 bit-identical。
- TileLang：既有 DSL 的开发效率，又能控制到 bit-level 精度。

#### 3.3 Batch-Invariant Deterministic Kernels

**Why deterministic？**
- 训练可复现性对调试硬件/软件故障至关重要。
- 出现 loss spike 时，确定性结果有助于定位数值根因。

**如何保证 Batch Invariance（任意 token 的输出与其在 batch 中的位置无关）？**
- **Attention**：放弃 split-KV（会跨 SM 分配单序列计算）。采用双内核策略：内核 A 单 SM 处理整序列保证吞吐；内核 B 多 SM 处理最后 partially-filled wave，但通过精心设计累积顺序保持一致。
- **MatMul**：用 DeepGEMM 替换 cuBLAS，放弃传统 split-k。

**如何保证 Determinism？**
- **Attention Backward**：不用 `atomicAdd`（浮点加不可结合），改为每 SM 独立累积缓冲区，最后全局确定性求和。
- **MoE Backward**：多 rank 并发写同一缓冲区时，设计单 rank 内 token 顺序预处理 + 跨 rank 缓冲区隔离。

#### 3.4 FP4 Quantization-Aware Training (QAT)

**格式**：**MXFP4（E2M1）**，以 $1 \times 32$ tiles 为粒度配 scale。

**QAT 策略**：
- FP32 master weights → 量化到 FP4 → **无损反量化到 FP8 (E4M3)** → 前向计算。
- "无损"原理：FP8 (E4M3) 比 FP4 (E2M1) 多 2 个指数位，动态范围更大。经验验证当前权重满足 scale 比值阈值。
- 反向：梯度对 FP8 权重计算，STE（Straight-Through Estimator）回传至 FP32 master weights。
- **关键**：整个 QAT 流程**完全复用现有 FP8 训练框架**，无需修改反向管道。

**应用**：
- **MoE expert weights**：权重占 GPU 显存大头。
- **CSA indexer 的 QK path**：QK activations 在 FP4 中缓存、加载、相乘。
- **Index scores**：从 FP32 量化到 BF16，top-k selector 加速 **2×**，召回率保持 **99.7%**。

> **Why A not B？为什么用 FP4→FP8 无损反量化，而不是修改训练框架支持 FP4 计算？**
> - 修改训练框架支持原生 FP4 计算需要重写大量 kernel 和反向传播逻辑。
> - FP4→FP8 反量化只需在数据加载时做一次格式转换，现有 FP8 kernel 完全复用。
> - FP8 的动态范围足够吸收 FP4 的细粒度 scale，经验验证"无损"。

#### 3.5 Training Framework — 四大子系统

| 子系统 | 核心设计 | 关键收益 |
|--------|---------|---------|
| **Muon 高效实现** | 混合 ZeRO 桶分配（Dense 参数限制最大并行度 + MoE 每层独立优化）；BF16 Newton-Schulz；梯度通信用两阶段 reduce-scatter + BF16 量化 | ZeRO 与 Muon 兼容；通信量减半 |
| **mHC 内存高效实现** | 融合 kernel；选择性重算；调整 DualPipe 1F1B 重叠以适应 mHC 增加的流水线通信 | wall-time overhead 仅占重叠 pipeline stage 的 **6.7%** |
| **Contextual Parallelism** | 两阶段通信：rank 间传递最后 $m$ 条未压缩 KV → 压缩 → All-gather | 解决 CSA/HCA 压缩 KV 长度在各 rank 不一致的问题 |
| **Extended Auto-diff for Checkpointing** | Tensor 级激活检查点：TorchFX 追踪全图，自动找出重算某 tensor 的**最小子图** | 开发者只需标注需检查点的 tensor，系统自动推导最优重算路径 |

#### 3.6 Inference Framework — KV Cache 异构设计

**问题**：CSA/HCA/SWA 三种注意力导致不同层的 KV 大小、更新规则完全不同，无法直接套用 PagedAttention。

**方案**：设计为**双轨管理**：

1. **Classical KV Cache**：管理 CSA/HCA 压缩后的 KV。
   - 每请求分配多个 block，每 block 覆盖 $\mathrm{lcm}(m, m')$ 个原始 token。
   - 产生 $k_1 = \mathrm{lcm}(m,m')/m$ 条 CSA compressed token 和 $k_2 = \mathrm{lcm}(m,m')/m'$ 条 HCA compressed token。

2. **State Cache**：管理 SWA 和尚未就绪压缩的未压缩 tail tokens。
   - 每请求预分配固定大小 block。
   - SWA 段：存最近 $n_{win}$ 个 token 的 KV。
   - CSA/HCA 段：存未压缩的 pending tail states。

> **图表说明**：Figure 6（Page 22-23）展示了 KV Cache Layout：
> - 上方为 Classical KV Cache（蓝色 block，存储压缩后的 CSA/HCA KV）。
> - 下方为 State Cache（绿色 block，存储 SWA 窗口和未压缩 tail）。
> - 不同层之间用不同颜色/大小标注，清晰展示了各层 KV 结构的差异。
> - 右侧标注了共享前缀（shared prefix）的复用策略：相同前缀的 KV 可以共享，减少重复 prefill。

> ⚠️ **如图片显示不清晰**：Figure 6 跨越 PDF 第 22-23 页，包含 KV Cache 结构示意图和可能的效率对比表格。建议从 PDF 截取高清版本替换 DSV4_page22.png 和 DSV4_page23.png。

---

### §4 Pre-Training — 数据、配置与稳定性

#### 4.1 数据构造

| 数据类别 | 说明 | 变化（vs V3） |
|---------|------|-------------|
| **网页数据** | 过滤批量自动生成与模板化内容 | 更严格清洗 |
| **数学与编程** | 核心组成部分 | 新增 agentic 数据 |
| **多语言** | 更大规模多语言语料 | 规模扩大 |
| **长文档** | 科学论文、技术报告 | 重点新增 |
| **总量** | **32T（Flash）/ 33T（Pro）tokens** | — |

- Tokenizer：DeepSeek-V3 tokenizer，词汇表 **128K**。
- 受 Ding et al. (2024) 启发，将不同来源文档打包进适当长度序列，最小化截断。
- 与 V3 不同：预训练期间采用 **sample-level attention masking**。

*斜体注：exact data mix ratios、网页数据过滤阈值、去重策略等未披露。*

#### 4.2 训练设置

**模型配置**（核心参数对比）：

| 配置项 | Flash | Pro |
|--------|-------|-----|
| 层数 | 43 | 61 |
| Hidden dim | 4096 | 7168 |
| CSA compression rate ($m$) | 4 | 4 |
| HCA compression rate ($m'$) | 128 | 128 |
| Routed experts | 256 | 384 |
| 激活专家数/ token | 6 | 6 |
| 总参数量 | **284B** | **1.6T** |
| 激活参数量 | **13B** | **49B** |

**训练超参数**：

| 超参数 | Flash | Pro |
|--------|-------|-----|
| 训练 token | 32T | 33T |
| Batch size (max) | 75.5M | 94.4M |
| Peak LR | $2.7 \times 10^{-4}$ | $2.0 \times 10^{-4}$ |
| Sequence length | 4K → 16K → 64K → **1M** | 同左 |
| Warmup steps | 2000 | 2000 |
| Optimizer | Muon + AdamW（Embedding/Head/RMSNorm） | 同左 |
| MTP loss weight | 0.3（衰减期 0.1） | 同左 |

#### 4.3 训练稳定性 — Anticipatory Routing + SwiGLU Clamping

**Anticipatory Routing（预判路由）**

万亿参数 MoE 训练中，loss spike 始终与 MoE 层异常值相关，且路由机制会加剧异常值。简单回滚无法预防复发。

**原理**：解耦主干网络与路由网络的同步更新。
- step $t$ 使用当前参数 $\theta_t$ 计算特征。
- 但路由索引**基于历史参数** $\theta_{t-\Delta t}$ 预计算并缓存。

**实现优势**：
- 预计算仅需单前向传播。
- 通过流水线执行和 EP 通信的计算重叠，额外 wall-clock 开销限制在约 **20%**。
- **动态触发**：自动检测 loss spike → 短回滚 → 激活 Anticipatory Routing → 运行一段时间后自动恢复标准训练。
- **效果**：以可忽略的总体额外开销避免 loss spike，且不损害模型性能。

**SwiGLU Clamping**
- 线性分量钳制到 **[−10, 10]**；门控分量上限 **10**。
- 有效消除异常值，训练更稳定，且不损害性能。

#### 4.4 预训练评测结果

**关键结论**：Flash（13B 激活）尽管参数远小于 V3.2（37B 激活），仍在广泛基准上超越 V3.2；Pro（49B 激活）几乎全面领先。

| 评测维度 | V3.2-Base | Flash-Base | Pro-Base |
|---------|-----------|------------|----------|
| MMLU | 87.8 | 88.7 | **90.1** |
| Simple-QA verified | 28.3 | 30.1 | **55.2** |
| FACTS Parametric | 27.1 | 33.9 | **62.6** |
| HumanEval | 62.8 | 69.5 | **76.8** |
| MATH | 60.5 | 57.4 | **64.5** |
| LongBench-V2 | 40.2 | 44.7 | **51.5** |

*注：BigCodeBench 上 Flash（56.8）< V3.2（63.9），说明代码 benchmark 上对压缩注意力仍有挑战。*

---

### §5 Post-Training — 两段式后训练管线

后训练采用 **两阶段范式**：
1. **Specialist Training**：独立培养各领域专家（数学、代码、agent、instruction following）。
2. **On-Policy Distillation (OPD)**：用 reverse KL 将各领域专家的知识蒸馏到统一模型。

#### 5.1 Specialist Training

**流程**：Base 模型 → **领域特定 SFT** → **GRPO（Group Relative Policy Optimization）** 对齐。

**GRM（Generative Reward Model）**：actor 自身充当评判模型，把内部推理能力融入评估，仅需极少人工标注。

**Agentic 能力增强**：
- 新工具调用 schema：采用专用 `|DSML|` token 和 XML 格式，减少转义失败和调用错误。
- **Interleaved Thinking**：工具调用场景下完整保留全部历史推理内容（跨 user turn 不清空）；普通对话仍丢弃之前推理痕迹。
- **Quick Instruction**：向输入序列追加 `<|action|>`、`<|query|>` 等特殊 token，直接复用已计算的 KV cache，降低首 token 时延（TTFT）。

*斜体注：SFT 精确样本量、GRPO group size/KL penalty/temperature/batch size 等未披露。*

![465](Pasted%20image%2020260616105003.png)

> **图表说明**：Figure 7（Page 32）展示了 Thinking 策略的两种模式：
> - (a) Thinking with tools：工具调用场景下的思维链保留策略——完整保留全部历史推理内容。
> - (b) Thinking without tools：普通对话场景下的思维链管理——新用户消息到达时仍丢弃之前推理痕迹。

> ⚠️ **如图片显示不清晰**：Figure 7 位于 PDF 第 32 页。当前预渲染截图（DSV4_page32.png）可能不够清晰，内含 Table 2-5 的文本表格，建议从 PDF 截取高清版本。

#### 5.2 On-Policy Distillation (OPD)

**动机**：传统 weight-merging 或 mixed RL 合并不同领域专家时易导致**性能退化**（catastrophic forgetting + distribution shift）。OPD 通过在统一参数空间内、于学生模型自身生成的轨迹上对齐教师输出分布，避免该退化。

**教师模型**：共使用 **超过十个** 领域专家模型。

**目标函数（Reverse KL）**：

$$
\mathcal{L}_{\mathrm{OPD}}(\theta) = \sum_{i=1}^{N} w_i \cdot D_{KL}\big(\pi_\theta \parallel \pi_{E_i}\big)
$$

- 最小化 reverse KL（而非 forward KL）意味着学生**保守地**在教师分布内学习，不会产生教师分布外的"幻觉"。
- $w_i$：各专家权重，由相对重要性决定。

**Teacher Scheduling**（工程关键）：
- 所有教师权重卸载到**集中式分布式存储**，按需通过类 ZeRO 的参数分片加载。
- 数据调度时**按教师索引排序**，确保每个 mini-batch 只加载一次对应教师 head，任意时刻 GPU 上最多只驻留一个教师 prediction head。
- 所有参数/隐藏态的加载/卸载均在后台**异步**进行，不阻塞计算关键路径。

**Full-Vocabulary OPD 的高效实现**：
- 先前工作常将 full-vocabulary KL 简化为逐 token 估计，但会导致**梯度估计方差高、训练不稳定**。
- DeepSeek-V4 采用 **full-vocabulary logit distillation**：
  - 缓存教师的**最后一层隐藏状态**（而非完整 logits，$|V| > 100k$）。
  - 训练时实时通过对应 prediction head 重建完整 logits。
  - 用 **TileLang kernel** 加速 KL 计算并抑制动态内存分配。

> **Why A not B？为什么不用 forward KL 而用 reverse KL？**
> - Forward KL：$D_{KL}(\pi_{teacher} \|\| \pi_{student})$，要求学生分布覆盖教师分布的所有支撑集，容易让学生产生教师分布外的输出（"发散"）。
> - Reverse KL：$D_{KL}(\pi_{student} \|\| \pi_{teacher})$，迫使学生只在教师分布的高概率区域内分配概率质量，更"保守"、更稳定，避免多专家合并时的分布外退化。

*斜体注：OPD 总蒸馏步数、学习率、教师权重 $w_i$ 的具体计算方式未披露。*

#### 5.3 RL & OPD Infrastructure

| 基础设施 | 核心设计 | 关键收益 |
|---------|---------|---------|
| **FP4 Quantization** | Rollout 及仅推理的前向传播应用 MXFP4；训练阶段无损 FP4→FP8 反量化复用现有 FP8 框架 | 降低内存带宽与采样延迟 |
| **Preemptible & Fault-Tolerant Rollout** | Token 级**写前日志（WAL）**：每生成一个 token 立即追加到 WAL；抢占时保存 KV cache，恢复时继续解码 | 避免确定性重生成带来的**长度偏置**（短回复更可能在打断中存活） |
| **Scaling RL for 1M Context** | 将 rollout 数据解耦为轻量级元数据和重量级逐 token 字段；通过**共享内存数据加载器**消除节点内数据冗余 | 显著降低 CPU/GPU 内存压力 |
| **Sandbox (DSec)** | Apiserver / Edge / Watcher 三组件，基于 3FS；支持 Function Call / Container / microVM / fullVM 四种底层 | 单集群管理 **数十万个并发沙箱实例** |

#### 5.4 后训练评测结果

**与闭源前沿模型对比**：

| 评测维度 | V4-Pro-Max 表现 | 与前沿模型关系 |
|---------|----------------|---------------|
| Knowledge（SimpleQA, Chinese-SimpleQA） | 大幅优于开源模型 | **开源 SOTA** |
| Knowledge（MMLU-Pro, HLE, GPQA） | 有提升 | 仍距 Gemini-3.1-Pro 有差距 |
| Reasoning（标准 benchmark） | 优于 GPT-5.2 / Gemini-3.0-Pro | 略逊 GPT-5.4 / Gemini-3.1-Pro，差距约 **3–6 个月** |
| Code Agent | 显著提升 | — |

**内部评测集结果**：

| 任务 | 对手 | 结果 |
|------|------|------|
| Agentic Search vs. RAG | RAG 基线 | V4-Pro 胜率 **61.7%**（869 题） |
| 中文功能性写作 | Gemini-3.1-Pro | 胜率 **62.65%**（3170 题） |
| 中文创意性写作（质量） | Gemini-3.1-Pro | 胜率 **77.48%** |
| 复杂指令遵循与多轮写作 | Claude-Opus-4.5 | 胜率 46.9%（略处下风） |

> **关键发现**：V4-Pro-Max 在开源模型中达到 **SOTA**，但与 GPT-5.4 / Gemini-3.1-Pro 仍有差距，论文坦承差距约 **3–6 个月**。

---

### §6 Conclusion, Limitations, and Future Directions

**论文自我声明的局限**：
1. 知识评测上仍有差距（MMLU-Pro、HLE、GPQA 等）。
2. 推理评测上仍略逊 GPT-5.4 和 Gemini-3.1-Pro。
3. 长上下文交互延迟需要进一步优化。
4. 复杂指令遵循（196 题评测）对 Claude-Opus-4.5 略处下风。

**未来方向**：
1. **新维度的模型稀疏性**：在 MoE 与稀疏注意力之外，探索更稀疏的 embedding modules。
2. **低延迟长上下文系统**：持续研究低延迟架构与系统技术。
3. **长程多轮 Agentic 任务**：重视并持续迭代探索。
4. **多模态能力**：正在融入模型。
5. **更优的数据策展与合成策略**。

---

## 个人深度思考

### 方法优雅性排名

1. **CSA/HCA 混合注意力 — 分而治之的典范**：CSA 负责"压缩+稀疏选择"（精度与效率的平衡），HCA 负责"重度压缩"（极致效率），SWA 负责"局部细粒度"（精度补偿）。三者各司其职，没有一个模块试图做所有事。这比单一的全局稀疏注意力（如 NSA）或单一的 KV eviction（如 H2O）都要更灵活。

2. **mHC 的数学严谨性**：将残差连接约束在 Birkhoff polytope 上，不仅解决了数值稳定性，还保留了表达能力（双随机矩阵对乘法封闭）。这比简单的 gradient clipping 深刻得多——它是从**结构保证**而非**数值补救**的角度解决问题。

3. **OPD 的 reverse KL 选择**：这是一个极其关键但容易被忽视的设计。Forward KL 会让学生"发散"去覆盖所有教师分布的支撑集，而 reverse KL 让学生"保守地"停留在教师分布的高概率区域内。这解释了为什么 OPD 能避免 weight-merging 的退化——它从根本上阻止了分布外的"幻觉"输出。

### 最值得关注的三个数字

1. **KV Cache 降至 2%**（1M context, BF16 GQA8 基线）：这个数字定义了长上下文模型的可行性边界。没有它，1M token 的在线 serving 在工程上不可行。

2. **Anticipatory Routing 额外开销仅 ~20%**：以极低成本解决万亿参数 MoE 的 loss spike 问题，体现了**系统与算法协同设计**的价值——不是改模型，而是改更新机制。

3. **OPD 使用超过十个教师模型**：这暗示了后训练的规模已经非常大，且涉及的领域极其细分。未来模型的竞争可能不仅是预训练的规模，更是后训练的专家覆盖广度。

### 可复现的洞察

1. **SwiGLU Clamping [-10, 10]**：极其简单但有效的稳定技巧，任何使用 SwiGLU 的大型模型训练都可以尝试。

2. **两阶段后训练（Specialist → OPD）**：可作为多领域 LLM 对齐的通用范式——先让各领域独立达到高水准，再用蒸馏统一，比直接在混合数据上训练更可控。

3. **样本级 attention masking**（而非文档级）在长文档预训练中可能更重要，因为它允许跨文档的 attention 学习（如果文档相关），同时仍通过 packing 保持效率。

---

## 8. 关键引用

```bibtex
@article{deepseek2026v4,
  title={DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
  author={{DeepSeek-AI}},
  journal={arXiv preprint},
  year={2026},
  note={Preview version, https://huggingface.co/collections/deepseek-ai/deepseek-v4},
}
```

---

## 附录：论文未披露的关键信息

| 参数类别 | 缺失信息 | 说明 |
|---------|---------|------|
| **数据混合** | 各类数据的精确混合比例 | 仅提及类别，未给占比 |
| **数据过滤** | 网页数据过滤阈值、评分模型 | 提到清洗但未给量化标准 |
| **预训练超参** | 全局训练步数、精确 batch size schedule、梯度裁剪阈值 | 仅给最大值和 warmup 步数 |
| **MoE 路由** | Hash routing 的预定义 hash 函数具体形式 | 仅描述按 token ID 分配 |
| **稀疏注意力** | CSA 与 HCA 在模型各层的交错比例/分配规则 | 未披露具体层分配 |
| **mHC 初始化** | $\alpha_l$ 与静态偏置 $S_l$ 的初始化数值 | 仅提及"小值" |
| **Muon 超参** | Newton-Schulz 迭代的详细数值分析、与 AdamW 的消融对比 | 未给出收敛速度倍数 |
| **GRPO 超参** | group size、KL penalty 系数、temperature、batch size、学习率、训练步数 | 仅引用"prior research" |
| **GRM 架构** | 具体网络结构、参数量、训练数据规模 | 未披露 |
| **SFT 数据** | 各领域精确样本量、epoch、数据格式 | 未披露 |
| **OPD 超参** | 总蒸馏步数、学习率、教师权重 $w_i$ 的具体计算方式 | 未披露 |
| **课程学习** | 是否采用课程学习或具体调度策略 | 未提及 |
| **硬件信息** | GPU 数量、型号、集群规模、总训练 FLOPs、训练天数 | 未披露 |
| **评测细节** | Appendix B 内部评测集的 prompt 模板、shot 设置、评分者身份、采样参数 | 完全未披露 |
| **系统实测** | EP Overlap 的实际计算通信比（C/B）、TileLang 与手写 CUDA 的端到端对比 | 仅给理论阈值 |
| **KV Cache 存储** | On-Disk KV Cache Storage 的具体格式、命中率、延迟数据 | Page 23 文本截断 |
