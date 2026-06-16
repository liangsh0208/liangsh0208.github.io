---
created: 2026-06-15
published: 2024-12-27
paper: https://arxiv.org/abs/2412.19437
code: https://github.com/deepseek-ai/DeepSeek-V3
authors: DeepSeek-AI (DeepSeek)
tags:
  - LLM
  - MoE
  - MLA
  - FP8
  - Multi-Token-Prediction
  - GRPO
  - 推理蒸馏
  - 训练效率
---

# DeepSeek-V3 Technical Report

## 一句话总结

DeepSeek-V3 是一个 671B 总参数、37B 激活参数的 MoE 语言模型，在 14.8T token 上预训练，总训练成本仅 2.788M H800 GPU 小时（约 $5.576M），通过 MLA、无辅助 loss 负载均衡、Multi-Token Prediction、FP8 训练等创新，在开源模型中达到 SOTA，性能媲美 GPT-4o 和 Claude-3.5-Sonnet。

![](DSV3_fig1_benchmark.png)

> **Figure 1**: DeepSeek-V3 与各类基线模型在多个基准测试上的性能对比。DeepSeek-V3 在知识、代码、数学等任务上全面领先开源模型，并与顶级闭源模型持平。

---

## 1. 研究背景与动机

### 1.1 问题定义

大语言模型（LLM）正在快速迭代，逐步逼近 AGI。开源模型（DeepSeek、LLaMA、Qwen、Mistral 等）也在奋力追赶闭源模型（GPT-4o、Claude-3.5-Sonnet）。**核心问题是**：如何在保持强性能的同时，实现经济的训练和推理成本？

### 1.2 现有方法的不足

- **Dense 模型训练成本极高**：如 LLaMA-3.1 405B 需要海量计算资源，每 token 激活全部参数。
- **MoE 负载均衡带来性能折损**：传统 MoE 依赖辅助 loss（auxiliary loss）强制负载均衡，过大的辅助 loss 会损害模型性能（Wang et al., 2024a）。
- **KV Cache 内存瓶颈**：标准 Multi-Head Attention（MHA）的 KV Cache 随序列长度线性增长，严重制约长上下文推理效率。
- **训练通信瓶颈**：跨节点专家并行引入大量 all-to-all 通信，计算-通信比约为 1:1，效率低下。
- **低精度训练验证不足**：FP8 混合精度训练虽然被提出，但尚未在超大规模模型上验证可行性。

### 1.3 本文核心思路

DeepSeek-V3 在 DeepSeek-V2 的 MLA + DeepSeekMoE 架构基础上，实现了四项关键创新：

1. **无辅助 loss 负载均衡**（Auxiliary-Loss-Free Load Balancing）：用可学习的专家偏置动态调整路由，避免辅助 loss 对模型性能的伤害。
2. **Multi-Token Prediction（MTP）**：每个位置额外预测未来 token，增强训练信号密度，同时可用于推理阶段投机解码加速。
3. **FP8 混合精度训练框架**：首次在超大规模模型上验证 FP8 训练的有效性和稳定性。
4. **DualPipe + 极致工程优化**：实现前向/反向计算与 all-to-all / 流水线通信的完全重叠，消除流水线气泡。

---

## 2. 数据准备与策展（Data Curation）

### 2.1 数据来源分类与规模

| 数据维度 | 详情 |
|---------|------|
| **总量** | 14.8T 高质量多样化 token |
| **语言** | 以英语和中文为主，扩展了多语言覆盖 |
| **领域增强** | 相比 DeepSeek-V2，提升了数学和编程样本比例 |
| **去冗余** | 优化数据处理管线以降低冗余，同时保持语料多样性 |

### 2.2 数据处理与质量增强

#### Fill-in-Middle（FIM）策略

受 DeepSeekCoder-V2 启发，采用 Prefix-Suffix-Middle（PSM）框架：

```
<|fim_begin|> f_pre <|fim_hole|> f_suf <|fim_end|> f_middle <|eos_token|>
```

- FIM 比例：0.1（即 10% 的训练数据使用 FIM 格式）
- 在文档级别应用，作为 pre-packing 过程的一部分
- **关键发现**：FIM 不损害 next-token prediction 能力，同时使模型能基于上下文准确预测中间文本

#### Tokenizer

- **算法**：Byte-level BPE
- **词表大小**：128K
- **改进**：
  - 修改 pretokenizer 和训练数据以优化多语言压缩效率
  - 新增结合标点和换行符的 token
  - **问题**：这种组合 token 在多行 prompt（无末尾换行符）时可能引入 token boundary bias
  - **解决**：训练期间随机拆分一定比例这类组合 token，让模型接触更多特殊情况

### 2.3 文档打包策略

- 采用 document packing 保持数据完整性
- **不**使用跨样本 attention masking（不同于 Ding et al., 2024）

### 2.4 隐私与合规处理

论文未详细披露去标识化、PHI 移除等隐私合规细节，仅提到"高质量和多样化"的数据筛选。

### 2.5 数据验证

- 消融实验中，MTP 策略在 Small MoE（15.7B）和 Large MoE（228.7B）上均一致提升性能
- 无辅助 loss 负载均衡策略在同等训练量下，验证损失从 2.258→2.253（1B MoE），2.085→2.080（3B MoE）

---

## 3. 训练管线详解（Training Pipeline）

### 3.1 基础模型与预训练

#### 模型架构超参数

| 参数 | 值 |
|------|-----|
| Transformer 层数 | 61 |
| 隐藏维度 | 7168 |
| 注意力头数 $n_h$ | 128 |
| 每头维度 $d_h$ | 128 |
| KV 压缩维度 $d_c$ | 512 |
| Query 压缩维度 $d_c'$ | 1536 |
| RoPE 解耦维度 $d_h^R$ | 64 |
| MoE：共享专家数 $N_s$ | 1 |
| MoE：路由专家数 $N_r$ | 256 |
| 每 token 激活路由专家数 $K_r$ | 8 |
| 专家中间维度 | 2048 |
| 总参数 | 671B |
| 每 token 激活参数 | 37B |
| MTP 深度 $D$ | 1 |

**注**：前 3 层使用标准 FFN，其余层全部替换为 MoE 层。

#### 训练超参数

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW |
| $\beta_1$ | 0.9 |
| $\beta_2$ | 0.95 |
| weight_decay | 0.1 |
| 最大序列长度 | 4K |
| 训练 token 总量 | 14.8T |

#### 学习率调度（三段式）

```
Step 0-2K:     线性 warmup → 2.2×10⁻⁴
Step 2K-10T:   恒定 2.2×10⁻⁴
Step 10T-14.3T: 余弦衰减 → 2.2×10⁻⁵
Final 500B:    先恒定 2.2×10⁻⁵ (333B)，再恒定 7.3×10⁻⁶ (167B)
```

#### Batch Size 调度

- 前 469B tokens：从 3072 逐步增加到 15360
- 剩余训练：固定 15360

#### 梯度裁剪

- 梯度裁剪范数：1.0

#### 负载均衡与 MTP 权重调度

| 阶段 | Token 范围 | 偏置更新速度 $\gamma$ | MTP 权重 $\lambda$ |
|------|-----------|----------------------|-------------------|
| 主训练 | 前 14.3T | 0.001 | 0.3（前 10T）→ 0.1（后 4.8T） |
| 收尾 | 最后 500B | 0.0 | 0.1 |

- 序列级平衡 loss 系数 $\alpha$ = 0.0001（仅防极端不平衡）
- 节点限制：$M=4$（每 token 最多路由到 4 个节点）

### 3.2 长上下文扩展

采用两阶段扩展（基于 YaRN）：

| 阶段 | 上下文长度 | 步数 | Batch Size | 学习率 |
|------|-----------|------|-----------|--------|
| Phase 1 | 4K → 32K | 1000 | 1920 | 7.3×10⁻⁶ |
| Phase 2 | 32K → 128K | 1000 | 480 | 7.3×10⁻⁶ |

**YaRN 超参数**：$s=40$, $\alpha=1$, $\beta=32$，缩放因子 $\sqrt{t} = 0.1\ln s + 1$
- 仅对解耦共享 key $\mathbf{k}_t^R$ 应用 YaRN

### 3.3 IFT / 指令微调（SFT）

#### 数据构造

- **总量**：1.5M 指令微调实例

**推理数据（Reasoning Data）**：
- 领域：数学、代码竞赛、逻辑谜题
- 来源：内部 DeepSeek-R1 模型生成
- **问题**：R1 生成数据存在过度思考、格式不佳、长度过长等问题
- **解法**：
  1. 先用 SFT+RL 训练领域专家模型（expert model）
  2. 生成两类 SFT 样本：`<problem, original_response>` 和 `<system_prompt, problem, R1_response>`
  3. 系统 prompt 精心设计，引导模型产生含反思和验证机制的回答
  4. RL 阶段用高温采样生成融合 R1 模式和原始数据的回答
  5. RL 完成后用拒绝采样（rejection sampling）筛选高质量 SFT 数据

**非推理数据（Non-Reasoning Data）**：
- 领域：创意写作、角色扮演、简单问答
- 来源：DeepSeek-V2.5 生成 + 人工审核

#### SFT 训练配置

- 在 DeepSeek-V3-Base 上微调 **2 个 epoch**
- 学习率：余弦衰减，从 $5 \times 10^{-6}$ 降至 $1 \times 10^{-6}$
- 序列打包（sequence packing）+ 样本掩码（sample masking）确保相互隔离

### 3.4 RL 阶段

#### 奖励模型设计

**Rule-Based RM**：
- 适用：可用规则验证答案的问题（数学题、LeetCode 等）
- 机制：要求模型在特定格式中给出最终答案（如框内），用规则验证正确性
- 优势：不易被操纵或利用

**Model-Based RM**：
- 适用：自由形式答案的问题（创意写作等）
- 训练：从 DeepSeek-V3 SFT checkpoint 训练
- **增强**：偏好数据不仅包含最终奖励，还包含导致奖励的思维链（CoT），缓解 reward hacking

#### Group Relative Policy Optimization（GRPO）

类似 DeepSeek-V2，放弃与策略模型同等大小的 Critic 模型，改为从组内得分估计 baseline。

对于每个问题 $q$，从旧策略 $\pi_{\theta_{old}}$ 采样 $G$ 个输出 $\{o_1, o_2, \cdots, o_G\}$，然后优化策略 $\pi_\theta$：

$$\begin{split}\mathcal{J}_{GRPO}(\theta)&=\mathbb{E}{[q\sim P(Q),\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta_{old}}(O|q)]}\\&\frac{1}{G}\sum_{i=1}^{G}\left(\min\left(\frac{\pi_{\theta}(o_{i}|q)}{\pi_{\theta_{old}}(o_{i}|q)}A_{i},\text{clip}\left(\frac{\pi_{\theta}(o_{i}|q)}{\pi_{\theta_{old}}(o_{i}|q)},1-\varepsilon,1+\varepsilon\right)A_{i}\right)-\beta\mathbb{D}_{KL}\left(\pi_{\theta}||\pi_{ref}\right)\right),\end{split}$$

其中 $\mathbb{D}_{KL}$ 定义为：

$$\mathbb{D}_{KL}\left(\pi_{\theta}||\pi_{ref}\right)=\frac{\pi_{ref}(o_{i}|q)}{\pi_{\theta}(o_{i}|q)}-\log\frac{\pi_{ref}(o_{i}|q)}{\pi_{\theta}(o_{i}|q)}-1,$$

各符号含义如下：

| 符号 | 含义 |
|------|------|
| $\varepsilon$ | PPO 裁剪超参数 |
| $\beta$ | KL 散度系数 |
| $\pi_{ref}$ | 参考模型（SFT 模型） |
| $A_i$ | 优势函数，来自组内奖励归一化 |

优势计算：

$$A_{i}=\frac{r_{i}-{\operatorname{mean}(\{r_{1},r_{2},\cdots,r_{G}\})}}{{\operatorname{std}(\{r_{1},r_{2},\cdots,r_{G}\})}}.$$

**RL 数据来源**：编程、数学、写作、角色扮演、问答等多样化 prompt。

### 3.5 DeepSeek-R1 知识蒸馏

**核心方法**：将长 CoT（Chain-of-Thought）推理模型的能力蒸馏到标准 LLM 中。

**Pipeline**：
1. 使用专家模型生成融合 R1 模式和原始数据的推理样本
2. SFT 阶段注入这些样本
3. 通过系统 prompt 引导模型学习反思和验证模式
4. 平衡模型准确度与输出长度

**效果**：在 LiveCodeBench 和 MATH-500 上均有显著提升，但会显著增加平均回答长度。

### 3.6 Self-Rewarding（生成式奖励模型）

- 在通用场景中，难以用硬编码规则构造反馈
- 采用 Constitutional AI 方法，利用 DeepSeek-V3 自身的投票评估结果作为反馈源
- DeepSeek-V3 在 RewardBench 上达到与 GPT-4o-0806 和 Claude-3.5-Sonnet-1022 最佳版本相当的性能
- 通过投票（voting）进一步提升判断能力

---

## 4. 方法

### 4.1 核心思想与架构

![](DSV3_fig2_architecture.png)

> **Figure 2**: DeepSeek-V3 基础架构示意图。采用 MLA 实现高效推理，DeepSeekMoE 实现经济训练。

#### Multi-Head Latent Attention（MLA）

**直觉**：通过低秩联合压缩 Key-Value，将 KV Cache 从 $O(d_h n_h)$ 降至 $O(d_c + d_h^R)$，同时用 RoPE 解耦保持位置信息。

**完整公式推导**：

**Step 1: KV 联合压缩**

$$\boxed{\mathbf{c}_t^{KV}} = W^{DKV}\mathbf{h}_t$$

其中 $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ 是压缩的 KV 潜在向量；$W^{DKV} \in \mathbb{R}^{d_c \times d}$ 为下投影矩阵。

**Step 2: KV 上投影与解耦 RoPE Key**

$$[\mathbf{k}_{t,1}^{C}; \mathbf{k}_{t,2}^{C}; \ldots; \mathbf{k}_{t,n_h}^{C}] = \mathbf{k}_t^C = W^{UK}\mathbf{c}_t^{KV}$$

$$\boxed{\mathbf{k}_t^R} = \operatorname{RoPE}({W^{KR}}\mathbf{h}_t)$$

$$\mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C; \mathbf{k}_t^R]$$

**Step 3: Value 上投影**

$$[\mathbf{v}_{t,1}^{C}; \mathbf{v}_{t,2}^{C}; \ldots; \mathbf{v}_{t,n_h}^{C}] = \mathbf{v}_t^C = W^{UV}\mathbf{c}_t^{KV}$$

**Step 4: Query 低秩压缩**

$$\mathbf{c}_t^Q = W^{DQ}\mathbf{h}_t$$

$$[\mathbf{q}_{t,1}^{C}; \ldots; \mathbf{q}_{t,n_h}^{C}] = \mathbf{q}_t^C = W^{UQ}\mathbf{c}_t^Q$$

$$[\mathbf{q}_{t,1}^{R}; \ldots; \mathbf{q}_{t,n_h}^{R}] = \mathbf{q}_t^R = \operatorname{RoPE}({W^{QR}}\mathbf{c}_t^Q)$$

$$\mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^C; \mathbf{q}_{t,i}^R]$$

**Step 5: 注意力计算**

$$\mathbf{o}_{t,i} = \sum_{j=1}^{t} \operatorname{Softmax}_{j}\left(\frac{\mathbf{q}_{t,i}^{T}\mathbf{k}_{j,i}}{\sqrt{d_{h}+d_{h}^{R}}}\right)\mathbf{v}_{j,i}^{C}$$

$$\mathbf{u}_t = W^O[\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; \ldots; \mathbf{o}_{t,n_h}]$$

**KV Cache 节省分析**：
- 标准 MHA：缓存 $2 \times d_h n_h = 2 \times 128 \times 128 = 32768$ 维向量
- MLA：缓存 $d_c + d_h^R = 512 + 64 = 576$ 维向量
- **压缩比约 57×**

#### DeepSeekMoE 与无辅助 Loss 负载均衡

**基本架构**：

$$\mathbf{h}_t' = \mathbf{u}_t + \sum_{i=1}^{N_s} \operatorname{FFN}^{(s)}_{i}\left(\mathbf{u}_t\right) + \sum_{i=1}^{N_r} g_{i,t} \operatorname{FFN}^{(r)}_{i}\left(\mathbf{u}_t\right)$$

$$g_{i,t} = \frac{g'_{i,t}}{\sum_{j=1}^{N_r} g'_{j,t}}$$

$$g'_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \operatorname{Topk}(\{s_{j,t}|1\leqslant j\leqslant N_{r}\},K_{r}) \\ 0, & \text{otherwise} \end{cases}$$

$$s_{i,t} = \operatorname{Sigmoid}\left({\mathbf{u}_{t}}^{T}\mathbf{e}_{i}\right)$$

**与 DeepSeek-V2 的区别**：使用 Sigmoid 计算 affinity，并对选中 affinity 做归一化得到 gating value。

**无辅助 Loss 负载均衡**：

为每个专家引入可学习偏置 $b_i$，加到 affinity score 上用于 top-K 路由：

$$g'_{i,t} = \begin{cases} s_{i,t}, & s_{i,t}+b_{i}\in\operatorname{Topk}(\{s_{j,t}+b_{j}|1\leqslant j\leqslant N_{r}\},K_{r}) \\ 0, & \text{otherwise}. \end{cases}$$

**关键设计**：
- 偏置 $b_i$ **仅用于路由决策**，不改变 gating value（仍为原始 $s_{i,t}$ 的归一化）
- 每步训练结束后，监控全 batch 专家负载：
  - 过载专家：$b_i \leftarrow b_i - \gamma$
  - 欠载专家：$b_i \leftarrow b_i + \gamma$
- $\gamma$ 为偏置更新速度（前 14.3T 设为 0.001，最后 500B 降为 0.0）

**互补序列级辅助 Loss**（仅防极端不平衡）：

$$\mathcal{L}_{\mathrm{Bal}} = \alpha \sum_{i=1}^{N_r} f_i P_i$$

$$f_i = \frac{N_r}{K_r T} \sum_{t=1}^{T} \mathds{1}\left(s_{i,t}\in\operatorname{Topk}(\{s_{j,t}|1\leqslant j\leqslant N_{r}\},K_{r})\right)$$

$$s'_{i,t} = \frac{s_{i,t}}{\sum_{j=1}^{N_r} s_{j,t}}, \quad P_i = \frac{1}{T}\sum_{t=1}^{T} s'_{i,t}$$

其中 $\alpha = 0.0001$ 极小，$\mathbb{1}(\cdot)$ 为指示函数。

**专家专攻化分析**（Figure 9 / Appendix C）：
- 无辅助 loss 策略允许 batch-wise 负载均衡（而非 sequence-wise）
- 专家在不同 domain 上呈现更强的专攻模式
- batch-wise 约束更灵活，不强制每序列内均衡，允许专家按 domain 特化

#### Multi-Token Prediction（MTP）

![](DSV3_fig3_mtp.png)

> **Figure 3**: MTP 实现示意图。在每个深度保持完整的因果链，顺序预测额外 token。

**直觉**：传统 next-token prediction 训练信号稀疏；MTP 让每个位置额外预测未来 $D$ 个 token，增加训练信号密度并可能改善表示预规划（representation pre-planning）。

**与 Gloeckle et al. (2024) 的区别**：
- Gloeckle：并行预测 $D$ 个额外 token，使用独立输出头
- DeepSeek-V3：**顺序预测**，每个深度保持完整因果链

**第 $k$ 个 MTP 模块**：

$$\mathbf{h}_{i}^{\prime k}=M_{k}[\operatorname{RMSNorm}(\mathbf{h}_{i}^{k-1});\operatorname{RMSNorm}(\operatorname{Emb}(t_{i+k}))],$$

$$\mathbf{h}_{1:T-k}^{k}=\operatorname{TRM}_{k}(\mathbf{h}_{1:T-k}^{\prime k}),$$

$$P_{i+k+1}^{k}=\operatorname{OutHead}(\mathbf{h}_{i}^{k}).$$

**MTP 训练目标**：

$$\mathcal{L}_{\text{MTP}}^{k}=\operatorname{CrossEntropy}(P_{2+k:T+1}^{k},t_{2+k:T+1})=-\frac{1}{T}\sum_{i=2+k}^{T+1}\log P_{i}^{k}[t_{i}],$$

$$\mathcal{L}_{\text{MTP}}=\frac{\lambda}{D}\sum_{k=1}^{D}\mathcal{L}_{\text{MTP}}^{k}.$$

**共享设计**：
- 所有 MTP 模块共享主模型的 embedding 层和输出头
- $M_k \in \mathbb{R}^{d \times 2d}$ 为可学习的投影矩阵

**推理阶段**：
- MTP 可与投机解码（speculative decoding）结合
- 第二 token 预测接受率在各类生成主题中达到 **85%-90%**
- 实现 **1.8× TPS（Tokens Per Second）** 提升

### 4.2 关键设计选择（Why A not B？）

| 设计选择 | 本文方案 | 替代方案 | 工程权衡 |
|---------|---------|---------|---------|
| 负载均衡 | 无辅助 loss + 偏置动态调整 | 纯辅助 loss | 避免辅助 loss 损害主目标，batch-wise 均衡允许专家专攻化 |
| MTP 预测方式 | 顺序因果链 | 并行独立头 | 保持因果结构，与自回归解码更一致；推理时可丢弃不增加成本 |
| KV 压缩 | MLA 低秩压缩 + RoPE 解耦 | MQA/GQA | MLA 在压缩率和性能间取得更好平衡（压缩比约 57×） |
| 精度框架 | FP8 核心 GEMM + BF16 关键算子 | 纯 BF16 | 加速训练 + 减少显存；需精心设计量化策略和累加精度 |
| 专家路由 | Sigmoid + 归一化 gating | Softmax gating | Sigmoid 更稳定，归一化在 top-K 间重新分配权重 |

---

## 5. 实验结果

### 5.1 实验设置

**基线模型**：
- DeepSeek-V2-Base
- Qwen2.5 72B Base
- LLaMA-3.1 405B Base

**评测框架**：内部 HAI-LLM 评估框架，统一评测设置。

**推理配置**：
- 标准基准：MMLU、DROP、GPQA、SimpleQA 等采用 simple-evals 框架 prompt
- 代码/数学：LiveCodeBench（2024.08-11）、AIME 2024、CNMO 2024
- AIME/CNMO：temperature=0.7，16 次运行取平均
- MATH-500：greedy decoding
- 最大输出长度：8192 tokens

### 5.2 主实验结果（Base 模型）

**DeepSeek-V3-Base 关键数字**（部分精选指标）：

| 基准 | 指标 | DeepSeek-V2 | Qwen2.5 72B | LLaMA-3.1 405B | **DeepSeek-V3** |
|------|------|-------------|-------------|----------------|-----------------|
| Pile-test | BPB | 0.606 | 0.638 | **0.542** | 0.548 |
| BBH | EM | 64.1 | 75.3 | 80.1 | **83.9** |
| MMLU | EM | 71.4 | 80.4 | **88.6** | 87.1 |
| MMLU-Pro | EM | 36.6 | 55.0 | **73.3** | 70.3 |
| DROP | F1 | 69.0 | 75.8 | 81.7 | **89.0** |
| HumanEval | Pass@1 | 40.2 | 57.3 | 73.2 | **82.6** |
| LiveCodeBench-Base | Pass@1-COT | 19.4 | 35.4 | 54.1 | **69.4** |
| MATH | EM | 36.6 | 59.4 | 64.6 | **70.0** |
| GSM8K | EM | 81.2 | 88.7 | 91.0 | **93.0** |

**关键结论**：
- DeepSeek-V3-Base 在绝大多数基准上成为最强开源模型
- 尤其在代码和数学上远超对手
- **仅用 37B 激活参数**，优于 LLaMA-3.1 405B（激活 405B）

### 5.3 Chat 模型评测

| 基准 | DeepSeek-V2.5 | Qwen2.5 72B | LLaMA-3.1 405B | Claude-3.5-Sonnet | GPT-4o | **DeepSeek-V3** |
|------|---------------|-------------|----------------|---------------------|--------|-----------------|
| MMLU | 78.5 | 80.4 | 88.6 | **88.5** | 87.2 | **88.5** |
| MMLU-Pro | 52.4 | 55.0 | 73.3 | **75.9** | 72.0 | 75.9 |
| GPQA | 37.1 | 45.0 | 49.0 | 59.5 | 53.6 | **59.1** |
| DROP (3-shot) | 80.4 | 82.4 | 86.0 | 88.3 | 81.5 | **91.6** |
| FRAMES | 65.4 | 75.2 | 78.1 | 79.8 | 80.5 | **82.8** |
| LongBench v2 | 34.1 | 42.5 | 52.7 | 54.4 | 50.1 | **57.7** |
| SimpleQA | 15.6 | 17.1 | **29.1** | 24.8 | 24.8 | 24.9 |
| C-SimpleQA | 40.4 | 59.1 | 63.1 | 64.7 | 68.0 | **84.5** |
| HumanEval-Mul | 62.2 | 68.9 | 77.4 | 81.7 | 78.0 | **82.6** |
| LiveCodeBench | 24.3 | 36.3 | 54.1 | 61.0 | 53.6 | **61.5** |
| AIME 2024 | 16.7 | 30.3 | 23.3 | 16.0 | 9.3 | **39.2** |
| MATH-500 | 56.0 | 71.8 | 73.8 | 78.3 | 74.6 | **85.5** |
| Arena-Hard | 50.0 | 65.0 | 69.7 | **85.2** | 80.4 | **85.5** |
| AlpacaEval 2.0 | 36.4 | 42.7 | 44.0 | 52.0 | 51.1 | **70.0** |

**关键结论**：
- DeepSeek-V3 全面领先开源模型，在 Arena-Hard 上首次突破 85%
- 数学 AIME 2024 达 39.2%，甚至超过 o1-preview
- 代码 LiveCodeBench 达 61.5%，算法任务全面领先
- 长上下文任务（FRAMES、LongBench v2）表现顶级

### 5.4 消融实验

#### MTP 消融（Table 4）

| 基准 | Small MoE Baseline | Small MoE + MTP | Large MoE Baseline | Large MoE + MTP |
|------|-------------------|-----------------|-------------------|-----------------|
| BBH | 39.0 | **41.4** | 70.0 | **70.7** |
| MMLU | 50.0 | **53.3** | 67.5 | 66.6 |
| DROP | 39.2 | **41.3** | 68.5 | **70.6** |
| HumanEval | 20.7 | **26.8** | 44.5 | **53.7** |
| GSM8K | 25.4 | **31.4** | 72.3 | **74.0** |
| MATH | 10.7 | **12.6** | 38.6 | **39.8** |

- MTP 在大多数基准上一致提升性能
- **注意**：推理时丢弃 MTP 模块，推理成本不变

#### 无辅助 Loss 负载均衡消融（Table 5）

| 基准 | Small Aux-Loss | Small Aux-Loss-Free | Large Aux-Loss | Large Aux-Loss-Free |
|------|---------------|---------------------|---------------|---------------------|
| BBH | 37.3 | **39.3** | 66.7 | **67.9** |
| MMLU | 51.0 | **51.8** | 68.3 | 67.2 |
| DROP | 38.1 | **39.0** | **67.1** | **67.1** |
| HumanEval | 22.0 | **22.6** | 40.2 | **46.3** |
| GSM8K | 27.1 | **29.6** | 70.7 | **74.5** |
| MATH | 10.9 | **11.1** | 37.2 | **39.6** |

- 无辅助 loss 策略在大多数基准上优于纯辅助 loss 方法
- 尤其在大规模模型上提升更明显

#### Batch-Wise vs Sequence-Wise 负载均衡

| 方法 | 1B MoE 验证 Loss | 3B MoE 验证 Loss |
|------|-----------------|-----------------|
| Sequence-wise Aux Loss | 2.258 | 2.085 |
| **Aux-Loss-Free (Batch-wise)** | **2.253** | **2.080** |
| Batch-wise Aux Loss | 2.253 | 2.080 |

- Batch-wise 均衡（无论有无辅助 loss）一致优于 Sequence-wise
- 证明 batch-wise 的灵活性允许更好的专家专攻化

### 5.5 训练成本

| 阶段 | H800 GPU Hours | 美元（$2/GPUh） |
|------|---------------|----------------|
| 预训练 | 2,664K | $5.328M |
| 上下文扩展 | 119K | $0.238M |
| 后训练 | 5K | $0.01M |
| **总计** | **2,788K** | **$5.576M** |

- 每 trillion tokens 仅需 **180K H800 GPU hours**
- 在 2048 H800 集群上，每 trillion tokens 约 3.7 天
- 预训练阶段在不到两个月内完成

---

## 6. 基础设施亮点

### 6.1 DualPipe 流水线并行

![](DSV3_fig5_dualpipe.png)

> **Figure 5**: DualPipe 调度示意图。将前向/反向 chunk 分为 attention、all-to-all dispatch、MLP、all-to-all combine、PP communication 等组件，实现计算-通信重叠。

**核心思想**：将每个 chunk 分为四个组件（attention、all-to-all dispatch、MLP、all-to-all combine），反向时进一步拆分为 backward-for-input 和 backward-for-weights。通过精心调度使通信被计算隐藏。

**效果**：即使无重通信负担场景，DualPipe 仍具效率优势。

### 6.2 FP8 混合精度训练

![](DSV3_fig6_fp8.png)

> **Figure 6**: FP8 混合精度框架示意图。核心 GEMM 用 FP8，embedding、output head、gating、norm、attention 保持更高精度。

**策略**：
- **核心 GEMM**：FP8
- **保持高精度的算子**：embedding、output head、gating、normalization、attention operators
- **细粒度量化**：tile-wise（$1 \times N_c$）或 block-wise（$N_c \times N_c$）分组
- **提升累加精度**：部分和提升到 CUDA Cores 计算
- **尾数优先**：相比指数更重视尾数精度
- **在线量化**：动态计算量化参数，无需额外存储

**效果**：FP8 训练模型相对 loss 误差始终低于 **0.25%**。

### 6.3 极致内存节省

1. **重计算 RMSNorm 和 MLA 上投影**：减少激活显存
2. **EMA 参数存 CPU**：减少 GPU 显存占用
3. **MTP 模块共享 embedding/output head**：在相同流水线阶段物理共享参数

### 6.4 推理部署

- **Prefilling**：采用专家冗余部署策略避免推理负载不均衡
- **Decoding**：优化 all-to-all 通信和内存访问
- 端到端生成速度达到 DeepSeek-V2 的 **2 倍以上**

---

## 7. 局限性与讨论

### 7.1 论文自我分析的局限

1. **部署单元较大**：推荐部署单元相对较大，对小型团队构成负担
2. **生成速度仍有提升空间**：尽管已是 V2 的两倍以上
3. **蒸馏增加回答长度**：DeepSeek-R1 蒸馏带来性能提升的同时也显著增加输出长度

### 7.2 深层局限分析

| 局限 | 根因分析 |
|------|---------|
| 英文事实知识弱于 GPT-4o | 训练 token 更多分配给中文知识，SimpleQA 落后但 C-SimpleQA 大幅领先 |
| 工程任务弱于 Claude-3.5 | SWE-Bench 和 Aider 上仍落后 Claude-Sonnet，但大幅领先开源模型；反映工程能力需要更多长 CoT 和工具调用训练 |
| 负载均衡的 domain shift 风险 | Batch-wise 均衡在训练时很灵活，但推理时 domain 分布偏移可能导致负载不均衡；已通过专家冗余部署缓解 |
| Reward hacking 风险 | Model-based RM 中 CoT 奖励设计可以缓解，但无法完全消除；Self-rewarding 也依赖模型自身判断 |

### 7.3 未来方向

1. **架构持续优化**：进一步改善训练和推理效率，追求无限上下文长度支持；突破 Transformer 架构局限
2. **数据持续迭代**：在更全面的维度上驱动数据 scaling，探索更多训练信号源
3. **深度思考能力**：扩展推理长度和深度，提升模型智能和问题解决能力
4. **评估体系完善**：防止过度优化固定基准集，建立更全面多维的评估方法

---

## 8. 个人思考

### 8.1 方法的优雅之处

1. **负载均衡的工程智慧**：用极小的偏置动态调整替代了沉重的辅助 loss，这是一个"四两拨千斤"的设计。batch-wise vs sequence-wise 的分析尤其深刻——不只是提出方法，更解释了**为什么**它有效（专家专攻化）。

2. **MTP 的双重价值**：训练时增强信号密度，推理时可直接用于投机解码，这种训练-推理一致性设计非常优雅。接受率 85-90% 的数据也很扎实。

3. **FP8 的工程闭环**：不是简单用低精度，而是从量化策略（tile-wise / block-wise）、累加精度、在线量化、甚至硬件设计建议形成完整闭环。

4. **R1 蒸馏的方法论**：不是直接蒸馏输出，而是先将 R1 的反思/验证模式内化到 RL 策略中，再用 rejection sampling 筛选高质量 SFT 数据——这比 naive 蒸馏高明得多。

### 8.2 与相关工作的关联

- **MLA**：继承自 DeepSeek-V2，相比 GQA/MQA 在低秩压缩上更激进，效果也更优
- **DeepSeekMoE**：细粒度专家 + 共享专家范式，相比 GShard 更细粒度
- **GRPO**：继承自 DeepSeek-V2 / Shao et al. (2024)，弃用 Critic 模型的思路非常大胆
- **投机解码**：MTP 与 EAGLE 思路相似，但 MTP 目标偏向训练增强而非纯推理优化

### 8.3 最值得关注的数字

| 数字 | 意义 |
|------|------|
| **$5.576M 总训练成本** | 证明了超大规模模型训练的经济可行性，对传统"大模型只有巨头能玩"认知的冲击 |
| **180K H800 GPUh / 1T tokens** | 每万亿 token 的训练成本极低，工程优化的极致体现 |
| **37B 激活 / 671B 总参数** | 仅 5.5% 参数激活即可达到 GPT-4o 级别性能，MoE 效率的有力证明 |
| **1.8× TPS 提升（MTP）** | 训练信号增强 + 推理加速的双重收益 |
| **AIME 2024: 39.2%** | 甚至超过 o1-preview，非长 CoT 模型的数学天花板 |
| **Arena-Hard: 85.5%** | 首个在该基准突破 85% 的开源模型 |

### 8.4 可复现的洞察

1. **Batch-wise > Sequence-wise**：如果训练 MoE，负载均衡的粒度应该放在 batch 级别而非 sequence 级别，这允许专家专攻化且不牺牲均衡性。

2. **偏置路由 > 辅助 loss**：用可学习的轻量级偏置替代辅助 loss，在超大规模验证中展现了鲁棒性。

3. **MTP 是"免费午餐"**：训练时加 MTP 模块，推理时丢弃——不增加推理成本但提升训练效果。$D=1$ 的设置也很务实。

---

## 9. 关键引用

```bibtex
@article{deepseekai2024deepseekv3,
  title={DeepSeek-V3 Technical Report},
  author={{DeepSeek-AI}},
  journal={arXiv preprint arXiv:2412.19437},
  year={2024},
  url={https://arxiv.org/abs/2412.19437}
}
```

---

## 附录：论文未披露的关键信息

| 参数类别 | 缺失信息 | 说明 |
|---------|---------|------|
| 预训练数据构成比例 | 各语言/领域比例 | 仅提到"数学和编程比例提升"，无具体百分比 |
| 数据清洗细节 | 去重策略、质量评分指标 | 提到"去冗余"但未给出方法 |
| RL 超参数 | GRPO 的 $\varepsilon$、$\beta$、$G$（组大小）、温度 | 未披露具体数值 |
| RL 训练步数 | RL 阶段总步数/轮数 | 仅提到"数百个 RL steps" |
| SFT 数据详细配比 | 推理数据 vs 非推理数据比例 | 未给出各域精确样本数 |
| 推理部署配置 | Prefill/Decode 的并行策略细节 | 仅提到"专家冗余部署" |
| FP8 量化参数 | tile/block 的具体分组大小 $N_c$ | 提到概念但未给具体值 |
| DualPipe 具体参数 | chunk 大小、流水线 stages 数量 | 仅描述思想未给配置 |
| EMA 存 CPU 的具体影响 | 训练速度 trade-off | 提到优化但未量化 |
| 长上下文扩展的 YaRN 衰减策略 | attention scaling 的具体实现 | 提到复用 V2 配置 |
| 训练鲁棒性机制 | 无 irrecoverable loss spike 的具体原因 | 仅陈述事实未分析 |

---

> **免责声明**：本笔记数字均来自论文原文，若与官方后续更新有出入，请以最新版本为准。部分公式中的符号解释基于原文上下文推导。
