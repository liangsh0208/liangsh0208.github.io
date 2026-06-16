---
created: 2026-06-15
published: 2026-02-26
paper: https://arxiv.org/abs/2602.21548
code: https://github.com/deepseek-ai/3FS
authors: Yongtong Wu, Shaoyuan Chen, Yinmin Zhong, Rilin Huang, Yixuan Tan, Wentao Zhang, Liyue Zhang, Shangyan Zhou, Yuxuan Liu, Shunfeng Zhou, Mingxing Zhang, Xin Jin, Panpan Huang (Peking University, Tsinghua University, DeepSeek-AI)
tags:
  - LLM-Inference
  - KV-Cache
  - PD-Disaggregation
  - Storage-Bandwidth
  - System-Optimization
  - Agentic-LLM
  - DeepSeek
---

# DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference

## 一句话总结

DualPath 通过在 PD-Disaggregated 推理架构中引入 **storage-to-decode** 双路径 KV-Cache 加载，将原本闲置的 decode 引擎存储网卡带宽利用起来，消除了 prefill 引擎的单侧 SNIC 饱和瓶颈；配合 CNIC 中心流量隔离和自适应调度，在三个模型（DS 27B/660B、Qwen 32B）的 agentic 工作负载下实现了离线推理吞吐 **1.87×**、在线服务吞吐 **1.96×** 的提升。

![](DualPath_fig1.png)

> **Figure 1**: 左侧为现有 PD-Disaggregated 架构中的瓶颈——prefill 引擎的存储网卡（SNIC）带宽饱和，而 decode 引擎 SNIC 闲置；右侧为 DualPath 的双路径加载方案，通过 storage-to-decode 路径将 KV-Cache 先加载到 decode 引擎，再经 RDMA 计算网络转发到 prefill 引擎，实现全局带宽池化。

---

## 1. 研究背景与动机

### 1.1 问题定义

LLM 推理正从单轮交互（单轮聊天/推理）快速演进到 **Agentic 多轮交互范式**：模型通过多轮对话与外部环境（浏览器、代码解释器、数据库等）交互，逐轮累积上下文。每轮交互仅追加少量新 token（通常数百），但上下文可能累积到数十万甚至上百万 token。

这使得推理负载特征发生根本变化：

- **高 KV-Cache 命中率**：典型 agentic 场景下，$\geq 95\%$ 的 token 可命中 KV-Cache，只有新追加的 token 需要预填充计算
- **极低的 cache-compute ratio**：IO 负载远高于计算负载。以 DeepSeek-V3.2 在 32.7k 上下文、平均追加 429 token 的场景为例，cache-compute ratio（kv-cache 加载量 / 计算量）约为 **22 GB/PFLOP**；对于未做 MLA 优化的模型（如 Qwen2.5-32B），该比例高达 **117–267 GB/PFLOP**
- **计算网络与存储网络隔离**：现代 DGX SuperPOD 中，每台节点有 8 块 GPU，每个 GPU 绑一个 400Gbps 计算 NIC（CNIC，东西向），另有一个独立的 400Gbps 存储 NIC（SNIC，南北向）

### 1.2 现有方法的不足

现代推理系统普遍采用三层架构：

1. **PD-Disaggregation**（Prefill-Decode 分离）—— prefill 和 decode 分别由独立 GPU 处理
2. **Layerwise Prefill**（逐层预填充）—— 每层只加载该层的 KV-Cache，突破 HBM 容量限制
3. **外部分布式存储**（如 3FS）—— 持久化 KV-Cache 以支持多轮复用

这三层架构的组合带来了一个严重但此前未被充分关注的问题：**存储网络带宽利用极度不对称**。在现有设计中，prefill 引擎必须从远程存储加载全部 KV-Cache，导致 prefill 侧的 SNIC 持续饱和；而 decode 引擎由于只负责生成 token 并将新 KV-Cache 写入存储，其 SNIC 大部分时间处于闲置状态。

**简单增加 prefill 侧带宽代价极高**，且在通用集群中往往不可行。因此需要将 decode 引擎闲置的 SNIC 带宽利用起来。

### 1.3 本文核心思路

DualPath 的核心洞察是：**KV-Cache 加载不必局限于 prefill 引擎**。论文提出双路径加载：

- **PE Read Path（传统）**：storage → prefill engine HBM → decode engine buffer
- **DE Read Path（新型）**：storage → decode engine buffer → prefill engine HBM (via RDMA over CNIC)

通过动态选择两条路径，系统将所有引擎（prefill + decode）的 SNIC 带宽聚合为全局可调度资源，同时利用 CNIC 的计算网络传输能力来弥补 decode 到 prefill 的转发开销。

---

## 2. 为什么已有方法不够？

| 方法 | 核心思路 | 不足 |
|---|---|---|
| **Mooncake** (Qin et al., 2025) | 分布式 DRAM 缓存池 + affinity-aware 调度 | 仅适用于内存富裕场景；RL rollout 阶段 DRAM 被训练状态占用，无法缓存大量 KV-Cache；DRAM 成本远高于 SSD |
| **HCache / CachedAttention** (Gao et al., 2024/2025) | 减少 KV-Cache 数据量 | 未解决引擎间 SNIC 带宽不均衡的根本问题 |
| **TARDIS / Phoenix** (Hu et al., 2025; Yan et al., 2025) | 降低检索开销 | 同上，仍是单一路径优化 |
| **KVPR / TailorKV** (Jiang et al., 2025; Yao et al., 2025) | 重计算重叠 + 混合量化 | 减轻 PCIe 带宽压力，未触及跨引擎 SNIC 不均衡 |

DualPath 与这些方法正交：即使叠加重计算或量化技术，双路径加载仍可进一步释放瓶颈。

---

## 3. DualPath 系统架构

### 3.1 核心组件

- **Inference Engines**：每块 GPU 为一个引擎，分为 Prefill Engines (PEs) 和 Decode Engines (DEs)
- **Traffic Manager (§5)**：负责 (1) Host↔Device (H2D/D2H)、(2) PE↔DE KV-Cache 传输、(3) 通过 SNIC 读写存储
- **Request Scheduler (§6)**：中央调度器，负责将请求分配到 (PE, DE) 对，并动态选择加载路径

### 3.2 双路径加载详解

DualPath 为每个 PE 和 DE 分配了少量 DRAM 作为 **Buffer**（PE Buffer / DE Buffer）。

#### PE Read Path（图 4(a)）

1. 将命中 token 的 KV-Cache 从存储读入 **PE Buffer**（Label 1-2）
2. 在 attention 层计算前，将该层 KV-Cache 转入 **PE HBM**（3-4）
3. 计算完成后，将该层完整 KV-Cache（命中 + 新计算）转入 **DE Buffer**（5-7）
4. 上述过程逐层重复 $n_{layer}$ 次，传输与计算重叠

#### DE Read Path（图 4(b)）

1. 将命中 token 的 KV-Cache 从存储读入 **DE Buffer**（Label 1-2）
2. PE 预填充时，从 DE Buffer 经 RDMA/PCIe 读取该层 KV-Cache 到 PE HBM（3-5），与计算重叠
3. 该层计算完成后，仅将新追加 token 的 KV-Cache（miss token）转回 DE Buffer，与已有命中 KV-Cache 合并
4. 重复 $n_{layer}$ 次

#### Decode Phase

PE 完成预填充后，final prompt KV-Cache（包括通过 PE path 加载的 + 通过 DE path 加载的）在 DE Buffer 中准备就绪：

- DE 分配 HBM，执行 H2D 传输（Label 8-9 in Fig 4(a)；6-7 in Fig 4(b)）
- 释放 CPU 内存，开始 decode 自回归生成
- 每当生成满一个 block（如 64 token），立即持久化到磁盘

#### Block Layout 设计

- **Full Block**：包含所有层的 KV-Cache，用于**与存储交互**（读/写）
- **Layer Block**：仅包含一层的 KV-Cache，用于**引擎间逐层流传输**

这种分层设计避免了手动内存布局转换：$n$ 个 Layer Block 拼接即为一个 Full Block。

![](DualPath_fig4.png)

> **Figure 4**: 双路径加载数据流示意。(a) PE Read Path：KV-Cache 从存储 → PE Buffer → PE HBM → DE Buffer；(b) DE Read Path：KV-Cache 从存储 → DE Buffer → PE HBM，仅 miss token 的 KV-Cache 返回 DE Buffer。关键差异在于前者需要完整 KV-Cache 从 PE 传 DE，后者 DE已持有命中 KV-Cache，只需回传 miss token 的 KV-Cache。

### 3.3 Bottleneck-Free 理论分析

论文给出了在小众常见 P/D 比例下系统不会引入 CNIC 或 DRAM 瓶颈的**理论边界**。

**符号定义**：

| 符号 | 含义 |
|---|---|
| $P$ | prefill 节点数 |
| $D$ | decode 节点数 |
| $g$ | 每节点 GPU 数 |
| $B$ | 单个计算 NIC 带宽 |
| $s \cdot B$ | 每节点存储 NIC 总带宽（$s \leq g$） |
| $M$ | 每节点内存带宽 |
| $n_{layer}$ | 模型层数 |

**PE CNIC 读方向**（所有对之间总流量）：

$$
2 \times T_p \times Dg = 2Bs/g \leq B \quad \text{(因 } s \leq g \text{)}
$$

**PE CNIC 写方向**：

$$
(T_p + T_c) \times Dg = Bs/g \times (1 + D/P) \leq B
$$

推导得约束：

$$
P/D \geq \frac{s}{g - s}
$$

**DE CNIC 读方向**：

$$
(T_p + T_c \times 2) \times Pg = s/g \times (P/D + 2) \times B \leq B
\Rightarrow P/D \leq \frac{g - 2s}{s}
$$

**DE CNIC 写方向**：

$$
P/D \leq \frac{g - s}{2s}
$$

**DE DRAM 压力**（半双工求和）：

$$
(3 + 2P/D)Bs \leq M \Rightarrow P/D \leq \frac{M/Bs - 3}{2}
$$

综合 bounds：

$$
\frac{s}{g - s} \leq \frac{P}{D} \leq \min\left\{ \frac{g - 2s}{s}, \frac{g - s}{2s}, \frac{M/Bs - 3}{2} \right\}
$$

代入实际值 $(g=8, s=1, M \approx 500 \text{ GB/s}, Bs \approx 50 \text{ GB/s})$：

$$
\frac{1}{7} \leq P/D \leq \frac{7}{2}
$$

**覆盖绝大多数实际部署配置。** 这是 DualPath 的根本设计保证：在常见 P/D 比下引入 DE Read Path 不会引入计算网络或内存瓶颈。

*斜体注：论文未披露当 P/D 超出此范围时的降级策略（如自动回退到纯 PE Read Path）。*

---

## 4. CNIC-Centric 流量管理

### 4.1 为什么需要流量隔离？

直接引入 DE Read Path 会在计算网络（CNIC）和 PCIe 上增加 KV-Cache 传输流量。模型推理中存在大量**亚毫秒级突发集体通信**（如 Expert Parallelism 的 AllToAll、Tensor/Context Parallel 的 ReduceScatter/AllGather），这些通信对延迟极度敏感。

现有 GPU 数据传输方案的问题：

- **GPUDirect Storage**：直接从存储 backend 读入 GPU HBM，不经过 CNIC QoS
- **CUDA Copy Engine**：直接通过 PCIe 复制 host↔device，同样绕过计算网络 QoS
- **现有 GPU 不支持 PCIe QoS**（Richter et al., 2016），无法通过软件流量整形来隔离模型推理和 KV-Cache 传输

### 4.2 核心设计：所有 GPU 流量经 CNIC

**DualPath 要求所有进出 GPU 的数据——包括本地 H2D/D2H copy——都必须经过 GPU 绑定的 CNIC，使用 GPUDirect RDMA 数据路径。**

具体来说：
- 读 KV-Cache 时：先从存储 backend 读入 host DRAM，然后通过 RDMA Write 由 GPU 的 CNIC 完成本地 H2D copy
- 写 KV-Cache 时：先由 CNIC 将数据从 GPU HBM 写到 host DRAM，再经 SNIC 持久化到存储

这个"绕路"是目前 **唯一能在生产环境中确保 KV-Cache 流量不劣化关键推理通信** 的实用方法。

### 4.3 流量隔离实现

**InfiniBand**：利用 Virtual Lanes (VLs) 做严格隔离。
- 模型推理通信 → **高优先级 VL**（分配 ~99% 带宽）
- KV-Cache 传输 → **低优先级 VL**（保留 ~1% 防饿死）

论文配置：
- `qos_max_vls = 4`
- `qos_high_limit = 240`
- 高优先级 WRR: `0:192, 1:192, 2:0, 3:192`
- 低优先级 WRR: `0:192, 1:192, 2:64, 3:192`

**RoCE v2 / 未来互联**：
- 通过 Traffic Class (TC) + DSCP 标记配合硬件队列实现等效隔离
- UnifiedBus、Ultra Ethernet 的 QoS 机制天然兼容此设计

### 4.4 CNIC-Assisted Copy 的性能优势

论文测试发现，**CNIC-assisted H2D/D2H 在处理大量小数据块时优于 CUDA copy engine**：

- `cudaMemcpyAsync` 单次提交延迟开销：**5–7 μs**
- RDMA Write work request（用户态 mmio 写 NIC 寄存器）：**~1 μs**
- 利用 **doorbell batching**（Kalia et al., 2016）进一步摊销提交开销

*斜体注：CUDA driver 为闭源，无法进一步拆解 5–7 μs 的开销组成。*

---

## 5. 自适应请求调度器

调度分为两级：

1. **Inter-Engine Scheduling**：将请求分配给 (PE, DE) 对，并选择 Read Path（PE or DE）
2. **Intra-Engine Scheduling**（仅 PE）：决定哪些请求进入当前 forward batch

### 5.1 跨引擎调度（Inter-Engine Scheduling）

#### 引擎组织

- 引擎按组管理（group），每组只有一个 **Leader Engine**（rank 0）与中央调度器交互
- 同节点上所有引擎保证在同一组
- 每组引擎全为 PEs 或全为 DEs

#### PE 调度（Algorithm 1）

每个引擎 $e$ 定期上报：
- $seq_e$：已分配但未完成的请求数
- $tok_e$：这些请求的总 token 数（作为 GPU + NIC + 磁盘负载的代理）
- $read\_q_{n(e)}$：节点 $n(e)$ 的磁盘读取队列长度

**引擎分类**（给定阈值 $\alpha$ = 磁盘队列短阈值，$\beta$ = 未完成 token 上限）：

| 类别 | 条件 | 调度策略 |
|---|---|---|
| $C_1$（过载） | $tok_e > \beta$ | 不分配新请求 |
| $C_2$（低磁盘负载） | $read\_q_{n(e)} \leq \alpha \land tok_e \leq \beta$ | **优先分配**（避免 SNIC 闲置） |
| $C_3$（高磁盘负载） | $read\_q_{n(e)} > \alpha \land tok_e \leq \beta$ | $C_2$ 为空时分配 |

在可用类别内，选择 $\arg\min tok_e$ 的引擎，以实现 token 数量均衡。

#### DE 调度（两阶段）

**Phase 1（跨组）**：有全局等待队列 + 每组私有队列。向 token 总量最小的组分配请求。

**Phase 2（组内）**：
1. 计算组内所有 DE 剩余 HBM 的总和，从私有队列头部遍历，确定可调度请求数上限（假设无 HBM 碎片）
2. 计算 token 高阈值：$Z = 1.05 \times (\sum_{r \in R} len_r + \sum_{e \in E} tok_e) / |E|$
3. 遍历私有队列，在剩余 HBM 充足的 DE 中：
   - 优先分配给 **低 token DE**（$tok_e + len(r) \leq Z$），以平衡 GPU/NIC 压力；内部分配选 $seq_e$ 最小的，平衡请求数
   - 否则分配给 **高 token DE** 中 $tok_e$ 最小的，减少 HBM 耗尽和抢占风险

#### KV-Cache Read 路径选择

选择 **磁盘读取队列更短的一侧** 读 KV-Cache。论文指出未来可将一个请求拆分为两部分分别从 PE 和 DE 读取。

### 5.2 引擎内调度（Intra-Engine Scheduling）

仅 PE 需要引擎内调度。DE 将所有分配到的请求都放入 forward batch。

在 Data Parallelism 配置下（如 MLA 模型），每个 GPU 服务不同请求子集。Attention 层所有 GPU 必须同步后才能进入 FFN 层，因此需要**最小化 GPU bubble**。

**Compute-Quota-Based 批处理**：
- 每个请求用二元组 $(cached, bsz)$ 描述：$cached$ = 已有 KV-Cache 的 token 数；$bsz$ = 本次需计算的 token 数
- 从理论计算量预测 attention 层执行时间（通过提前 profiling 建立回归关系）
- 按 FIFO 顺序加入请求，只要预测 attention 时间不超过 **compute quota 阈值（默认 300ms）**
- 若加入请求会超出 quota，对该请求做 **binary search 缩小 $bsz$**，进行 chunked prefill

![](DualPath_fig6.png)

> **Figure 6**: 左：基于 compute quota 的 batch 选择；右：应用 compute quota 前后 GPU timeline 对比。通过将 attention 执行时间拉平，显著减少了 GPU 等待 bubble。

---

## 6. 实验结果

### 6.1 实验设置

**测试集群**：
- 每节点 8 × NVIDIA Hopper GPUs，双路 CPU
- 8 × 400Gbps RDMA NIC（CNIC）+ 1 × 400Gbps 存储 NIC（SNIC）
- InfiniBand 互联，计算网络与存储网络物理隔离
- 分布式存储 3FS（无内部 DRAM cache，可饱和 SNIC 400Gbps）

**模型**：

| 模型 | 架构 | KV-Cache 优化 | 并行策略 |
|---|---|---|---|
| DeepSeek V3.2 660B | MoE + Sparse Attention (MLA) | MLA 压缩 | EP + DP |
| DS 27B（内部实验模型）| MoE + Sparse Attention | MLA 压缩 | EP + DP |
| Qwen2.5-32B | Dense + GQA | GQA（比 MLA 大） | DP only |

**数据集**：从生产 agentic RL 训练工作负载收集的 3 个 trace 数据集，各 500 条轨迹：

| MaxLen | 平均轮数 | Append | Gen | Total | Context |
|---|---|---|---|---|---|
| 32K | 60 | 608 | 148 | 28,639 | 17,183 |
| 48K | 106 | 474 | 172 | 42,607 | 25,120 |
| 64K | 157 | 429 | 176 | 55,958 | 32,721 |

- Append / Gen：每轮平均追加 / 生成 token 数
- Context：所有轮中平均上下文 length
- **KV-Cache 命中率约 98.7%**（64K 场景）

**Baselines**：
- **SGL(MC)**：SGLang + HiCache + Mooncake Store + Mooncake Transfer Engine
- **Basic**：无修改的内部推理框架
- **Oracle**：绕过所有磁盘读、D2H/H2D 和跨 PD KV-Cache 传输（理论上限）

**P/D 比例**：DS 660B 默认 2P4D；Qwen 32B 默认 1P2D；DS 27B 默认 1P1D。

**SLO（在线服务）**：TTFT $\leq$ 4s，TPOT $\leq$ 50ms。

### 6.2 主实验结果

#### 离线推理（Batch）

![](DualPath_fig7.png)

> **Figure 7**: 离线推理 Job Completion Time (JCT) 随 agent 数量和 Max Agent Length 的变化。上：DS 27B；中：DS 660B；下：Qwen 32B。SGL(MC) 在部分大配置上未跑通（N/A）。

**核心数据**：
- **DS 660B**：DualPath 相比 Basic 最高提升 **1.87×**，非常接近 Oracle 性能
- **DS 27B**：最高提升 **1.78×**，但与 Oracle 仍有 1.09–1.85× 差距（1P1D 的 SNIC 带宽不足）
- **Qwen 32B**：趋势与 DS 27B 类似
- 随着 batch size 和 MaxLen 增大，DualPath 收益递增（IO 占比更高）

#### 在线服务

![](DualPath_fig10.png)

> **Figure 10**: 在线服务延迟指标随 agent arrival rate (APS) 的变化。上图：DS 27B；下图：DS 660B。阴影区表示实验结束前 150s 的波动范围。超出 SLO 阈值的数据点被省略。

| 模型 | DualPath APS 容量 vs Basic | TTST | TPOT |
|---|---|---|---|
| DS 27B | **1.67×** | 与 Basic 持平 | 与 Basic 持平 |
| DS 660B | **2.25×** | 与 Basic 持平 | 与 Basic 持平 |

- DualPath 的 TTST（第二 token 延迟）与 Basic 持平，说明没有在 decode 阶段引入额外开销
- TPOT 也与 Basic 持平，确认 CNIC 流量隔离有效
- SGL(MC) TTST 异常偏低（可能是实现 bug，前两 token 几乎同时到达）

#### 不同 P/D 比例验证

![](DualPath_fig8.png)

> **Figure 8**: DS 27B 离线推理在不同 Prefill-Decode 比例下的表现（64K context）。

| 配置 | 对比 | 本质 |
|---|---|---|
| Basic 1P1D ≈ Basic 1P2D | Basic 只用 PE 的 SNIC | 存储带宽相同 |
| DualPath 1P1D ≈ Basic 2P1D | DualPath 用所有节点 SNIC ≈ Basic 用两 PE 节点 SNIC | **存储带宽相同** |
| DualPath 2P1D ≈ DualPath 1P2D | DualPath 聚合所有节点带宽 | **存储带宽相同** |
| DualPath 平均加速 | 1.64×（最高 2.46×） |

这验证了 **存储带宽是真正的瓶颈**：只要系统可用的 SNIC 总带宽相同，性能就大致相同。

#### Append/Generation Length 变化

![](DualPath_fig9.png)

> **Figure 9**: 左：变化 Append Length（DS 660B, 64K, 1024 agents）；右：变化 Generation Length。随着 Append length 增加（计算压力上升），Basic 性能逐渐逼近 DualPath 和 Oracle，说明瓶颈从 IO 转向计算。不同 append scale 下 DualPath 相对 Basic 提升 **1.82–1.99×**。

### 6.3 消融实验

![](DualPath_fig12.png)

> **Figure 12 (右)**：离线推理消融实验（DS 660B, 64K MAL）。

| 技术组件 | 相比 Basic 的 JCT 降低 | 说明 |
|---|---|---|
| + Layerwise Prefill | **17.21%** | 缓解 PE HBM 瓶颈，隐藏传输开销 |
| + Dual-Path Loading | **38.19%** | 核心增益，聚合了分布式存储带宽 |
| + Scheduling Algorithm | **45.62%** | 在双路径基础上做负载均衡，最大化 NIC 利用率 |

#### 负载均衡效果

- **SNIC 流量**：Max/Avg 比率从 Round-Robin 的 **1.53** 改善到调度的 **1.18**（图 13）
- **Attention 执行时间**：Max/Avg 比率低至 **1.06**，前 5% 的任务过程中 GPU 气泡最小化（图 14）

### 6.4 大规模实验

| 场景 | 配置 | 规模 | 结果 |
|---|---|---|---|
| 离线推理 | 2P4D→48P96D | 1,152 GPUs, 48K agents | JCT: 3,167s → 3,201s（**近乎线性扩展**） |
| 在线服务 | 2P4D→44P88D | 1,152 GPUs | APS: 0.4 → **8.8**（**22× 吞吐**，延迟基本持平） |

调度器 CPU 占用始终低于 10 核，不构成瓶颈。

*斜体注：大规模实验未经过 P/D 比例和并行策略的精细调优，因此未在小规模基础上进一步降低 JCT 或提升 APS；但大规模部署在碎片化减少和突发请求调度上有潜力。*

---

## 7. 讨论与局限

### 7.1 Working Set 与存储成本分析

在线服务中，KV-Cache 的 working set 可估算为：

$$
W \approx \lambda \cdot \bar{T} \cdot total\_len_{avg} / 2
$$

其中 $\lambda$ = agent 到达率 (APS)，$\bar{T}$ = 平均 JCT。

以 DS 660B 为例：
- $\lambda = 0.1$ APS 时，$W \approx 69$ GB
- $\lambda = 0.45$ APS 时，$W \approx 681$ GB

实际生产环境中，由于工具调用延迟和到达间隔不为零，$\bar{T}$ 会进一步增大。若 JCT 增加 $r$ 倍：
- APS 容量增加 $r$ 倍（空闲时间不占用推理资源）
- Working set 扩大 $r^2$ 倍
- 所需存储成本按 $r^3$ 增长

这意味着 agentic 部署的**存储成本可能随吞吐非线性膨胀**，在当前设定下已接近可用存储边界。

### 7.2 论文自我分析的局限

1. **P/D 比例与并行策略调优依赖先验**：离线推理负载高度动态（如 RL rollout 前半段 prefill 压力大，后半段轻），当前采用固定 P/D 比。作者建议未来使用**模拟器或在线自适应调整**。
2. **调度器 TTFT 大比例场景仍有优化空间**：在超大规模下 TTFT 百分位还有提升余地。
3. **小模型 P-D transfer 开销显著**：DS 27B 的 TPOT 显著高于 Oracle，说明基础 PE→DE 传输本身对小模型影响大。

### 7.3 DualPath 的深层局限

- **DE Read Path 的额外 H2D**：为保证 TTFT 占比不至于过高，DE Buffer 在 decode 阶段需要额外的 H2D 传输；如果 generation 长度更长，这部分开销占比会减小。论文在典型 agentic 场景（gen short）下可接受。
- **对物理网络拓扑的假设**：Bottleneck-free 分析假设 PCIe switch 合理配对、计算网络无拥塞；在非标准拓扑中受益可能下降。
- **跨路径分片请求未实现**：当前一个请求只从一条路径读 KV-Cache，细粒度分片理论上可进一步均衡负载。
- **无 DRAM cache**：DualPath 可以与 DRAM 缓存（如 Mooncake）正交结合，但论文指出增益有限——因为当已经消除了 CPU/PCIe/存储带宽瓶颈后，DRAM 加速的边际收益递减。

---

## 8. 与相关工作的关联

| 工作 | 关联点 | 差异 |
|---|---|---|
| [[Mooncake]] (Qin et al., 2025) | 分布式 KV-Cache 存储 | Mooncake 用 DRAM 缓存 + affinity 调度；DualPath 直接走 SSD，利用 decode SNIC 带宽，大幅降低 DRAM 需求 |
| [[TokenLake]] (Wu et al., 2025) | Segment-level prefix cache pool | TokenLake 细粒度缓存复用；DualPath 解决的是缓存加载带宽不均衡问题 |
| [[LayerKV]] / PrefillOnly | Layerwise prefill | DualPath 复用了该范式作为基础组件 |
| [[Strata]] (Xie et al., 2025) | 分层存储 + cache-aware 调度 | Strata 侧重层次化缓存；DualPath 重新设计了跨引擎的数据路径 |
| [[DistServe]] / Splitwise | PD-Disaggregation | DualPath 在不改变 PD 分离架构的前提下，从数据路径角度解决 SNIC 不均衡 |

---

## 9. 核心引用

```bibtex
@article{wu2026dualpath,
  title={DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference},
  author={Wu, Yongtong and Chen, Shaoyuan and Zhong, Yinmin and Huang, Rilin and Tan, Yixuan and Zhang, Wentao and Zhang, Liyue and Zhou, Shangyan and Liu, Yuxuan and Zhou, Shunfeng and Zhang, Mingxing and Jin, Xin and Huang, Panpan},
  journal={arXiv preprint arXiv:2602.21548},
  year={2026},
  url={https://arxiv.org/abs/2602.21548}
}
```

---

## 10. 关键图片汇总

| 图号 | 文件名 | 说明 |
|---|---|---|
| Fig 1 | `DualPath_fig1.png` | 现有瓶颈 vs DualPath 架构对比（P0 必须） |
| Fig 3 | `DualPath_fig3.png` | 硬件趋势 + batch size 对吞吐的影响 |
| Fig 4 | `DualPath_fig4.png` | 双路径加载数据流（P0 必须） |
| Fig 5 | `DualPath_fig5.png` | 跨引擎 PE 调度示意 |
| Fig 6 | `DualPath_fig6.png` | 引擎内 compute-quota 调度 |
| Fig 7 | `DualPath_fig7.png` | 离线推理主实验结果（P0 必须） |
| Fig 8 | `DualPath_fig8.png` | P/D 比例影响 |
| Fig 9 | `DualPath_fig9.png` | Append/Generation length 变化 |
| Fig 10 | `DualPath_fig10.png` | 在线服务延迟（P0 必须） |
| Fig 12 | `DualPath_fig12.png` | TTFT 分解 + 消融实验（P1 推荐） |
| Fig 13 | `DualPath_fig13.png` | SNIC 负载均衡效果 |

---

## 11. 个人思考

### 11.1 方法的优雅之处

- **问题定位精准**：没有试图从头设计一套推理系统，而是在已成为事实标准的 PD-Disaggregation + Layerwise Prefill + External Storage 架构上，发现了一个被忽略的**资源结构性浪费**（decode SNIC 闲置），并用最小的架构改动解决它。
- **工程权衡清晰**：选择绕过 CNIC 而不是绕过存储网络，不是性能最优但 QoS 最可控；所有技术决策都有"Why A not B"的明确论证（如 §5.2 中 GPUDirect Storage vs CNIC-assisted copy 的对比）。
- **理论-实践闭环**：第四章的理论分析给出了 bottleneck-free 的 P/D 比范围，实验验证了在此范围内性能确实符合预期。

### 11.2 最值得关注的数字

1. **1.87× / 1.96×**：离线/在线吞吐提升。考虑到这是一个纯系统层优化、不改模型、不改训练，**系统层 2× 提升尤为难得**。
2. **45.62%**（消融中 DualPath loading + scheduling 的 JCT 降低）：确认了存储带宽不均衡是核心瓶颈，而非计算或内存的局部问题。
3. **$\frac{1}{7} \leq P/D \leq \frac{7}{2}$**：bottleneck-free 边界。当前部署基本都落在此区间内。
4. **从 Ampere→Blackwell，IO-compute ratio 下降 14.4×**：这是驱动整个问题的硬件根本趋势。随着 GPU FLOPS 继续增长远快于网络带宽，IO 瓶颈会越来越尖锐。

### 11.3 对后续工作的启发

- **Agentic 推理的存储经济学**：§7.2 的 working set 分析指出，在线吞吐增加 $r$ 倍时存储成本按 $r^3$ 膨胀。这意味着未来 agentic 推理成本的真正想象力可能在存储侧，而不只是 GPU-hour。KV-Cache 压缩、稀疏化、分层冷温热存储将是极具价值的方向。
- **CNIC 作为系统瓶颈组件**：论文将 CNIC 从" mere network interface "升级为 GPU 所有流量的 QoS 仲裁者。这暗示未来推理系统的设计应以网络 QoS 为 first-class citizen。
- **RL Rollout 场景的适配性**：论文多次提到 RL rollout 阶段 DRAM 被训练状态占用、必须依赖外部存储，这使得 DualPath 的设计在训练场景中同样适用。

---

## 附录：论文未披露的关键信息

| 参数类别 | 缺失信息 | 论文是否提及但未给数字 |
|---|---|---|
| 代码开源 | DualPath 本身代码未开源 | 否，仅基于内部框架改造 5K 行 |
| DE Buffer 大小 | 每个 DE 分配的 DRAM buffer 精确大小 | 提及"a small amount of DRAM"，无具体数值 |
| Layer Block size | tokens 维度（`block_size`）取值 | 提及"64 tokens"（持久化 block），但未说明传输 block |
| 阈值调参 | $\alpha$、$\beta$、compute quota 的绝对数值公式 | $\alpha$ = 3 秒可读的 token 数；$\beta$ = 5 秒处理量；quota = 300ms；给出了相对定义 |
| InfiniBand QoS 延迟 | 高低优先级 VL 的具体延迟差异 | 否 |
| 传输协议 | PE↔DE 间 RDMA 的具体协议（IB Verbs / RC / UD） | 否 |
| 降级策略 | 当 P/D 超出 bottleneck-free 范围时的行为 | 否 |
| 小模型 overhead**根源** | DS 27B 的 P-D transfer 开销相对 Oracle 为何显著 | 指出"leave as future work" |
