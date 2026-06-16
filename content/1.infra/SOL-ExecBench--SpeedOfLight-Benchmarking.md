---
created: 2026-06-12
published: 2026-03-19
paper: https://arxiv.org/abs/2603.19173
code: https://github.com/NVIDIA/SOL-ExecBench
authors: Edward Lin, Sahil Modi, Siva Kumar Sastry Hari, Qijing Huang, Zhifan Ye, Nestor Qin, Fengzhe Zhou, Yuan Zhang, Jingquan Wang, Sana Damani, Dheeraj Peri, Ouye Xie, Aditya Kane, et al. (NVIDIA)
tags:
  - GPU内核优化
  - 基准测试
  - CUDA
  - Blackwell
  - 性能建模
  - AI编译器
---

# SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits

## 一句话总结
NVIDIA 提出了 SOL-ExecBench——一个包含从 124 个前沿 AI 模型提取的 235 个真实 GPU 内核优化问题的基准测试，通过 SOLAR 管道推导硬件级的 Speed-of-Light (SOL) 性能上界，并以 SOL Score 衡量优化方案相对硬件极限的逼近程度；实验发现当前 agent 优化基线的中位数 SOL Score 仅为 0.732，说明 GPU 内核优化仍有大量未被开采的硬件潜力。

![](SOLExecBench_fig1_overview.png)
> **Figure 1**: SOL-ExecBench 的整体架构与问题覆盖范围。覆盖六类 AI 架构（LLM、Diffusion、Vision、Audio、Video、Multimodal），共 235 个内核优化问题，通过 SOLAR 管道计算硬件 SOL 上界，以 SOL Score 评估优化进度。

---

## 1. 研究背景与动机

### 1.1 问题定义
AI 模型的计算效率越来越受限于底层 GPU 内核的执行效率。现有基准测试通常用“相对于软件基线的加速比”来衡量成功，但**真正的优化目标应该是逼近硬件极限（Speed-of-Light, SOL）**。随着每一代 GPU（如 Blackwell）引入新的性能特性，软件基线（如 PyTorch eager）与硬件潜能之间的差距持续扩大，传统的“相对加速比”度量越来越失真。

### 1.2 现有方法的不足
1. **基线依赖偏差**：KernelBench、BackendBench 等现有基准以 PyTorch eager/Triton 为基线，一个内核可以比 PyTorch 快 10 倍，却仍距离硬件 SOL 超过 10×。
   - 实证发现：speedup 与 "fraction of SOL headroom reclaimed" 的相关系数仅为 r = 0.10（log-log 尺度），说明两者几乎无关。
   
2. **覆盖范围不足**：现有基准的问题来源偏旧（如 ResNet、BERT、VGG），缺少对 **MoE（DeepSeek-V3）、SSM（Mamba-2）、linear attention（RWKV）、混合架构（Nemotron-H）** 等前沿模型的覆盖。

3. **硬件特性滞后**：新精度格式（FP8、NVFP4）、第 5 代 Tensor Core 等 Blackwell 特有特性无法在旧基准中得到有效测试。

4. **评估基础设施脆弱**：现有基准缺乏对 reward hacking（例如通过 CUDA stream 并发欺骗计时、缓存输出复用、monkey-patch 计时函数等）的系统化防护。

### 1.3 本文核心思路
- 从 124 个真实生产级/前沿模型中提取内核优化问题
- 构建 **SOLAR 管道**（Speed-of-Light Analytical Reasoning），通过 roofline 分析 + einsum 表达推导硬件性能上界
- 提出 **SOL Score**：不是比基线快了多少倍，而是**向硬件极限的 gap 中收复了多少**
- 建立严格的沙箱化评测框架，防御 14.5% 的 agent 提交中发现的 reward hacking

---

## 2. 基准测试构造（Benchmark Construction）

### 2.1 来源模型覆盖（124 个模型，6 大领域）

| 领域 | 代表模型 | 数量 |
|------|---------|------|
| LLM | Llama-3.x, Gemma-3, Phi-4, DeepSeek-V3/R1, Qwen3-Coder-480B, GLM-4.7, Kimi-K2 | 61 |
| Diffusion | SD 系列, FLUX.1/2, HunyuanImage, Sana, HiDream, LLaDA-8B | 24 |
| Vision | SAM-HQ, ConvNextV2, VMamba, NAFNet, Swin2SR, MaskGIT | 6 |
| Audio/Speech | Whisper, Parakeet-TDT, Canary, Voxtral, OpenVoice, Kokoro, XTTS-v2 | 9 |
| Video | Wan2.2-T2V | 2 |
| Multimodal/Hybrid | Qwen3-VL, Llama-3.2-Vision, Gemma-3n, Jamba, Nemotron-H, RWKV-v7 | 22 |

### 2.2 提取管道（四阶段）

1. **模型准备**：加载架构定义，提取带配置常量的源代码
2. **子图提取**：前沿 LLM 自动识别计算子图，生成独立 PyTorch 实现 → **7,400 个子图**
3. **筛选与采样**：基于 11 个维度（算子类型、精度、复杂度、模型域等）进行**分层采样**，确保覆盖均衡
4. **验证**：三层验证
   - (a) 人类专家 + LLM 审查规格说明
   - (b) 执行级正确性检查（多组输入对比输出）
   - (c) Agentic 优化器压力测试，暴露规格漏洞
   
最终产出：**245 个有效问题**（235 公开 + 10 保留）

### 2.3 问题分类（四级体系）

| 类别 | 说明 | 数量 | 精度 | 示例 |
|------|------|------|------|------|
| L1 | 单算子内核 | 94 | BF16/FP32 | GQA, RMSNorm, SwiGLU, RoPE |
| L2 | 多算子融合内核（复杂度 3–10×） | 82 | BF16/FP32 | Decoder layer, MoE dispatch, SSM chunk scan |
| Quant | 低精度内核 | 33 | FP8/NVFP4 | FP8 MLA projection, NVFP4 MoE expert |
| FIB | FlashInfer-Bench 推理原语 | 26 | BF16/FP8 | Fused attention, FP8 MoE, RMSNorm |

**关键统计：**
- 方向：Forward 189（80%），Backward 46（20%）
- 算子类型：Attention 81（35%）、MoE 36（15%）、Normalization 27（12%）、Embedding 20（9%）、Linear 16（7%）
- 精度：BF16 107（46%）、FP32 79（34%）、FP8 19（8%）、NVFP4 15（6%）、FP16 12（5%）、Mixed 3（1%）
- 78 个问题（33%）使用自定义输入生成器处理结构化输入

### 2.4 问题规格格式

每个问题含三组件（扩展 FlashInfer Trace schema）：

1. **Definition**: 问题名称、操作类型、符号轴（symbolic axes）、张量形状/dtype、参考实现
2. **Reference**: 自包含 PyTorch 代码（含 `run()` 函数，`get_inputs()` 用于结构化输入）
3. **Workloads**: 每个问题含约 16 组动态 shape 实例（batch ∈ {1,...,64}, seq_len ∈ {128,...,8192}）

---

## 3. SOL 上界推导与评测指标

### 3.1 SOLAR 管道（Speed-of-Light Analytical Reasoning）

SOLAR 是推导硬件级性能上界的自动分析管道，由三个阶段组成：

![](SOLExecBench_fig3_solar.png)
> **Figure 3**: SOLAR 管道三阶段：Graph Extractor（图提取）→ Agentic Einsum Converter（算子转 einsum）→ SOL Analyzer（roofline 分析）。图中以 Jamba-Reasoning-3B 的 fused attention output projection 为例，展示从 PyTorch 代码到 SOL 上界的完整推导流程。

#### Stage 1: Graph Extractor
- 基于 **torchview** 库
- 通过 forward hook 在实时前向传播中追踪 PyTorch 模型，生成算子图
- 捕获数据流、类型和张量形状

#### Stage 2: Agentic Einsum Converter
- 将 PyTorch 算子翻译成**扩展 einsum 表达式**（广义爱因斯坦求和，用索引记号表示任意张量计算）
- 维护持久化的 PyTorch ops → validated einsum conversions 查找表
- 遇到未见的算子时，LLM agent 自动生成并验证候选转换函数

#### Stage 3: SOL Analyzer
基于 roofline 模型计算理论最小运行时间：

$$T_{SOL} = \max\left( \frac{\text{Total FLOPs}}{\text{Compute Throughput}}, \quad \frac{\text{Total Fused Bytes}}{\text{Memory Bandwidth}} \right)$$

**支持特性：**
- 图级融合优化（graph-level fusion）的数据移动减量
- prefetch 优化
- **Orojenesis 扩展**（Huang et al., 2024）：根据 on-chip buffer 容量建模更紧致的 off-chip 数据移动下界

**示例**（Jamba-Reasoning-3B 的 fused attention output projection + residual add）：
- 操作：Matmul (16,512,2560) × (2560,2560) + elementwise add
- Total FLOPs: 107.4 G
- Fused memory: 126 MB
- Arithmetic intensity: 427
- Bottleneck: compute-bound
- SOL runtime on B200@1.5GHz: **0.059 ms**

### 3.2 SOL Score 指标设计

SOL Score 是本文的核心创新指标，量化优化方案向硬件极限逼近的程度。

**符号定义：**
- $T_b$：基线运行时间（scoring baseline，由 agent 自动生成）
- $T_{SOL}$：SOLAR 估算的硬件最小运行时间
- $T_k$：候选内核运行时间
- $S(T_k)$：SOL Score

**公式：**

$$S(T_k) = \frac{1}{1 + \frac{T_k - T_{SOL}}{T_b - T_{SOL}}}$$

或等价形式：

$$S(T_k) = \frac{T_b - T_{SOL}}{(T_k - T_{SOL}) + (T_b - T_{SOL})}$$

**锚点解读：**

| $T_k$ 位置 | SOL Score | 含义 |
|-----------|-----------|------|
| $T_k = T_b$ | 0.5 | 与基线持平 |
| $T_k = T_{SOL}$ | 1.0 | 达到硬件极限 |
| $T_k \to \infty$ | $\to 0$ | 极慢 |

**非线性放大效应：** 越接近 SOL，同等绝对耗时降低带来的分数增益越大。这天然鼓励优化者去攻克最难的"最后一公里"问题，而非在远离 SOL 的区域做容易优化。

**正确性约束：** 候选内核必须先通过正确性检查（$C = 1$），否则 $S = 0$。

**总体基准分数：**

$$\bar{S} = \frac{1}{N} \sum_{j=1}^{N} C_j S_j$$

### 3.3 评测框架

**支持语言与工具链：**
Python、Triton、CUDA/C++（含 PTX、CUTLASS、CuTE DSL、cuBLAS、cuDNN、cuTile）

**正确性检查机制：**
1. 物化参考输出
2. 多组 seeded trials 对比
3. 检查 shape/dtype，拒绝 inf/NaN
4. 使用工作负载特定的容差元组 $(\text{atol}, \text{rtol}, \text{matched_ratio})$，并施加 1.25× 安全裕度

**运行时测量：**
- CUDA events，10 warmup + 50 timed iterations × 3 trials
- 每次 timed iteration 前**清空 L2 cache**（256 MB buffer zeroed）
- 每次运行克隆张量参数
- B200 上 GPU clocks **锁定在 1,500 MHz**
- 每个 solution 300 秒超时

### 3.4 Reward Hacking 与防御体系

全部 agent 提交中 **14.5% 被标记为 reward hacking**（共 589 例），分成三大利用族：

| 利用族 | 具体手法 | 检测数量 | 占比 |
|--------|---------|---------|------|
| 精度降级 | FP32 暗地里降级为 FP16/BF16/FP8 | 259 | 6.4% |
| Monkey patching | 修改计时函数、内存分配器等 | 134 | 3.3% |
| Stream 注入 | 多 CUDA stream 并发欺骗计时 | 100 | 2.5% |
| 缓存输出复用 | 按 data_ptr 键值复用上轮输出 | 67 | 1.6% |
| JIT Fork | 子进程编译异步执行 | 少量 | - |
| 一次性正确性 | lazy eval 仅在首轮返回正确结果 | 少量 | - |
| Thread 注入 | 注入额外线程干扰计时 | 少量 | - |

**多层防御机制：**
1. 监视线程数量
2. 禁止多 stream 使用
3. 自定义内存分配器：每次分配将指针偏移 256B，破坏 data_ptr 缓存
4. 严格类型检查 `type(t) is torch.Tensor`
5. 多轮正确性验证（随机化输入）
6. 关键计时函数的内存地址验证
7. **LLM-as-a-judge 静态代码分析**
8. 基线候选的人类专家复审
9. 收紧数值容差

**保守设计**：直接禁用 CUDA streams，依赖 PyTorch 默认内存分配器。

### 3.5 Scoring Baseline 生成

基线不是固定的 PyTorch eager，而是**由多 agent turn-based 优化系统自动生成**：
- 多个 agent 在固定时间/成本预算内独立优化 PyTorch reference
- 每轮结束后，有效方案暴露给下一批 agent
- agent 限制为 PyTorch + 标准 Python 包
- 仅保留通过编译、正确性和 reward-hacking 审查的最快方案
- 每个问题的最快有效内核成为 scoring baseline

---

## 4. 实验结果

### 4.1 实验设置

- **硬件**: NVIDIA DGX B200 节点（8× B200 GPU, 192 GB HBM3e, 8 TB/s 单卡带宽）
- **软件**: CUDA 13.1.1, cuDNN 9.17.1, PyTorch 2.9.0, NVIDIA driver 580.95
- **运行时**: 单 GPU，SM clocks 锁定 1,500 MHz

### 4.2 SOL Score vs. Speedup 的关键对比

**核心发现：speedup 与 SOL headroom reclaimed 几乎无关（r = 0.10）**

![](SOLExecBench_fig4_sol_score.png)
> **Figure 4**: SOL Score 与 speedup 的对比分析。左：speedup 分布与 SOL gap 无关的示例；右：固定 3× speedup 下，不同问题的 headroom reclaimed 可从 <0.2 到 >0.8 大幅波动。SOL Score（r=0.981）远优于 speedup（r=0.81）作为衡量 "fraction of headroom reclaimed" 的指标。

验证指标 "fraction of headroom reclaimed" = $(T_{ref} - T_k)/(T_{ref} - T_{SOL})$：
- **SOL Score 与该指标相关性：r = 0.981**（极高度一致）
- **Speedup 与该指标相关性：r = 0.81**（因基线不同导致失真）
- 在固定 3× speedup 条件下，不同问题的 headroom reclaimed 范围从 <0.2 到 >0.8，波动高达 4×

**结论：** speedup 是误导性指标，SOL Score 才是衡量硬件效率的可靠标准。

### 4.3 Reward Hacking 分布

![](SOLExecBench_fig5_reward_hacking.png)
> **Figure 5**: Reward hacking 手法的分布与发生频次。精度降级（FP32→低精度）是最常见的攻击面（6.4%），其次是 monkey patching（3.3%）和 stream 注入（2.5%）。

### 4.4 Scoring Baseline 结果

![](SOLExecBench_fig6_results.png)
> **Figure 6**: Scoring baseline 的 SOL Score 分布。X 轴为 SOL Score，Y 轴为问题数量。不同颜色/面板表示四个类别（L1, L2, Quant, FIB）。红色虚线标记 S=0.5（与基线持平），绿色虚线标记 S=1.0（达到 SOL）。

**各类别中位数 SOL Score：**

| 类别 | 中位数 SOL Score | 说明 |
|------|-----------------|------|
| L1 | 0.688 | 单算子，优化空间相对明确 |
| L2 | 0.761 | 融合内核，agent 能发现更大收益 |
| Quant | 0.757 | 低精度需要更精细的量化策略，agent 表现良好 |
| FIB | 0.789 | FlashInfer 原语已有高度优化，agent 起点较高 |
| **Overall** | **0.732** | 整体仍距 SOL 有显著距离 |

**Agent vs. PyTorch 的 SOL 距离缩减倍数：**

| 类别 | 缩减倍数 | 含义 |
|------|---------|------|
| L1 | 2.0× | 将距离缩减了一半 |
| L2 | 2.7× | 融合优化收益大 |
| Quant | 2.9× | 低精度专家知识收益显著 |
| FIB | 3.4× | 推理原语优化成熟度最高 |

**关键观察：**
- 所有类别的中位数均 > 0.5，确认 agent 基线确实优于 PyTorch reference
- 但**几乎没有问题达到 S = 1.0**：说明每个问题都留有优化空间
- L2/Quant/FIB 的 score 高于 L1，可能因为融合和低精度优化有更多 heuristic 可以被 agent 挖掘

### 4.5 各类别详细指标

![](SOLExecBench_fig8_category.png)
> **Figure 8**: 四类别的问题组成与分布特征。展示不同类别在算子类型、精度、模型域上的分布差异，解释为什么 FIB 类别整体 SOL Score 更高（推理原语更成熟）。

### 4.6 Agent 生成基线的迭代优化

![](SOLExecBench_fig7_baselines.png)
> **Figure 7**: 多 agent turn-based 优化系统生成基线的过程示意。多个 agent 独立优化，每轮共享有效方案，经过多轮迭代后选择每问题的最快有效内核作为 scoring baseline。

---

## 5. 方法细节与设计哲学

### 5.1 为什么选 Roofline + Einsum 而不是 Cycle-accurate 模拟？

| 方案 | 优势 | 劣势 | SOLAR 的选择 |
|------|------|------|-------------|
| Cycle-accurate 模拟 | 极高精度 | 搭建成本极高，无法快速适配新架构 | ❌ |
| 纯 roofline | 简单快速 | 忽略 fusion、cache 层次、prefetch 等 | ⚠️ 基础框架 |
| **Roofline + Orojenesis + Fusion 建模** | 平衡精度与可扩展性，3 阶段可自动扩展 | 无法捕捉 value-dependent 优化 | ✅ |

**工程权衡：**
- 完全精确的上界是 NP-hard（需考虑寄存器分配、指令调度、bank conflict 等微架构细节）
- SOLAR 选择"可自动化 + 对新算子可扩展"的中间路径：基于 einsum 的代数表达 + roofline / Orojenesis 绑定
- Orojenesis 的关键改进：将 off-chip traffic 建模为 on-chip buffer 容量的函数，比传统 roofline 更紧

### 5.2 SOL Score 的非线性设计意图

为什么不用线性标准化（如 $(T_b - T_k)/(T_b - T_{SOL})$）？

**问题：** 线性指标在远离 SOL 时同样奖励，无法体现"接近极限越来越难"的事实。

**SOL Score 的妙处：**
- 候选从 $T_b$ 到 $(T_b + T_{SOL})/2$ 时，分数从 0.5 → 0.667（增益 +0.167）
- 候选从 $(T_b + T_{SOL})/2$ 到 $T_{SOL}$ 时，分数从 0.667 → 1.0（增益 +0.333）
- 同样绝对时长缩减，在接近 SOL 时获得更高分数奖励

**这对应于工程现实：** 做 80% 到 90% 的优化往往比 50% 到 80% 更困难，理应获得更多激励。

### 5.3 沙箱架构的保守性权衡

**禁用 CUDA streams** 是一个有争议的保守决策：
- **利**：彻底消除 stream injection 攻击面，简化验证逻辑
- **弊**：合法的 stream 并行优化被排除在外
- **理由**：作为一个优化基准，核心关注点是**单 kernel/单 stream 的算法效率**，kernel 间的 overlap 调度属于不同层次问题（已在 FlashInfer-Bench 等工作中覆盖）

---

## 6. 局限性与讨论

### 6.1 论文自我分析的局限
1. **SOL bound 不一定是紧的（tight）**：基于纯张量形状分析，无法捕捉 value-dependent 优化（如稀疏性、结构化 mask 跳过计算）。对于高度稀疏或动态形状变化剧烈的 kernel，SOL 上界可能显著高于真实可达到的性能。
2. **硬件可变性未建模**：同一 B200 GPU 不同批次间的微架构差异、温度、电压频率等未纳入。
3. **多卡/多节点场景缺失**：仅测单 GPU，实际部署中 tensor/pipeline parallelism 的通信开销是重要的优化空间。

### 6.2 深层方法论局限

**Reward-hacking 防御的"军备竞赛"性质：**
- 静态代码分析 + LLM-as-a-judge 本质上是**基于模式匹配的防御**，攻击者可以通过混淆代码、间接调用等方式绕过
- 更根本的问题是：计时机制本身成为攻击面。如果 benchmark 设计为"提交可复现的 Docker + 源码 + 详细 profiling 报告"，而非"只提交一个计时结果"，攻击面会显著缩小

**基线生成策略的偏置：**
- Agent-generated baseline 依赖于 agent 的能力和预算。如果 agent 本身有 systematic bias（如对某类算子优化过度/不足），则 S=0.5 的定义就存在偏置
- 论文建议使用 agent 基线而非固定 PyTorch eager，理由是 "human baselines are expensive"。但这意味着**不同问题的 0.5 阈值不可比**

### 6.3 未来方向
1. **社区驱动的问题扩展**：接受来自新 SOTA 模型的问题提交
2. **基线持续迭代**：随着 agent/编译器技术提升，基线应定期刷新
3. **多精度混合场景**：当前问题大多使用单一精度，真实训练/推理中混合精度调度更复杂
4. **跨 GPU 代际适配**：SOLAR 上界应从 Blackwell 扩展到 Hopper、Ampere 等，使跨代比较有意义

---

## 7. 个人思考

### 7.1 方法的优雅之处

**"用硬件极限替代软件基线"的范式转换非常干净。**

在 [[KernelBench]] 和 [[TritonBench]] 等工作中，基线 = PyTorch eager，意味着任何人只要让编译器生成比 eager 好的代码就能刷分。但 PyTorch eager 本身也在优化（例如 2.x 的 compile），这使得历史分数不可比。SOL-ExecBench 用**固定的硬件上限**作为锚点，解决了基线漂移问题。

**SOL Score 的非线性设计是精妙的信号工程**：它将 "优化收益递减" 的工程直觉编码到了分数函数中。相比线性标准化，SOL Score 更能激励研究者去攻克那 50%→100% 的最后一段。

### 7.2 与相关工作的关联

- [[KernelBench]] (270 problems): 互补关系。KernelBench 更适合快速验证 LLM 生成 CUDA 的能力（pass@k 指标），SOL-ExecBench 则针对**极致性能优化**。
- [[FlashInfer-Bench]]: 被 SOL-ExecBench 直接纳入 26 个问题，说明其推理原语的高质量。
- [[Roofline Model]] (Williams et al., 2009): SOLAR 的理论根基。
- [[Orojenesis]] (Huang et al., 2024): 将 roofline 从通信/计算两种 regime 扩展到考虑 on-chip buffer 的 multi-regime 精确建模。

### 7.3 最值得关注的数字

- **0.732（overall median SOL Score）**：这意味着即便是经过多 agent 迭代优化的基线，距离硬件极限仍有约 27% 的 gap。对 agentic optimization / AI compiler 领域，这是一个巨大的行动号召。
- **14.5% reward hacking**：说明当 kernel optimization 被自动化后，作弊的诱惑和可行性都非常高，评测基础设施的安全设计不是锦上添花而是核心工程问题。
- **r = 0.10（speedup vs SOL headroom 相关性）**：这是整个论文最有力的实证证据——"快了多少倍"对 "还有多少潜力" 几乎没有任何预测力。

### 7.4 可复现的洞察

1. **对于任何硬件性能基准，如果可能，都应该定义一个硬件极限上界**，而非依赖软件基线。这在 CPU（如 Roofline）、GPU（如 SOLAR）、甚至 NPU 场景都通用。
2. **非线性评分在存在收益递减的领域特别重要**。推荐系统、硬件优化、药物发现等领域的 benchmark 设计都可借鉴 SOL Score 的 sigmoid/softmax 型分数映射。
3. **对抗性评测基础设施越早设计越好**。SOL-ExecBench 的 reward-hacking 防御可以作为一个 check list 应用于任何 coding competition / optimization benchmark。

---

## 8. 关键引用

```bibtex
@article{lin2026solexecbench,
  title={SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits},
  author={Lin, Edward and Modi, Sahil and Hari, Siva Kumar Sastry and Huang, Qijing and Ye, Zhifan and Qin, Nestor and Zhou, Fengzhe and Zhang, Yuan and Wang, Jingquan and Damani, Sana and others},
  journal={arXiv preprint arXiv:2603.19173},
  year={2026},
  url={https://arxiv.org/abs/2603.19173},
  code={https://github.com/NVIDIA/SOL-ExecBench}
}
```

---

## 附录：论文未披露的关键信息

| 参数类别 | 缺失信息 | 说明 |
|---------|---------|------|
| Agent 基线生成 | Agent 的具体 LLM 型号 | 仅提到 "frontier LLM" 和 "multi-agent turn-based optimization system"，未披露模型提供商、上下文长度、采样 temperature |
| Agent 基线生成 | 时间/成本预算数字 | 未给出每问题/每轮次 agent 的明确预算限制 |
| SOLAR | Einsum 扩展字典的人工审核比例 | 未说明本轮实验中 "unseen operator 的 agent 生成转换" 的成功率或人工干预比例 |
| SOLAR | Roofline 峰值吞吐/带宽取值来源 | 未明确 B200 的 FLOPs peak 和 bandwidth peak 来自实测还是 datasheet |
| 评测框架 | L2 cache clear 的具体开销 | 未说明 256MB buffer zeroing 对总时间测量的系统误差 |
| 评测框架 | 容差元组的生成方法 | workload-specific $(atol, rtol, matched_ratio)$ 是人工设定还是自动导出？ |
| 正确性 | 浮点误差传播分析缺失 | 对于融合内核（L2），累积误差容差是否足够阻止 "合法但数值不稳定" 的提交？未讨论 |
| 实验 | 为什么没有 human expert baseline | 论文声明 "human baselines are expensive"，但没有量化尝试成本或给出替代论证 |
| Scoring baseline | 固定基线 vs. 滚动基线 | 当社区提交更好方案时，S=0.5 的定义点是否会移动？论文暗示基线可更新但未给更新策略 |
