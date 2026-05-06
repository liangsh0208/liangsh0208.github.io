# DeepSeek V4 训练实现深度解析

> 文档基于开源代码仓库 `cann` 解析，涵盖模型架构、训练框架、并行策略、优化器等所有核心模块的实现原理。

---

## 目录

1. [项目概述与架构定位](#1-项目概述与架构定位)
2. [模型架构详解](#2-模型架构详解)
   - 2.1 超参数配置 (`args.py`)
   - 2.2 多超连接 (mHC)
   - 2.3 动态稀疏注意力 (DSA)
   - 2.4 索引器辅助损失 (Indexer Loss)
   - 2.5 专家混合 (MoE)
   - 2.6 多Token预测 (MTP)
   - 2.7 完整前向传播流程
3. [训练框架与数据 Pipeline](#3-训练框架与数据-pipeline)
4. [并行策略详解](#4-并行策略详解)
5. [优化器与内存优化](#5-优化器与内存优化)
6. [NPU 融合算子系统](#6-npu-融合算子系统)
7. [Checkpoint 与权重转换](#7-checkpoint-与权重转换)

---

## 1. 项目概述与架构定位

本仓库是一个**基于 torchtitan 的昇腾 NPU 训练插件**，采用"最小化 fork + 动态 monkey-patch"的架构设计。它不 fork 整个 torchtitan，而是在其之上通过运行时 patch 注入 NPU 特化逻辑。

### 目录结构

| 目录 | 职责 |
|------|------|
| `torchtitan-npu/` | NPU 训练插件（核心代码） |
| `cann-recipes-train/` | 权重转换、量化工具与配置 |

### 核心设计哲学

- **继承上游能力**：Trainer 循环、checkpointing、metrics、FSDP/TP/PP/EP/CP 全部复用 torchtitan upstream
- **运行时注入**：通过 `torchtitan.components.optimizer.build_optimizers = _build_optimizers_wrapper` 等方式 monkey-patch，替换为 NPU 版本
- **Converter 系统**：在模型构建后通过 registry 将 eager PyTorch 实现替换为 NPU 融合算子（如 `npu_sparse_attn_shared_kv`）

---

## 2. 模型架构详解

模型代码位于 `torchtitan_npu/models/deepseek_v4/model/`。

### 2.1 超参数配置 (`args.py`)

`DeepSeekV4ModelArgs` 定义了完整的模型超参数：

```python
dim: int = 4096                    # 模型隐藏维度
n_heads: int = 64                   # 注意力头数
head_dim: int = 512                 # 每头维度
rope_head_dim: int = 64             # RoPE 维度（head_dim 中参与旋转的部分）
q_lora_rank: int = 1024             # Q 投影的 LoRA 降维秩
o_lora_rank: int = 1024             # O 投影的 LoRA 降维秩
o_groups: int = 8                   # O 投影分组数
window_size: int = 128              # 滑动窗口大小
compress_ratios: Tuple[int] = (...) # 逐层压缩比率配置
index_n_heads: int = 64             # 索引器头数
index_head_dim: int = 128          # 索引器每头维度
index_topk: int = 512               # 索引器 top-k 选择数
hc_mult: int = 4                    # mHC 超连接扩展倍数
num_mtp_modules: int = 0            # MTP 模块数（配置中通常为 1）
```

**关键压缩比率配置**：
- `compress_ratio=1`：该层为**标准滑动窗口注意力**，无压缩
- `compress_ratio=4`：**重叠压缩**，使用 `Indexer` + `Compressor`，压缩 token 之间存在重叠
- `compress_ratio=128`：**标准压缩**，将 128 个 token 压缩为 1 个 KV

43 层模型的压缩比率配置规律：前两层为 `1, 1`（标准注意力），之后是 `4, 128` 交替出现，即：

```
(1, 1, 4, 128, 4, 128, 4, 128, ...)
```

这种设计让模型的不同层采用不同粒度的稀疏注意力，浅层用更细粒度的局部关注，深层用粗粒度的全局压缩关注。

---

### 2.2 多超连接 (mHC - multi-HyperConnection)

mHC 是 DeepSeek V4 的核心创新之一，替代了传统的残差连接。它让每个 Transformer block 的输入在**超连接空间**中扩展为多个并行分支，通过 Sinkhorn 算法计算分支权重后再聚合。

#### 数学原理

输入 `x` 的形状为 `[B, S, hc_mult=4, D]`。处理流程：

**1. HcPre（前处理）**

```python
x_flat = x.flatten(2).float()                         # [B, S, hc_mult*D]
rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
mixes = F.linear(x_flat, hc_fn) * rsqrt                # 学习超连接混合权重
pre, post, comb = torch_hc_split_sinkhorn(mixes, ...)  # Sinkhorn 归一化
y = torch.sum(pre.unsqueeze(-1) * x, dim=2)           # 按 pre 权重聚合到 [B, S, D]
```

`HcSplitSinkhorn` 将 `mixes` 拆分为三部分：
- `pre`: `hc_mult` 个前置权重，通过 sigmoid 归一化到 `(0, 1]`，用于**聚合输入**
- `post`: `hc_mult` 个后置权重，通过 sigmoid 归一化，用于**重扩展输出**
- `comb`: `hc_mult x hc_mult` 的耦合矩阵，通过 softmax + **Sinkhorn-Knopp 迭代**（默认 20 次）进行行列双随机归一化

Sinkhorn-Knopp 算法确保 `comb` 矩阵的每一行和、每一列和都等于 1，这在超连接空间中实现了流量的守恒约束：

```python
comb = comb.softmax(-1) + eps
col_sum = comb.sum(-2, keepdim=True)
comb = comb / (col_sum + eps)
for _ in range(sinkhorn_iters - 1):
    row_sum = comb.sum(-1, keepdim=True)
    comb = comb / (row_sum + eps)
    col_sum = comb.sum(-2, keepdim=True)
    comb = comb / (col_sum + eps)
```

**2. HcPost（后处理）**

在 Attention/MoE 输出后，将单分支输出重新扩展回超连接空间：

```python
y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
    comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
)
```

这里 `x` 是当前层的输出（单分支 `[B, S, D]`），`residual` 是原始的 hyper-connection 输入（多分支 `[B, S, hc_mult, D]`）。通过 `post` 权重和 `comb` 耦合矩阵，输出被重新扩展为 `[B, S, hc_mult, D]`。

**3. HcHead（模型头部）**

在最终输出层，将多分支聚合为单一输出用于预测 logits：

```python
pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
```

与 block 内的 HcPre 不同，HcHead 只有前置聚合（`pre`），没有复杂的 Sinkhorn comb 矩阵。

#### 为什么用 mHC？

传统残差连接是 `output = F(x) + x`，而 mHC 将输入扩展为 4 个并行超通道，每个通道可以独立地与其他通道耦合。这提供了比标准残差更丰富的信息流路径，同时通过 Sinkhorn 约束保持了数值稳定性。

---

### 2.3 动态稀疏注意力 (DSA - Dynamic Sparse Attention)

DSA 是 DeepSeek V4 最具特色的架构组件，它结合了**滑动窗口局部注意力**和**压缩 KV 全局注意力**，并通过一个**可学习的索引器 (Indexer)** 来选择需要参与全局关注的压缩 token。

#### 整体架构

单个 Attention 模块被拆分为三个子模块：

```
PreAttention  →  InnerAttention  →  PostAttention
(可编译 eager)   (NPU 融合算子)     (可编译 eager)
```

这种拆分的原因是 `InnerAttention` 包含昇腾 NPU 融合算子（`npu_sparse_attn_shared_kv` 等），无法被 `torch.compile` 编译，所以将其隔离在外层两个可编译模块之间。

#### 2.3.1 PreAttention：投影与压缩

```python
class PreAttention(nn.Module):
    def __init__(self, layer_id, args):
        self.wq_a = nn.Linear(dim, q_lora_rank, bias=False)   # Q LoRA 降维
        self.q_norm = RMSNorm(q_lora_rank)
        self.wq_b = nn.Linear(q_lora_rank, n_heads * head_dim, bias=False)  # Q LoRA 升维
        self.wkv = nn.Linear(dim, head_dim, bias=False)      # 单头 KV (MQA)
        self.kv_norm = RMSNorm(head_dim)
        
        # compress_ratio=4 时同时有 Indexer + Compressor
        if compress_ratio == 4:
            self.compressor = Compressor(args, compress_ratio, head_dim)
            self.indexer = Indexer(args, compress_ratio)
        elif compress_ratio > 1:
            self.compressor_128 = Compressor(args, compress_ratio, head_dim)
```

**PreAttention.forward 流程**：

1. **Q 投影（LoRA 路径）**：`x → wq_a → q_norm → wq_b → [B, S, n_heads, head_dim]`
2. **Q 归一化**：对 Q 做 RMSNorm 风格的逐 token 归一化（平方均值倒数）
3. **RoPE 应用**：将 Q 拆分为 nope 部分和 rope 部分，仅对 rope 部分应用旋转位置编码
4. **KV 投影**：`x → wkv → kv_norm → [B, S, head_dim]`，应用 RoPE
5. **压缩 KV**（`compress_ratio > 1`）：
   - `compress_ratio=4`：调用 `Compressor` 生成压缩 KV，同时调用 `Indexer` 生成索引查询
   - `compress_ratio=128`：仅调用 `Compressor_128`

#### 2.3.2 Compressor：局部聚合压缩

`Compressor` 将连续 token 局部聚合成压缩 token：

```python
def forward(self, x, freqs_cis):
    kv = self.wkv(x)              # [B, S, coff * head_dim]
    score = self.wgate(x)         # [B, S, coff * head_dim]
    
    # 按 compress_ratio 分块: [B, S//ratio, ratio, coff*D]
    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio)) + self.ape  # 加可学习 APE
    
    if overlap:  # ratio=4 时启用
        kv = overlap_transform(kv)      # 生成 2*ratio 个重叠位置
        score = overlap_transform(score, float("-inf"))
    
    # softmax 加权聚合
    kv = (kv * score.softmax(dim=2)).sum(dim=2)  # [B, S//ratio, head_dim]
    kv = self.norm(kv)
    kv_rot = apply_rotary_emb(kv[..., -rope_head_dim:], freqs_cis)
    kv = torch.cat([kv[..., :-rope_head_dim], kv_rot], dim=-1)
    return kv
```

**重叠压缩 (`overlap_transform`)**：

当 `compress_ratio=4` 时，Compressor 生成 `2 * ratio = 8` 个压缩位置。其中后 4 个位置是标准压缩（token [0,1,2,3] → pos0, [4,5,6,7] → pos1...），前 4 个位置是**重叠压缩**（token [0,1,2,3] → pos0, [1,2,3,4] → pos1, ...），即相邻压缩块之间 overlap 了 3/4 的 token。

```python
def overlap_transform(self, tensor, value=0):
    # tensor: [b, s, r, 2d]
    new_tensor = tensor.new_full((b, s, 2*r, d), value)
    new_tensor[:, :, r:] = tensor[:, :, :, d:]       # 标准压缩部分 -> 后半段
    new_tensor[:, 1:, :r] = tensor[:, :-1, :, :d]   # 重叠压缩部分 -> 前半段（偏移1位）
    return new_tensor
```

重叠压缩的目的是让相邻压缩 token 共享更多上下文信息，缓解粗粒度压缩导致的位置边界效应。

#### 2.3.3 Indexer：学习选择压缩 Token

`Indexer` 只在 `compress_ratio=4` 的层存在。它的任务是**学习哪些压缩位置对当前查询最重要**。

```python
class Indexer(nn.Module):
    def __init__(self, args, compress_ratio):
        self.wq_b = nn.Linear(q_lora_rank, n_heads * head_dim, bias=False)
        self.weights_proj = nn.Linear(dim, n_heads, bias=False)  # 头级重要性权重
        self.compressor = Compressor(args, compress_ratio, head_dim, rotate=True)
        
    def forward(self, x, qr, freqs_cis, hadamard_mat, offset):
        q = self.wq_b(qr)           # [B, S, n_heads, head_dim]
        q_rope = apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)
        q = rotate_activation(q, hadamard_mat)    # Hadamard 旋转
        
        k = self.compressor(x, freqs_cis)          # 压缩 KV
        k = rotate_activation(k, hadamard_mat)    # Hadamard 旋转
        
        weights = self.weights_proj(x) * scale     # [B, S, n_heads]
        return q, k, weights
```

Indexer 的 Q 和 K 都经过 **Hadamard 变换**（随机旋转），这是为了让点积注意力在压缩后的低维空间中保持近似内积不变性。Hadamard 矩阵是一个正交矩阵（满足 $H^T H = nI$），对输入做 Hadamard 变换相当于做一次无损的随机旋转。

#### 2.3.4 LiCompute & SparseAttention

**LiCompute** 在 Indexer 产出的压缩 Q/K 上计算注意力分数，选择 top-k 压缩位置：

```python
class LiCompute(nn.Module):
    def forward(self, q_indexer, k_indexer, weights, seqlen, offset):
        # einsum: [B, S, n_heads, head_dim] @ [B, S//ratio, head_dim] -> [B, S, n_heads, S//ratio]
        index_score = torch.einsum("bshd,btd->bsht", q_indexer, k_indexer)
        # ReLU + 头级加权
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, S//ratio]
        # causal mask: 只保留当前位置之前的压缩块
        mask = torch.arange(S//ratio) >= (base + 1) // ratio
        index_score += torch.where(mask, min_val, 0)
        # top-k 选择
        topk_idxs = index_score.topk(min(index_topk, end_pos // ratio), dim=-1)[1]
        compress_topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        return compress_topk_idxs, index_score
```

**SparseAttention** 执行实际的稀疏注意力计算：

```python
def forward(self, query_states, kv_states, attn_sink, kv_compress, compress_topk_idxs):
    # 1. 滑动窗口 topk
    topk_idxs = get_window_topk_idxs(window_size, bsz, seqlen)  # [B, S, window_size]
    
    # 2. 拼接压缩 topk
    if compress_ratio > 1:
        topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
    
    # 3. 构建索引 mask（-inf 表示不参与计算的位置）
    index_mask = torch.full((..., kv_len + 1), min_val).scatter_(-1, topk_idxs, 0)
    
    # 4. 注意力计算 + sink token
    attn_weights = query @ kv.T * scale
    attn_weights = attn_weights + index_mask[..., :-1]  # mask 掉非 topk 位置
    combined_logits = torch.cat([attn_weights, attn_sink], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1)
    scores = probs[..., :-1]  # 去掉 sink
    attn_output = scores @ kv_states
```

**稀疏注意力的组合逻辑**：

每个 query token 实际上只关注两类 token：
1. **滑动窗口内**的最近 `window_size=128` 个 token（局部注意力）
2. **Top-k 压缩 token**（全局注意力，通过 Indexer 学习选择）

所有其他位置的 KV 被 `index_mask` 置为 `-inf`，在 softmax 中贡献为 0，因此实际计算量大大减少。

**注意**：还有一个 `attn_sink` 参数。这实际上是一个可学习的 sink token（类似 attention sink 机制），用于给注意力分布提供一个稳定的初始锚点，减少长序列中的注意力漂移问题。

#### 2.3.5 PostAttention：输出投影

```python
class PostAttention(nn.Module):
    def __init__(self, args):
        self.wo_a = nn.Linear(n_heads * head_dim // n_groups, n_groups * o_lora_rank, bias=False)
        self.wo_b = nn.Linear(n_groups * o_lora_rank, dim, bias=False)
```

PostAttention 同样使用 LoRA 风格的两段投影：
1. `wo_a`：将多头输出按 `n_groups=8` 分组降维到 `o_lora_rank=1024`
2. `wo_b`：从 LoRA 空间升维回 `dim=4096`

此外，PostAttention 对输出应用了**逆 RoPE**（`inverse=True`），这是因为在 InnerAttention 中 Q 和 KV 都应用了 RoPE，为了使注意力计算保持等变性，输出需要进行逆旋转来恢复原始坐标系。

```python
o_nope, o_rope = torch.split(o, [head_dim - rd, rd], dim=-1)
o_rope = apply_rotary_emb(o_rope, freqs_cis, inverse=True)  # 逆 RoPE
```

---

### 2.4 索引器辅助损失 (Indexer Loss)

Indexer 的选择能力不能仅靠最终任务的端到端梯度来训练（梯度信号太弱且稀疏），因此设计了一个**辅助损失**来显式监督 Indexer 的选择质量。

#### 损失定义

```python
class DSAIndexerLoss(nn.Module):
    def forward(self, selected_main_attn_dist, index_score, topk_indices, loss_scale):
        # 将 index_score softmax 为概率分布
        index_score = F.softmax(index_score, dim=-1, dtype=torch.float32)
        # 主注意力的 softmax 分布（已按 score 归一化）
        selected_main_attn_dist = F.normalize(selected_main_attn_dist, p=1, dim=-1)
        # KL 散度: 让 index_score 逼近主注意力分布
        loss = F.kl_div(
            (index_score + 1e-10).log(),
            selected_main_attn_dist + 1e-10,
            reduction="none"
        ).sum(dim=-1).mean()
        return loss * loss_scale
```

**直观理解**：Indexer 选择的 top-k 压缩位置的 softmax 分数分布，应该与真实主注意力（在所有压缩位置上的 softmax 分数分布）尽可能一致。这是一个**蒸馏**关系：用主注意力的完整分布作为 teacher，监督 Indexer 的近似选择。

#### AutoScaler：与主损失同 scale

在大规模分布式训练中，辅助损失需要与主损失保持相同的梯度缩放比例，否则会因为梯度累积/DP 规模不同而导致训练不稳定。

```python
class DSAIndexerLossAutoScaler(torch.autograd.Function):
    main_loss_backward_scale: torch.Tensor = None
    
    @staticmethod
    def forward(ctx, output, aux_loss):
        ctx.save_for_backward(aux_loss)
        return output  # 不改变前向输出
    
    @staticmethod
    def backward(ctx, grad_output):
        (loss,) = ctx.saved_tensors
        scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
        # 将 indexer loss 的梯度缩放到与主 loss 相同的 scale
        scaled_grad = torch.ones_like(loss) * scale
        return grad_output, scaled_grad
```

使用方式（在 block forward 中）：

```python
loss = self.cal_index_loss(...)
x = DSAIndexerLossAutoScaler.apply(x, loss)  # x 不变，但 backward 时产生 indexer loss 梯度
```

`main_loss_backward_scale` 在 Trainer 初始化时被设置，计算公式考虑了梯度累积步数和 CP/DP 度数。

#### 分布式损失同步

在并行训练时，Indexer loss 需要在所有参与同一层计算的 rank 上同步。代码中通过 monkey-patch 替换了 logging helper：

```python
def distributed_track_dsa_indexer_metrics(total_acc_steps):
    # 全局 All-Reduce AVG
    dist.all_reduce(dsa_indexer_losses, op=dist.ReduceOp.AVG)
    # 补偿 PP: AVG 除以了 WORLD_SIZE，但有效 rank 只有 WORLD_SIZE // PP
    pp_degree = parallel_dims.pp if parallel_dims.pp_enabled else 1
    dsa_indexer_losses *= pp_degree
    # 最终平均
    loss = dsa_indexer_losses.sum() / num_layers / total_acc_steps
```

---

### 2.5 专家混合 (MoE)

MoE 实现位于 `moe.py`，继承自 torchtitan 的基类但做了大量定制。

#### 路由策略

**三层 Hash 路由**：

```python
class TokenChoiceTopKRouter(TokenChoiceTopKRouter):
    def __init__(self, ..., layer_id, args):
        self.hash = layer_id < args.n_hash_layers  # 前 n_hash_layers=3 层用 hash
        if self.hash:
            # 每个 token ID 映射到 top_k 个固定专家
            tid2eid = nn.Parameter(
                torch.stack([torch.randperm(self.top_k) for _ in range(vocab_size)]),
                requires_grad=False,
            )
```

前 3 层使用**基于 token ID 的固定哈希路由**：每个 vocab ID 预先随机分配 `top_k=6` 个专家。这样做的原因是：
1. 浅层 embedding 空间的语义 token 分布相对稳定
2. 避免可学习门控在训练初期的不稳定性
3. 减少路由计算开销

**深层可学习路由**：

```python
def forward(self, x, input_ids, expert_bias):
    scores = self.gate(x)
    if self.score_func == "sqrtsoftplus":
        scores = _softplus_stable(scores.to(torch.float32)).sqrt()
    # ...
    if self.hash:
        selected_experts = self.tid2eid[input_ids.flatten()]
    else:
        selected_experts = (scores + expert_bias).topk(self.top_k, dim=-1)[1]
```

路由分数函数使用 `sqrtsoftplus`：`sqrt(softplus(x))`。相比于 sigmoid 或 softmax，它的特点是：
- 输出始终为正，但不会像 softmax 那样强制概率和为 1
- `sqrt` 根号变换压缩了大分数的动态范围，避免少数 expert 被过度选中

**路由归一化**：

```python
if self.route_norm:
    denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
    top_scores = top_scores / denominator
top_scores = top_scores * self.route_scale  # route_scale=1.5
```

`route_norm` 使选中专家的分数归一化为和为 1（但路由决策本身不依赖归一化后的分数），然后通过 `route_scale=1.5` 放大最终门控值，增加 expert 输出的区分度。

#### GroupedExperts 与分组矩阵乘

```python
# use_grouped_mm=True 时，专家权重被打包为组内矩阵
self.experts.swiglu_limit = moe_args.swiglu_limit
```

当 `use_grouped_mm=True` 时，所有专家的 `w1`/`w2`/`w3`（SwiGLU FFN 的三个矩阵）被打包为分组张量，通过 `npu_grouped_matmul` 一次性完成所有专家的前向计算，避免 for-loop 逐个调用专家的 Python 开销。

#### Score Before/After Experts

```python
if self.score_before_experts:
    routed_input = (routed_input * top_scores.reshape(-1, 1)).to(x.dtype)
    # 门控与输入融合后送入专家
else:
    # 标准路径: 专家输出后再加权
    out_experts = torch.bmm(top_scores, routed_output_unsorted)
```

`score_before_experts` 控制门控的应用位置。为 true 时，先将 token 按其路由分数缩放后再送入专家（可以与 matmul 融合），输出后直接求和；为 false 时，专家计算原始输入，输出后再用 bmm (batch matmul) 加权求和。

---

### 2.6 多Token预测 (MTP - Multi-Token Prediction)

MTP 模块用于在一次前向传播中预测多个未来 token，增加训练信号密度。

```python
class MTPModule(DeepSeekV4TransformerBlock):
    def __init__(self, layer_id, model_args):
        super().__init__(layer_id, model_args)
        self.enorm = RMSNorm(dim)
        self.hnorm = RMSNorm(dim)
        self.e_proj = nn.Linear(dim, dim, bias=False)   # token embedding 投影
        self.h_proj = nn.Linear(dim, dim, bias=False)   # 上一层输出投影
    
    def forward(self, input_offset, prev_embed, input_ids, freqs_cis, ...):
        input_offset = self.enorm(input_offset)
        prev_embed = self.hnorm(prev_embed)
        x = self.e_proj(input_offset) + self.h_proj(prev_embed)
        x = x.unsqueeze(2).repeat(1, 1, hc_mult, 1)  # 扩展为 hyper-connection 空间
        # 之后与普通 Transformer block 完全一致
        ...
```

MTP 模块的核心输入：
- `input_offset`: 偏移后的真实 token embedding（如第 `n+1` 个 token 的 embedding）
- `prev_embed`: 上一层（主模型或前一个 MTP 模块）的输出

MTP 模块通过两个投影将真实 token 信息和模型预测信息融合，然后送入与普通 block 相同的 Attention + MoE 计算流程。

#### MTP Loss

```python
def multi_token_cross_entropy_loss(preds, labels, job_config):
    # preds[0] 是主模型预测
    main_loss = cross_entropy_loss(preds[0], labels[:, :seq_len])
    mtp_loss = 0
    for offset, pred in enumerate(preds[1:], 1):
        loss = cross_entropy_loss(pred, labels[:, offset:offset+seq_len])
        loss = loss / num_mtp_modules
        mtp_loss += loss
    return main_loss + mtp_loss * mtp_loss_weight  # mtp_loss_weight=0.3
```

每个 MTP 模块的预测都参与损失计算，但最终求和时会用 `mtp_loss_weight=0.3` 降低 MTP loss 的权重，确保主模型的预测仍然是主导训练目标。

#### DataLoader Patch for MTP

为了让 dataloader 产出足够长的序列以支持 MTP，代码对 `seq_len` 做了临时扩展：

```python
# hf_datasets.py
def mtp_build_text_dataloader(...):
    if num_mtp_modules > 0:
        original_seq_len = job_config.training.seq_len
        job_config.training.seq_len += num_mtp_modules  # 临时增加
        dataloader = build_text_dataloader(...)         # 生成更长序列
        job_config.training.seq_len = original_seq_len  # 恢复原值
        return dataloader
```

这意味着如果 `seq_len=4096` 且 `num_mtp_modules=1`，dataloader 实际产出长度为 4097 的序列，其中最后 1 个 token 用于 MTP 模块的偏移标签。

---

### 2.7 完整前向传播流程

```python
# DeepSeekV4Model.forward
seq_len = tokens.shape[1] - num_mtp_modules  # 主序列长度
input_ids = tokens[:, :seq_len].detach().long()
h = self.tok_embeddings(tokens[:, :seq_len])  # [B, seq_len, dim]
h = h.unsqueeze(2).repeat(1, 1, hc_mult, 1)   # [B, seq_len, hc_mult, dim]

# --- 主模型前向 (n_layers blocks) ---
for layer in main_layers:
    h = layer(h, input_ids, freqs_cis, hadamard_mat, attention_masks)

# HcHead 聚合 + RMSNorm + Output
h = self.hc_head(h, ...)
h = self.norm(h)
output = self.output(h.float())
output_list = [output]

# --- MTP 模块前向 ---
for mtp_id in range(num_mtp_modules):
    token_offset = tokens[:, mtp_id+1 : mtp_id+1+seq_len]  # 偏移真实 token
    input_offset = self.tok_embeddings(token_offset)
    h = mtp_layer(input_offset, prev_embed, input_ids, freqs_cis, ...)
    h = self.hc_head(h, ...)
    prev_embed = h
    output = self.output(self.norm(h))
    output_list.append(output)

return output_list  # len = 1 + num_mtp_modules
```

---

## 3. 训练框架与数据 Pipeline

### 3.1 训练入口 (`entry.py`)

```python
if __name__ == "__main__":
    config = ConfigManager().parse_args()
    
    # 应用各种 monkey patches
    _patch_for_garbage_collection_run()
    _patch_for_parallel_dims_build_mesh()
    _patch_torchtitan_model_reshape_for_broadcast()
    
    if config.model.name == "deepseek_v4":
        _patch_train_step_for_dsv4_indexer_loss()   # 每步记录 indexer loss
        _patch_init_for_dsa_set_loss_scale()       # 设置 AutoScaler 的 loss scale
        _patch_for_train_npu_memory()              # 设置 NPU 显存限制
    
    if config.compile.enable:
        import inductor_npu_ext  # NPU 编译后端扩展
    
    trainer = Trainer(config)
    trainer.train()
```

### 3.2 TrainSpec 注册 (`__init__.py`)

模型通过 `TrainSpec` 协议注册到 torchtitan：

```python
def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=DeepSeekV4Model,
        parallelize_fn=parallelize_deepseek_v4,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        ...
    )
```

这样 torchtitan 的通用 Trainer 就能直接消费 DeepSeek V4 模型，无需修改 Trainer 本身的代码。

### 3.3 激活重计算 Patch

```python
class _patched_apply_full_ac:
    """AC wrapper that skips indexing loss computation during initial forward pass."""
```

当 `activation_checkpoint.mode="full"` 时， checkpointing 会保存前向传播的部分中间结果。自定义 patch 确保**在 AC 保存阶段跳过 indexer loss 计算**，因为在 checkpoint 的 recompute 阶段会重新计算这些 loss，无需在第一次 forward 浪费计算。

---

## 4. 并行策略详解

配置文件采用 **纯 FSDP + 大 EP** 策略：

```toml
[parallelism]
data_parallel_shard_degree = -1    # 自动推断 = 128
tensor_parallel_degree = 1         # 未启用
pipeline_parallel_degree = 1       # 未启用
expert_parallel_degree = 128       # EP128
context_parallel_degree = 1        # 未启用
```

### 4.1 FSDP 数据并行 (`apply_fsdp`)

```python
apply_fsdp(
    model,
    dp_mesh,            # 主 FSDP mesh
    edp_mesh=edp_mesh,  # MoE 参数的独立 FSDP mesh "efsdp"
    ep_degree=128,
    ...
)
```

**核心设计**：MoE 参数（expert weights）和稠密参数（attention, norms, embeddings）使用**不同的 FSDP mesh**：
- 稠密参数：`dp_mesh`（所有数据并行 rank）
- MoE 参数：`edp_mesh`（仅 EP group 内的 rank，大小为 WORLD_SIZE / EP）

这意味着一个 256 专家的 MoE 层，在 EP128 下，每个 EP rank 只负责 2 个专家，而这 2 个专家的权重还进一步通过 `efsdp` mesh 在 EP group 内 shard。

### 4.2 Expert Parallel (`apply_moe_ep_tp`)

```python
apply_moe_ep_tp(
    model,
    tp_mesh=...,        # 可为 None（V4 配置中 None）
    ep_mesh=ep_mesh,    # EP128
    use_deepep=False,   # 使用标准 EP 后端
)
```

EP 的核心操作是 `all_to_all_single`：
1. **Token Dispatch**：每个 rank 根据路由结果，将 token 发送到拥有对应专家的 rank
2. **专家计算**：本地执行（通过分组 matmul）
3. **Token Combine**：将计算结果 `all_to_all` 发送回原 rank

NPU 优化：`npu_expert_parallel` converter 将标准的 dispatch/combine 替换为 `npu_moe_token_permute`/`unpermute`，在 NPU 上直接完成 token 的 permute/unpermute，减少 CPU 侧索引开销。

### 4.3 Tensor Parallel 计划（即使未启用）

虽然主配置 `tp=1`，但代码中定义了完整的 TP plan。这里体现了 V4 模型 TP 的复杂性：

**`PrepareModuleInputOutputWithBwdAllReduce`**：

```python
class PrepareModuleInputOutputWithBwdAllReduce(PrepareModuleInputOutput):
    """在指定输入上注册 backward allreduce hook"""
    def _attach_bwd_hook_fn(self, module, inputs):
        def _allreduce_grad_hook(grad):
            torch.distributed.all_reduce(grad, op=SUM, group=self.group)
            return grad
        inp.register_hook(_allreduce_grad_hook)
```

这个自定义 plan 用于处理 DSA 中某些输入张量在 TP 下需要梯度同步的场景。例如 `kv_states` 在 TP 下是 Replicate 的，但由于 Q 是 TP shard 的，Q@K.T 的反向传播时 K 的梯度需要 allreduce。

### 4.4 关键通信优化：独立 ProcessGroup

```python
def _patch_for_parallel_dims_build_mesh():
    """强制 EP 和 FSDP 使用独立的 ProcessGroup，使 all-to-all 和 all-gather 可以重叠。"""
```

PyTorch 默认会为所有的 collectives 创建共享的 NCCL/HCCL communicator。这会导致 EP all-to-all 和 FSDP all-gather/reduce-scatter **串行执行**（因为它们竞争同一个 HCCL stream）。patch 强制创建**分离的 ProcessGroup**，使两类通信可以在不同的 HCCL stream 上**并发重叠**，显著提升训练吞吐。

### 4.5 `torch.compile` 策略

由于 `InnerAttention` 包含 NPU 融合算子（无法编译），编译策略是**按模块细粒度包裹**：

```python
if transformer_block.moe_enabled:
    # MoE 层: FSDP(GroupedExperts) 会 graph break，所以不能编译整个 block
    for attr_name, submod in block.named_children():
        if attr_name == "inner_attention":
            continue  # 跳过不可编译的 NPU 融合算子
        if attr_name == "experts":
            continue  # 跳过 token dispatch/combine
        setattr(block, attr_name, torch.compile(submod, ...))
else:
    # 非 MoE 层可以直接编译整个 block
    transformer_block = torch.compile(transformer_block, ...)
```

---

## 5. 优化器与内存优化

### 5.1 SwapOptimizer：CPU-GPU 状态交换

```python
class SwapOptimizersContainer(OptimizersContainer):
    """将 AdamW 的 exp_avg/exp_avg_sq 常驻在 CPU pinned memory，
        step() 时以 slice 为单位流水线化 load-update-offload。"""
```

**核心机制**：

1. **初始化**：Adam 状态（`exp_avg`, `exp_avg_sq`）创建在 GPU，但立即 `resize_(0)` 释放显存，同时在 CPU pinned memory 创建副本。

2. **Forward/Backward**：GPU 上只有模型参数和梯度，**无优化器状态**，节省约 ~8 bytes/param 显存。

3. **Step 流水线（关键）**：

```python
def swap_optimizer_step(self, closure=None):
    params_list = [p for p in self.param_groups]
    swap_count = 0
    for i, param in enumerate(params_list):
        if param.grad is None:
            continue
        
        # 1. 预加载下一批参数的状态到 GPU（异步）
        if swap_count == 0:
            swap_count = pipeline_load_param(self.swap_numel, params_list, i, swap_count)
        
        # 2. 等待加载完成
        wait_swap_to_device_event(param)
        
        # 3. 参数更新（使用 torch._fused_adamw_）
        param_update(param, state, group)
        
        # 4. 异步 offload 回 CPU
        with stream(swap_to_host_stream):
            wait_param_update_event(param)
            swap_states_to_host(param)
            swap_count -= param.numel()
```

**三个 CUDA/HCCL stream**：
- `current_stream`：执行实际的 `torch._fused_adamw_` 参数更新
- `swap_to_device_stream`：将下一批参数的 Adam 状态从 CPU 加载到 GPU（`copy_` non_blocking）
- `swap_to_host_stream`：将已完成更新的参数的 Adam 状态从 GPU 卸载到 CPU

通过 `record_event` / `wait_event` 在三者之间建立依赖关系，实现**流水线 overlap**。

**效果**：将 Adam 状态的显存占用从 ~16 bytes/param 降至接近 0（更新瞬间的峰值约为 ~8 bytes/param / swap_times）。

### 5.2 Muon 混合优化器

```python
def build_muon_hybrid_optimizers(model_parts, optimizer_config, ...):
    """2D 矩阵参数 → Muon optimizer
        非 2D 参数（embeddings, norms, biases）→ AdamW"""
```

Muon (Momentum Orthogonalization) 是一种针对矩阵参数设计的优化器，它通过在梯度更新前对梯度做正交化处理（类似 Newton 法的近似），使参数更新方向更"干净"。

代码中虽然保留了 Muon 实现，但**主配置使用 AdamW**（`swap_optimizer=true`），Muon 主要用于未来实验。

### 5.3 LR Scheduler

```python
# 三段式调度
1. Linear warmup: warmup_steps=400
2. Stable: lr = base_lr = 1e-5
3. Cosine decay: decay_steps = round(steps * 0.8) - warmup_steps
   lr = base_lr * (min_lr_factor + (1 - min_lr_factor) * 0.5 * (1 + cos(pi * progress)))
```

最终学习率不会降到 0，而是 clamp 到 `min_lr_factor * base_lr = 0.01 * 1e-5 = 1e-7`。

---

## 6. NPU 融合算子系统

代码通过 **Converter Registry** 在模型构建后做算子替换：

| Converter | 文件 | 替换目标 |
|-----------|------|----------|
| `deepseek_v4_sfa` | `converters/kernels/deepseek_v4_sfa.py` | `SparseAttention.forward` → `npu_sparse_attn_shared_kv`; `LiCompute.forward` → `npu_lightning_indexer`; `LiLoss.forward` → 融合 `SparseLightningIndexerGradKLLoss` |
| `npu_mhc_pre` | `converters/kernels/mhc_prepost.py` | `HcPre/Post/Head` → Triton-based NPU 实现 |
| `npu_gmm` | `converters/kernels/gmm.py` | `GroupedExperts.forward` → `npu_grouped_matmul` (含 SwiGLU 融合) |
| `npu_expert_parallel` | `converters/kernels/expert_parallel.py` | Token dispatch/combine → `npu_moe_token_permute/unpermute` |
| `npu_rms_norm` | `converters/kernels/rms_norm.py` | RMSNorm → NPU 融合版本 |
| `npu_rope` | `converters/kernels/rope.py` | RoPE compute → NPU 融合版本 |

这些 converter 的注册风格通常是：

```python
@register_converter("deepseek_v4_sfa")
def apply_deepseek_v4_sfa(model):
    for module in model.modules():
        if isinstance(module, SparseAttention):
            module.forward = wraps(module.forward)(npu_sparse_attn_wrapper)
```

即通过 Python 的 monkey-patch 在 eager 实现和 NPU 算子之间建立桥接。

---

## 7. Checkpoint 与权重转换

### 7.1 DCP + HF 双格式保存

```python
# checkpoint_patch.py
class CheckpointManager:
    def save(self, ...):
        # DCP (PyTorch Distributed Checkpoint): 保存 sharded 状态
        # HF (safetensors): 在 rank 0 gather EP expert 权重后保存
```

训练产出同时包含：
1. **DCP 格式**：直接用于训练 resumption，保留所有 sharding 信息
2. **HuggingFace 格式**：用于推理部署，`state_dict_adapter` 负责 key 映射和 expert weight remapping

### 7.2 权重格式映射 (`state_dict_adapter.py`)

```python
class DeepSeekV4StateDictAdapter(DeepSeekV3StateDictAdapter):
    def convert_hf_to_titan(self, state_dict):
        # 映射 HF key 到 torchtitan 内部 key
        # "layers.{i}.attn.wq_a.weight" → "layers.{i}.attention.pre_attention.wq_a.weight"
        # per-layer compressor/indexer 根据 compress_ratios 动态映射
        # expert weights: "standard" ↔ "gmm" (grouped) 格式转换
```

### 7.3 FP8 → BF16 转换

```python
# convert_model.py
def convert_model(input_path, output_path, target_dtype="bf16"):
    # 1. 读取 HF FP8 checkpoint (含 scale_inv)
    # 2. fp8 dequantize: weight = fp8_weight * (1 / scale_inv)
    # 3. 可选量化到 W8A8-INT 或 W4A8-MX
    # 4. 更新 model_index.json
```

---

## 总结

DeepSeek V4 的训练实现展现了现代大模型训练的完整技术栈：

1. **架构创新**：mHC 替代残差、DSA 压缩 KV + 索引器 top-k 选择、Hash + 可学习混合路由、MTP 多 token 预测
2. **内存工程**：SwapOptimizer CPU offload、AC 全量激活检查点、独立 FSDP mesh 减少 MoE 显存压力
3. **并行策略**：纯 FSDP128 + EP128（无 TP/PP），EP 和 FSDP 独立 ProcessGroup 重叠通信
4. **NPU 特化**：Converter 系统替换 eager 为融合算子（sparse attn、grouped matmul、MoE permute）
5. **训练稳定性**：Indexer loss 辅助监督、AutoScaler 保持梯度 scale 一致、分布式 loss 同步

在 BF16 + 4K 序列 + MTP1 配置下，该实现预期在昇腾 A3 64 卡集群达到 **~1100 tokens/GPU/s** 的吞吐。
