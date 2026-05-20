---
created: 2026-05-06
---

# Megatron-LM MoE混合专家架构

> 【源码定位】|【阅读建议】|【前置知识】
> - **源码定位**: `megatron/core/transformer/moe/` - 包含router.py, token_dispatcher.py, experts.py等
> - **阅读建议**: 重点理解论文核心概念: EP+TP混合并行、TopK路由、共享专家
> - **前置知识**: DeepSeek-V3论文、All2All通信、负载均衡理论

---

## 1. 模块概述

Megatron-LM的MoE实现是生产级深度集成，核心特性：
- **混合并行**: Expert Parallel (EP) + Tensor Parallel (TP) 组合
- **路由算法**: TopK/Greedy/Sinkhorn/Expert Choice多种策略
- **负载均衡**: aux_loss/seq_aux_loss/global_aux_loss多级均衡
- **共享专家**: DeepSeek-V3风格永久激活专家

### 【重点】MoE模块组织

| 文件 | 功能 | 规模 |
|------|------|------|
| `moe_layer.py` | MoE层封装 | 31106字节 |
| `router.py` | 路由算法实现 (TopK/Sinkhorn/专家偏置) | 72901字节 |
| `token_dispatcher.py` | All2All token分发与收集 | 68560字节 |
| `experts.py` | 专家前向/后向 (TP分组或TP辅助并行) | 38225字节 |
| `shared_experts.py` | DeepSeek共享专家实现 | 15817字节 |
| `moe_utils.py` | 辅助损失计算、工具函数 | 34216字节 |

---

## 2. 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MoE层架构 (Megatron-LM DeepSeek风格)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   MoE Layer输入: [sequence_length, batch_size, hidden_size]                │
│       │                                                                     │
│       ▼ 1. 路由器计算                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Router (TopKRouter / SinkhornRouter / ExpertChoiceRouter)          │   │
│   │  ┌───────────────────────────────────────────────────────────────┐  │   │
│   │  │ Step 1: 计算路由分数 (线性投影)                               │  │   │
│   │  │   logits = hidden @ W_router  → [num_tokens, num_experts]    │  │   │
│   │  │                    (W_router形状: [num_experts, hidden_size]) │  │   │
│   │  │ Step 2: 应用负载均衡 (可选expert偏置)                         │  │   │
│   │  │   if moe_router_enable_expert_bias: logits += expert_bias     │  │   │
│   │  │ Step 3: TopK选择 (或Sinkhorn/Expert Choice)                   │  │   │
│   │  │   probs, indices = topk(softmax(logits), k=topk)              │  │   │
│   │  │   → probs: [num_tokens, topk], indices: [num_tokens, topk]   │  │   │
│   │  │ Step 4: 归一化                                                │  │   │
│   │  │   probs = probs / sum(probs, dim=-1, keepdim=True)            │  │   │
│   │  └───────────────────────────────────────────────────────────────┘  │   │
│   │       │ Output: Routing map [num_tokens, num_experts], Prs.        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼ 2. 路由映射处理                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Token Dispatcher (ExpertAllToAllTokenDispatcher)                   │   │
│   │  ┌───────────────────────────────────────────────────────────────┐  │   │
│   │  │ 本地重排 + All-To-All 发送到目标专家                         │  │   │
│   │  │                                                               │  │   │
│   │  │ 本地重排: 将tokens按目标expert分组                           │  │   │
│   │  │   [token1→exp2, token2→exp5, ...] → [exp1_batch, exp2_batch] │  │   │
│   │  │                                                               │  │   │
│   │  │ All-To-All: 跨EP组的节点间token交换                          │  │   │
│   │  │   EP组内各rank发送分配给远程expert的tokens                   │  │   │
│   │  │   接收本地需处理的expert的tokens                             │  │   │
│   │  │   → 输出: [local_tokens, num_local_experts, hidden]        │  │   │
│   │  └───────────────────────────────────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼ 3. 专家计算                                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Expert处理 (TP并行)                                                │   │
│   │                                                                    │   │
│   │  专家分布: 4个节点, EP=4, TP=2, 64个专家 → 每节点16个专家         │   │
│   │                                   → 每2卡TP组共享8个专家            │   │
│   │                                                                    │   │
│   │  Node 0 (EP group 0):           Node 1 (EP group 1):              │   │
│   │  ┌────┐┌────┐                 ┌────┐┌────┐                       │   │
│   │  │GPU0││GPU1│                 │GPU2││GPU3│                       │   │
│   │  │TP0 ││TP1 │                 │TP0 ││TP1 │                       │   │
│   │  │E0-7││E0-7│                 │E16-23││E16-23│                   │   │
│   │  └────┘└────┘                 └────┘└────┘                       │   │
│   │    ↑ AllGather/Reduce-Scatter within TP (同03章TP模式)          │   │
│   │                                                                    │   │
│   │  每个专家: ColumnParallel(FC1) → Activation → RowParallel(FC2)  │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼ 4. 结果汇聚                                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Token反向交换                                                      │   │
│   │  - All-To-All将专家输出送回原始token位置                           │   │
│   │  - 按probs权重加权聚合                                             │   │
│   │ output = sum(probs[i] * expert_output[i] for i in topk)            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼ 5. 共享专家 (可选，DeepSeek风格)                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Shared Experts (shared_experts.py)                                 │   │
│   │  - 每token除 routed experts 外，还经过1个 permanent shared expert  │   │
│   │  - 结果与MoE输出相加                                               │   │
│   │  final_output = moe_output + shared_expert_output                  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼ 6. 辅助损失计算                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Aux Loss (moe_utils.py)                                            │   │
│   │  - load_balancing_loss: 确保各expert token数均衡                  │   │
│   │  - z_loss: 防止router输出过度尖锐                                │   │
│   │  loss += moe_aux_loss_coeff * (load_balancing_loss + z_loss)     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心概念与实现

### 3.1 TopK路由实现

```python
# 文件: megatron/core/transformer/moe/router.py

class TopKRouter(Router):
    """TopK路由实现。
    
    算法步骤:
    1. 线性投影计算路由分数
    2. (可选) 应用专家偏置(expert bias)
    3. TopK选择专家
    4. (可选) 负载均衡辅助损失
    """
    
    def __init__(self, config: TransformerConfig, ...):
        super().__init__(config, ...)
        self.topk = config.moe_router_topk
        self.score_function = config.moe_router_score_function  # softmax/sigmoid
        
        # 专家偏置 (动态调整expert利用率)
        self.enable_expert_bias = config.moe_router_enable_expert_bias
        
    def gating(self, input: torch.Tensor) -> torch.Tensor:
        """计算路由门控分数。
        
        Args:
            input: [num_tokens, hidden_size]
        Returns:
            logits: [num_tokens, num_moe_experts]
        """
        # 线性投影: [num_tokens, hidden] @ [hidden, num_experts].T
        logits = F.linear(input, self.weight, self.bias)
        return logits
    
    def routing(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行TopK路由选择。
        
        Returns:
            scores: [num_tokens, topk] 选中专家的权重
            indices: [num_tokens, topk] 选中专家的索引
        """
        # (可选) 应用训练时的噪声/抖动
        if self.training and self.config.moe_router_jitter_noise > 0:
            logits = logits * self._generate_noise(logits)
        
        # 应用专家偏置 (如果启用)
        if self.enable_expert_bias and self.training:
            self._maintain_float32_expert_bias()
            logits = logits + self.expert_bias
        
        # 计算分数 (softmax/sigmoid)
        if self.score_function == 'softmax':
            scores_all = F.softmax(logits, dim=-1, dtype=torch.float32)
        else:  # sigmoid
            scores_all = torch.sigmoid(logits)
            
        # TopK选择
        scores, indices = torch.topk(scores_all, k=self.topk, dim=-1)
        
        # 归一化 (确保topk权重和为1)
        if self.config.moe_router_normalize_topk_prob:
            scores = scores / scores.sum(dim=-1, keepdim=True)
            
        # 构建routing_map (用于dispatch)
        routing_map = torch.zeros_like(logits, dtype=torch.bool)
        routing_map.scatter_(1, indices, True)
        
        # 应用aux loss (负载均衡)
        if self.training and self.is_aux_loss_enabled():
            scores = self._apply_aux_loss(
                scores, scores_all, routing_map
            )
            
        return scores, indices
    
    def _apply_aux_loss(self, probs, scores_for_aux_loss, routing_map):
        """应用辅助负载均衡损失。
        
        关键指标: 负载均衡度 = max(tokens_per_expert) / min(tokens_per_expert)
        目标: 使各expert处理的token数接近均匀分布
        """
        # 统计每个expert的token数
        tokens_per_expert = routing_map.sum(dim=0)
        
        # 计算辅助损失 (类似DeepSeek/MoE论文的负载均衡损失)
        # loss_aux = num_experts * sum(fraction_expert_i * mean_routing_prob_i)
        aux_loss = compute_load_balancing_loss(
            probs=probs,
            tokens_per_expert=tokens_per_expert,
            topk=self.topk,
        )
        
        # 通过MoEAuxLossAutoScaler注册到全局
        save_to_aux_losses_tracker(
            'load_balancing_loss',
            aux_loss,
            self.layer_number,
            self.config.num_layers,
        )
        
        return probs
```

### 3.2 Token分发实现

```python
# 文件: megatron/core/transformer/moe/token_dispatcher.py

class ExpertAllToAllTokenDispatcher:
    """EP组内All2All token分发器。
    
    处理流程:
    1. 本地重排: 根据routing_map组织token发往各expert的buffer
    2. All2All: 跨EP rank交换token
    3. 本地batching: 为本地管理的experts批量处理
    """
    
    def __init__(self, config: TransformerConfig, ...):
        self.num_experts = config.num_moe_experts
        self.ep_size = parallel_state.get_expert_model_parallel_world_size()
        # 每EP rank管理的专家数
        self.num_local_experts = self.num_experts // self.ep_size
        
    def dispatch(self, hidden: torch.Tensor, router_probs: torch.Tensor, 
                 routing_map: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """分发tokens到目标experts。
        
        Args:
            hidden: [num_tokens, hidden_size]
            router_probs: [num_tokens, topk] 权重
            routing_map: [num_tokens, num_experts] bool掩码
            
        Returns:
            dispatched_input: [num_local_tokens, num_local_experts, hidden]
            tokens_per_expert: 各专家接收的token数 (用于计算aux loss)
        """
        # Step 1: 统计每expert的token数
        num_tokens_per_expert = routing_map.sum(dim=0, dtype=torch.int32)
        
        # Step 2: 计算本地需要接收的token数 (用于分配buffer)
        # All-Gather各rank的tokens_per_expert，确定buffer大小
        
        # Step 3: 本地重排 - 按目标expert分组
        # permuted_input: 重排后的token [total_tokens_for_dispatch, hidden]
        # permuted_probs: 对应权重
        
        # Step 4: All-To-All通信
        # 输入: [EP, tokens_for_rank_i, hidden]
        # 输出: [EP, tokens_from_rank_i, hidden] (实际是2D tensor [E*M, H])
        
        # Step 5: unpack到[num_local_experts, max_tokens, hidden]
        
        return hidden_states, tokens_per_expert, ...
    
    def combine(self, expert_output: torch.Tensor, ...):
        """收集各expert输出并复原顺序。
        
        步骤是dispatch的逆过程:
        1. All-To-All将输出送回源rank
        2. 按原token顺序unpermute
        3. 按router_probs加权求和
        """
        # All-To-All回到源rank
        # Unpermute恢复原始token顺序
        # 加权聚合
        output = torch.zeros_like(input_hidden)
        for k in range(topk):
            output += router_probs[:, k:k+1] * expert_k_output
            
        return output
```

### 3.3 共享专家 (DeepSeek-V3风格)

```python
# 文件: megatron/core/transformer/moe/shared_experts.py

class SharedExpert(MegatronModule):
    """共享专家实现 (DeepSeek-V3风格)。
    
    特性:
    - 所有token必经过此expert (非routed)
    - 可与routed MoE专家相加或级联
    - 通常使用更大FFN维度
    """
    
    def __init__(self, config: TransformerConfig, ...):
        super().__init__()
        
        # 共享专家通常是单个大型MLP
        # 或使用TP并行的标准专家结构
        self.num_shared_experts = config.num_shared_experts  # 通常1个
        
        # 创建共享专家 (使用标准TP并行)
        self.shared_experts = torch.nn.ModuleList([
            SharedExpertMLP(config) for _ in range(self.num_shared_experts)
        ])
        
        # 融合权重 (控制共享专家贡献)
        self.shared_expert_gate = torch.nn.Linear(
            config.hidden_size, 1  # gate分数
        ) if config.moe_shared_expert_gate else None
    
    def forward(self, hidden_states):
        """共享专家前向。"""
        # 所有token经过共享专家
        shared_output = sum(expert(hidden_states) for expert in self.shared_experts)
        
        # 可选: 学习gate控制融合强度
        if self.shared_expert_gate is not None:
            gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
            shared_output = shared_output * gate
            
        return shared_output

# 在MoE层中使用:
class MoELayer:
    def forward(self, hidden_states):
        # 1. Routed MoE输出
        moe_output = self.moe_experts(hidden_states)
        
        # 2. Shared Expert输出
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            # 相加 (DeepSeek-V3风格) 或 级联
            output = moe_output + shared_output
        else:
            output = moe_output
            
        return output
```

### 3.4 与nanotron对比

| 对比项 | Megatron-LM | nanotron |
|--------|-------------|----------|
| **MoE支持** | 原生深度集成 | 基础或需外部 |
| **路由算法** | TopK/Greedy/Sinkhorn/Expert Choice | 通常只TopK |
| **EP并行** | 原生专家并行 | 需手动实现 |
| **共享专家** | DeepSeek-V3完整实现 | 无 |
| **负载均衡** | 多级aux_loss (local/global/seq) | 简单aux_loss |
| **Token限制** | 容量因子控制，支持token drop | 基础实现 |
| **All2All优化** | 精细实现，支持异步 | 简单实现 |
| **与TP集成** | EP+TP混合，专家内TP | 通常分离 |

---

## 4. 配置参数

| 参数名 | 类型 | 典型值 | 说明 |
|--------|------|--------|------|
| `num_moe_experts` | int | 64/256/512 | 总专家数 |
| `moe_router_topk` | int | 2/4/8 | 激活专家数 |
| `expert_model_parallel_size` | int | 1-64 | EP并行度 |
| `moe_router_load_balancing_type` | str | "aux_loss" | 负载均衡类型 |
| `moe_aux_loss_coeff` | float | 0.01-0.1 | 辅助损失系数 |
| `moe_router_score_function` | str | "softmax" | 打分函数 |
| `moe_router_enable_expert_bias` | bool | False | 启用专家偏置 |
| `num_shared_experts` | int | 0-4 | 共享专家数 |
| `moe_expert_capacity_factor` | float | 1.25-2.0 | 容量溢出缓冲 |

---

## 5. 常见问题与排查

**Q: Load balancing loss异常高**

```python
# 诊断: 检查moe_aux_loss_coeff配置
# 典型范围0.01-0.1，过大抑制太多
# 检查expert bias是否在合理范围
config = TransformerConfig(
    moe_aux_loss_coeff=0.01,
    moe_router_enable_expert_bias=True,  # 自动调整
)
```

**Q: All2All通信性能瓶颈**

```bash
# 诊断: 使用nsys查看通信耗时
nsys profile -t cuda,nvtx,osrt train.py ...

# 优化: 增加EP size但减小TP，或检查节点间带宽
```

---

## 6. 参考资料

- **核心文件**: `megatron/core/transformer/moe/` (moe_layer.py, router.py, token_dispatcher.py)
- **交叉引用**: [02_并行状态管理](02_并行状态管理与进程组.md), [08_优化器](08_优化器与分布式优化.md)
- **论文**: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (MoE架构)
- **开源参考**: `deepseek-ai/DeepSeek-V3` repo (MoE实现细节)
