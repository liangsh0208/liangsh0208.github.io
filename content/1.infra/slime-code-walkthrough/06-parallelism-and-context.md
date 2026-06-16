---
title: "Slime 代码走读 — 06. 并行策略与权重同步"
date: 2026-06-15T12:00:00+08:00
draft: false
---

> 本文档解析 slime 的多维并行策略（TP/PP/CP/DP/EP/VP）、上下文并行（CP）的工具实现，以及 Megatron 权重到 SGLang Engine 的同步机制。

---

## 一、并行策略总览

slime 基于 Megatron-LM 构建，支持六种并行维度：

| 并行维度 | 参数 | 说明 | 代码位置 |
|---------|------|------|---------|
| **TP** | `--tensor-model-parallel-size` | Tensor Parallel：把 attention/FFN 的权重按列或行切分 | Megatron Core |
| **PP** | `--pipeline-model-parallel-size` | Pipeline Parallel：按层切分到不同 rank | Megatron Core |
| **VP** | `--virtual-pipeline-model-parallel-size` | Virtual Pipeline：同一 rank 交替执行不同 pipeline stage | Megatron Core |
| **CP** | `--context-parallel-size` | Context Parallel：序列维度的并行（ring attention） | `cp_utils.py` |
| **DP** | `--data-parallel-size`（自动计算） | Data Parallel：不同 rank 处理不同 sample | Megatron Core |
| **EP** | `--expert-model-parallel-size` | Expert Parallel（MoE）：不同 expert 到不同 rank | Megatron Core |

### 本任务（306872698）的并行配置

```bash
--tensor-model-parallel-size 8    # TP=8（每 8 GPU 一个张量并行组）
--pipeline-model-parallel-size 1  # PP=1（所有层在一个 rank）
--context-parallel-size 1         # CP=1（单 rank 处理完整序列）
--expert-model-parallel-size 1    # EP=1（非 MoE 模型）

总 GPU = 32
world_size = 4 nodes × 8 GPU = 32
TP=8, PP=1, CP=1, VP=None

DP = world_size / (TP × PP × CP) = 32 / 8 = 4
即 4 个 data parallel 组，每组 8 GPU 做 TP
```

---

## 二、Context Parallel（CP）详解

CP 是序列维度的并行：把一个长序列（如 32768 tokens）切到 N 个 rank 上。在 `slime/backends/megatron_utils/cp_utils.py` 中实现了全套工具。

### 2.1 核心问题

Ring Attention 中的 `context-parallel` 需要：**前向时 local attention 计算后 allgather，反向时 reduce-scatter**。具体到 GRPO 场景中还有额外挑战：**log probs 需要在 CP 上对齐**（因为 PPO 的 `old_log_probs` 和 `new_log_probs` 必须逐 token 对应）。

### 2.2 `get_logits_and_tokens_offset_with_cp()` — 计算各 rank 的切片位置

```python
def get_logits_and_tokens_offset_with_cp(total_length, response_length, qkv_format="thd", max_seq_len=None):
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    assert cp_size > 1
    
    prompt_length = total_length - response_length
    
    # chunk_size = ceil(total_length / (2 * cp_size))
    # 把整个序列切成 2 * cp_size 个段，按 zigzag 分配
    if qkv_format == "thd":
        chunk_size = (total_length + 2 * cp_size - 1) // (2 * cp_size)
    else:
        chunk_size = (max_seq_len + 2 * cp_size - 1) // (2 * cp_size)
    
    # cp_rank 负责的两个 chunk
    # chunk_0: 正序分配的段
    # chunk_1: 逆序分配的段
    chunk_0 = (cp_rank * chunk_size, (cp_rank + 1) * chunk_size)
    chunk_1 = ((2 * cp_size - cp_rank - 1) * chunk_size, (2 * cp_size - cp_rank) * chunk_size)
    
    # response 对应的 logits 区间（prompt_length-1 到 total_length-1）
    logits_0 = (max(chunk_0[0], prompt_length - 1), min(chunk_0[1], total_length - 1))
    logits_1 = (max(chunk_1[0], prompt_length - 1), min(chunk_1[1], total_length - 1))
    
    # tokens 对应的区间（logits 需要 shift by 1）
    token_0 = (logits_0[0] + 1, logits_0[1] + 1) if logits_0[0] < logits_0[1] else (0, 0)
    token_1 = (logits_1[0] + 1, logits_1[1] + 1) if logits_1[0] < logits_1[1] else (0, 0)
    
    return chunk_size, (chunk_0, chunk_1), (logits_0, logits_1), (token_0, token_1)
```

### 2.3 `all_gather_with_cp()` — CP 数据全聚合

```python
def all_gather_with_cp(tensor, total_length, response_length):
    """Gather response tensors from all CP ranks via differentiable all-reduce."""
    cp_group = mpu.get_context_parallel_group()
    cp_size = mpu.get_context_parallel_world_size()
    
    # 当前 rank 拥有的 response 段
    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(total_length, response_length)
    
    # 左右两部分
    chunk_0 = tensor[: logits_offset[0][1] - logits_offset[0][0]]
    chunk_1 = tensor[logits_offset[0][1] - logits_offset[0][0] :]
    
    # 构建全序列张量（空位用 0 填充）
    left  = torch.zeros(logits_offset[0][0] - (prompt_length - 1), ...)
    mid   = torch.zeros(logits_offset[1][0] - logits_offset[0][1], ...)
    right = torch.zeros(total_length - 1 - logits_offset[1][1], ...)
    
    full_tensor = torch.cat([left, chunk_0, mid, chunk_1, right])
    assert full_tensor.shape[0] == response_length
    
    # Differentiable all-reduce 聚合各 rank 的部分
    full_tensor = dist.nn.all_reduce(full_tensor, group=cp_group)
    
    return full_tensor
```

> `dist.nn.all_reduce` 是**可微**的：前向时 sum across ranks，反向时 gradient 自动 broadcast back。这比 `all_gather` 更省显存。

### 2.4 `slice_log_prob_with_cp()` — logprob 的 CP 切片

```python
def slice_log_prob_with_cp(log_prob, total_length, response_length, qkv_format="thd", max_token_len=None):
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:
        return log_prob
    
    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(total_length, response_length, ...)
    
    # 从完整 logprob 中提取当前 rank 负责的段
    chunk_1 = log_prob[logits_offset[0][0] - (prompt_length - 1) : logits_offset[0][1] - (prompt_length - 1)]
    chunk_2 = log_prob[logits_offset[1][0] - (prompt_length - 1) : logits_offset[1][1] - (prompt_length - 1)]
    
    return torch.cat([chunk_1, chunk_2])
```

### 2.5 `_allgather_cp_redistribute()` — allgather-CP 后的重分布

```python
def _allgather_cp_redistribute(res, logits, args, total_lengths, response_lengths, max_seq_lens=None):
    """
    allgather_CP 后，每个 rank 持有的是连续的全局段。
    但 zigzag ring attention 期望交错切分。
    这个函数把 allgather 的数据重新 slice 成 zigzag 需要的 local chunks。
    """
    cp_group = mpu.get_context_parallel_group()
    cp_rank = mpu.get_context_parallel_rank()
    
    logits_local_len = logits.size(1)
    chunk_start = cp_rank * logits_local_len
    chunk_end = chunk_start + logits_local_len
    
    for key, values in res.items():
        full_resps = []
        seq_start = 0
        for value, total_length, response_length in zip(values, total_lengths, response_lengths):
            prompt_length = total_length - response_length
            
            # 计算当前 rank 在全局中的覆盖
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1
            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            
            if e <= s:
                full_resp = torch.zeros(response_length, dtype=value.dtype, device=value.device, requires_grad=True)
            else:
                resp_start = s - logit_global_start
                resp_end = e - logit_global_start
                full_resp = F.pad(value, (resp_start, response_length - resp_end))
            
            full_resps.append(full_resp)
            seq_start += total_length
        
        # Differentiable all-reduce 全局聚合
        all_cat = torch.cat(full_resps, dim=0)
        all_cat = dist.nn.all_reduce(all_cat, group=cp_group)
        
        # 重新 slice 成 zigzag CP 模式
        new_values = []
        for idx, (full_resp, total_length, response_length) in enumerate(zip(all_cat.split(response_lengths, dim=0), ...)):
            new_values.append(slice_log_prob_with_cp(full_resp, total_length, response_length, ...))
        
        res[key] = new_values
```

---

## 三、Pipeline Parallelism（PP）

### 3.1 PP stage 间数据广播：`sync_actor_critic_data()`

```python
def sync_actor_critic_data(args, rollout_data, group):
    """
    在 PP 中 actor 和 critic 可能分布在不同 stage。
    广播 values (from critic) 和 log_probs / ref_log_probs (from actor)。
    """
    values = rollout_data.get("values")
    log_probs = rollout_data.get("log_probs" if not args.use_rollout_logprobs else "rollout_log_probs")
    ref_log_probs = rollout_data.get("ref_log_probs")
    
    handles = []
    
    # Values from Critic (src=1)
    if not values:
        values = [torch.empty_like(log_prob) for log_prob in log_probs]
    for value in values:
        handles.append(dist.broadcast(value, src=1, group=group, async_op=True))
    
    # Log probs from Actor (src=0)
    if args.kl_coef != 0 or args.use_kl_loss:
        if not log_probs:
            log_probs = [torch.empty_like(value) for value in values]
        if not ref_log_probs:
            ref_log_probs = [torch.empty_like(value) for value in values]
        for ref_log_prob, log_prob in zip(ref_log_probs, log_probs):
            handles.append(dist.broadcast(log_prob, src=0, group=group, async_op=True))
            handles.append(dist.broadcast(ref_log_prob, src=0, group=group, async_op=True))
    
    for handle in handles:
        handle.wait()
    
    rollout_data.update({...})
```

**场景**：当 `use_critic=True` 且 Actor 和 Critic 是不同模型时：
- Actor 在 PP stage 0 得到 `log_probs`, `ref_log_probs`
- Critic 在 PP stage 1 得到 `values`
- 两者需要互相广播到对方 stage

---

## 四、权重同步：Megatron → SGLang Engine

```
┌──────────────────────────────────────────────────────┐
│ Megatron Actor (TP=8, PP=1, DP=4)                     │
│  - 90B params distributed across 32 GPU (TP sharded)   │
│  - Adam states on CPU (--optimizer-cpu-offload)       │
└──────────────────────────────────────────────────────┘
                          │ update_weights()
                          ▼
┌──────────────────────────────────────────────────────┐
│ UpdateWeightFromDistributed / UpdateWeightFromTensor  │
│  - Gather TP sharded params → full param on rank0     │
│  - Convert Megatron format → HuggingFace format       │
│  - Quantize (int4/fp8, optional)                      │
│  - Broadcast to rollout engines (NCCL or memory copy)  │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│ SGLang Engine (TP=8)                                    │
│  - Receive HF format weights                           │
│  - Convert to SGLang internal format                   │
│  - Load into GPU memory                                │
│  - Rebuild CUDA Graphs                                │
└──────────────────────────────────────────────────────┘
```

### 4.1 更新策略选择

```python
# slime/backends/megatron_utils/actor.py
update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed
self.weight_updater = update_weight_cls(
    self.args, self.model,
    weights_getter=lambda: self.weights_backuper.get("actor"),
    model_name=...,
    quantization_config=...,
)
```

| 模式 | 类 | 通信方式 | 场景 |
|------|---|---------|------|
| Colocate | `UpdateWeightFromTensor` | 直接 `torch.Tensor` 内存复制 | 同 GPU/节点，不走网络 |
| 分离部署 | `UpdateWeightFromDistributed` | NCCL broadcast | 跨节点，走 IB / NVLink |

### 4.2 `UpdateWeightFromDistributed.update_weights()`

```python
@torch.no_grad()
def update_weights(self):
    self.weight_version += 1
    
    # ① 暂停 SGLang 生成 + flush cache
    if dist.get_rank() == 0:
        ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
        ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
    dist.barrier(group=get_gloo_group())
    
    # ② 非 expert 参数：gather TP → remove padding → convert to HF → buffer → broadcast
    for name, param in named_params_and_buffers(self.args, self.model):
        if ".experts." in name:
            continue
        param = all_gather_param(name, param)  # TP gather
        if self._is_pp_src_rank:  # PP src rank (DP=0, TP=0)
            converted = convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
            buffer_size = self._update_weight_from_distributed(name, param, converted, buffer_size)
    
    if converted_named_tensors:
        self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)
    dist.barrier(group=get_gloo_group())
    
    # ③ Expert 参数（MoE）
    for name, param in named_params_and_buffers(...):
        if ".experts." not in name:
            continue
        param = all_gather_param(name, param)
        # EP gather + broadcast
        ...
    
    # ④ 量化后处理
    if dist.get_rank() == 0 and self.quantization_config:
        post_process_weights(restore_weights_before_load=False, post_process_quantization=True, ...)
    
    # ⑤ 恢复生成
    if dist.get_rank() == 0:
        ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
    dist.barrier(group=get_gloo_group())
```

### 4.3 关键：`all_gather_param()` — TP 维度聚合

```python
def all_gather_param(name, param):
    if not is_tensor_parallel_param(param):  # 无 TP 维度的参数
        return param
    
    # Megatron TP 的 allgather
    tp_group = mpu.get_tensor_model_parallel_group()
    gathered = [torch.empty_like(param) for _ in range(mpu.get_tensor_model_parallel_world_size())]
    dist.all_gather(gathered, param, group=tp_group)
    param = torch.cat(gathered, dim=tp_sharded_dimension)
    return param
```

### 4.4 `convert_to_hf()` — Megatron 格式转 HuggingFace

```python
# slime/backends/megatron_utils/megatron_to_hf/hf_weight_iterator_direct.py
def convert_to_hf(args, model_name, name, param, quantization_config):
    # 去除 Megatron 的 padding
    param = remove_padding(param, ...)
    
    # 重命名参数名：Megatron 命名空间 → HuggingFace 命名空间
    name = megatron_to_hf_name_map(name, model_name)
    
    # 量化
    if quantization_config and quantization_config["quant_method"] == "compressed-tensors":
        param = quantize_param(param, quantization_config)
    
    return (name, param.cpu())  # 返回 CPU Tensor（用于 ray.put / broadcast）
```

### 4.5 `UpdateWeightFromTensor`（Colocate 模式）

```python
class UpdateWeightFromTensor:
    def update_weights(self):
        self.weight_version += 1
        
        if self._is_pp_src_rank:
            for name, param in named_params_and_buffers(self.args, self.model):
                param = all_gather_param(name, param)
                param = convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
                self._weight[name] = param
        
        if self._is_pp_src_rank:
            for name, param in self._weight.items():
                for rollout_engine in self.rollout_engines:
                    handle = rollout_engine.update_weight.remote(name, param)
                    handles.append(handle)
```

---

## 五、并行关系图解

```
32 GPU, 4 节点, 8 GPU/node, Qwen3-32B

Placement Group Bundles:
  bundle 0 (node 0): GPU 0-7
  bundle 1 (node 1): GPU 8-15
  bundle 2 (node 2): GPU 16-23
  bundle 3 (node 3): GPU 24-31

Colocate 模式:
  ┌─────────────────────────────────────┐
  │ bundle 0 (node 0)                   │
  │  GPU 0-7: SGLang Engine (TP=8) + Megatron Actor (TP=8, DP=0)
  │         分时复用（offload 切换）     │
  └─────────────────────────────────────┘
  ┌─────────────────────────────────────┐
  │ bundle 1-3 (node 1-3)               │
  │  GPU 8-31: 同 bundle 0 模式           │
  └─────────────────────────────────────┘
  DP=4, 每组 TP=8, 每组内 8 GPU 同时

分离部署模式:
  ┌─────────────────────────────────────┐
  │ bundle 0-1 (node 0-1): GPU 0-15     │
  │  SGLang Engines (rollout, TP=8, 占 2 组)
  └─────────────────────────────────────┘
  ┌─────────────────────────────────────┐
  │ bundle 2-3 (node 2-3): GPU 16-31    │
  │  Megatron Actors (train, TP=8, DP=4)
  └─────────────────────────────────────┘
```

---

## 六、并行策略选择建议

| 模型大小 | GPU 数 | 推荐策略 | 理由 |
|---------|--------|---------|------|
| ~7B | 4-8 | TP=4/8, DP=1 | 小模型，通信开销占主导 |
| 32B | 16-32 | TP=8, DP=2-4 | 需要较大 batch，DP 提升吞吐 |
| 70B+ | 64+ | TP=8, PP=2+, CP=2+ | 全长序列需要 CP，PP 降低显存 |
| MoE (mixtral) | 64+ | TP=4, EP=4, DP=4 | Expert 到不同节点，EP 通信最高效 |

**关键洞察**：

1. **TP=8 的约定**：H100/H20 节点内通常有 NVLink，TP 应该限制在节点内部，否则跨节点 TP 通信会慢 10x~
2. **DP 增加会增加 weight sync 通信**：每次 rollout 后需要把更新后的权重传到所有 SGLang engines，DP=4 意味着有 4 组独立更新的 Megatron 进程
3. **CP > 1 时 `get_batch()` 和 `loss.py` 都会启用 CP 路径**：log_probs 需要 `all_gather_with_cp`，这是隐藏的性能瓶颈
