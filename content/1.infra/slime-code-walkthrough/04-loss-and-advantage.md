---
title: "Slime 代码走读 — 04. Loss 计算与 Advantage 估计"
date: 2026-06-15T12:00:00+08:00
draft: false
---

> 本文档深入解析 slime 的训练核心：`loss.py` 中的 Policy Loss、Value Loss、Advantage Estimation，以及 `ppo_utils.py` 中的 KL 计算、GAE、GRPO 等功能。

---

## 一、Loss 文件总览：`slime/backends/megatron_utils/loss.py`

```
loss.py (1031 lines)
  ├── get_responses()              # 从 logits 中提取 response 段
  ├── _allgather_cp_redistribute() # CP 环境下全聚合 + 重分布
  ├── get_log_probs_and_entropy()  # 计算 log_probs + 熵
  ├── get_values()                 # 提取 value 预测
  ├── apply_opd_kl_to_advantages() # OPD KL penalty
  ├── compute_advantages_and_returns() # 策略无关的 adv 计算调度
  ├── vanilla_tis_function()       # Truncated Importance Sampling
  ├── icepop_function()            # ICEPOP off-policy correction
  ├── policy_loss_function()       # 策略梯度损失（PPO/GSPO/...）
  ├── value_loss_function()        # Critic value loss
  ├── sft_loss_function()         # SFT loss
  └── loss_function()              # 调度器：microbatch loss scaling + Megatron 适配
```

---

## 二、`compute_advantages_and_returns()` — 优势估计调度器

```python
def compute_advantages_and_returns(args, rollout_data):
    log_probs = rollout_data.get("log_probs")
    ref_log_probs = rollout_data.get("ref_log_probs")
    rewards = rollout_data.get("rewards")
    values = rollout_data.get("values")
    
    if not is_pipeline_last_stage():
        return  # 非 pipeline 末 stage 不计算
    
    # ① 计算 KL divergences
    if args.kl_coef == 0:
        kl = [torch.zeros_like(log_probs[i]) for i in range(len(log_probs))]
    else:
        kl = [compute_approx_kl(log_probs[i], ref_log_probs[i]) for i in range(len(log_probs))]
    
    # ② 根据算法选择 advantage estimator
    match args.advantage_estimator:
        case "grpo" | "gspo":
            get_grpo_returns(rewards, kl)
        case "ppo":
            get_advantages_and_returns_batch(values, rewards, kl, gamma, lambd)
        case "reinforce_plus_plus":
            get_reinforce_plus_plus_returns(rewards, kl, loss_masks, response_lengths, total_lengths)
        case "reinforce_plus_plus_baseline":
            get_reinforce_plus_plus_baseline_advantages(rewards, kl, loss_masks)
    
    # ③ 可选：OPD token-level KL penalty
    if args.use_opd:
        apply_opd_kl_to_advantages(args, rollout_data, advantages, log_probs)
    
    # ④ 可选：advantage normalization across DP group
    if args.normalize_advantages:
        # gather all advantages across DP ranks
        # compute masked mean + std
        # distributed_masked_whiten()
```

### 2.1 GRPO Returns：`utils/ppo_utils.py:get_grpo_returns()`

```python
def get_grpo_returns(rewards: torch.Tensor, kl: list[torch.Tensor]):
    """最简单：returns = reward.expand(kl.shape)"""
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns
```

**GRPO 核心**（`advantages = returns`）：
- 没有 value model
- 每个 sample 的 advantage 就是它的 group-normalized return
- 优势来自 reward 的组内相对值（分母再除以标准差）

### 2.2 PPO GAE：`utils/ppo_utils.py:get_advantages_and_returns_batch()`

```python
def get_advantages_and_returns_batch(total_lengths, response_lengths, values_list, rewards_list, gamma, lambd):
    if cp_size > 1:
        # 先 all_gather CP ranks 的数据
        full_values_list = [all_gather_with_cp(v, total_len, resp_len) for v, ... in values_list]
        full_rewards_list = [all_gather_with_cp(r, total_len, resp_len) for r, ... in rewards_list]
    
    # pad 到 max_len [B, T]
    full_values = torch.zeros(B, max_len, device=device, dtype=dtype)
    full_rewards = torch.zeros(B, max_len, device=device, dtype=dtype)
    
    for i in range(B):
        full_values[i, :response_lengths[i]] = full_values_list[i]
        full_rewards[i, :response_lengths[i]] = full_rewards_list[i]
    
    # 选择 GAE 实现
    if not chunked:
        full_advantages, full_returns = vanilla_gae(rewards, values, gamma, lambd)
    else:
        full_advantages, full_returns = chunked_gae(rewards, values, gamma, lambd, chunk_size=128)
```

**原始 GAE**（O(T) 序列依赖）：
```python
def vanilla_gae(rewards, values, gamma, lambd):
    B, T = rewards.shape
    lastgaelam = torch.zeros(B)
    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else 0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        adv_rev.append(lastgaelam)
    full_advantages = torch.stack(adv_rev[::-1], dim=1)
    full_returns = full_advantages + values
```

**Chunked GAE**（参考文献：FlashLinearAttention 的 O(T/C) sequential + O(C²) parallel）：

```python
def chunked_gae(rewards, values, gamma, lambd, chunk_size=128):
    # 1. 计算 deltas
    next_values = torch.cat([values[:, 1:], torch.zeros(B, 1)], dim=1)
    deltas = rewards + gamma * next_values - values
    
    # 2. 逆序序列，把 backward GAE 变成 forward scan
    deltas_rev = torch.flip(deltas, dims=[1])
    
    # 3. Pad 到 chunk_size 倍数
    # 4. reshape: [B, n_chunks, chunk_size]
    # 5. 构建 intra-chunk 扫描矩阵 M (triangular Toeplitz)
    M = w ** (col - row)  # [chunk_size, chunk_size]
    
    # 6. 并行计算 intra-chunk 结果: S_local = deltas @ M
    deltas_flat = deltas_rev.reshape(B * n_chunks, chunk_size)
    S_local_flat = deltas_flat @ M
    S_local = S_local_flat.reshape(B, n_chunks, chunk_size)
    
    # 7. 外部保留传播：s_prev 跨 chunk 传递
    for c in range(n_chunks):
        S_global = S_local[:, c, :Lc] + s_prev.unsqueeze(1) * pow_vec[:Lc]
        S_rev[:, start:end] = S_global
        s_prev = S_global[:, -1]  # 下个 chunk 的初始状态
    
    # 8. flip back
    advantages = torch.flip(S_rev, dims=[1])
    returns = advantages + values
```

> **Chunked GAE 的意义**：在长序列（max_response_len=8192）下，O(T) 的反向扫描序列依赖会成为瓶颈。Chunked GAE 将依赖降到 O(T/C)，同时每个 chunk 内部完全并行化。

### 2.3 `compute_approx_kl()` — KL 散度近似

```python
@torch.compile(dynamic=True)
def compute_approx_kl(log_probs, log_probs_base, kl_loss_type, importance_ratio=None):
    log_ratio = log_probs.float() - log_probs_base.float()
    
    if kl_loss_type == "k1":
        kl = log_ratio
    elif kl_loss_type == "k2":
        kl = log_ratio**2 / 2.0
    elif kl_loss_type in ["k3", "low_var_kl"]:
        # k3 / low_var_kl 是 Schulman 博客推荐的无偏估计
        log_ratio = -log_ratio
        kl = log_ratio.exp() - 1 - log_ratio
    
    if importance_ratio is not None:
        kl = importance_ratio * kl  # DeepSeek-V3.2 unbiased estimation
    
    if kl_loss_type == "low_var_kl":
        kl = torch.clamp(kl, min=-10, max=10)
    
    return kl
```

---

## 三、`policy_loss_function()` — 策略损失核心

```python
def policy_loss_function(args, batch, logits, sum_of_sample_mean):
    """
    Returns: (loss_tensor, reported_loss_dict)
    """
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]
    
    # ① 重新计算当前策略的 log_probs
    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits, args=args, ...,
        with_entropy=True,
    )
    log_probs = log_probs_and_entropy["log_probs"]
    entropy = log_probs_and_entropy["entropy"]
    
    # ② 计算 PPO KL = old_log_probs - new_log_probs
    if args.advantage_estimator == "gspo":
        # GSPO 需要 full sequence logprobs（先 all_gather CP）
        full_log_probs = [all_gather_with_cp(p, total_len, resp_len) for p in log_probs]
        ppo_kl = compute_gspo_kl(full_log_probs, full_old_log_probs, local_log_probs, loss_masks)
    else:
        log_probs = torch.cat(log_probs)
        old_log_probs = torch.cat(old_log_probs)
        ppo_kl = old_log_probs - log_probs  # per-token KL
    
    # ③ PPO clipped policy gradient
    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)
    
    # ④ Optional: OPSM sequence masking
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(args, full_log_probs, full_old_log_probs, advantages, loss_masks)
        pg_loss = pg_loss * opsm_mask
    
    # ⑤ Optional: TIS (Truncated Importance Sampling)
    if args.use_tis or args.get_mismatch_metrics:
        ois = (-ppo_kl).exp()  # importance ratio
        pg_loss, modified_response_masks, tis_metrics = vanilla_tis_function(
            args, pg_loss, batch["log_probs"], batch["rollout_log_probs"], loss_masks
        )
        # sum_of_sample_mean 也需要用修改后的 mask 重新计算
        sum_of_sample_mean = get_sum_of_sample_mean(..., modified_response_masks)
    
    # ⑥ 聚合到标量
    pg_loss      = sum_of_sample_mean(pg_loss)
    pg_clipfrac  = sum_of_sample_mean(pg_clipfrac)
    ppo_kl       = sum_of_sample_mean(ppo_kl)
    entropy_loss = sum_of_sample_mean(torch.cat(entropy))
    
    # ⑦ 总 loss
    loss = pg_loss - args.entropy_coef * entropy_loss
    
    # ⑧ KL 正则（与 PPO KL 不同，直接对 ref model KL）
    if args.use_kl_loss:
        ref_log_probs = torch.cat(batch["ref_log_probs"])
        kl = compute_approx_kl(log_probs, ref_log_probs, args.kl_loss_type)
        kl_loss = sum_of_sample_mean(kl)
        loss = loss + args.kl_loss_coef * kl_loss
    
    return loss, {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
        "kl_loss": kl_loss.clone().detach() if args.use_kl_loss else None,
    }
```

### 3.1 `compute_policy_loss()` — PPO Clipped Loss

```python
@torch.compile(dynamic=True)
def compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None):
    """
    ratio = π_new / π_old = exp(-ppo_kl)
    pg_losses1 = -ratio * advantages                      [unclipped]
    pg_losses2 = -ratio.clamp(1-ε, 1+ε_high) * advantages  [clipped]
    pg_loss    = max(pg_losses1, pg_losses2)              [PPO objective]
    """
    ratio = (-ppo_kl).exp()
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()  # 裁剪比例
    
    # Optional: Dual-clip PPO (China AI Labs 常用)
    if eps_clip_c is not None:
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1
    
    return pg_losses, clipfrac
```

### 3.2 `vanilla_tis_function()` — 截断重要性采样

```python
def vanilla_tis_function(args, pg_loss, train_log_probs, rollout_log_probs, loss_masks):
    # ratio = π_train / π_rollout
    ratio = torch.exp(train_log_probs - rollout_log_probs)
    tis_abs = torch.abs(ratio - 1)
    
    # 截断：ratio 超出 [tis_clip_low, tis_clip] → 权重置 0（拒绝采样）
    tis_weights = torch.clamp(ratio, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != ratio).float()
    
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, {
        "tis": ratio.detach(),
        "tis_clipfrac": tis_clipfrac.detach(),
        "tis_abs": tis_abs.detach(),
    }
```

### 3.3 `compute_opsm_mask()` — 序列级掩码

```python
def compute_opsm_mask(args, full_log_probs, full_old_log_probs, advantages, loss_masks):
    """
    Off-Policy Sequence Masking (OPSM)
    当 advantage < 0 且序列 KL > delta 时，整个序列被 mask
    """
    ppo_kl = ((old_log_prob - log_prob) * loss_mask).sum() / loss_mask.sum()
    mask = ((advantage < 0) & (seq_kl > args.opsm_delta)).float()
    # mask = 0 → 整个序列的 loss = 0
    return 1 - mask, opsm_clipfrac
```

---

## 四、`value_loss_function()` — Critic Value Loss

```python
def value_loss_function(args, batch, logits, sum_of_sample_mean):
    old_values = torch.cat(batch["values"])     # rollout 时保存的 value
    
    _, values = get_values(logits, args, ...)    # 当前 critic 的 value 预测
    values = torch.cat([v.flatten() for v in values["values"]])
    
    returns = torch.cat(batch["returns"])
    
    # PPO value clipping
    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)  # 取 max of clipped / unclipped
    
    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())
    
    return loss, {
        "value_loss": loss.detach(),
        "value_clipfrac": values_clipfrac.detach(),
    }
```

---

## 五、`loss_function()` — Megatron 适配层

```python
def loss_function(args, batch, num_microbatches, logits):
    # ① 选择 loss 类型
    match args.loss_type:
        case "policy_loss": func = policy_loss_function
        case "value_loss":  func = value_loss_function
        case "sft_loss":     func = sft_loss_function
        case "custom_loss":  func = load_function(args.custom_loss_function_path)
    
    # ② 计算 loss
    loss, log = func(args, batch, logits, sum_of_sample_mean)
    
    # ③ allgather-CP 死锁保护：rank 可能无 loss token → 加 0 * logits.sum()
    if args.allgather_cp and cp_size > 1:
        loss = loss + 0 * logits.sum()
    
    # ④ Megatron gradient scaling
    global_batch_size = batch.get("dynamic_global_batch_size", args.global_batch_size)
    if not args.calculate_per_token_loss:
        loss = loss * num_microbatches / global_batch_size * dp_size
    else:
        loss = loss * cp_size
    
    # ⑤ 构造 Megatron logging dict
    num_tokens = sum(loss_mask.sum() for loss_mask in batch["loss_masks"])
    normalizer = num_tokens if args.calculate_per_token_loss else 1
    
    return (
        loss,
        normalizer,
        {
            "keys": list(log.keys()),
            "values": torch.tensor([num_samples_or_tokens] + list(log.values()), device=logits.device),
        },
    )
```

---

## 六、`get_log_probs_and_entropy()` — Log Prob 计算细节

```python
def get_log_probs_and_entropy(logits, args, unconcat_tokens, total_lengths, response_lengths, with_entropy=False):
    log_probs_list = []
    entropy_list = []
    
    for logits_chunk, tokens_chunk in get_responses(logits, args, ...):
        log_prob, entropy = calculate_log_probs_and_entropy(
            logits_chunk,
            tokens_chunk,
            mpu.get_tensor_model_parallel_group(),
            with_entropy=with_entropy,
            chunk_size=args.log_probs_chunk_size,
        )
        log_probs_list.append(log_prob.squeeze(-1))
        entropy_list.append(entropy)
    
    res = {"log_probs": log_probs_list}
    if with_entropy:
        res["entropy"] = entropy_list
    
    # CP 场景：allgather → redistribute → zigzag ring attention layout
    if args.allgather_cp:
        _allgather_cp_redistribute(res, logits=logits, ...)
    
    return torch.empty((0,), device=logits.device), res
```

**`calculate_log_probs_and_entropy()`**（`utils/ppo_utils.py`）：

```python
def calculate_log_probs_and_entropy(logits, tokens, tp_group, with_entropy=False, chunk_size=-1):
    logits = logits.contiguous()
    entropy = None
    
    if chunk_size > 0:
        # 分 chunk 计算，避免 vocab 过大导致 OOM
        num_chunks = (logits.size(0) - 1) // chunk_size + 1
        tokens_chunks = tokens.chunk(num_chunks, dim=0)
        logits_chunks = logits.chunk(num_chunks, dim=0)
        log_probs = []
        for tokens_chunk, logits_chunk in zip(tokens_chunks, logits_chunks):
            log_prob = compute_log_probs(logits_chunk.clone(), tokens_chunk, tp_group)
            log_probs.append(log_prob)
        log_prob = torch.cat(log_probs, dim=0)
        
        if with_entropy:
            entropys = []
            for _, logits_chunk in zip(tokens_chunks, logits_chunks):
                entropy = compute_entropy_from_logits(logits_chunk.clone(), tp_group)
                entropys.append(entropy)
            entropy = torch.cat(entropys, dim=0)
    else:
        log_prob = compute_log_probs(logits.clone(), tokens, tp_group)
        if with_entropy:
            entropy = compute_entropy_from_logits(logits.clone(), tp_group)
    
    return log_prob, entropy
```

**`compute_log_probs()`** — Megatron Fused Cross Entropy：

```python
def compute_log_probs(logits, tokens, process_group):
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
    logits = logits.unsqueeze(1)   # [seq_len, 1, vocab_size]
    tokens = tokens.unsqueeze(1) # [seq_len, 1]
    return -fused_vocab_parallel_cross_entropy(logits, tokens, process_group)
```

> `fused_vocab_parallel_cross_entropy` 是 Megatron 的全词表并行融合 kernel：TP rank 只计算词表子集的 cross entropy，然后用 allreduce 聚合。

---

## 七、Loss 计算完整数据流

```
Megatron Pipeline Forward Step
  │
  ├── model(input_ids=tokens, packed_seq_params=packed_seq_params)
  │     └── output_tensor ← Megatron GPTModel.forward() [TP/PP/CP]
  │
  ├── partial(loss_function, args, batch, num_microbatches)
  │     │
  │     ├── policy_loss_function()
  │     │     ├── get_log_probs_and_entropy()
  │     │     │     ├── get_responses()  [从 logits 切 response 段]
  │     │     │     └── calculate_log_probs_and_entropy()
  │     │     │           ├── compute_log_probs() [fused cross entropy]
  │     │     │           └── compute_entropy_from_logits() [_VocabParallelEntropy]
  │     │     │
  │     │     ├── full_log_probs = all_gather_with_cp()? [GSPO 时需要]
  │     │     ├── compute_gspo_kl() 或 ppo_kl = old_log_probs - log_probs
  │     │     ├── compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high)
  │     │     ├── OPSM/TIS（可选）
  │     │     ├── sum_of_sample_mean()  [按 token 数归一化]
  │     │     └── loss = pg_loss - entropy_coef * entropy + kl_loss_coef * kl
  │     │
  │     ├── Megatron gradient scaling: loss *= num_microbatches / GBS * dp_size
  │     ├── allgather-CP 死锁保护: loss += 0 * logits.sum()
  │     └── return (loss, normalizer, {"keys": [...], "values": torch.tensor([...])})
```

---

## 八、关键设计洞察

1. **Megatron Pipeline 的 loss_func 返回值** 不是单纯的 loss tensor，而是 `(loss, normalizer, log_dict)`：
   - `loss` 已经被 scale 过（gradient scaling）
   - `normalizer` 告诉 pipeline engine 除的是什么
   - `log_dict` 被 pipeline engine collect 并 allreduce

2. **sum_of_sample_mean** 是核心的归一化函数：
   - `calculate_per_token_loss=True`：loss / (sum of valid tokens)
   - `calculate_per_token_loss=False`：loss / (number of samples)
   - 在 CP 场景下，它还会处理 CP 切分导致的局部/全局归一化差异

3. **CP allgather CP redistribute**：这是 `allgather_cp` 模式下的关键操作——CP rank 先 allgather 全局 logits，计算完整的 response log probs，然后再 redist slice 回 local chunk。这是为了让 `old_log_probs` 和 `new_log_probs` 能逐 token 对齐做比率计算。

4. **TIS 和 OPSM 都是 on-policy 保护机制**：
   - TIS（Truncated Importance Sampling）在 token 级别拒绝 off-policy 数据
   - OPSM（Off-Policy Sequence Masking）在 sequence 级别拒绝偏离过大的样本
