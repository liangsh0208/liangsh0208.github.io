---
title: "Slime GRPO 训练日志全解析（306872698）"
date: 2026-06-15T12:00:00+08:00
draft: false
---

> 本笔记基于 **任务 306872698**（`dc_slime_qwen3-32b_4node`）及其 slime 源码，逐行拆解每一条训练日志的含义与代码来源。

---

## 一、任务概览

| 字段 | 值 |
|------|-----|
| **任务 ID** | 306872698 |
| **任务名称** | `dc_slime_qwen3-32b_4node` |
| **基础模型** | [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) |
| **训练框架** | slime |
| **算法** | GRPO（Group Relative Policy Optimization） |
| **资源配置** | 4 节点 × 8 GPU = **32 × H20-3e** |
| **prompt 数据** | `dapo-math-17k.jsonl` |
| **评测数据** | `aime_2024_problems.jsonl` |
| **总 rollout** | 1000 |
| **状态** | complete ✅ |

### 关键超参

```bash
--num-rollout 1000
--rollout-batch-size 32          # 每轮 32 个 prompt
--n-samples-per-prompt 8           # 每个 prompt 8 条 response
--global-batch-size 256            # 训练 global batch
--rollout-max-response-len 8192   # 生成长度上限
--advantage-estimator grpo         # GRPO（无 value model）
--eps-clip 0.2 --eps-clip-high 0.28
--lr 1e-6 --lr-decay-style constant
--tensor-model-parallel-size 8
--use-dynamic-batch-size
```

> 注：`kl-loss-coef=0.00`、`entropy-coef=0.00`，即 KL 与熵正则虽有计算但权重为 0，属于**纯 policy gradient** 训练。

---

## 二、训练主循环（`train.py`）

```python
for rollout_id in range(args.num_rollout):       # 1000 轮
    rollout_data = generate(rollout_id)           # ① SGLang 采样
    actor_model.train(rollout_id, rollout_data)   # ② Megatron 训练
    eval(rollout_id)                              # ③ 周期性评测
```

每个 rollout 的完整链路：

```
generate → log_rollout → compute log_probs/ref_log_probs
→ compute_advantages → train_one_step × N_steps → log_perf
```

---

## 三、Rollout 阶段日志（`rollout/` 前缀）

**源码位置**：`slime/backends/megatron_utils/data.py:log_rollout_data()`

该函数在 PP last stage + TP rank 0 上执行，汇总 DP 各 rank 数据后打印。

| 指标名 | 代码来源 | 含义 |
|--------|---------|------|
| `rollout/log_probs` | `get_log_probs_and_entropy()` | **当前 Actor 策略**对生成 response 的对数概率（采样时的旧策略） |
| `rollout/ref_log_probs` | Reference Model forward | **SFT Reference Model**的对数概率，用于 KL 计算 |
| `rollout/rollout_log_probs` | 与 log_probs 相同 | 存入 rollout_data，训练时直接复用为 `old_log_probs` |
| `rollout/entropy` | `calculate_log_probs_and_entropy()` | **策略熵**。越高越随机，越低越确定 |
| `rollout/returns` | `compute_advantages_and_returns()` → `get_grpo_returns()` | **回报值**，GRPO 中 = `reward - kl_penalty` |
| `rollout/advantages` | 同上 | **优势值**。GRPO 中 `advantages = returns`，是 PPO loss 的核心信号 |
| `rollout/raw_reward` | Reward Model (`rm-type: deepscaler`) | DeepScaler reward。数学题答对≈1，答错≈0（或连续值） |
| `rollout/rewards` | raw_reward 经 KL 修正后 | 用于计算 advantage 的实际奖励 |
| `rollout/response_lengths` | SGLang 输出 | 每条 response 的 token 长度 |
| `rollout/total_lengths` | prompt + response | 完整序列长度 |
| `rollout/truncated` | SGLang 截断标记 | 被 `max_response_len=8192` 截断的样本数 |
| `rollout/truncated_ratio` | `truncated / count` | **截断比例**，越高说明模型越容易触顶生成长文本 |
| `rollout/repetition_frac` | `has_repetition()`（zlib 压缩比） | **复读/重复文本比例**。超过阈值视为 repetition |
| `rollout/response_len/mean` | 对 response_lengths 求均值 | 平均回答长度 |
| `rollout/response_len/max/min/median` | 统计量 | 回答长度极值与中位数 |
| `rollout/zero_std/count_0` | GRPO 分组统计 | **零标准差组数**：同一 prompt 的 8 个样本 reward 全相同（通常是全错），GRPO **无法提供有效梯度** |
| `rollout/zero_std/count_1` | 同上 | 标准差非零的组数，即**有区分度的组数** |

> 💡 **GRPO 特殊点**：`advantage_estimator=grpo` 时，不依赖 value model，advantages 直接由回报计算得出。

---

## 四、训练步骤日志（`train/` 前缀）

**源码位置**：`slime/backends/megatron_utils/model.py:train()` → 每步 `train_one_step()` 后打印

```python
log_dict = {
    f"train/{role_tag}{key}": val.mean().item()
    for key, val in loss_dict.items()
}
log_dict[f"train/{role_tag}grad_norm"] = grad_norm
log_dict[f"train/{role_tag}lr-pg_{id}"] = opt_param_scheduler.get_lr(param_group)
log_dict["train/step"] = rollout_id * num_steps_per_rollout + step_id
logger.info(f"{role_tag}step {accumulated_step_id}: {log_dict}")
```

| 指标名 | 来源 | 含义 |
|--------|------|------|
| `train/loss` | `loss_function()` | **总损失** = `pg_loss - entropy_coef * entropy_loss + kl_loss_coef * kl_loss` |
| `train/pg_loss` | `compute_policy_loss()` | **策略梯度损失**（PPO clipped surrogate loss） |
| `train/entropy_loss` | `calculate_log_probs_and_entropy()` | **平均 token 熵**。加了负号参与总 loss：越高 → loss 越小 |
| `train/pg_clipfrac` | `(ratio.clamp != ratio).mean()` | **PPO 裁剪比例**。过高说明策略变化过快 |
| `train/ppo_kl` | `old_log_probs - log_probs` | **近似 KL 散度**，监控新旧策略偏离程度 |
| `train/kl_loss` | `compute_approx_kl(log_probs, ref_log_probs)` | **KL 正则损失**（本任务权重为 0，实际为 0） |
| `train/grad_norm` | `optimizer.get_grad_norm()` | **梯度范数**。爆炸/消失均可能指示训练异常 |
| `train/lr-pg_0` | `opt_param_scheduler.get_lr()` | **参数组 0 学习率**（本任务 constant 1e-6） |
| `train/lr-pg_1` | 同上 | 参数组 1 学习率（通常 weight decay 不同） |
| `train/train_rollout_logprob_abs_diff` | `(old_log_probs - rollout_log_probs).abs()` | **训练策略 vs 采样策略的 logprob 绝对差**。异步训练时可能变大 |

---

## 五、性能日志（`perf/` 前缀）

**源码位置**：`slime/utils/train_metric_utils.py:log_perf_data_raw()`

基于 `Timer`（`slime/utils/timer.py`，单例模式）收集各阶段耗时后计算。

| 指标名 | 含义 | 计算方式 |
|--------|------|---------|
| `perf/rollout_time` | 一次 rollout 总耗时（生成 256 条 response） | `Timer("rollout_time")` |
| `perf/log_probs_time` | 计算旧策略 log_probs 耗时 | `Timer("log_probs_time")` |
| `perf/log_probs_tflops` | log_probs 计算效率 | `total_fwd_flops / log_probs_time` |
| `perf/ref_log_probs_time` | Reference model 计算耗时 | `Timer("ref_log_probs_time")` |
| `perf/ref_log_probs_tflops` | ref model 效率 | `total_fwd_flops / ref_log_probs_time` |
| `perf/actor_train_time` | Actor 训练总耗时 | `Timer("actor_train_time")` |
| `perf/actor_train_tflops` | Actor 训练有效 TFLOPS | `3 × total_fwd_flops / actor_train_time`（×3 for fwd+bwd） |
| `perf/actor_train_tok_per_s` | Actor 训练吞吐 | `sum(seq_lens) / actor_train_time` |
| `perf/train_time` | 训练主阶段耗时 | `Timer("train_time")` |
| `perf/train_wait_time` | 等待时间（数据准备、同步） | `Timer("train_wait_time")` |
| `perf/step_time` | 整轮 step 时间 | `train_wait_time + train_time` |
| `perf/wait_time_ratio` | 等待时间占比 | `train_wait_time / step_time` |
| `perf/data_preprocess_time` | rollout 数据预处理时间 | `Timer("data_preprocess_time")` |
| `perf/update_weights_time` | SGLang 权重更新时间 | `Timer("update_weights_time")` |
| `perf/save_model_time` | 模型保存耗时 | `Timer("save_model_time")` |
| `perf/tokens_per_gpu_per_sec` | 每 GPU 每秒 token 数 | 总吞吐 / GPU 数 |
| `perf/effective_tokens_per_gpu_per_sec` | 有效 token（非 padding）吞吐 | 更真实的 FLOPS 利用率 |
| `perf/longest_sample_tokens_per_sec` | 最长样本 token 处理速度 | 评估 memory-bound 瓶颈 |

---

## 六、评测日志（`eval/` 前缀）

每 `eval_interval=20` 个 rollout 跑一次 AIME2024 评测。

| 指标名 | 含义 |
|--------|------|
| `eval/aime` | **AIME 2024 评测得分**（pass@k / accuracy） |
| `eval/aime/response_len/mean` | 评测时回答平均长度 |
| `eval/aime/response_len/max/min/median` | 长度统计量 |
| `eval/aime/truncated_ratio` | 被 `eval_max_response_len=16384` 截断的比例 |
| `eval/aime/repetition_frac` | 评测回答复读比例 |

---

## 七、关键源码链路映射

```
train.py rollout loop (1000 轮)
  │
  ├── rollout_manager.generate(rollout_id)
  │     └── SGLang 采样 32 prompt × 8 samples
  │     └── rollout_data (tokens, log_probs, ref_log_probs, rewards…)
  │
  ├── data.py: log_rollout_data()
  │     └── 打印 rollout/* 日志
  │     └── gather_log_data("rollout", …)   ← DP rank 聚合
  │
  ├── loss.py: compute_advantages_and_returns()
  │     └── get_grpo_returns(reward, kl)   ← GRPO: returns = reward - kl
  │     └── advantages = returns              ← GRPO 无 value model
  │
  ├── model.py: train()
  │     └── train_one_step() × num_microbatches
  │           └── loss_function()
  │                 ├── policy_loss_function()
  │                 │     ├── get_log_probs_and_entropy()   ← 当前策略 logp
  │                 │     ├── compute_policy_loss()          ← PPO clipped loss
  │                 │     └── return (loss, reported_loss)
  │           └── logger.info(f"step {id}: {log_dict}")       ← train/* 日志
  │
  └── train_metric_utils.py: log_perf_data_raw()              ← perf/* 日志
```

---

## 八、训练有效性速查表

| 监控目标 | 关注指标 | 正常范围 / 期望趋势 |
|---------|---------|-------------------|
| **能力提升** | `rollout/raw_reward` → `eval/aime` | 逐步上升 |
| **策略稳定** | `train/pg_clipfrac` | 通常 < 0.2，过高表示变化过快 |
| **策略变化** | `train/ppo_kl` | 小正值，稳定或缓慢增长 |
| **梯度健康** | `train/grad_norm` | 不出现 NaN/Inf，不趋近于 0 |
| **学习效率** | `rollout/zero_std/count_0` | 越低越好（高说明 reward 无区分度） |
| **生成质量** | `rollout/truncated_ratio` | 适中；过高=触顶，过低=太短 |
| **重复问题** | `rollout/repetition_frac` | 越低越好 |
| **系统效率** | `perf/actor_train_tflops` | 接近理论峰值比例 |
| **瓶颈定位** | `perf/wait_time_ratio` | 过高说明 generate/train 流水线不均衡 |

---

## 九、RL 训练性能优化要点 & MFU 有效性分析

### 9.1 Pretrain vs RL：性能本质差异

| 维度 | Pretrain / SFT | RL Training (GRPO/PPO) |
|------|---------------|----------------------|
| **计算模式** | 单一 fwd + bwd，高度规律 | 多阶段异步流水线：generate → log_probs → ref_log_probs → advantage → train → update_weights |
| **时间构成** | 90%+ 是计算密集型的 matmul | generate 阶段 decode 是 **memory-bound**，train 才是 compute-bound |
| **并行策略** | DP × TP × PP × CP，统一调度 | **两套系统共存**：SGLang (generate) + Megatron (train) |
| **瓶颈** | 计算效率、通信带宽 | **pipeline overlap**、显存容量、weight sync 开销 |
| **GPU 利用率含义** | CUDA core 忙碌程度 | 可能是 Memory/PCIe 瓶颈，CUDA core 在等 KV cache |

从 slime 的 `train.py` 主循环可以清晰看到这一差异：

```python
for rollout_id in range(args.num_rollout):
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))   # SGLang generate
    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))          # Megatron train
    actor_model.update_weights()                                            # weight sync to SGLang
```

**一个 rollout 周期内，GPU 时间被三个完全不同的阶段瓜分**，它们的计算特征、瓶颈类型、甚至所占用的 GPU 子系统都不一样。

---

### 9.2 为什么传统 MFU 在 RL 中失效？

#### 传统 MFU 的计算方式

看 slime 中的实现（`slime/utils/flops_utils.py` + `train_metric_utils.py`）：

```python
# flops_utils.py: calculate_fwd_flops() → 只算 transformer layer fwd 的 matmul FLOPs
# train_metric_utils.py
log_dict["perf/actor_train_tflops"] = 3 * total_fwd_flops / log_dict["perf/actor_train_time"]
```

它只计算了：
$$
\text{TFLOPS} = \frac{3 \times \text{total\_fwd\_flops}}{\text{actor\_train\_time}}
$$

**三个致命问题**：

| 问题 | 说明 |
|------|------|
| **只覆盖 train 阶段** | generate 阶段通常占 **30% ~ 60%** 的 rollout 时间，却完全不在公式里 |
| **忽略 log_probs/ref_log_probs 耗时** | 全序列 forward 也吃时间，但计时器与 train 分离 |
| **Pipeline Bubble 被掩盖** | `perf/train_wait_time` 和 `perf/wait_time_ratio` 直接暴露了 GPU 空转，但 MFU 算不出来 |

> ⚠️ **结论**：即便把 train 的 MFU 从 40% 拉到 60%，如果 generate 占 50%、weight sync 占 20%，端到端提升可能不到 10%。

---

### 9.3 RL 中真正有效的指标组合

| 层级 | 指标 | 意义 |
|------|------|------|
| **端到端吞吐** | `rollout_batch_size × n_samples_per_prompt / perf_step_time` | 最硬的业务指标 |
| **阶段分解** | `perf/rollout_time` vs `perf/train_time` vs `perf/update_weights_time` | 定位瓶颈在哪 |
| **等待占比** | `perf/wait_time_ratio` | pipeline 做得好不好的直接信号 |
| **train 子效率** | `perf/actor_train_tflops` | 在 train 阶段内部，近似传统 MFU，**仅用于横向对比不同 TP/PP/recompute 配置** |
| **generate 效率** | decode tokens/s (SGLang 自带) | memory-bound 阶段看 FLOPs 无意义 |
| **显存效率** | memory overhead / KV cache reuse | 序列长短差异大时极易 OOM |

> 💡 **核心结论**：在 RL 中，"时间" 比 "FLOPs" 更重要。

---

### 9.4 六大性能优化方向 & 具体操作建议

#### 方向 1：Pipeline Overlap — 让 generate 和 train "并行"

**问题**：当前 rollout 的 generate 完成后才开始 train，train 完成后才做下一轮的 generate，时间串行。

**优化动作**：

| 操作 | 具体做法 | 代码/配置位置 |
|------|---------|-------------|
| **独立部署** | SGLang 和 Megatron 使用不重叠的 GPU（`--colocate` 设为 false） | `shell/run-qwen3-32B.sh: --colocate` |
| **显存分时复用** | Colocate 场景下，train 前 offload SGLang 权重，train 完再 onload | `slime/ray/rollout.py: EngineGroup.offload()` |
| **Async train** | 让 train 和下一轮的 generate 重叠 | `actor_model.async_train()` 已经是基础，可进一步解耦 `ray.get` 同步点 |
| **评测算子解耦** | eval 单独走一批 GPU，不要和 train/generate 抢资源 | 修改 `eval_interval` 和 placement group 分配 |

**关键代码**：
```python
# slime/ray/rollout.py
class EngineGroup:
    needs_offload: bool = False  # True when GPUs overlap with megatron
    # needs_offload=True 时，每次 rollout 都要做 offload → onload → weight sync，expensive
```

**建议**：如果 GPU 总数 >= 32，优先分离部署；如果 < 16，colocate 时必须做好显存分时和 CUDA graph 缓存。

---

#### 方向 2：Dynamic Batch Size — 均衡微批次 token 数

**问题**：RL 生成的 response 长度差异极大（50 token ~ 8192 token），固定 microbatch 导致短序列 padding 严重。

**已有配置**：
```bash
--use-dynamic-batch-size
--max-tokens-per-gpu 20480
```

**代码逻辑**（`data.py:get_data_iterator()`）：
```python
# 按 batch 中各 sample 的 total_lengths 计算最少需要多少个 microbatch
num_microbatches = get_minimum_num_micro_batch_size(
    samples[start:end], args.max_tokens_per_gpu * cp_size
)
# 用 seqlen_balanced_partitions 尽量让每个 microbatch 内的 token 总数相近
partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)
```

**调参建议**：

| 参数 | 作用 | 调大/调小 |
|------|------|----------|
| `--max-tokens-per-gpu` | 每 GPU 同时处理的最大 token 数 | 显存充足时调大 → 减少 microbatch 数 → 减少 forward_boundary 开销 |
| `--micro-batch-size` | 固定模式下每 microbatch 的 sample 数 | dynamic 模式下自动计算，通常不需要手动设 |
| `--global-batch-size` | 每 rollout 训练的总 sample 数 | 256 是本任务配置，需保证能被 DP size × microbatch 整除 |

**优化效果**：减少 padding → 提升 `perf/effective_tokens_per_gpu_per_sec` → 减少 bubble。

---

#### 方向 3：CPU Offload 的精细控制

**配置**：
```bash
--optimizer-cpu-offload               # Adam 状态 offload 到 CPU
--overlap-cpu-optimizer-d2h-h2d     # overlap 数据传输
--recompute-granularity full        # 全重计算，用 FLOPs 换显存
--recompute-method uniform
```

**决策框架**：

```
显存使用情况 ──→ 当前 batch 是否能跑满？
     │
     ├─ 是（GPU 计算饱和）→ 不要开 CPU offload，PCIe 反而引入瓶颈
     │
     └─ 否（OOM 或 batch 被迫改小）→ 检查 offload 能否释放足够显存
             │
             ├─ 能放大 batch → 开 offload 赚
             └─ 只能省几 GB 但 batch 不变 → 开 offload 亏（D2H/H2D  penalty）
```

**加速技巧**：`--overlap-cpu-optimizer-d2h-h2d` 让优化器 states 的 CPU↔GPU 传输和 compute overlap，力求零额外耗时。

---

#### 方向 4：SGLang 生成效率优化

**配置**：
```bash
--sglang-mem-fraction-static 0.7     # 只给 SGLang 70% 显存
--sglang-cuda-graph-bs 1 2 4 8 ...  # 预编译 CUDA graphs
```

**优化动作**：

| 手段 | 说明 |
|------|------|
| **增大 CUDA Graph 覆盖范围** | 本任务已经预编译了 1 到 256 的 batch size graph，避免运行时编译开销 |
| **Radix Attention / Prefix Cache** | AIME 数学题的 prompt 有大量共享前缀，KV cache 复用率越高，decode 越快 |
| **调整 temperature / top_p** | 过高 temperature 导致采样更发散，难以命中 cache；但过低会降低探索效率，需权衡 |
| **连续批处理 (Continuous Batching)** | SGLang 自带，确保生成期间有新请求立即插入，不浪费 GPU cycles |

**监控指标**：SGLang 自带 `decode_tokens_per_sec`，单独观测。如果过低（< 100 tok/s for H20），重点查 KV cache hit rate。

---

#### 方向 5：Weight Sync 加速

slime 支持两种 weight sync 方式：

```python
# 方式 A：内存直接更新（快，但需显存/PCIe 充足）
engine.update_weights.remote(weights)

# 方式 B：从磁盘刷新（慢，但省显存）
engine.update_weights_from_disk.remote(self.model_path)
```

**优化动作**：

| 场景 | 推荐方式 | 额外加速手段 |
|------|---------|-------------|
| Colocate（SGLang 和 Megatron 同 GPU）| 方式 A（内存更新） | 用 `torch_memory_saver` 做显存 margin 预留 |
| 分离部署（跨节点）| 方式 B 或 RDMA P2P | 用 `--actor-num-nodes=1` 把 Megatron actor 和 SGLang 放同节点 |
| 超大模型（>70B）| 增量更新 / shard 负载均衡 | 只传输改动的 expert / layer，减少通信量 |

**关键代码**（`slime/backends/megatron_utils/actor.py`）：
```python
logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
```
用 `torch_memory_saver` 控制训练时给 SGLang 预留多少显存，避免 OOM 同时保证 weight sync 直接走内存。

---

#### 方向 6：减少 Non-Compute 等待

**从日志直接定位**：

```python
# 端到端效率
rollout_e2e_throughput = (args.rollout_batch_size * args.n_samples_per_prompt) / perf_step_time

# pipeline 效率
train_efficiency = perf_train_time / perf_step_time

# 时间占比分解
generate_ratio = perf_rollout_time / perf_step_time
train_ratio    = perf_train_time / perf_step_time
wait_ratio     = perf_wait_time / perf_step_time
sync_ratio     = perf_update_weights_time / perf_step_time
```

**目标状态**：

| 占比 | 健康范围 | 异常信号 |
|------|---------|---------|
| `train_ratio` | 40% ~ 60% | < 30% 说明 train 严重被卡住 |
| `generate_ratio` | 30% ~ 50% | > 60% 说明 decode 太慢 |
| `wait_ratio` | < 15% | > 30% 说明 Ray 同步或数据准备有问题 |
| `sync_ratio` | < 5% | > 10% 说明 weight update 成了瓶颈 |

**加速动作**：

1. **Ray Internal**：减少 `ray.get` 阻塞调用，改用 `ray.wait` + callback
2. **数据预处理异步化**：`perf/data_preprocess_time` 如果高，把 tokenization + advantage compute 放到 CPU 线程池
3. **GC 调参**：`--manual-gc` + `--manual-gc-interval` 对齐各 rank 的 GC 时机，避免 staggered pause

---

### 9.5 MFU 在 RL 中到底该怎么用？

| 场景 | MFU 是否有意义 | 正确的用法 |
|------|-------------|-----------|
| **Train 阶段内部** | ✅ 有意义 | `perf/actor_train_tflops` 近似反映 Megatron 计算效率，用于**横向对比不同并行配置** |
| **Rollout generate 阶段** | ❌ 无意义 | decode 是 memory-bound，看 decode token/s 更合理 |
| **跨阶段比较** | ❌ 无意义 | 不同阶段的 FLOPs 不在一个量纲 |
| **端到端优化目标** | ❌ 失效 | 应该看 **wall-clock samples/s** 和 **各阶段时间占比** |

> 🎯 **最终建议**：在 RL 训练的性能优化中，把 **"减少非计算等待时间"** 放在第一位，**"提升计算密集阶段的 MFU"** 放在第二位。即便 train MFU 只有 40%，如果 wait_ratio 接近 0，端到端吞吐也可能是满分。相反，train MFU 60% 但 wait_ratio 50%，意味着有一半时间 GPU 在空转。

---

## 十、结论

任务 `306872698` 的日志体系非常完整，覆盖从 **生成 → 指标汇总 → 优势计算 → 策略训练 → 性能 profiling → 评测** 的全链路：

- **`rollout/*`** — 反映模型**生成质量**（reward、长度分布、截断、重复、零标准差组数）
- **`train/*`** — 反映**学习动力学**（loss 各分量、梯度、学习率、策略偏移）
- **`perf/*`** — 反映**工程效率**（各阶段耗时、TFLOPS、吞吐、等待占比）
- **`eval/*`** — 反映**下游能力**（AIME2024 评测，每 20 rollout 一次）

通过这套日志，可以：
1. 用 `rollout/raw_reward` 与 `eval/aime` 判断训练是否有效提升模型能力
2. 用 `train/pg_clipfrac` 与 `train/ppo_kl` 监控策略稳定性
3. 用 `perf/actor_train_tflops` **在 train 子阶段内部**评估计算效率
4. 用 `perf/wait_time_ratio` 和 **阶段时间占比分解**定位真正的系统瓶颈
5. 用 `rollout/zero_std/count_0` 识别 GRPO 分组失效问题
