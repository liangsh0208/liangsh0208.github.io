---
title: "Slime 代码走读 — 02. Rollout 系统：从 RolloutManager 到 SGLang Engine"
date: 2026-06-15T12:00:00+08:00
draft: false
---

> 本文档完整走读 slime 的 Rollout 系统，覆盖 `RolloutManager` → `RolloutServer` → `ServerGroup` → `SGLangEngine` 四层架构，以及异步生成 `generate_rollout_async()` 的实现细节。

---

## 一、Rollout 系统架构总览

```
RolloutManager (Ray Actor, 全局协调)
  └── servers: dict[str, RolloutServer]
        └── server_groups: list[ServerGroup]
              └── all_engines: list[SGLangEngine]  [Ray Actor]
                    ├── router_ip + router_port      [sglang router]
                    └── CUDA Graph + KV Cache
```

| 组件 | 代码位置 | 角色 |
|------|---------|------|
| `RolloutManager` | `slime/ray/rollout.py:349` | 调度中心：管理所有 rollout、评测、数据转换 |
| `RolloutServer` | `slime/ray/rollout.py:210` | 单模型实例 = router + N 个 engine groups |
| `ServerGroup` | `slime/ray/rollout.py:38` | 的一组同构 SGLang engines（同 TP size） |
| `RolloutRayActor` | `slime/backends/sglang_utils/sglang_engine.py` | 实际的 SGLang engine 进程 |
| `sglang router` | 外部进程 | HTTP 负载均衡 + prefix cache routing |

---

## 二、RolloutManager：`slime/ray/rollout.py:349`

```python
@ray.remote
class RolloutManager:
    def __init__(self, args, pg):
        # ① 数据加载
        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        # ② 动态加载 rollout 函数和 eval 函数
        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)

        # ③ 启动 rollout servers
        self.servers = start_rollout_servers(args, pg)

        # ④ 可选：启动健康监控（fault tolerance）
        self._health_monitors = [RolloutHealthMonitor(group, args) for srv in servers for group in srv.server_groups]
```

### 2.1 `generate()` 方法

```python
def generate(self, rollout_id):
    data, metrics = self._get_rollout_data(rollout_id)
    _log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
    data = self._convert_samples_to_train_data(data)
    return self._split_train_data_by_dp(data, self.train_parallel_config["dp_size"])
```

**逻辑拆解**：

| 步骤 | 函数 | 说明 |
|------|------|------|
| 1 | `_get_rollout_data()` | 调用 `generate_rollout()` 获取 samples |
| 2 | `_log_rollout_data()` | 记录 rollout 指标（reward、长度、截断比等） |
| 3 | `_convert_samples_to_train_data()` | 提取 tokens/loss_masks/rewards/sample_indices |
| 4 | `_split_train_data_by_dp()` | 按 DP size 切分数据，每 rank 一份 `ray.put(Box)` |

### 2.2 `_get_rollout_data()`

```python
def _get_rollout_data(self, rollout_id):
    if self.args.load_debug_rollout_data:
        # 调试模式：直接从磁盘加载
        data = torch.load(...)
    else:
        # 正常模式：调用用户配置的 rollout 函数
        data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source)
        metrics = data.metrics
        data = data.samples

        # 如果 sample 数不是 global_batch_size 的整数倍，trim
        if len(data) % global_batch_size != 0:
            trim_len = (len(data) // global_batch_size) * global_batch_size
            data = data[:trim_len]

    # dynamic global batch size: 让每轮只做一次 training step
    if self.args.use_dynamic_global_batch_size:
        self._dynamic_global_batch_size = self._compute_dynamic_global_batch_size(len(data))

    return data, metrics
```

### 2.3 `_convert_samples_to_train_data()`

```python
def _convert_samples_to_train_data(self, samples):
    raw_rewards, rewards = self._post_process_rewards(samples)

    train_data = {
        "tokens":         [sample.tokens for sample in samples],
        "response_lengths":  [sample.response_length for sample in samples],
        "rewards":        rewards,           # 经 norm 后的 reward
        "raw_reward":     raw_rewards,
        "truncated":      [1 if sample.status == TRUNCATED else 0 for sample in samples],
        "loss_masks":     [sample.loss_mask for sample in samples],
        "sample_indices": [sample.index for sample in samples],
        "rollout_log_probs": [sample.rollout_log_probs for sample in samples],
        "multimodal_train_inputs": [...],
        "teacher_log_probs": [...],
    }
    return train_data
```

### 2.4 `_split_train_data_by_dp()`

```python
def _split_train_data_by_dp(self, data, dp_size):
    if self.args.balance_data:
        # sequence length balancing: 让每个 DP rank 的总 token 数尽可能接近
        partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
    else:
        # Round-robin 切分
        partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

    for i in range(dp_size):
        rollout_data = {}
        for key in ["tokens", "response_lengths", "rewards", ...]:
            rollout_data[key] = [data[key][j] for j in partitions[i]]

        rollout_data_refs.append(Box(ray.put(rollout_data)))
    return rollout_data_refs  # 每个元素发到一个 DP rank
```

**`Box` 的作用**：因为 `ray.put()` 返回的是 `ObjectRef`，而 `ray.get(ObjectRef)` 时需要有类型包裹。`Box` 是一个简单的 dataclass，避免类型系统的复杂性。

---

## 三、RolloutServer / ServerGroup：`slime/ray/rollout.py:38`

### 3.1 `ServerGroup` — 引擎组

```python
@dataclasses.dataclass
class ServerGroup:
    args: Any
    pg: Any  # (placement_group, reordered_bundle_indices, reordered_gpu_ids)
    all_engines: list      # 每 ranks-per-node 一个 engine
    num_gpus_per_engine: int
    needs_offload: bool    # True 表示和 Megatron 同 GPU（colocate）
    model_path: str | None
    rank_offset: int       # 本组之前的 engine 总数
    gpu_offset: int        # 本组之前的 GPU 数
    worker_type: str       # "regular", "prefill", "decode", "placeholder"
```

**`needs_offload` 判定逻辑**：

```python
# 在 start_rollout_servers() 中
group_abs_start = rollout_pg_offset + gpu_offset
needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus
```

- 如果 rollout GPU 和 Megatron GPU 有重叠（colocate），则 `needs_offload=True`
- 意味着 rollout 结束后要 `release_memory_occupation()` 释放显存
- train 完成后要 `resume_memory_occupation()` 恢复权重 + KV cache

### 3.2 `ServerGroup.start_engines()` — 启动引擎

```python
def start_engines(self, port_cursors=None):
    for i in range(len(self.all_engines)):
        if self.all_engines[i] is not None:
            continue

        global_rank = self.rank_offset + i

        # 设置运行时环境变量
        env_vars = {
            "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",      # 显存分时复用
            "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
            ...
        }

        # 创建 Ray Actor
        rollout_engine = RolloutRayActor.options(
            num_gpus=0.2,           # 资源声明：占 0.2 GPU（会被 PG 绑定到具体 GPU）
            scheduling_strategy=PlacementGroupSchedulingStrategy(...),
            runtime_env={"env_vars": env_vars},
        ).remote(
            self.args,
            rank=global_rank,
            base_gpu_id=base_gpu_id,
            num_gpus_per_engine=self.num_gpus_per_engine,
            sglang_overrides=self.sglang_overrides,
        )
```

**`num_gpus=0.2` 不表示只占 0.2 GPU**：

- Ray 的 `num_gpus` 是**调度粒度**，`0.2` 意思是 "这个 actor 和一个完整 GPU 的 1/5 资源关联"
- 但实际上 SGLang 进程内部会通过 `cudaSetDevice` 独占一个或多个 GPU
- `0.2` 配合 `placement_group_bundle_index` 让多个 SGLang engine 共享一个 PG bundle（即一个节点）

### 3.3 端口分配

```python
addr_and_ports = {
    rank: {
        "host": get_addr(),
        "port": get_port(),           # HTTP server port
        "nccl_port": get_port(),       # SGLang 内部 NCCL
        "dist_init_addr": f"{addr}:{port}",  # PyTorch init
    }
}
```

每个 engine 需要 3-4 个端口（HTTP + NCCL + dist_init + dp_attention 端口）。通过 Ray actor 本身获取节点 IP 和空闲端口。

---

## 四、异步生成：`slime/rollout/sglang_rollout.py`

### 4.1 入口：`generate_rollout()`

```python
def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output
    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    data_source.add_samples(aborted_samples)  # partial rollout 时收集中断样本
    return output
```

### 4.2 `generate_rollout_async()` — 核心异步生成逻辑

```python
async def generate_rollout_async(args, rollout_id, data_source):
    state = GenerateState(args)  # 单例：存储 tokenizer / semaphore / dp_rank 计数
    
    target_data_size = args.rollout_batch_size  # 目标 prompt 数
    
    while len(data) < target_data_size:
        # ① 提交生成请求（不等待完成）
        while state.remaining_batch_size < target_data_size:
            samples = data_source(args.over_sampling_batch_size)  # 取一批 prompt
            state.submit_generate_tasks(samples)  # create_task async
        
        # ② 等待任意一个任务完成
        done, state.pendings = await asyncio.wait(
            state.pendings, return_when=asyncio.FIRST_COMPLETED
        )
        
        for task in done:
            group = task.result()  # [Sample, Sample, ...]，长度 = n_samples_per_prompt
            
            # ③ 动态采样过滤器（可选）
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                continue
            
            # ④ 保留有效样本
            if len(data) < target_data_size:
                data.append(group)
    
    # ⑤ 中断剩余请求
    aborted_samples = await abort(args, rollout_id)
    
    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples
```

**关键**：这是一个**异步流水线**——不等待所有请求一起完成，而是来一个处理一个，达到目标数量后中断剩余请求。

### 4.3 `GenerateState` — 全局生成状态（单例）

```python
class GenerateState(metaclass=SingletonMeta):
    def __init__(self, args):
        self.tokenizer = load_tokenizer(args.hf_checkpoint)
        self.processor = load_processor(args.hf_checkpoint)
        
        # 控制并发度：同时处理的请求数
        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        
        self.sampling_params = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
        )
        
        # dp_rank balancing：SGLang DP attention 的负载均衡
        self.dp_counts = [0] * (args.sglang_dp_size or 1)
```

### 4.4 `generate()` — 单样本生成

```python
async def generate(args, sample, sampling_params):
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    
    # 异步 HTTP post（带有 Semaphore 限制并发）
    async with state.semaphore:
        output = await post(url, payload, headers)
    
    # 解析输出
    sample.tokens = sample.tokens + new_response_tokens
    sample.response_length += len(new_response_tokens)
    sample.response += output["text"]
    
    # 获取 logprobs（return_logprob=True）
    if "output_token_logprobs" in output["meta_info"]:
        new_response_log_probs = [item[0] for item in output["meta_info"][output_token_logprobs]]
        sample.rollout_log_probs += new_response_log_probs
    
    # 获取 routed_experts（MoE，可选）
    if "routed_experts" in output["meta_info"]:
        sample.rollout_routed_experts = np.frombuffer(pybase64.b64decode(output["routed_experts"]))
```

### 4.5 `generate_and_rm()` — 生成 + 奖励计算

```python
async def generate_and_rm(args, sample, sampling_params):
    # 1. 生成
    sample = await generate(args, sample, sampling_params)
    
    # 2. 单样本 reward 计算（异步）
    if sample.reward is None:
        sample.reward = await async_rm(args, sample)
    
    return sample
```

**Reward model 集合**：位于 `slime/rollout/rm_hub/`，支持：
- `deepscaler.py` — DeepScaler 数学 reward
- `math_utils.py` — 数学题目的规则 reward（数值/符号/等价判断）
- `gpqa.py` / `f1.py` / `ifbench.py` — 其他评测集

### 4.6 `generate_and_rm_group()` — 组级生成

```python
async def generate_and_rm_group(args, group, sampling_params):
    # 同一 prompt 的 n_samples_per_prompt 个 Sample 并发生成
    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        
        # 确定性推理：每个 sample 有固定的 seed
        if args.sglang_enable_deterministic_inference:
            current_sampling_params["sampling_seed"] = state.group_sampling_seeds[idx]
        
        tasks.append(asyncio.create_task(generate_and_rm(args, sample, current_sampling_params)))
    
    group = await asyncio.gather(*tasks)
    
    # 整组 reward（如 group-level奖赏）
    if args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards):
            sample.reward = reward
    
    return group
```

---

## 五、`abort()` — 中断与残差样本收集

```python
async def abort(args, rollout_id):
    # 1. 设置 aborted flag
    state.aborted = True
    
    # 2. 查询所有 engine URL
    response = await get("http://router/workers")
    urls = [worker["url"] for worker in response["workers"]]
    
    # 3. 并行发送 abort 请求
    abort_tasks = [post(f"{url}/abort_request", {"abort_all": True}) for url in urls]
    await asyncio.gather(*abort_tasks, return_exceptions=True)
    
    # 4. 收集已完成的 partial samples
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
    
    return aborted_samples
```

** Partial Rollout 场景**：
- 如果达到 `target_data_size` 后还有 pending 请求，调用 `abort()`
- SGLang 引擎会收到 abort 信号，停止生成
- 已经生成的部分 response 被收集，用于下一 rollout 继续
- `start_rollout_id` 标记首次生成的 rollout ID

---

## 六、Rollout → Train 数据转换图解

```
Rollout 输出（list[Sample]）
  │
  ├── Sample 0: tokens=[tok0, tok1, ...], response="...", reward=1.0
  ├── Sample 1: tokens=[tok0, tok1, ...], response="...", reward=0.0
  └── ... （共 rollout_batch_size × n_samples_per_prompt = 256 条）
  │
  ▼ _convert_samples_to_train_data()
  │
RolloutBatch（dict[str, list]）
  │
  ├── "tokens": [torch.tensor(...), torch.tensor(...)]          # prompt+response 拼在一起
  ├── "response_lengths": [R1, R2, ...]                        # 各 response 长度
  ├── "total_lengths": [T1, T2, ...]                           # prompt_length + response_length
  ├── "rewards": [1.0, 0.0, ...]                                # 归一化后的 reward
  ├── "raw_reward": [1.0, 0.0, ...]                            # 原始 reward
  ├── "loss_masks": [[1,1,1,...], [1,1,0,...]]                 # response token 有效性
  ├── "rollout_log_probs": [torch.tensor(...), torch.tensor(...)] # SGLang 返回的 logprob
  ├── "truncated": [0, 1, ...]                                   # 是否被截断
  └── "sample_indices": [0, 1, ...]                              # 原始 sample 索引
  │
  ▼ _split_train_data_by_dp(dp_size=4)
  │
DP 0: ray.put(Box(partition=[0,4,8,...]))  → rank 0
DP 1: ray.put(Box(partition=[1,5,9,...]))  → rank 1
DP 2: ray.put(Box(partition=[2,6,10,...])) → rank 2
DP 3: ray.put(Box(partition=[3,7,11,...])) → rank 3
```

---

## 七、调用链总结

```
RolloutManager.generate()
  → _get_rollout_data()
      → generate_rollout() [slime/rollout/sglang_rollout.py]
          → generate_rollout_async() [async]
              → state.submit_generate_tasks()
                  → generate_and_rm_group() [async]
                      → generate_and_rm() [async]
                          → generate() [HTTP call to sglang]
                              → post() [slime/utils/http_utils]
                                  → SGLang Router → SGLang Engine
                          → async_rm() [reward model]
                              → deepscaler.py / math_utils.py
              → asyncio.wait() → done tasks
              → call_dynamic_filter() [optional]
              → metric_gatherer.collect()
              → abort() [cancel remaining]
      → _compute_dynamic_global_batch_size() [optional]
      → trim_samples() [optional]
  → _log_rollout_data() [metrics logging]
  → _convert_samples_to_train_data() [extract fields]
  → _split_train_data_by_dp() [distribute to ranks]

RolloutManager.eval()
  → eval_rollout() [slime/rollout/sglang_rollout.py]
      → eval_rollout_single_dataset() [async]
          → Dataset (prompts loaded from JSONL)
          → copy.deepcopy() per sample × n_eval_samples
          → generate_and_rm() [async]
          → collect rewards / truncated / samples
      → _log_eval_rollout_data() [eval metrics]
```

---

## 八、关键设计洞察

1. **异步并发是关键**：`asyncio.create_task()` 让一组内的 N 个 sample 并发生成，而不是串行等待
2. **Semaphore 防浪涌**：限制同时发向 SGLang router 的请求数，避免 router OOM
3. **SGLang Router 做负载均衡**：router 层面实现 continuous batching，而不是在 slime 代码中
4. **DP rank 负载均衡**：通过 `dp_rank_context` 追踪 SGLang DP attention 的各 rank 负载，均分请求
5. **Abort 收集 partial**：支持超大规模 rollout（比如 oversampling 2x，然后 abort 50%）
6. **`Box(ray.put())` 是内存引用**：训练 Actor 通过 `ray.get()` 才能从 Object Store 取数据，避免大 tensor 直接序列化进 RPC
