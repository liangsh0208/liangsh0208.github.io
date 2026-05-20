---
created: 2026-05-06
---

# Rollout 数据处理流程分析

> 本文档详细分析 Slime 框架中 Rollout 数据的完整处理流程
> 分析路径: `/Users/danchen/Documents/RL_fw/slime/slime`

---

## 目录

1. [Rollout 数据处理概览](#1-rollout-数据处理概览)
2. [阶段一: 数据加载与预处理](#2-阶段一-数据加载与预处理)
3. [阶段二: 生成请求与推理](#3-阶段二-生成请求与推理)
4. [阶段三: 奖励计算](#4-阶段三-奖励计算)
5. [阶段四: 数据分片与传输](#5-阶段四-数据分片与传输)
6. [阶段五: 数据迭代器构建](#6-阶段五-数据迭代器构建)
7. [阶段六: Context Parallel 数据切分](#7-阶段六-context-parallel-数据切分)
8. [阶段七: 训练批次构建](#8-阶段七-训练批次构建)
9. [数据流总结图](#9-数据流总结图)

---

## 1. Rollout 数据处理概览

Rollout 数据处理是 RLHF 训练的核心环节，负责将从原始数据到可训练张量的完整转换。

### 1.1 数据处理流水线

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Rollout 数据处理流水线                               │
└─────────────────────────────────────────────────────────────────────────────┘

[原始数据]
    │
    ▼
┌──────────────────┐
│ 1. 数据加载      │  Dataset类
│   - JSONL/Parquet│
│   - Tokenize     │
│   - 长度过滤     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 2. 数据源管理    │  DataSource类
│   - Sample分组   │
│   - 缓冲区管理   │
│   - 状态持久化   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 3. 生成请求      │  SGLang引擎
│   - 批量推理     │
│   - log_prob计算 │
│   - 多模态处理   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 4. 奖励计算      │  RM Hub
│   - 规则验证     │
│   - 模型评分     │
│   - 自定义RM     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 5. 数据分片      │  process_rollout_data
│   - DP分片       │
│   - ByteBuffer   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 6. 迭代器构建    │  DataIterator
│   - 动态batch    │
│   - 序列均衡     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 7. CP数据切分    │  cp_utils
│   - Zigzag切分   │
│   - AllGather    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 8. 训练批次      │  get_batch
│   - PackedSeq    │
│   - Loss Mask    │
│   - 多模态输入   │
└──────────────────┘
```

### 1.2 核心数据结构

```python
# Sample: 单条样本
@dataclass
class Sample:
    group_index: int           # 组索引
    index: int                 # 全局索引
    prompt: str                # 输入提示
    tokens: list[int]          # Token IDs
    response: str              # 生成的响应
    response_length: int       # 响应长度
    reward: float              # 奖励值
    loss_mask: list[int]       # 损失掩码
    status: Status             # 状态

# RolloutBatch: 批次数据字典
RolloutBatch = dict[str, list[torch.Tensor] | list[int] | list[float] | list[str]]
```

---

## 2. 阶段一: 数据加载与预处理

**核心文件**: `utils/data.py`

### 2.1 Dataset 类结构

```python
class Dataset:
    def __init__(
        self,
        path,                    # 数据路径 (JSONL/Parquet)
        tokenizer,               # HuggingFace Tokenizer
        processor,               # 多模态 Processor
        max_length,              # 最大长度限制
        *,
        prompt_key="text",       # prompt字段名
        multimodal_keys=None,    # 多模态字段
        label_key=None,          # 标签字段
        metadata_key="metadata", # 元数据字段
        seed=42,
        apply_chat_template=False,
    ):
        # 1. 读取文件
        # 2. 构建 conversation 格式
        # 3. 应用 chat template
        # 4. 处理多模态输入
        # 5. 过滤超长样本
```

### 2.2 数据加载流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         数据加载流程                                         │
└─────────────────────────────────────────────────────────────────────────────┘

read_file(path)
    │
    ├─ 解析路径切片语法: path@[start:end]
    │
    ├─ JSONL格式:
    │   └─ 逐行读取 ─▶ json.loads()
    │
    └─ Parquet格式:
        └─ pyarrow ─▶ batch迭代 ─▶ to_pylist()

         ▼

_build_messages(data, prompt_key, as_conversation, multimodal_keys)
    │
    ├─ 文本 prompt:
    │   └─ 直接返回或包装为 [{"role": "user", "content": prompt}]
    │
    └─ 多模态 prompt:
        ├─ 解析占位符: <image>, <video>, <audio>
        ├─ 构建多模态内容列表
        └─ 验证占位符与数据数量匹配

         ▼

apply_chat_template (可选)
    │
    └─ tokenizer.apply_chat_template(prompt, tools, ...)

         ▼

process_vision_info (多模态)
    │
    └─ processor(text=prompt, images=...)

         ▼

filter_long_prompt(origin_samples, tokenizer, processor, max_length)
    │
    ├─ 纯文本样本: tokenizer批量编码
    ├─ 多模态样本: processor单独处理
    └─ 过滤: len(input_ids) > max_length

         ▼

Dataset.samples  ←─ list[Sample]
```

### 2.3 文件格式支持

| 格式 | 扩展名 | 读取方式 | 特点 |
|------|--------|----------|------|
| JSONL | `.jsonl` | 逐行解析 | 流式读取，内存友好 |
| Parquet | `.parquet` | PyArrow列式读取 | 高效压缩，适合大数据集 |

### 2.4 路径切片语法

```python
# 支持路径切片
"/path/to/data.jsonl@[100:500]"  # 只读取第100-500行
"/path/to/data.jsonl@[:1000]"    # 只读取前1000行
"/path/to/data.jsonl@[1000:]"    # 从第1000行开始读取
```

---

## 3. 阶段二: 生成请求与推理

**核心文件**: `rollout/sglang_rollout.py`

### 3.1 生成流程架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         生成请求流程                                         │
└─────────────────────────────────────────────────────────────────────────────┘

generate_rollout(args, rollout_id, data_source, evaluation=False)
    │
    └─► generate_rollout_async()
        │
        ├─► data_source.get_samples(num_samples)
        │   └─ 获取 list[list[Sample]]
        │
        ├─► state.submit_generate_tasks(samples)
        │   │
        │   └─► for group in samples:
        │           asyncio.create_task(generate_and_rm_group())
        │
        ├─► asyncio.wait() ─▶ 等待完成
        │
        └─► abort(args, rollout_id) ─▶ 清理未完成任务
```

### 3.2 单样本生成流程

```python
async def generate(args: Namespace, sample: Sample, sampling_params: dict) -> Sample:
    """
    单样本生成流程
    """
    # 1. Tokenize
    if processor and multimodal:
        processor_output = processor(text=sample.prompt, ...)
        prompt_ids = processor_output["input_ids"][0]
        sample.multimodal_train_inputs = {...}
    else:
        prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)

    # 2. 构建请求
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
        "return_routed_experts": True,  # MoE路由信息
    }

    # 3. 发送到SGLang
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    output = await post(url, payload)

    # 4. 解析响应
    sample.tokens = prompt_ids + new_response_tokens
    sample.response = output["text"]
    sample.response_length = len(new_response_tokens)
    sample.rollout_log_probs = new_response_log_probs
    sample.rollout_routed_experts = routed_experts_array

    # 5. 更新状态
    sample.update_from_meta_info(args, output["meta_info"])

    return sample
```

### 3.3 多模态数据处理

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         多模态数据处理                                       │
└─────────────────────────────────────────────────────────────────────────────┘

_sample.multimodal_inputs = {"images": [PIL.Image, ...]}

         ▼

process_vision_info(prompt, processor)
    │
    ├─ 解析prompt中的 <image> 占位符
    ├─ 加载图像数据
    └─ 返回 processor_kwargs

         ▼

processor(text=prompt, images=...)
    │
    ├─ input_ids: [1, seq_len]
    ├─ pixel_values: [num_images, C, H, W]
    └─ image_sizes: [num_images, 2]

         ▼

sample.multimodal_train_inputs = {
    "pixel_values": tensor,
    "image_sizes": tensor,
}
```

### 3.4 生成参数配置

```python
sampling_params = {
    "temperature": args.rollout_temperature,      # 采样温度
    "top_p": args.rollout_top_p,                  # Nucleus采样
    "top_k": args.rollout_top_k,                  # Top-K采样
    "max_new_tokens": args.rollout_max_response_len,
    "stop": args.rollout_stop,                    # 停止词
    "stop_token_ids": args.rollout_stop_token_ids,
    "skip_special_tokens": args.rollout_skip_special_tokens,
}
```

---

## 4. 阶段三: 奖励计算

**核心文件**: `rollout/rm_hub/__init__.py`, `rollout/rm_hub/math_utils.py`

### 4.1 奖励模型类型

| RM类型 | 函数 | 描述 |
|--------|------|------|
| `math` | `grade_answer_verl()` | 数学答案验证 (SymPy) |
| `deepscaler` | `get_deepscaler_rule_based_reward()` | DeepScaler规则 |
| `dapo` | `compute_score_dapo()` | DAPO评分 |
| `f1` | `f1_score()` | F1分数 |
| `gpqa` | `compute_gpqa_reward()` | GPQA评估 |
| `ifbench` | `compute_ifbench_reward()` | IFEval评估 |
| `remote_rm` | `remote_rm()` | 远程RM服务 |
| `custom` | 用户自定义函数 | 完全可定制 |

### 4.2 奖励计算流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         奖励计算流程                                         │
└─────────────────────────────────────────────────────────────────────────────┘

async_rm(args, sample)
    │
    ├─► 自定义RM路径?
    │   └─► load_function(args.custom_rm_path)
    │
    ├─► 提取 boxed 答案 (可选)
    │   └─► extract_boxed_answer(response)
    │
    └─► 根据rm_type分发:
        │
        ├─ "remote_rm" ──► HTTP POST 到 args.rm_url
        │
        ├─ "math" ──► grade_answer_verl(response, label)
        │   │
        │   ├─ extract_boxed_answer(solution_str)
        │   │   └─ 提取 \boxed{} 内的内容
        │   │
        │   └─ grade_answer_sympy(given, ground_truth)
        │       ├─ _normalize(expr)  # 标准化表达式
        │       ├─ split_tuple(expr) # 处理元组
        │       └─ are_equal_under_sympy() # SymPy验证
        │
        ├─ "f1" ──► f1_score(response, label)
        │
        ├─ "gpqa" ──► compute_gpqa_reward(response, label, metadata)
        │
        └─ "random" ──► random.randint(0, 1)
```

### 4.3 数学答案验证详解

```python
# math_utils.py 核心逻辑

def grade_answer_verl(solution_str, ground_truth):
    """
    数学答案验证流程
    """
    # 1. 提取答案
    given_answer = extract_answer(solution_str)  # 提取 \boxed{} 内容

    # 2. 标准化
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    # 3. 直接字符串匹配
    if ground_truth_normalized == given_normalized:
        return True

    # 4. SymPy符号验证
    return are_equal_under_sympy(ground_truth_normalized, given_normalized)

def _normalize(expr: str) -> str:
    """
    表达式标准化
    """
    # 移除单位、百分号、货币符号
    # 处理LaTeX命令: \frac, \sqrt, \tfrac, \dfrac
    # 处理混合数: "7 3/4" → "7+3/4"
    # 移除空格、转小写
    # 尝试LaTeX解析
```

### 4.4 批量奖励计算

```python
async def batched_async_rm(args, samples: list[Sample]) -> list[float]:
    """
    批量奖励计算 (支持 group_rm)
    """
    if args.custom_rm_path:
        # 自定义函数负责批量逻辑
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples)

    # 并行计算每个样本
    tasks = [async_rm(args, sample) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
```

---

## 5. 阶段四: 数据分片与传输

**核心文件**: `utils/data.py:process_rollout_data()`

### 5.1 分片流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         数据分片流程                                         │
└─────────────────────────────────────────────────────────────────────────────┘

主进程 (Rollout Manager)
    │
    ├─ generate_rollout() ─▶ list[list[Sample]]
    │
    ├─ 将Sample转换为RolloutBatch字典
    │   │
    │   ├─ "tokens": [tensor, ...]
    │   ├─ "response_lengths": [int, ...]
    │   ├─ "total_lengths": [int, ...]
    │   ├─ "rewards": [float, ...]
    │   ├─ "loss_masks": [tensor, ...]
    │   └─ "rollout_log_probs": [tensor, ...]
    │
    ├─ 按DP rank分片
    │   │
    │   └─ partition = get_seqlen_balanced_partitions(...)
    │       └─ 使用 Karmarkar-Karp 算法均衡序列长度
    │
    └─ 存储到 Box (Ray共享内存)

         ▼

训练进程 (每个DP rank)
    │
    └─► process_rollout_data(args, rollout_data_ref, dp_rank, dp_size)
        │
        ├─ ray.get(rollout_data_ref[dp_rank].inner)
        │   └─ 从Ray共享内存获取本地分片
        │
        └─ rollout_data.pop("partition")  # 移除分片信息
```

### 5.2 process_rollout_data 函数

```python
def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    """
    数据分片处理

    Args:
        args: 配置参数
        rollout_data_ref: Ray ObjectRef列表 (每个DP rank一个)
        dp_rank: 当前DP rank
        dp_size: DP world size

    Returns:
        RolloutBatch: 本rank的数据分片
    """
    # 1. 从Ray获取数据
    assert len(rollout_data_ref) == dp_size
    rollout_data = ray.get(rollout_data_ref[dp_rank].inner)

    # 2. 应用分片索引
    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    # 3. 保存序列长度 (用于性能分析)
    Timer().seq_lens = total_lengths

    # 4. 按分片索引筛选
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data
```

### 5.3 序列长度均衡

```python
# utils/seqlen_balancing.py

def get_seqlen_balanced_partitions(
    seqlen_list: list[int],
    k_partitions: int,
    equal_size: bool
) -> list[list[int]]:
    """
    使用 Karmarkar-Karp 算法均衡分区

    目标: 使每个分区的总token数尽可能接近

    Args:
        seqlen_list: 每个样本的序列长度
        k_partitions: 分区数量
        equal_size: 是否要求每个分区大小相等

    Returns:
        partitions: k_partitions个索引列表
    """
    # 使用最大差分法 (Largest Differencing Method)
    partitions = karmarkar_karp(seqlen_list, k_partitions, equal_size)
    return partitions
```

---

## 6. 阶段五: 数据迭代器构建

**核心文件**: `backends/megatron_utils/data.py`

### 6.1 DataIterator 类

```python
class DataIterator:
    """
    Micro-batch迭代器

    支持两种模式:
    1. 固定大小连续切片
    2. 动态索引调度 (序列长度均衡)
    """

    def __init__(
        self,
        rollout_data: RolloutBatch,
        micro_batch_size: int | None = None,       # 固定模式
        micro_batch_indices: list[list[int]] | None = None,  # 动态模式
    ):
        self.rollout_data = rollout_data
        self.micro_batch_size = micro_batch_size
        self.micro_batch_indices = micro_batch_indices
        self.offset = 0

    def get_next(self, keys: Sequence[str]) -> dict:
        """获取下一个micro-batch"""
        batch = {}
        for key in keys:
            vals = self.rollout_data.get(key, None)
            if vals is None:
                batch[key] = None
            else:
                if self.micro_batch_indices is not None:
                    # 动态模式: 按索引选取
                    indices = self.micro_batch_indices[self.offset]
                    batch[key] = [vals[i] for i in indices]
                else:
                    # 固定模式: 连续切片
                    batch[key] = vals[self.offset : self.offset + self.micro_batch_size]

        # 更新偏移量
        if self.micro_batch_indices is not None:
            self.offset += 1
        else:
            self.offset += self.micro_batch_size

        return batch

    def reset(self):
        """重置迭代器"""
        self.offset = 0
        return self
```

### 6.2 迭代器构建流程

```python
def get_data_iterator(
    args: Namespace,
    model,
    rollout_data: RolloutBatch,
) -> tuple[list[DataIterator], list[int]]:
    """
    构建数据迭代器和micro-batch调度

    Returns:
        data_iterators: 每个VPP stage一个迭代器
        num_microbatches: 每个step的micro-batch数量
    """
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1

    num_local_samples = len(rollout_data["total_lengths"])
    num_local_gbs = global_batch_size // dp_size
    num_steps_per_rollout = num_local_samples // num_local_gbs

    if not args.use_dynamic_batch_size:
        # ========== 固定模式 ==========
        num_microbatches = [
            num_local_gbs // args.micro_batch_size
            for _ in range(num_steps_per_rollout)
        ]
        data_iterator = [
            DataIterator(rollout_data, args.micro_batch_size)
            for _ in range(vpp_size)
        ]
    else:
        # ========== 动态模式 ==========
        # 1. 计算每个step的micro-batch数量
        for i in range(num_steps_per_rollout):
            start, end = i * num_local_gbs, (i + 1) * num_local_gbs
            samples = rollout_data["total_lengths"][start:end]
            num_mb = get_minimum_num_micro_batch_size(
                samples, args.max_tokens_per_gpu * cp_size
            )
            num_microbatches.append(num_mb)

        # 2. All-reduce取最大值 (确保DP rank之间同步)
        dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=dp_group)

        # 3. VPP约束: micro-batch数必须能被 vpp_size 整除
        if vpp_size > 1:
            num_microbatches = clamp(
                num_microbatches // microbatch_group_size_per_vp_stage
                * microbatch_group_size_per_vp_stage,
                min=1
            )

        # 4. 构建均衡的索引调度
        micro_batch_indices = []
        for i, num_mbs in enumerate(num_microbatches):
            start, end = i * num_local_gbs, (i + 1) * num_local_gbs
            samples = rollout_data["total_lengths"][start:end]
            # 使用Karmarkar-Karp均衡
            partitions = get_seqlen_balanced_partitions(
                samples, num_mbs, equal_size=False
            )
            # 转换为全局索引
            for partition in partitions:
                micro_batch_indices.append([idx + start for idx in partition])

        data_iterator = [
            DataIterator(rollout_data, None, micro_batch_indices)
            for _ in range(vpp_size)
        ]

    return data_iterator, num_microbatches
```

### 6.3 动态Batch大小计算

```python
def get_minimum_num_micro_batch_size(
    total_lengths: list[int],
    max_tokens_per_gpu: int
) -> int:
    """
    使用First-Fit算法计算最小micro-batch数量

    目标: 每个micro-batch的总token数不超过 max_tokens_per_gpu
    """
    batches = []  # 每个元素是一个batch的当前token总数

    for length in total_lengths:
        # 尝试放入已有batch
        for i in range(len(batches)):
            if batches[i] + length <= max_tokens_per_gpu:
                batches[i] += length
                break
        else:
            # 需要新batch
            batches.append(length)

    return len(batches)
```

---

## 7. 阶段六: Context Parallel 数据切分

**核心文件**: `backends/megatron_utils/cp_utils.py`

### 7.1 Context Parallel 概述

Context Parallel (CP) 用于处理超长序列，将序列分割到多个GPU上并行计算。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Context Parallel 切分示意                                 │
└─────────────────────────────────────────────────────────────────────────────┘

原始序列 (长度 L):
┌─────────────────────────────────────────────────────────────────┐
│  P  P  P  P  R  R  R  R  R  R  R  R  R  R  R  R  R  R  R  R  R  │
│  [  Prompt  ]  [              Response              ]            │
└─────────────────────────────────────────────────────────────────┘
       prompt_len                         response_len

CP=4时的Zigzag切分:
┌────────────────────────────────────────────────────────────────┐
│ CP Rank 0: [chunk_0]     [chunk_7]                             │
│ CP Rank 1: [chunk_1]     [chunk_6]                             │
│ CP Rank 2: [chunk_2]     [chunk_5]                             │
│ CP Rank 3: [chunk_3]     [chunk_4]                             │
└────────────────────────────────────────────────────────────────┘

每个CP Rank持有两块数据:
- chunk_0, chunk_7 for rank 0
- chunk_1, chunk_6 for rank 1
- ...
```

### 7.2 切分计算函数

```python
def get_logits_and_tokens_offset_with_cp(
    total_length: int,
    response_length: int,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
):
    """
    计算Context Parallel切分偏移量

    Args:
        total_length: prompt + response 总长度
        response_length: response 长度
        qkv_format: "thd" (packed) 或 "bshd" (batched)

    Returns:
        chunk_size: 每块大小
        chunk_offsets: (chunk_0, chunk_1) 的全局偏移
        logits_offsets: (logits_0, logits_1) 的偏移
        token_offsets: (token_0, token_1) 的偏移
    """
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    prompt_length = total_length - response_length

    # 计算切块大小
    if qkv_format == "thd":
        chunk_size = (total_length + 2 * cp_size - 1) // (2 * cp_size)
    else:
        chunk_size = (max_seq_len + 2 * cp_size - 1) // (2 * cp_size)

    # Zigzag切分
    # Chunk 0: 前半部分
    chunk_0 = (cp_rank * chunk_size, (cp_rank + 1) * chunk_size)
    # Chunk 1: 后半部分 (倒序)
    chunk_1 = ((2 * cp_size - cp_rank - 1) * chunk_size,
               (2 * cp_size - cp_rank) * chunk_size)

    # Logits需要-1偏移 (预测下一个token)
    logits_0 = (max(chunk_0[0], prompt_length - 1),
                min(chunk_0[1], total_length - 1))
    logits_1 = (max(chunk_1[0], prompt_length - 1),
                min(chunk_1[1], total_length - 1))

    return chunk_size, (chunk_0, chunk_1), (logits_0, logits_1), token_offsets
```

### 7.3 Token切分函数

```python
def slice_with_cp(
    tokens: torch.Tensor,
    pad_value: int | float,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """
    将token序列按CP模式切分

    输入: [L] 或 [B, L]
    输出: [2*chunk_size] (包含两个块)
    """
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()

    if cp_size == 1:
        return tokens  # 无需切分

    token_len = len(tokens)
    chunk_size = (token_len + 2 * cp_size - 1) // (2 * cp_size)

    # 填充到 2 * cp_size * chunk_size
    pad = 2 * cp_size * chunk_size - token_len
    tokens = F.pad(tokens, (0, pad), value=pad_value)

    # 切出两块
    start_1, end_1 = chunk_size * cp_rank, chunk_size * (cp_rank + 1)
    start_2, end_2 = chunk_size * (2 * cp_size - cp_rank - 1), chunk_size * (2 * cp_size - cp_rank)

    return torch.cat([tokens[start_1:end_1], tokens[start_2:end_2]])
```

### 7.4 Log Prob All-Gather

```python
def all_gather_with_cp(
    tensor: torch.Tensor,
    total_length: int,
    response_length: int,
) -> torch.Tensor:
    """
    从各个CP rank收集完整的log_prob序列

    每个rank可能有部分数据，通过zero填充和all_reduce重建完整序列
    """
    cp_group = mpu.get_context_parallel_group()
    cp_size = mpu.get_context_parallel_world_size()

    if cp_size == 1:
        return tensor

    # 计算各chunk的偏移
    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(...)

    prompt_length = total_length - response_length

    # 构建完整序列 (用zero填充空缺)
    chunk_0 = tensor[:logits_offset[0][1] - logits_offset[0][0]]
    chunk_1 = tensor[logits_offset[0][1] - logits_offset[0][0]:]

    full_tensor = torch.zeros(response_length, ...)
    # 放置数据到正确位置
    # ...

    # All-reduce合并所有rank的数据
    full_tensor = dist.nn.all_reduce(full_tensor, group=cp_group)

    return full_tensor
```

---

## 8. 阶段七: 训练批次构建

**核心文件**: `backends/megatron_utils/data.py:get_batch()`

### 8.1 get_batch 函数

```python
def get_batch(
    data_iterator: DataIterator,
    keys: Sequence[str],
    pad_multiplier: int = 128,
    qkv_format: str = "thd",
    allgather_cp: bool = False,
) -> dict:
    """
    构建训练用的micro-batch

    Returns:
        dict containing:
        - "tokens": [1, T_padded] 输入token
        - "unconcat_tokens": list[Tensor] 原始token列表
        - "packed_seq_params": PackedSeqParams for FlashAttention
        - "full_loss_masks": [1, T_padded] 损失掩码
        - 其他请求的字段
    """
    assert "tokens" in keys
    batch = data_iterator.get_next(keys)

    tokens = batch["tokens"]
    pad_token_id = 0
    pad_size = mpu.get_tensor_model_parallel_world_size() * pad_multiplier

    # 保存原始token (用于后续log_prob计算)
    batch["unconcat_tokens"] = tokens

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    if qkv_format == "bshd":
        # ========== Batch模式 ==========
        max_seqlen = batch["max_seq_lens"][0]
        tokens = [slice_with_cp(t, pad_token_id, "bshd", max_seqlen) for t in tokens]
        tokens = torch.stack(tokens)
        packed_seq_params = None

    elif qkv_format == "thd":
        # ========== Packed模式 ==========
        if allgather_cp:
            # DSA模式: 先concat再切分
            cu_seqlens_list = [0]
            for t in tokens:
                cu_seqlens_list.append(cu_seqlens_list[-1] + t.size(0))

            tokens = torch.cat(tokens, dim=0)

            # 全局填充
            global_pad_size = cp_size * pad_size
            pad = (global_pad_size - tokens.size(0) % global_pad_size) % global_pad_size
            if pad != 0:
                tokens = F.pad(tokens, (0, pad), value=pad_token_id)

            cu_seqlens = torch.tensor(cu_seqlens_list, device="cuda")
            tokens = tokens.chunk(cp_size, dim=0)[cp_rank]
        else:
            # Zigzag模式
            tokens = [slice_with_cp(t, pad_token_id, "thd") for t in tokens]

            cu_seqlens = [0]
            for t in tokens:
                cu_seqlens.append(cu_seqlens[-1] + t.size(0))

            tokens = torch.cat(tokens)

            # 填充
            pad = (pad_size - tokens.size(0) % pad_size) % pad_size
            if pad != 0:
                tokens = F.pad(tokens, (0, pad), value=pad_token_id)

            cu_seqlens = torch.tensor(cu_seqlens).cuda() * cp_size

        # 创建 PackedSeqParams
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )

        tokens = tokens.unsqueeze(0)

    batch["tokens"] = tokens
    batch["packed_seq_params"] = packed_seq_params

    # ========== 构建 Loss Mask ==========
    loss_masks = []
    for loss_mask, total_length, response_length in zip(
        batch["loss_masks"],
        batch["total_lengths"],
        batch["response_lengths"],
    ):
        prompt_length = total_length - response_length
        # 左填充 prompt_length-1, 右填充 1
        loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)

        if not allgather_cp:
            loss_mask = slice_with_cp(loss_mask, 0, qkv_format, max_seqlen)

        loss_masks.append(loss_mask)

    # 合并和填充loss_masks
    if qkv_format == "thd":
        loss_masks = torch.cat(loss_masks)
        loss_masks = F.pad(loss_masks, (0, pad), value=0).unsqueeze(0)

    batch["full_loss_masks"] = loss_masks

    # ========== 处理多模态输入 ==========
    multimodal_train_inputs = batch.get("multimodal_train_inputs")
    if multimodal_train_inputs is not None:
        multimodal_data = {}
        multimodal_num_items = {}

        for mm_input_dict in multimodal_train_inputs:
            if mm_input_dict is not None:
                for key, mm_tensor in mm_input_dict.items():
                    if key not in multimodal_data:
                        multimodal_data[key] = mm_tensor
                        multimodal_num_items[key] = [mm_tensor.size(0)]
                    else:
                        multimodal_data[key] = torch.cat(
                            [multimodal_data[key], mm_tensor], dim=0
                        )
                        multimodal_num_items[key].append(mm_tensor.size(0))

        batch["multimodal_train_inputs"] = multimodal_data
        batch["multimodal_num_items"] = multimodal_num_items

    return batch
```

### 8.2 PackedSeqParams 结构

```python
@dataclass
class PackedSeqParams:
    """
    FlashAttention需要的打包序列参数

    用于支持变长序列的高效注意力计算
    """
    cu_seqlens_q: torch.Tensor     # Query累积长度 [num_seqs + 1]
    cu_seqlens_kv: torch.Tensor    # KV累积长度 [num_seqs + 1]
    max_seqlen_q: int              # Query最大长度
    max_seqlen_kv: int             # KV最大长度
    qkv_format: str                # "thd" 或 "bshd"
```

### 8.3 Loss Mask 处理

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Loss Mask 处理示意                                   │
└─────────────────────────────────────────────────────────────────────────────┘

原始序列:
┌─────────────────────────────────────────────────────────────────┐
│  P  P  P  P  R  R  R  R  R  R  R  R  R  R  R  R  R  R  R  R  R  │
│  [  Prompt  ]  [              Response              ]            │
└─────────────────────────────────────────────────────────────────┘

loss_mask (response部分):
                          [  1  1  1  1  1  1  1  1  1  1  1  1  1  1  ]

扩展后的 full_loss_mask:
┌─────────────────────────────────────────────────────────────────┐
│  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  0  │
│  [  padding  ] [           有效损失区域          ] [pad]       │
└─────────────────────────────────────────────────────────────────┘
   prompt_len-1          response_len                    1

说明:
- 前面填充 prompt_len-1 个0 (prompt部分不计算损失)
- 最后填充1个0 (padding token)
- 中间是response部分的原始loss_mask
```

---

## 9. 数据流总结图

### 9.1 完整数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Rollout 数据完整流                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐
│ JSONL/      │
│ Parquet     │
└──────┬──────┘
       │ read_file()
       ▼
┌─────────────┐     ┌──────────────────┐
│ dict (raw)  │────►│ Sample           │
│             │     │ - prompt         │
│             │     │ - label          │
│             │     │ - metadata       │
│             │     │ - multimodal     │
└──────┬──────┘     └────────┬─────────┘
       │                     │
       │ Dataset             │
       ▼                     │
┌─────────────┐              │
│ tokenizer   │              │
│ processor   │              │
└──────┬──────┘              │
       │                     │
       │ apply_chat_template │
       │ + process_vision    │
       ▼                     │
┌─────────────┐              │
│ Sample      │◄─────────────┘
│ - prompt    │  (已处理)
│ - tokens    │
│ - multimodal│
│   _inputs   │
└──────┬──────┘
       │
       │ DataSource.get_samples()
       ▼
┌─────────────┐
│ list[list   │
│ [Sample]]   │  (按组组织)
└──────┬──────┘
       │
       │ generate_and_rm_group()
       ▼
┌─────────────┐     ┌──────────────────┐
│ SGLang      │────►│ Sample           │
│ Engine      │     │ - response       │
│             │     │ - tokens (full)  │
│ /generate   │     │ - rollout_log_   │
│             │     │   probs          │
└──────┬──────┘     │ - status         │
       │            └────────┬─────────┘
       │                     │
       │ async_rm()          │
       ▼                     │
┌─────────────┐              │
│ RM Hub      │              │
│ - math      │              │
│ - f1        │              │
│ - custom    │              │
└──────┬──────┘              │
       │                     │
       ▼                     │
┌─────────────┐              │
│ Sample      │◄─────────────┘
│ (完整)      │  - reward
│             │  - loss_mask
└──────┬──────┘
       │
       │ 转换为 RolloutBatch
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RolloutBatch                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ "tokens":            [Tensor, Tensor, ...]     # 每个样本的token ids       │
│ "total_lengths":     [int, int, ...]           # prompt + response长度     │
│ "response_lengths":  [int, int, ...]           # response长度              │
│ "rewards":           [float, float, ...]       # 奖励值                    │
│ "loss_masks":        [Tensor, Tensor, ...]     # 损失掩码                  │
│ "rollout_log_probs": [Tensor, Tensor, ...]     # 推理时log概率             │
│ "multimodal_train_   [dict, dict, ...]         # 多模态数据                │
│  inputs":                                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       │ process_rollout_data() - DP分片
       ▼
┌─────────────┐
│ RolloutBatch│  (每个DP rank的本地数据)
│ (本地分片)  │
└──────┬──────┘
       │
       │ get_data_iterator() - 构建迭代器
       ▼
┌─────────────┐
│ DataIterator│
│ + micro_    │
│   batch_    │
│   indices   │
└──────┬──────┘
       │
       │ get_batch() - 构建训练批次
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Training Batch                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ "tokens":            Tensor[1, T_padded]        # 填充后的输入             │
│ "unconcat_tokens":   list[Tensor]               # 原始token列表            │
│ "packed_seq_params": PackedSeqParams            # FlashAttention参数       │
│ "full_loss_masks":   Tensor[1, T_padded]        # 损失掩码                 │
│ "multimodal_train_   dict[str, Tensor]          # 多模态数据               │
│  inputs":                                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       │ 送入Megatron训练
       ▼
┌─────────────┐
│ train_one_  │
│ step()      │
└─────────────┘
```

### 9.2 关键函数调用链

```
主训练循环:
│
├─ generate_rollout()
│   ├─ data_source.get_samples()           # 获取samples
│   ├─ generate_and_rm_group() [async]
│   │   ├─ generate()
│   │   │   └─ POST /generate               # SGLang推理
│   │   └─ async_rm()
│   │       └─ grade_answer_verl()          # 奖励计算
│   └─ 返回 list[list[Sample]]
│
├─ process_rollout_data()                   # DP分片
│   └─ ray.get(rollout_data_ref[dp_rank])
│
├─ get_data_iterator()
│   ├─ get_seqlen_balanced_partitions()     # 序列均衡
│   └─ DataIterator(...)
│
└─ train()
    └─ for step_id in num_steps:
        ├─ get_batch()
        │   ├─ data_iterator.get_next()
        │   ├─ slice_with_cp()              # CP切分
        │   └─ 构建 PackedSeqParams
        │
        └─ train_one_step()
            ├─ forward_backward_func()
            ├─ loss_function()
            └─ optimizer.step()
```

---

## 附录: 文件与函数索引

| 阶段 | 文件 | 核心函数 | 功能 |
|------|------|----------|------|
| 数据加载 | `utils/data.py` | `Dataset.__init__()` | 数据集初始化 |
| | | `read_file()` | 文件读取 |
| | | `_build_messages()` | 构建对话消息 |
| | | `filter_long_prompt()` | 长度过滤 |
| 数据源 | `rollout/data_source.py` | `RolloutDataSource.get_samples()` | 获取样本 |
| | | `RolloutDataSourceWithBuffer` | 带缓冲数据源 |
| 生成 | `rollout/sglang_rollout.py` | `generate_rollout()` | 主生成入口 |
| | | `generate()` | 单样本生成 |
| | | `generate_and_rm_group()` | 批量生成 |
| 奖励 | `rollout/rm_hub/__init__.py` | `async_rm()` | 奖励计算 |
| | `rollout/rm_hub/math_utils.py` | `grade_answer_verl()` | 数学验证 |
| 分片 | `utils/data.py` | `process_rollout_data()` | DP分片 |
| 迭代器 | `backends/megatron_utils/data.py` | `DataIterator` | 迭代器类 |
| | | `get_data_iterator()` | 迭代器构建 |
| 均衡 | `utils/seqlen_balancing.py` | `get_seqlen_balanced_partitions()` | 序列均衡 |
| | | `karmarkar_karp()` | KK算法 |
| CP切分 | `backends/megatron_utils/cp_utils.py` | `slice_with_cp()` | Token切分 |
| | | `all_gather_with_cp()` | All-Gather |
| | | `get_logits_and_tokens_offset_with_cp()` | 偏移计算 |
| 批次构建 | `backends/megatron_utils/data.py` | `get_batch()` | 训练批次构建 |
| | | `get_sum_of_sample_mean()` | 归约函数 |