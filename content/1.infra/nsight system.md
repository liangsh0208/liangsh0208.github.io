---
created: 2026-05-19
tags:
  - infra
---



nv 开发社区： https://developer.nvidia.cn/nsight-systems
nvsys官网： https://docs.nvidia.com/nsight-systems/UserGuide/index.html
nvcu doc: https://docs.nvidia.com/nsight-compute/
[Nsight Tools Tutorials](https://developer.nvidia.com/tools-tutorials)



# 入门线路
**如何在大模型训练和推理的性能优化实践中，用好 `nsys`, `ncu` 和 `PyTorch Profiler`**。

下面我会介绍一套深入实战的方法论和操作细节，希望能帮你解决一些实际项目中遇到的难题。

### 🧭 系统性调优流程

面对大模型复杂的训推流程，一个清晰的“三步走”方法论是最高效的：

1.  **宏观定位：`PyTorch Profiler` + `nsys`**
2.  **微观诊断：`ncu`**
3.  **验证迭代**

### 🔧 工具实战详解

#### 1. `nsys` (Nsight Systems)：宏观时间线定位

它的核心价值在于回答：“我的程序时间花在哪儿了？”，帮助你快速缩小问题范围，找到哪个环节是“路障”。

*   **PyTorch Profiler**负责宏观的PyTorch层性能，而 **`nsys`** 则深入到更底层的CUDA API、内核执行和CPU/GPU交互进行系统级分析。

*   **典型瓶颈分析**：我结合 `nsys` 的系统级视图和几个常见的性能指标来分析以下几种典型的瓶颈问题。
    *   **CPU/预处理瓶颈 (DataLoader)**：如果GPU使用率忽高忽低、跳动很大，往往意味着GPU在等待CPU或数据传输。`nsys`时间轴可以帮助你确认GPU是否长时间空闲。
    *   **显存瓶颈**：在推理时，如果你观察到模型吞吐量没有随 `batch_size` 线性增长，且`nvidia-smi`显示显存用满，很可能是**KV-Cache**过大引起了**分页碎片化**或**容量超限**。启用`cuda_malloc_async`内存池和KV Cache压缩是最直接的优化方向。
    *   **小Kernel过多**：如果GPU利用率看似很高但吞吐量低，可能问题不在“计算”，而在“调度”。Transformer模型中大量的ElementWise、Softmax等细粒度Kernel会累积可观的调用开销。

*   **NVTX标注**：面对大模型训练中细碎复杂的Kernel调用，借助NVTX（NVIDIA Tools Extension）进行标注能极大地提升`nsys`的可读性。通过 PyTorch 的 `torch.cuda.nvtx` 模块，你可以把代码的逻辑阶段标记出来。
    ```python
    import torch.cuda.nvtx as nvtx
    with nvtx.range("forward"): output = model(input)
    with nvtx.range("backward"): loss.backward()
    ```
    > 在运行 `nsys` 时，务必添加 `--trace=nvtx` 参数，才能在时间线上看到这些标注。

*   **多节点分布式训练**：如果你使用`megatron-lm`或`DeepSpeed`进行多节点训练，直接运行`nsys`默认只采集单GPU的数据。正确的做法是在**所有节点**上都启动 `nsys`，并确保通过 SSH 等工具设置了正确的环境变量，以接收来自主节点的 `CUDA_VISIBLE_DEVICES` 等设置。

#### 2. `PyTorch Profiler`：宏观到微观的桥梁

`PyTorch Profiler`是定位PyTorch层面性能瓶颈最直接的工具。

*   **组合 `schedule` 实现采样**：对于长时间的训推任务，全部采集会产生巨大文件。推荐使用 `schedule` 进行周期性采样，只保留最有价值的数据。
    ```python
    from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

    # 跳过前5步(wait)，预热2步(warmup)，记录6步(active)，重复2次(repeat)
    my_schedule = schedule(wait=5, warmup=2, active=6, repeat=2)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=my_schedule,
        on_trace_ready=tensorboard_trace_handler('./log'), # 结果自动保存
        record_shapes=True,
        profile_memory=True,      # 开启内存分析
        with_stack=True,          # 开启堆栈信息，便于追踪代码行
    ) as prof:
        for step in range(steps):
            train_one_step() 
            prof.step()           # 通知profiler进入下一步
    ```
    > 可以利用 `prof.export_chrome_trace("trace.json")` 将结果导出，在 Chrome 浏览器的 `chrome://tracing` 或 Perfetto 工具中直观地观察。

*   **内存分析高级技巧**：Profiler还提供了一些高级方法来应对显存（OOM）问题。
    *   通过设置 `profile_memory=True` 并结合 `record_shapes`，可以按内存占用排序，定位显存大户。
    *   使用 `torch.cuda.memory._dump_snapshot()` 生成内存快照文件，再用 `pickle` 加载分析，对于追踪训练循环中的内存泄漏问题尤为有效。

#### 3. `ncu` (Nsight Compute)：微观内核剖析

当瓶颈锁定到具体的CUDA Kernel（如FlashAttention或自定义算子）时，就需要用 `ncu` 来分析它“**为什么慢**”。

*   **核心指标解读**：`ncu`提供了大量底层硬件计数器，熟练掌握其中几个关键指标，就能准确判断内核瓶颈：
    *   **计算或内存吞吐量受限问题**：通过`Compute Workload`和`Memory Workload`等`SpeedOfLight`相关指标，判断内核是“计算受限”(Compute-Bound)还是“访存受限”(Memory-Bound)。
    *   **低占用率问题**：如果内核是访存受限，低`Occupancy`(占用率)往往意味着无法通过并行来隐藏访存延迟。例如，某个Kernel的实测活跃Warp数远低于SM的理论上限，就说明存在显著的资源限制（如寄存器使用过多）。
    *   **线程束发散问题**：在包含大量分支判断的逻辑中，如果`Branch Divergence`指标很高，意味着SIMD效率低下。这通常提示你需要重构代码，或通过数据预处理尽量避免发散。

*   **高级用法**：对于一些高级场景，`ncu`也有一些便捷的过滤选项。
    *   **聚焦特定Kernel**：大型模型会调用成千上万个Kernel，使用 `--kernel-regex` 参数可以只分析目标内核。
    *   **自动化闭环优化**：学术界已有研究（如CudaForge）将 `ncu` 输出的硬件指标作为反馈信号，驱动LLM Agent自动迭代生成和优化CUDA内核。

### 📈 实战案例：优化一个Transformer模块的推理延迟

假设我们遇到了一个具体问题：一个基于Transformer的大模型推理服务延迟过高。

1.  **第一步：宏观定位 (`PyTorch Profiler`)**
    *   使用PyTorch Profiler抓取几个推理step的trace。
    *   重点关注`aten::_scaled_dot_product_attention`或`FlashAttention`等自注意力相关算子的耗时。如果它们的耗时占比畸高，我们就定位到了主要矛盾。

2.  **第二步：系统级验证 (`nsys`)**
    *   运行 `nsys profile --trace=cuda,nvtx --output=attention_profile python inference.py`。
    *   在nsys的时间线视图里，验证GPU执行流是否与Profiler的结果一致。

3.  **第三步：内核深入分析 (`ncu`)**
    *   既然问题指向了Attention，我们用 `ncu` 重点分析对应的FlashAttention内核。
    *   使用 `ncu --set full --kernel-regex "flash_attn" python inference.py` 采集详细报告。
    *   打开报告，重点关注`SpeedOfLight`和`MemoryWorkload`指标。
        *   **假设分析**：如果发现`MemoryWorkload` (内存吞吐量)指标很高，且`Compute Workload` (计算吞吐量)相对较低，我们就判定该内核是 **“显存带宽受限”(Memory-Bound)**。
    *   **分析结论与优化**：若判断是显存带宽受限，优化的核心思路就是“减少访存”。
        *   **算子融合**：检查并简化Kernel，将多个简单的访存操作合并，或尝试使用[FlashAttention-3](https://github.com/Dao-AILab/flash-attention)等最新的、更激进的融合kernel。
        *   **KV-Cache优化**：对于大序列场景，尝试使用更高效的存储格式（如PagedAttention），或压缩KV-Cache的数据精度（如FP8），以减轻对显存带宽的压力。
        *   **模型重写**：如果优化后仍受限，可以考虑使用更低精度的数据类型（如FP8），这能在不增加带宽压力的情况下“传输”更多数据。

### 💎 总结

用一句话来总结这几个工具的分工就是：
> `PyTorch Profiler`负责**规划路线图**（宏观定位问题）；`nsys`是负责**查看实时交通路况**（系统级资源调度）；而`ncu`则是**事故现场的鉴定专家**（微观分析内核）。

在实际调优中，遵循“PyTorch Profiler ➡️ nsys ➡️ ncu”的流程，就像用导航先定位拥堵路段，再用路况看拥堵原因，最后派专家现场分析一样，能够帮助你高效地解决各类性能问题。

