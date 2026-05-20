---
created: 2026-05-06
---

# SGLang 源码分析笔记

> **版本锚定**：本系列笔记基于 SGLang 代码仓库 2026 年 4 月左右的快照编写。由于项目迭代速度快，部分文件路径和接口可能随版本变化，建议结合代码库最新状态交叉验证。

## 文档概览

| 章节 | 文件 | 一句话摘要 |
|------|------|-----------|
| 第零章 | [00_仓库整体架构介绍.md](00_仓库整体架构介绍.md) | 全局鸟瞰：仓库结构、多进程架构、核心数据流、并行策略总览 |
| 第一章 | [01_服务启动与初始化.md](01_服务启动与初始化.md) | Engine 启动全流程：子进程 fork、ZMQ 通信初始化、模型加载、KV Cache 预分配 |
| 第二章 | [02_服务入口与API层.md](02_服务入口与API层.md) | HTTP/gRPC 服务、OpenAI 兼容 API、请求/响应数据结构、流式输出 |
| 第三章 | [03_请求预处理.md](03_请求预处理.md) | TokenizerManager：分词、多模态处理、采样参数解析、约束解码初始化、LoRA/会话管理 |
| 第四章 | [04_调度系统.md](04_调度系统.md) | Scheduler 核心：事件循环、连续批处理、前缀感知调度、PrefillAdder 预算管理、结果处理 |
| 第五章 | [05_KV_Cache与前缀缓存.md](05_KV_Cache与前缀缓存.md) | RadixAttention：基数树前缀匹配、LRU/LFU/FIFO 淘汰、MHATokenToKVPool / MLATokenToKVPool |
| 第六章 | [06_模型执行.md](06_模型执行.md) | ModelRunner：前向传播、ForwardBatch 数据结构、CUDA Graph 优化、量化支持 |
| 第七章 | [07_Attention实现详解.md](07_Attention实现详解.md) | 可插拔注意力后端：FlashInfer、FlashAttention、FlashMLA、TorchNative、Triton 等 |
| 第八章 | [08_采样与输出处理.md](08_采样与输出处理.md) | Sampler：温度/Top-k/Top-p/Min-p 采样、PenalizerOrchestrator、约束解码、DetokenizerManager、推测解码 |
| 第九章 | [09_分布式与并行策略.md](09_分布式与并行策略.md) | TP/PP/DP/EP/CP/Disaggregation：通信原理、负载均衡、分离式推理 KV 传输 |
| 第十章 | [10_内核优化.md](10_内核优化.md) | sgl-kernel 库：注意力/MoE/GEMM/AllReduce/量化/元素级内核、JIT Triton 编译 |

## 按场景推荐阅读顺序

- **如果你关心推理延迟和吞吐**：**04 → 05 → 07 → 10**
- 如果你要添加新模型支持：**06 → 07 → 00 (8.1 工具调用)**
- 如果你要部署到生产环境：**01 → 02 → 09 → 00 (8.2 可观测性)**
- 如果你要优化长序列性能：**05 → 07 → 09 (CP章节)**
- 如果你要使用多模态能力：**03 → 06**
- 如果你要了解量化推理：**06 (第8节) → 10 (第5-6节)**

## 核心代码路径速查表

| 功能 | Python 路径 | CUDA/C++ 路径 |
|------|------------|---------------|
| Engine 启动 | `python/sglang/srt/entrypoints/engine.py` | - |
| HTTP Server | `python/sglang/srt/entrypoints/http_server.py` | - |
| gRPC Server | `python/sglang/srt/grpc/` | - |
| TokenizerManager | `python/sglang/srt/managers/tokenizer_manager.py` | - |
| Scheduler | `python/sglang/srt/managers/scheduler.py` | - |
| DetokenizerManager | `python/sglang/srt/managers/detokenizer_manager.py` | - |
| ModelRunner | `python/sglang/srt/model_executor/model_runner.py` | - |
| CUDA Graph | `python/sglang/srt/model_executor/cuda_graph_runner.py` | - |
| RadixCache | `python/sglang/srt/mem_cache/radix_cache.py` | - |
| Memory Pool | `python/sglang/srt/mem_cache/memory_pool.py` | - |
| FlashInfer 后端 | `python/sglang/srt/layers/attention/flashinfer_backend.py` | - |
| FlashMLA 后端 | `python/sglang/srt/layers/attention/flashmla_backend.py` | `sgl-kernel/csrc/attention/cutlass_mla_kernel.cu` |
| Sampler | `python/sglang/srt/sampling/sampler.py` | - |
| FP8 GEMM | `python/sglang/srt/layers/quantization/` | `sgl-kernel/csrc/gemm/fp8_gemm_kernel.cu` |
| GPTQ GEMM | - | `sgl-kernel/csrc/gemm/gptq/` |
| MoE Top-K | - | `sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu` |
| Custom AllReduce | - | `sgl-kernel/csrc/allreduce/custom_all_reduce.cu` |
| Function Call | `python/sglang/srt/function_call/` | - |
| Observability | `python/sglang/srt/observability/` | - |
| Compilation | `python/sglang/srt/compilation/` | - |

## 术语表

| 术语 | 全称/说明 | 首次出现 |
|------|----------|---------|
| **SGLang** | Structured Generation Language | - |
| **SRT** | SGLang Runtime | 第零章 |
| **RadixAttention** | 基于基数树的前缀缓存技术 | 第零章 |
| **ServerArgs** | 服务器启动参数配置类 | 第一章 |
| **PortArgs** | 进程间通信端口配置 | 第一章 |
| **TokenizerManager** | 请求预处理管理器（主进程） | 第一章 |
| **Scheduler** | 核心调度器（子进程） | 第一章 |
| **GenerateReqInput** | 原始 HTTP 生成请求输入 | 第二章 |
| **TokenizedGenerateReqInput** | 分词后的内部请求格式 | 第二章 |
| **Req** | 调度器内部请求对象 | 第四章 |
| **ScheduleBatch** | 调度器级别的批次对象 | 第四章 |
| **ModelWorkerBatch** | TP Worker 级别的批次对象 | 第四章 |
| **ForwardBatch** | 模型执行级别的 GPU 张量批次 | 第六章 |
| **ForwardMode** | 前向模式（EXTEND/DECODE/MIXED/IDLE） | 第四章 |
| **PrefillAdder** | Prefill 批次预算管理器 | 第四章 |
| **RadixCache** | 基数树前缀缓存 | 第五章 |
| **TreeNode** | RadixCache 的节点结构 | 第五章 |
| **ReqToTokenPool** | 请求到 Token 位置的映射池 | 第五章 |
| **MHATokenToKVPool** | 标准多头注意力的 KV Cache 池 | 第五章 |
| **MLATokenToKVPool** | DeepSeek MLA 压缩 KV Cache 池 | 第五章 |
| **EvictionStrategy** | 缓存淘汰策略（LRU/LFU/FIFO/SLRU） | 第五章 |
| **CUDA Graph** | CUDA 计算图捕获与重放优化 | 第六章 |
| **Piecewise CUDA Graph** | 分段计算图，支持动态 shape | 第六章 |
| **Attention Backend** | 注意力计算后端抽象 | 第七章 |
| **FlashMLA** | DeepSeek MLA 专用注意力后端 | 第七章 |
| **SamplingParams** | 采样参数配置 | 第八章 |
| **PenalizerOrchestrator** | 频率/存在/重复惩罚的调度器 | 第八章 |
| **GrammarManager** | 约束解码语法管理器 | 第三章 / 第八章 |
| **DetokenizerManager** | Token 反分词与增量解码进程 | 第八章 |
| **Speculative Decoding** | 推测解码加速技术 | 第八章 |
| **TP** | Tensor Parallel (张量并行) | 第九章 |
| **PP** | Pipeline Parallel (流水线并行) | 第九章 |
| **DP** | Data Parallel (数据并行) | 第九章 |
| **EP** | Expert Parallel (专家并行，MoE) | 第九章 |
| **CP** | Context Parallel (上下文并行) | 第九章 |
| **Disaggregation** | Prefill/Decode 分离式推理 | 第九章 |
| **EPLB** | Expert Load Balancing | 第零章 |
| **sgl-kernel** | C++/CUDA 高性能内核库 | 第十章 |
| **JIT Kernel** | Triton 即时编译内核 | 第十章 |

---

*本系列笔记为源码阅读整理，如有错漏，欢迎结合最新代码修正。*
