# Slime 训练框架技术文档库

本目录包含 Slime（清华开源 LLM 后训练框架）的完整技术文档，从架构设计到实现细节，覆盖训练全流程。

---

## 📚 文档目录

### 快速入门
| 文档 | 内容 | 建议阅读时间 |
|------|------|--------|
| **[00_整体架构概览](./00_整体架构概览.md)** | 框架定位、架构全景、核心概念、训练示例、参数手册 | 30 min |

### 核心模块详解
| 序号 | 文档 | 核心内容 | 关键源码定位 |
|------|------|----------|-------------|
| 01 | **[数据模块分析](./01_数据模块分析.md)** | Dataset、Rollout、DataBuffer、RM/Filter、Sample 类型 | `slime/utils/data.py`, `slime/rollout/*.py` |
| 02 | **[模型架构分析](./02_模型架构分析.md)** | Actor-Critic、ModelProvider、Bridge、参数冻结 | `slime/backends/megatron_utils/model_provider.py` |
| 03 | **[训练流程分析](./03_训练流程分析.md)** | 训练循环、损失函数（PPO/GRPO/SFT）、学习率调度 | `slime/backends/megatron_utils/loss.py` |
| 04 | **[通信与并行模块分析](./04_通信与并行模块分析.md)** | TP/PP/DP/CP/EP、进程组、Zigzag Ring Attention、FSDP | `slime/utils/distributed_utils.py` |
| 05 | **[调度与资源管理模块分析](./05_调度与资源管理模块分析.md)** | Ray Actor、Placement Group、RolloutManager、KV Cache 管理 | `slime/ray/rollout.py`, `slime/ray/placement_group.py` |
| 06 | **[Checkpoint 模块分析](./06_Checkpoint模块分析.md)** | 保存/加载、HF/Megatron 格式转换、在线权重同步 | `slime/backends/megatron_utils/checkpoint.py` |
| 07 | **[性能优化实践](./07_性能优化实践.md)** | 显存优化、计算优化、通信优化、配置调优、问题排查 | 综合指南 |

### 扩展与集成文档
| 主题 | 文档 | 用途 |
|------|------|------|
| Megatron-LM 集成与改造 | [Megatron-LM集成与改造.md](./Megatron-LM集成与改造.md) | Megatron 源码改造细节 |
| Rollout 数据流 | [Rollout_Data_Processing_Flow.md](./Rollout_Data_Processing_Flow.md) | Rollout 数据流完整处理流程 |
| 基础设施分析 | [Slime_Infrastructure_Analysis.md](./Slime_Infrastructure_Analysis.md) | 基础设施与部署分析 |
| 架构与执行流程 | [架构分析与执行流程.md](./架构分析与执行流程.md) | 高层架构与执行时序 |
| Moonlight SFT 训练 | [Moonlight_SFT_Training_Guide.md](./Moonlight_SFT_Training_Guide.md) | Moonlight 模型 SFT 训练指南 |
| Moonlight 16B SFT | [Moonlight_16B_SFT_Training_Script.md](./Moonlight_16B_SFT_Training_Script.md) | Moonlight-16B SFT 训练脚本 |

---

## 🚀 快速启动路径

### 路径1：快速上手（30分钟）
1. 阅读 **[00_整体架构概览](./00_整体架构概览.md)** - 了解框架全貌
2. 复制其中的 **GRPO 训练示例** 脚本进行首次训练
3. 参考 **参数手册** 进行微调

### 路径2：深度理解（1天）
1. 阅读 **数据模块（01）** + **模型架构（02）** - 理解数据流与模型结构
2. 阅读 **训练流程（03）** - 理解 PPO/GRPO 算法实现
3. 阅读 **调度模块（05）** - 理解 Ray 分布式工作原理

### 路径3：性能调优（按需）
1. 遇到 OOM → 直接查阅 **性能优化实践（07）§3 显存优化**
2. 训练速度慢 → 查阅 **并行分析（04）** 调整并行策略
3. 不稳定/崩溃 → 查阅 **Checkpoint（06）** 了解断点续训

---

## 📖 阅读建议

### 符号约定
| 符号 | 含义 |
|------|------|
| **【重点】** | 关键设计决策，必须理解 |
| **【实战】** | 可直接落地的配置或代码 |
| **【源码】** | 对应的具体实现文件和函数 |
| **⚠️ 注意** | 常见陷阱或易错点 |
| **💡 提示** | 进阶技巧或优化建议 |

### 代码示例约定
```python
# 伪代码 - 展示核心逻辑
"""
表意代码，省略了部分参数和异常处理，
仅用于说明核心概念和流程
"""

# 实战代码 (文件: train_ppo.py)
"""
可直接复制使用的完整配置或脚本
"""
```

---

## 🗂️ Slime 核心文件速查

### 入口脚本
| 文件 | 用途 |
|------|------|
| `train.py` | 同步训练主入口 |
| `train_async.py` | 异步训练主入口 |

### 核心模块（按代码量排序）
| 文件 | 行数 | 职责 |
|------|------|------|
| `slime/utils/arguments.py` | ~2200 | 参数解析（所有 args 定义） |
| `slime/ray/rollout.py` | ~1290 | RolloutManager 核心实现 |
| `slime/ray/placement_group.py` | ~200 | GPU 资源分配 |
| `slime/backends/megatron_utils/actor.py` | ~650 | Actor 训练 Actor |
| `slime/backends/megatron_utils/model.py` | ~790 | 训练核心逻辑 |
| `slime/backends/megatron_utils/loss.py` | ~1200 | PPO/GRPO/SFT 损失 |
| `slime/backends/sglang_utils/sglang_engine.py` | ~700 | SGLang 引擎封装 |

---

## 🔗 外部资源

### 官方资源
- **GitHub Repo**: https://github.com/THUDM/slime
- **官方文档**: https://thudm.github.io/slime/
- **架构博客**: [SGLang-Native Post-Training](https://lmsys.org/blog/2025-07-09-slime/)

### 依赖项目
- **Megatron-LM**: https://github.com/NVIDIA/Megatron-LM
- **SGLang**: https://github.com/sgl-project/sglang
- **Ray**: https://github.com/ray-project/ray

---

## 📝 文档更新日志

| 日期 | 版本 | 变更 |
|------|------|------|
| 2025-04-20 | v1.0 | 初始版本，完成核心 8 章 |

---

*本文档使用 Markdown 编写，建议使用支持 Mermaid 的编辑器查看架构图*
