# Nanotron 技术文档系列

> Nanotron 是 HuggingFace 开发的轻量级大规模语言模型预训练框架，支持 3D 并行（TP/PP/DP）。

---

## 文档目录

1. **00_整体架构与设计理念.md** - 3D并行架构、与生态关系、设计哲学
2. **01_配置系统与Config.md** - PretrainConfig详解、YAML配置
3. **02_模型并行引擎.md** - PipelineBlock、TP层、自动切分
4. **03_PP流水线并行.md** - PipelineEngine、1F1B调度、TensorPointer
5. **04_TP张量并行.md** - TP_Column/TP_Row线性层、AllReduce
6. **05_DP数据并行.md** - DDP/ZeRO、梯度同步
7. **06_优化器与学习率调度.md** - OptimizerBuilder、Warmup、Cosine
8. **07_Checkpoint与序列化.md** - CheckpointManager、版本兼容
9. **08_Dataset与数据处理.md** - CausalLMDataEngine、无限数据流
10. **09_Trainer训练循环.md** - DistributedTrainer、train_step
11. **10_实战配置指南.md** - 7B/14B/70B配置模板、SLURM启动

---

## 快速开始

```bash
# 安装
pip install nanotron

# 训练（使用示例配置）
python run_train.py --config-file examples/config_llama_7b.yaml
```

---

## 阅读路径

### 路径1：理解3D并行原理
1. 00_整体架构与设计理念
2. 03_PP流水线并行
3. 04_TP张量并行
4. 05_DP数据并行

### 路径2：配置和启动训练
1. 00_整体架构与设计理念
2. 01_配置系统与Config
3. 10_实战配置指南
4. 09_Trainer训练循环

### 路径3：修改模型/实验
1. 00_整体架构与设计理念
2. 02_模型并行引擎
3. 03_PP流水线并行
4. 09_Trainer训练循环

---

*更多文档读取中...*
