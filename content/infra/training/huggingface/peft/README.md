# PEFT 架构分析文档

> 基于 Hugging Face PEFT 库的深度架构分析，细粒度拆解各模块实现。

---

## 文档导航

| 序号 | 文档 | 内容概述 | 优先级 |
|------|------|----------|--------|
| 00 | [00_整体架构与设计理念.md](./00_整体架构与设计理念.md) | PEFT 总体架构、设计哲学、非侵入式集成 | ⭐⭐⭐ |
| 01 | [01_PeftModel包装器.md](./01_PeftModel包装器.md) | `get_peft_model`、基类设计、forward 包装 | ⭐⭐⭐ |
| 02 | [02_LoRA实现详解.md](./02_LoRA实现详解.md) | `LoraConfig`、`LoraLayer`、低秩分解、alpha/r 缩放 | ⭐⭐⭐ |
| 03 | [03_QLoRA量化训练.md](./03_QLoRA量化训练.md) | bitsandbytes、4-bit 量化、双重量化、分页优化器 | ⭐⭐ |
| 04 | [04_其他PEFT方法.md](./04_其他PEFT方法.md) | IA3、AdaLoRA、P-Tuning、Prompt Tuning 等 | ⭐⭐ |
| 05 | [05_多Adapter管理.md](./05_多Adapter管理.md) | `load_adapter`、`set_adapter`、组合策略 | ⭐⭐ |
| 06 | [06_Adapter合并与导出.md](./06_Adapter合并与导出.md) | `merge_and_unload`、`save_pretrained`、TIES/DARE | ⭐⭐ |
| 07 | [07_模型层映射表.md](./07_模型层映射表.md) | `LORA_TARGET_MODULES`、各模型架构映射 | ⭐⭐ |
| 08 | [08_实战配置指南.md](./08_实战配置指南.md) | LoRA/QLoRA 配置、训练脚本模板 | ⭐⭐⭐ |

---

## 快速索引

### 按主题

| 主题 | 相关文档 |
|------|----------|
| **架构理解** | 00, 01 |
| **LoRA 实现** | 02, 07 |
| **量化训练** | 03, 08 |
| **多任务/多Adapter** | 05, 06 |
| **其他方法** | 04 |

### 按开发阶段

| 阶段 | 推荐阅读 |
|------|----------|
| **入门** | 00 → 01 → 02 |
| **上手训练** | 02 → 07 → 08 |
| **量化部署** | 03 → 08 |
| **进阶技巧** | 04 → 05 → 06 |

---

## 源码定位

```
/Users/danchen/Documents/1.RL_fw/huggingface/peft/src/peft/
├── __init__.py              # 主入口
├── peft_model.py            # PeftModel 包装器
├── mapping.py               # 类型映射注册表
├── mapping_func.py          # get_peft_model 入口
├── config.py                # PeftConfig 配置基类
├── tuners/
│   ├── lora/                # LoRA 实现
│   ├── ia3/                 # IA3 实现
│   ├──adalora/              # AdaLoRA 实现
│   ├── prefix_tuning/       # 提示学习相关
│   └── ...
└── utils/
    ├── constants.py         # 模型层映射表
    ├── save_and_load.py     # 保存/加载
    └── merge_utils.py       # Adapter 合并算法
```

---

## 前置知识

- PyTorch 基础：`nn.Module`、`state_dict`、forward hook
- Transformers 库：`PreTrainedModel`、`AutoModel`、模型配置
- PEFT 基本概念：LoRA、Prompt Tuning 等参数高效微调方法
- Python 数据类 (`dataclass`) 和类型标注

---

## 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2026-04-20 | 初始拆分，从单文件重构为细粒度文档 |

---

## 参考资源

- **GitHub**: https://github.com/huggingface/peft
- **官方文档**: https://huggingface.co/docs/peft/
- **论文**: LoRA: Low-Rank Adaptation of Large Language Models

---

*文档生成日期: 2026-04-20*
*基于 PEFT 版本: 0.19.1.dev0*
