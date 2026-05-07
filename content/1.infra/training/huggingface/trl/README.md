# TRL (Transformers Reinforcement Learning) 架构文档

本目录包含 TRL 库的深度架构分析文档，按模块组织为细粒度子文档。

---

## 文档导航

### 核心架构

| 文档 | 内容 | 源码位置 |
|------|------|----------|
| [00_整体架构与设计理念.md](1.infra/training/huggingface/trl/00_整体架构与设计理念.md) | TRL 设计哲学、整体架构图、代码组织 | `trl/` |

### 核心训练器

| 文档 | 内容 | 源码位置 | 文件大小 |
|------|------|----------|----------|
| [01_SFTTrainer监督微调.md](01_SFTTrainer监督微调.md) | SFTConfig、packing、padding-free、VLM支持 | `trl/trainer/sft_trainer.py` | ~77KB |
| [02_DPOTrainer直接偏好优化.md](02_DPOTrainer直接偏好优化.md) | DPOConfig、reference model、multi-loss、MPO | `trl/trainer/dpo_trainer.py` | ~84KB |
| [03_GRPOTrainer组相对策略优化.md](03_GRPOTrainer组相对策略优化.md) | GRPOConfig、8种loss变体、DeepSeek-R1算法 | `trl/trainer/grpo_trainer.py` | ~151KB |
| [04_RewardTrainer奖励建模.md](04_RewardTrainer奖励建模.md) | RewardConfig、reward model训练 | `trl/trainer/reward_trainer.py` | - |
| [05_PPOTrainer近端策略优化.md](05_PPOTrainer近端策略优化.md) | PPOConfig、advantage计算、clip | `trl/experimental/ppo/` | 实验性 |

### 扩展功能

| 文档 | 内容 | 源码位置 |
|------|------|----------|
| [06_实验性训练器.md](06_实验性训练器.md) | KTO、RLOO、ORPO、NashMD等实验性方法 | `trl/trainer/`, `trl/experimental/` |
| [07_vLLM集成与生成.md](07_vLLM集成与生成.md) | Generator、vLLM配置、快速推理 | `trl/generation/` |
| [08_多模态VLM训练.md](08_多模态VLM训练.md) | VLM支持、图像处理、vision model | `trl/data_utils.py` |
| [09_实战配置指南.md](09_实战配置指南.md) | SFT/DPO/GRPO配置模板 | - |

---

## 快速参考

### 设计原则

1. **简单至上 (Simplicity Above All)**: 覆盖 90% 场景而非复杂的 100%
2. **代码重复但保持一致性**: 每个训练器自包含、独立可读
3. **显式优于隐式**: 避免 `hasattr` 防御式编程

### 核心依赖

```
Transformers ──► Accelerate ──► PEFT ──► vLLM ──► bitsandbytes
       ▲                                    ▲
       └────────── Datasets ────────────────┘
```

### 文件结构

```
trl/
├── trl/
│   ├── trainer/              # 核心训练器
│   │   ├── sft_trainer.py    # SFT
│   │   ├── dpo_trainer.py    # DPO
│   │   ├── grpo_trainer.py   # GRPO (~151KB, 最大)
│   │   ├── reward_trainer.py # 奖励模型
│   │   └── ...
│   ├── rewards/              # 内置奖励函数
│   ├── generation/           # vLLM集成
│   ├── experimental/         # 实验性功能
│   └── scripts/              # CLI脚本
├── docs/                     # 官方文档
└── tests/                    # 测试
```

---

## 按任务选择训练器

| 任务 | 推荐训练器 | 替代方案 |
|------|-----------|----------|
| 基础指令微调 | SFTTrainer | - |
| 偏好优化 (有对比数据) | DPOTrainer | ORPOTrainer |
| 推理/数学强化 | GRPOTrainer | RLOOTrainer |
| 视觉语言模型 | SFTTrainer | GRPOTrainer |
| 奖励模型训练 | RewardTrainer | - |
| 传统 RLHF | GRPOTrainer | PPOTrainer (实验性) |

---

## 关键论文索引

| 方法 | 论文 | 链接 |
|------|------|------|
| DPO | Direct Preference Optimization | [2305.18290](https://huggingface.co/papers/2305.18290) |
| GRPO | DeepSeekMath | [2402.03300](https://huggingface.co/papers/2402.03300) |
| DeepSeek-R1 | Incentivizing Reasoning | [2501.12948](https://huggingface.co/papers/2501.12948) |
| DAPO | Foundational RL Post-Training | [2503.14476](https://huggingface.co/papers/2503.14476) |
| Dr. GRPO | Surviving KL Divergence | [2503.20783](https://huggingface.co/papers/2503.20783) |
| KTO | Model Alignment without Preference Data | [2402.01306](https://huggingface.co/papers/2402.01306) |
| ORPO | Odds Ratio Preference Optimization | [2403.07691](https://huggingface.co/papers/2403.07691) |
| RLOO | Leave-One-Out RL | [2402.14740](https://huggingface.co/papers/2402.14740) |

---

## 源码定位速查

| 组件 | 路径 |
|------|------|
| 仓库根目录 | `/Users/danchen/Documents/1.RL_fw/huggingface/trl` |
| 核心包 | `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl` |
| 训练器 | `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/trainer` |
| 奖励函数 | `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/rewards` |
| 生成器 | `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/generation` |
| CLI脚本 | `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/scripts` |
| 实验功能 | `/Users/danchen/Documents/1.RL_fw/huggingface/trl/trl/experimental` |
| CLAUDE.md | `/Users/danchen/Documents/1.RL_fw/huggingface/trl/CLAUDE.md` |

---

## 使用建议

### 阅读顺序

1. **入门**: 先阅读 [00_整体架构与设计理念.md](1.infra/training/huggingface/trl/00_整体架构与设计理念.md) 了解设计哲学
2. **选择训练器**: 根据任务选择对应训练器文档
3. **配置参考**: 查看 [09_实战配置指南.md](09_实战配置指南.md) 获取配置模板
4. **深入了解**: 阅读具体训练器源码

### 代码阅读建议

- 每个训练器自包含设计，可独立理解
- 启动入口通常在 `_ Trainer.__init__` 和 `train()` 方法
- Loss 计算在 `_compute_loss` 方法中
- 参考 TRL 官方文档: https://huggingface.co/docs/trl

---

*本文档集重构自 `trl_架构分析.md`，采用细粒度子目录结构，便于按需查阅。*
