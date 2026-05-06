# Moonlight-16B-A3B SFT 训练脚本

## 训练环境

- **硬件**: 单机 8 卡 GPU
- **模型**: Moonlight-16B-A3B-Instruct (MoE)
- **任务**: SFT 监督微调

## 优化后的训练脚本

```bash
#!/bin/bash

# ============================================
# 清理环境
# ============================================
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

# ============================================
# 清理 save 目录（首次训练时取消注释）
# ============================================
# 如果是全新训练，确保 save 目录为空，避免残留状态文件
# rm -rf /mnt/mednas_new/danchen.lsh/4.result_slime/Moonlight-16B-A3B_slime/
# mkdir -p /mnt/mednas_new/danchen.lsh/4.result_slime/Moonlight-16B-A3B_slime/

set -ex

# ============================================
# 环境变量
# ============================================
export PYTHONBUFFERED=16

# NVLink 检测
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# 加载模型配置
source "/mnt/mednas_new/danchen.lsh/develop/slime/scripts/models/moonlight.sh"

# ============================================
# Checkpoint 配置
# ============================================
# 重要：--load 是必需的！代码要求 args.load 或 args.pretrained_checkpoint 至少有一个
# 首次训练：--load 指向 HF 模型目录（与 --hf-checkpoint 相同）
#           代码会自动检测是否为 Megatron 格式，如果是 HF 格式则用 bridge 加载
# 继续训练：--load 指向之前保存的 Megatron checkpoint 目录

CKPT_ARGS=(
   --hf-checkpoint /mnt/mednas_new/danchen.lsh/models/Moonlight-16B-A3B-Instruct
   --load /mnt/mednas_new/danchen.lsh/models/Moonlight-16B-A3B-Instruct
   # --ref-load /mnt/mednas_new/danchen.lsh/models/Moonlight-16B-A3B-Instruct_torch_dist
   --save /mnt/mednas_new/danchen.lsh/4.result_slime/Moonlight-16B-A3B_slime/
   --save-interval 20
)

# ============================================
# 并行策略配置 (核心优化)
# ============================================
# Moonlight-16B-A3B 是 MoE 模型
# TP=2 + EP=4 + ETP=1 = 8 GPU
# 非专家部分通过 TP=2 分片到 2 卡
# 专家部分通过 EP=4 分片到 4 组

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1

   # 选择性重计算（推荐，平衡显存和计算效率）
   --recompute-granularity selective

   # 动态批次
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096

   # 通信优化
   --use-distributed-optimizer
   # 注意：以下两个选项在当前版本有 bug，会导致重复 train() 调用时报错
   # 暂时禁用，等待上游修复
   # --overlap-grad-reduce
   # --overlap-param-gather
)

# ============================================
# SFT 训练配置
# ============================================
# 重要：--rollout-function-path 必须指定为 sft_rollout
#       默认值是 sglang_rollout，需要 sglang 参数（debug-train-only 会跳过 sglang 解析）
# --debug-train-only 跳过 sglang 引擎创建，纯 SFT 训练
# --rollout-global-dataset 启用全局数据集模式

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --rollout-global-dataset
   --prompt-data /mnt/mednas_new/danchen.lsh/datasets/sft_datasets/OpenHermes-2.5/openhermes2_5.jsonl
   --input-key conversations
   --rollout-shuffle
   --num-epoch 3
   --rollout-batch-size 128
   --global-batch-size 128

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

# ============================================
# 优化器配置
# ============================================
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

# ============================================
# Wandb 配置（可选）
# ============================================
# 如需使用 wandb，取消注释并设置 WANDB_KEY 环境变量
# 如需禁用 wandb，注释掉整个 WANDB_ARGS 或只保留空数组

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group moonlight-16B-A3B-sft
   # --wandb-key ${WANDB_KEY}
)

# ============================================
# 其他配置
# ============================================
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32

   # MoE 配置
   --moe-token-dispatcher-type alltoall
   --moe-grouped-gemm
)

# ============================================
# Ray 启动
# ============================================
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head \
   --node-ip-address ${MASTER_ADDR} \
   --num-gpus 8 \
   --disable-usage-stats \
   --dashboard-host=0.0.0.0 \
   --dashboard-port=8265

# ============================================
# 提交训练任务
# ============================================
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${MISC_ARGS[@]}
```

---

## 关键优化点说明

| 配置项 | 修改前 | 修改后 | 原因 |
|-------|-------|-------|------|
| `tensor-model-parallel-size` | 1 | **2** | 非专家参数分片，减轻单卡显存压力 |
| `expert-model-parallel-size` | 8 | **4** | 配合 TP 调整，TP×EP=8 |
| `recompute-granularity` | full + uniform + 1层 | **selective** | 修正冲突配置，平衡显存和计算 |
| `max-tokens-per-gpu` | 8192 | **4096** | MoE 激活显存更大，降低避免 OOM |
| `overlap-grad-reduce` | 无 | **禁用** | 当前版本有 bug，多次 train() 调用会报错 |
| `overlap-param-gather` | 无 | **禁用** | 当前版本有 bug，多次 train() 调用会报错 |
| `use-distributed-optimizer` | 默认启用 | **显式启用 + overlap** | 更好的显存利用 |
| `moe-token-dispatcher-type` | flex + deepep | **alltoall** | 基础配置更稳定 |
| `--colocate` | 启用 | **移除** | SFT 模式不需要 |
| `--rollout-function-path` | 默认 sglang | **sft_rollout** | SFT 训练必需！否则缺少 sglang 参数报错 |
| `--rollout-global-dataset` | 无 | **添加** | SFT 模式需要全局数据集 |
| `--load` | 未设置 | **指向 HF 模型目录** | 必需！首次训练与 --hf-checkpoint 相同路径 |

---

## 验证训练是否正常

### 1. 检查并行配置生效

训练启动后应该看到：

```
> building model ...
 > tensor model parallel size: 2
 > expert model parallel size: 4
 > sequence parallel: True
 > distributed optimizer: True
```

### 2. 监控 GPU 利用率

```bash
# 另开终端监控
watch -n 1 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv'
```

**正常指标**：
- GPU 利用率 > 85%
- 显存使用稳定在 70-90%

### 3. 训练日志关键指标

```
train/step: 100
train/loss: 2.34
train/grad_norm: 0.5
train/lr-pg_0: 1.0e-06
```

---

## 常见问题排查

### Q1: OOM 错误

```bash
# 降低 max-tokens-per-gpu
--max-tokens-per-gpu 2048

# 或增加重计算
--recompute-granularity full
```

### Q2: 训练速度慢

```bash
# 检查是否有通信瓶颈
export NCCL_DEBUG=INFO

# 确认 overlap 生效
# 日志中应该看到 "Overlap param gather enabled"
```

### Q3: AssertionError: args.load is not None

```bash
# 错误原因：--load 是必需参数
# 首次训练：--load 指向 HF 模型目录（与 --hf-checkpoint 相同）
--load /mnt/mednas_new/danchen.lsh/models/Moonlight-16B-A3B-Instruct

# 继续训练：--load 指向 Megatron checkpoint 目录
--load /mnt/mednas_new/danchen.lsh/4.result_slime/Moonlight-16B-A3B_slime/
```

### Q4: FileNotFoundError: iter_XXXXXXX 目录不存在

```bash
# 错误：No such file or directory: '.../iter_0000099'
# 原因：save 目录中有残留的 latest_checkpointed_iteration.txt 文件
#       但对应的 iter_XXXXXXX 目录不存在

# 解决方案1：清空 save 目录重新开始
rm -rf /mnt/mednas_new/danchen.lsh/4.result_slime/Moonlight-16B-A3B_slime/
mkdir -p /mnt/mednas_new/danchen.lsh/4.result_slime/Moonlight-16B-A3B_slime/

# 解决方案2：删除残留的状态文件
rm /mnt/mednas_new/danchen.lsh/4.result_slime/Moonlight-16B-A3B_slime/latest_checkpointed_iteration.txt
```

### Q5: Checkpoint 路径不存在

```bash
# 错误：args.load does not exist or is an empty directory
# 检查路径是否正确，确保目录非空
ls -la /path/to/checkpoint
```

---

## 配置选择参考

### 如果显存充足 (80GB A100/H100)

```bash
--max-tokens-per-gpu 6144
--recompute-granularity selective
```

### 如果显存紧张 (40GB A100)

```bash
--max-tokens-per-gpu 2048
--recompute-granularity full
```

### 如果要更快训练速度

```bash
--recompute-granularity full          # 牺牲计算换显存
--max-tokens-per-gpu 4096             # 保持这个值
# 确保 overlap 生效
```