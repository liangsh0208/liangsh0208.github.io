# Reward Manager 模块

**路径**：`verl/workers/reward_manager/`、`verl/utils/reward_score/`

Reward Manager 负责为 rollout 生成的回答计算奖励信号，支持规则函数、神经网络奖励模型、过程奖励等多种形式。

---

## 1. 抽象接口

**文件**：`verl/workers/reward_manager/abstract.py`

```python
class AbstractRewardManager:
    """奖励管理器基类"""

    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False
    ) -> torch.Tensor | dict[str, Any]:
        """
        输入：DataProto，包含：
            - batch["prompts"]：prompt token ids
            - batch["responses"]：response token ids
            - batch["attention_mask"]：完整序列 mask
            - non_tensor_batch["reward_model"]：{"ground_truth": ...}
            - non_tensor_batch["data_source"]：数据来源标识

        输出：
            - return_dict=False：reward_tensor (bs, response_len)
              最后一个有效 token 位置有非零值（outcome reward）
            - return_dict=True：包含 reward_tensor 和额外信息的字典
        """
        raise NotImplementedError

    def _extract_reward_from_rm_scores(self, data, return_dict):
        """如果 batch 中已有 rm_scores（由奖励模型计算），直接返回"""
        if "rm_scores" in data.batch:
            return data.batch["rm_scores"]
        return None
```

---

## 2. 注册机制

```python
REWARD_MANAGER_REGISTRY: dict[str, type[AbstractRewardManager]] = {}

def register(name: str):
    def wrapper(cls):
        REWARD_MANAGER_REGISTRY[name] = cls
        return cls
    return wrapper

# 使用方式
@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    ...
```

通过配置文件指定使用哪种 RewardManager：

```yaml
reward_model:
  reward_manager: "naive"   # naive / prime / dapo / batch
```

---

## 3. `NaiveRewardManager`

**文件**：`verl/workers/reward_manager/naive.py`

最常用的规则奖励管理器，对每条回答独立计算奖励。

```python
@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source"):
        self.compute_score = compute_score or default_compute_score
        # compute_score 是一个函数，签名：
        # compute_score(data_source, solution_str, ground_truth, extra_info) -> float

    def __call__(self, data: DataProto, return_dict=False):
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        for i in range(len(data)):
            # 解码 prompt 和 response
            prompt_str = tokenizer.decode(valid_prompt_ids)
            response_str = tokenizer.decode(valid_response_ids)

            # 获取 ground truth 和数据来源
            ground_truth = data[i].non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data[i].non_tensor_batch["data_source"]

            # 调用打分函数（规则匹配、正则表达式等）
            score = self.compute_score(data_source, response_str, ground_truth)

            # 将分数放在最后一个有效 token 位置（outcome reward）
            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor
```

---

## 4. 奖励函数（`utils/reward_score/`）

### 4.1 `default_compute_score`

```python
def default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """根据 data_source 路由到对应的打分函数"""
    if "gsm8k" in data_source:
        return gsm8k_reward(solution_str, ground_truth)
    elif "math" in data_source:
        return math_reward(solution_str, ground_truth)
    elif "code" in data_source:
        return code_reward(solution_str, ground_truth)
    ...
```

### 4.2 数学奖励

**文件**：`utils/reward_score/math_reward.py`、`math_verify.py`、`math_dapo.py`

```python
def gsm8k_reward(solution: str, ground_truth: str) -> float:
    """提取回答中的最后一个数字，与 ground_truth 比较"""
    answer = extract_answer(solution)
    return 1.0 if answer == ground_truth else 0.0

def math_reward(solution: str, ground_truth: str) -> float:
    """使用 sympy 进行符号数学等价验证"""
    pred = extract_boxed_answer(solution)
    return verify_math_equivalence(pred, ground_truth)
```

**DAPO 数学奖励**（`math_dapo.py`）：添加格式奖励，要求回答有特定结构（如 `<think>...</think>\n<answer>...</answer>`）。

### 4.3 代码奖励

**文件**：`utils/reward_score/prime_code/`

通过沙箱执行代码并运行测试用例：
```python
def code_reward(solution: str, test_cases: list) -> float:
    """在沙箱中执行代码，检查是否通过所有测试用例"""
    sandbox = Sandbox()
    results = [sandbox.run(solution, test) for test in test_cases]
    return sum(results) / len(results)  # 通过率
```

### 4.4 其他奖励函数

| 文件 | 用途 |
|------|------|
| `gsm8k.py` | GSM8K 数学题 |
| `geo3k.py` | 几何题（配合图像工具）|
| `rlla.py` | RLLA 论文的奖励 |
| `jpeg_compressibility.py` | 图像压缩质量奖励（扩散模型 RL）|
| `search_r1_like_qa_em.py` | 检索增强问答的精确匹配奖励 |
| `sandbox_fusion/` | 远程沙箱代码执行 |

---

## 5. `PRIMERewardManager`

**文件**：`verl/workers/reward_manager/prime.py`

PRIME（Process Reward with Implicit MEaning）使用过程奖励：不只在最终答案给分，而是对中间推理步骤也给出细粒度奖励。

```python
@register("prime")
class PRIMERewardManager(AbstractRewardManager):
    """
    结合 outcome reward 和 process reward：
    - outcome_reward：对最终答案打分（规则/RM）
    - process_reward：对每个中间 token 的分布变化打分
      （用隐式奖励模型，即参考策略的 log_prob 差异）
    """

    def __call__(self, data: DataProto, return_dict=False):
        # Outcome reward（来自规则函数）
        outcome_reward = compute_outcome_reward(data)

        # Process reward（基于 token-level log_prob 变化）
        process_reward = compute_process_reward(
            old_log_probs=data.batch["ref_log_prob"],
            new_log_probs=data.batch["rollout_log_probs"],
        )

        # 加权组合
        total_reward = (
            self.outcome_coeff * outcome_reward
            + self.process_coeff * process_reward
        )
        return total_reward
```

---

## 6. `DAPORewardManager`

**文件**：`verl/workers/reward_manager/dapo.py`

DAPO（Direct Advantage Policy Optimization）奖励：
- 过滤掉 rollout 中全部正确或全部错误的 batch（无学习信号）
- 对困难样本上采样，提升训练效率

```python
@register("dapo")
class DAPORewardManager(AbstractRewardManager):
    def __call__(self, data: DataProto, return_dict=False):
        rewards = compute_base_rewards(data)

        # 按 uid 分组
        for uid, indices in group_by_uid(data):
            group_rewards = rewards[indices]
            # 全对或全错 → clip 到 0（无信号）
            if group_rewards.all() or not group_rewards.any():
                rewards[indices] = 0.0

        return rewards
```

---

## 7. `BatchRewardManager`

**文件**：`verl/workers/reward_manager/batch.py`

批量奖励计算，将所有样本打包成一个大 batch 调用奖励模型（RM），提高 GPU 利用率：

```python
@register("batch")
class BatchRewardManager(AbstractRewardManager):
    def __call__(self, data: DataProto, return_dict=False):
        # 收集所有 response，打包成一个 batch
        all_responses = [data[i].batch["responses"] for i in range(len(data))]
        big_batch = pad_and_stack(all_responses)

        # 一次性调用 RM
        rm_scores = self.reward_model(big_batch)

        # 按原始顺序拆分返回
        return split_scores(rm_scores, data)
```

---

## 8. 奖励信号的数据流

```
RewardManager.__call__(batch)
        │
        ▼
reward_tensor: (bs, response_len)
  ├─ 规则奖励：只在最后一个有效 token 位置有值（outcome reward）
  │    例：[0, 0, 0, 1.0] 或 [0, 0, 0, -1.0]
  │
  └─ 过程奖励（PRIME）：每个 token 位置都有值
       例：[0.1, -0.2, 0.3, 0.8]
        │
        ▼
batch["token_level_scores"] = reward_tensor
        │
        ▼（apply_kl_penalty）
batch["token_level_rewards"] = token_level_scores - beta * KL_divergence
        │
        ▼（compute_advantage）
batch["advantages"], batch["returns"]
```

---

## 9. Reward 配置示例

```yaml
reward_model:
  reward_manager: "naive"
  compute_score: null   # null 时使用 default_compute_score

# 或者使用 RM 模型打分
reward_model:
  reward_manager: "naive"
  model:
    path: "/path/to/reward_model"
    tensor_model_parallel_size: 1
```
