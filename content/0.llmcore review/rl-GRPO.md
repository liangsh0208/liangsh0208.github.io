---
tags:
  - RL算法
---



paper: 
DeepSeekMath:  https://www.alphaxiv.org/abs/2402.03300

DeepSeek-R1: https://www.alphaxiv.org/abs/2501.12948

## GRPO

<alphaxiv-thinking-title title="提取GRPO算法细节" />

<alphaxiv-thinking-title title="整理公式与原理" />
GRPO（Group Relative Policy Optimization）是 DeepSeekMath 提出的 PPO 变体，核心改进是**舍弃了独立的 Critic（价值模型），改为利用同一问题下采样出的多个输出（Group）的相对奖励来估计基线（Baseline）**，从而显著降低显存占用和训练开销。

---

## 1. 从 PPO 到 GRPO 的动机

标准 PPO 在 LLM 强化学习中的目标函数为：

$$J_{PPO}(\theta) = \mathbb{E}_{q \sim P(Q), o \sim \pi_{\theta_{old}}(O|q)} \frac{1}{|o|} \sum_{t=1}^{|o|} \min\left( \frac{\pi_\theta(o_t|q,o_{<t})}{\pi_{\theta_{old}}(o_t|q,o_{<t})} A_t, \text{clip}(\cdot) A_t \right)$$

其中 $A_t$ 是通过 GAE（Generalized Advantage Estimation）计算的**优势函数**，需要维护一个与策略模型同等规模的 **Value Model** $V_\psi$。这给训练带来了两个问题：

- **显存与计算负担大**：Critic 模型通常和 Actor 一样大，显存开销翻倍；
- **逐 token 价值估计困难**：在数学推理等任务中，通常只有**最后一个 token** 能获得奖励模型的标量反馈，导致训练一个准确的逐 token 价值函数很困难。
![[附件/Pasted image 20260507112311.png]]
![[附件/Pasted image 20260507112551.png]]
- 上面截图是原始论文中的PPO 公式算法。 其中比较复杂的还是GAE 的计算。  GAE 的详细计算逻辑可以参考 PPO 原始paper： https://www.alphaxiv.org/abs/1707.06347 后边单独加一个章节解释一下。 
- reward 计算之后，一般的实现会再加一个KL 散度的抑制。防止模型更新太大。 GRPO 的实现把KL 散度放到了 优势函数计算的外部。



---

## 2. GRPO 的核心思想

GRPO 的关键改进如图 4 所示：

> **GRPO foregoes the value model, instead estimating the baseline from group scores, significantly reducing training resources.**

对于每个问题 $q$，GRPO 从旧策略 $\pi_{\theta_{old}}$ 中**采样一组输出** $\{o_1, o_2, \cdots, o_G\}$，然后用这 $G$ 个输出的**相对奖励**来计算优势，替代 Critic 模型提供的基线。

![[附件/Pasted image 20260507112335.png]]

---

## 3. GRPO 目标函数详解

GRPO 的优化目标定义为：

$$J_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)} \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min\left( \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})} \hat{A}_{i,t}, \text{clip}(\cdot) \hat{A}_{i,t} \right) - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right\}）$$

其中：

- $\pi_\theta$：当前策略模型；
- $\pi_{\theta_{old}}$：旧策略模型（采样用）；
- $\pi_{ref}$：参考模型（通常是 SFT 模型，用于 KL 约束）；
- $G$：每问题采样的输出数量（Group Size）；
- $\hat{A}_{i,t}$：第 $i$ 个输出第 $t$ 个 token 的**组相对优势**；
- $\varepsilon, \beta$：超参数。

与 PPO 公式 (2) 不同，GRPO **不在奖励中加入 KL 惩罚**，而是直接将 KL 散度加到损失函数中，避免优势计算复杂化。

KL 散度使用如下无偏估计器：

$$D_{KL}(\pi_\theta || \pi_{ref}) = \frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})} - \log \frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})} - 1$$

![[附件/Pasted image 20260507112453.png]]


---

## 4. 组相对优势 $\hat{A}_{i,t}$ 的计算

> *根据上面目标函数的公式可以看出。 GRPO 最大的创新和改动就是优势函数计算这一块。  原始PPO GAE 计算依赖于 value function的输出。 而GRPO 直接基于rward 做组内标准化。 
   对于奖励低于mean 的样本会提供一个负信号，对于奖励高于mean的样本会提供一个正向的信号。
   **GRPO 与 PPO 的唯一区别**在于 $\hat{A}_{i,t}$ 的计算方式：PPO 用 Critic/Value Model 通过 GAE 计算 $A_t$，而 GRPO 用同一问题下 $G$ 个输出的组相对奖励来计算 $\hat{A}_{i,t}$，从而省去了 Value Model。clip 机制本身保持不变。


根据监督信号的不同，GRPO 支持两种形式：

### 4.1 Outcome Supervision（结果监督）

每个输出 $o_i$ 的末端获得一个奖励 $r_i$。对组内奖励做标准化：

$$\tilde{r}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

然后**将输出内所有 token 的优势都设为该标准化奖励**：

$$\hat{A}_{i,t} = \tilde{r}_i$$

### *4.2 Process Supervision（过程监督）

> 论文主题还是介绍GRPO， 过程监督在这里重要性不太大。 

过程奖励模型对每个推理步骤的结束 token 给出奖励。设第 $i$ 个输出有 $K_i$ 个步骤，各步骤结束位置的奖励为 $r_{index(j)}^{(i)}$，同样做组内标准化：

$$\tilde{r}_{index(j)}^{(i)} = \frac{r_{index(j)}^{(i)} - \text{mean}(\mathbf{R})}{\text{std}(\mathbf{R})}$$

此时，token 级别的优势定义为**该 token 之后所有步骤标准化奖励的累加**：

$$\hat{A}_{i,t} = \sum_{index(j) \geq t} \tilde{r}_{index(j)}^{(i)}$$

这提供了更细粒度的训练信号。

---

## 5. 迭代 GRPO（Iterative RL）

如 Algorithm 1 所示，迭代 GRPO 进一步解决奖励模型与策略模型能力不匹配的问题：

1. 用当前策略采样新的训练数据；
2. 通过**回放机制**（replay，保留 10% 历史数据）持续训练旧奖励模型；
3. 将参考模型 $\pi_{ref}$ 更新为当前策略 $\pi_\theta$；
4. 用新奖励模型继续训练策略。

实验显示，迭代 RL 能持续提升性能，尤其是第一轮迭代带来显著提升。

>训练一轮后，做reward model 进行更新训练，然后再跑GRPO*
不是从头重新训练 reward model，而是使用 **replay mechanism**：每次更新时混入 **10% 的历史数据**，再叠加新采样的数据一起训练。这样既能适应新分布，又不会遗忘之前学过的判断标准。


![[附件/Pasted image 20260507123119.png]]
---

## 6. 与 PPO 的对比总结

| 维度 | PPO | GRPO |
|------|-----|------|
| **基线估计** | 需要独立的 Value Model | 用组内奖励均值/标准差估计 |
| **显存开销** | 高（Actor + Critic + Reward + Ref） | 低（省去 Critic） |
| **KL 约束** | 加入奖励 $r_t$ 中 | 直接加入损失函数 |
| **优势计算** | GAE + Value Model | 组相对标准化奖励 |
| **采样方式** | 单输出采样 | 每问题采样 $G$ 个输出 |

---

## 7. 统一视角下的 GRPO

论文将 SFT、RFT、DPO、PPO、GRPO 统一为如下梯度形式：

$$\nabla_\theta J_A(\theta) = \mathbb{E}_{(q,o) \sim D} \frac{1}{|o|} \sum_{t=1}^{|o|} GC_A(q, o, t, \pi_{ref}) \nabla_\theta \log \pi_\theta(o_t|q, o_{<t})$$

其中 $GC_A$ 为梯度系数。GRPO 的梯度系数为：

$$GC_{GRPO}(q, o, t, \pi_{ref}) = \hat{A}_{i,t} + \beta \left( \frac{\pi_{ref}(o_{i,t}|o_{i,<t})}{\pi_\theta(o_{i,t}|o_{i,<t})} - 1 \right)$$

相比 Online RFT（仅根据答案正确性赋予 0/1 系数），GRPO 通过奖励模型提供了**差异化的正负梯度系数**，并且 Process Supervision 还能提供步骤级细粒度信号，因此性能更优。




## PPO GAE 算法补充

ppo: https://www.alphaxiv.org/abs/1707.06347?chatId=019e007a-45a6-7392-bc4d-d97db2df92c0
- 原始论文计算公式
![[附件/Pasted image 20260507124340.png]]

![[附件/Pasted image 20260507123340.png]]



GAE 是一种**方差缩减**的优势函数估计方法，它通过引入参数 $\lambda$ 在偏差和方差之间进行权衡。

## 数学定义

论文中给出了GAE的截断版本（公式11）：

$$  
\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}  
$$

其中 **TD残差** $\delta_t$ 定义为（公式12）：

$$  
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)  
$$

## 参数说明

| 参数        | 含义              | 典型值                      |
| --------- | --------------- | ------------------------ |
| $\gamma$  | 折扣因子            | 0.99                     |
| $\lambda$ | GAE参数，控制偏差-方差权衡 | 0.95                     |
| $V(s)$    | 学习得到的状态价值函数     | -                        |
| $T$       | 轨迹片段长度（horizon） | 2048（MuJoCo）/ 128（Atari） |

## 计算流程

### 第一步：计算TD残差

对于每个时间步 $t$，计算：

$$  
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)  
$$

### 第二步：累积加权求和

从轨迹末端向前回溯计算优势估计：

$$  
\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}  
$$

### 具体展开

```
Â_0 = δ_0 + (γλ)δ_1 + (γλ)²δ_2 + ... + (γλ)^(T-1)δ_(T-1)
Â_1 = δ_1 + (γλ)δ_2 + (γλ)²δ_3 + ... + (γλ)^(T-2)δ_(T-1)
...
Â_(T-1) = δ_(T-1)
```

## 两种特殊情况

|$\lambda$ 值|效果|等价方法|
|---|---|---|
|$\lambda = 0$|$\hat{A}_t = \delta_t$|单步TD优势估计（高偏差，低方差）|
|$\lambda = 1$|$\hat{A}_t = \sum_{l=0}^{T-t-1} \gamma^l \delta_{t+l}$|蒙特卡洛优势估计（无偏差，高方差）|

> 当 $\lambda = 1$ 时，GAE退化为论文公式(10)的形式： $\hat{A}_t = -V(s_t) + r_t + \gamma r_{t+1} + \cdots + \gamma^{T-t+1}r_{T-1} + \gamma^{T-t}V(s_T)$


PPO 完整损失：
![[附件/Pasted image 20260507124033.png]]

- 下面这块更容易理解：


![[附件/Pasted image 20260507124202.png]]