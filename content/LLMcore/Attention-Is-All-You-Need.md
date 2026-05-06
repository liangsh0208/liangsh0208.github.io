
paper 地址： https://arxiv.org/pdf/1706.03762

- 复习基础的transformer 结构和最原始的实现公式算法。

## transformer 架构。
![[Pasted image 20260505230935.png]]
![[Pasted image 20260506160425.png]]
设计动机： 在 Transformer 之前，序列建模的主流方案是 RNN/LSTM。它们有两个根本缺陷：

- **无法并行**：隐状态 $h_t$ 依赖 $h_{t-1}$，必须逐步计算
- **长程依赖难学**：信号在序列中传播路径长，梯度容易消失

Transformer 的核心思想是**完全抛弃递归，用注意力机制直接建模任意两个位置之间的依赖关系**，路径长度从 $O(n)$ 降为 $O(1)$


## 输入处理

### Token Embedding

将离散 token 映射为 $d_{\text{model}} = 512$ 维稠密向量。编码器输入、解码器输入、解码器输出前的线性层**三者共享同一权重矩阵**，权重乘以 $\sqrt{d_{\text{model}}}$ 进行缩放。

### 位置编码（Positional Encoding）

模型无递归无卷积，本身不感知顺序，因此在 embedding 上**叠加**位置编码：

$$  
PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)  
$$
$$  
PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)  
$$

选择正弦/余弦函数的原因：对任意固定偏移 $k$，$PE_{pos+k}$ 可表示为 $PE_{pos}$ 的线性函数，让模型易于学习**相对位置**关系。实验也表明，可学习位置编码与此结果几乎相同（25.7 vs 25.8 BLEU），但正弦版本理论上可外推到训练时未见过的更长序列。
![[Pasted image 20260506155435.png]]


## attention

- 注意力的点积算法。 传统NLP 计算注意力机制，有多种计算方案，GPT 之后，基本上默认dot product算法了。
- 这是整个架构的基础计算单元。给定查询矩阵 $Q$、键矩阵 $K$、值矩阵 $V$，输出计算为：

$$  
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V  
$$

除以 $\sqrt{d_k}$ 是关键设计：当 $d_k$ 较大时，点积结果会很大，将 softmax 推入梯度极小的区域，缩放可有效缓解这一问题
![[Pasted image 20260505231320.png]]

llama的实现比较贴近原始论文： src/transformers/models/llama/modeling_llama.py

```python

@use_kernelized_func(apply_rotary_pos_emb)
class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
```


###  MHA
![[Pasted image 20260506110357.png]]
与其用一个注意力函数处理全维度，不如将 $Q, K, V$ 分别投影到 $h$ 个低维子空间，并行计算注意力后拼接：

$$  
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O  
$$

本文使用 $h=8$ 个头，每个头维度 $d_k = d_v = 64$。


- 下面是注意力以及多头的示意图：
![[Pasted image 20260506154204.png]]

## FFN
- 不过现在的模型，基本上已经转换为gelu的实现算法。
- 
![[Pasted image 20260506154309.png]]

gelu 示例代码
```python
class DeepseekV3MLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

```


## 论文中的模型参数配置

|参数|Base 模型|Big 模型|
|---|---|---|
|层数 $N$|6|6|
|模型维度 $d_{\text{model}}$|512|1024|
|前馈维度 $d_{ff}$|2048|4096|
|注意力头数 $h$|8|16|
|$d_k = d_v$|64|64|
|Dropout $P_{drop}$|0.1|0.3|
|标签平滑 $\epsilon_{ls}$|0.1|0.1|
|训练步数|100K|300K|
|参数量|65M|213M|

## 训练设置

- **硬件**：8 块 NVIDIA P100 GPU，单机训练
- **优化器**：Adam，$\beta_1 = 0.9$，$\beta_2 = 0.98$，$\epsilon = 10^{-9}$
- **学习率调度**：前 $warmup\_steps = 4000$ 步线性预热，之后按步数倒数平方根衰减：

$$  
lrate = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5},\ \text{step} \cdot warmup\_steps^{-1.5})  
$$

- **正则化**：残差 Dropout（每个子层输出）+ Label Smoothing $\epsilon_{ls} = 0.1$
    
    Regularization
- **推理**：Beam Search，beam size = 4，长度惩罚 $\alpha = 0.6$





## 附件
### 注意力具象化
![[Pasted image 20260506161406.png]]