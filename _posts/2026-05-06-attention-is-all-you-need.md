---
layout: post
title: "Attention Is All You Need"
date: 2026-05-06 00:00:00 +0800
categories: [论文]
---

paper 地址： https://arxiv.org/pdf/1706.03762

- 复习基础的 transformer 结构和最原始的实现公式算法。

![attention overview](/assets/images/Pasted%20image%2020260505230935.png)

## attention 算法

- 注意力的点积算法。传统 NLP 计算注意力机制有多种方案，GPT 之后基本默认 dot product 算法。

![dot product](/assets/images/Pasted%20image%2020260505231320.png)

- 下图是原始的点积算法。论文中进行了改进，加入了 $d_k$ 开方。

![scaled dot product](/assets/images/Pasted%20image%2020260505231448.png)

```python
# 示例代码占位
```
