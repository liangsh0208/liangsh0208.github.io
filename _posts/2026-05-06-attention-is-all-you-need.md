
---
layout: post
title: "Attention Is All You Need"
categories: 论文
---

paper 地址： https://arxiv.org/pdf/1706.03762

- 复习基础的transformer 结构和最原始的实现公式算法。
- 
![[Pasted image 20260505230935.png|435]]

## attention 算法

- 注意力的点积算法。 传统NLP 计算注意力机制，有多种计算方案，GPT 之后，基本上默认dot product算法了。
- 
![[Pasted image 20260505231320.png|679]]

- 下图是原始的点积算法。 论文中进行了改进，记性了 dk 开方
![[Pasted image 20260505231448.png|619]]

```
python

```
