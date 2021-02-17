---
title: "Attention Is All You Need"
date: Feb 10, 2021
---

今天就来读一读大名鼎鼎的Attention，一个在图像/文本都能用的机制。

下图为Transformer的结构。
<center>
<img src="../imgs/transformer.png">
</center>
在attention中，query, key,value经常被提到，这里有一个解释：[What exactly are keys, queries, and values in attention mechanisms?](https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms)

# Encoder and Decoder Stacks

**Encoder**: 6个不同的层，每层有两个子层。第一层，multi-head self-attention mechanism。第二层， 简单的positionwise fully connected feed-forward network。每两个子层间有residual connection，然后用layer normalization。即，每个子层的输出是：
$$
LayerNomrm(x+Sublayer(x))
$$
为了方便residual connection，输出维度都为:
$$
d_{model}=512
$$


**Decoder**: 6个不同的层。除了Encoder里有的两个子层，还加了第三个子层，对于encoder层的输出使用multi-head attention。也使用了residual connection 和 layer normalization。此外，对self-attention layer做了一些修改，使得其输出只与前面的位置有关。

# Attention

<center>
<img src="../imgs/attention.png">
</center>

**Scaled Dot-Product Attention**:
$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
其中，Q是query的矩阵，K是key的矩阵，V是value的矩阵，d_k是key的维度。

**除以d_k的原因**：怀疑在d_k很大的时候，函数会进入softmax梯度很小的区域，所以除以d_k。



**Multi-Head Attention**:
$$
MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O\\
where\ head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)
$$


**Appalications of Attention in this model**:

1. 在