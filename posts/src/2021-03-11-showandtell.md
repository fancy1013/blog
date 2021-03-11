---
title: "Show and Tell: Image Caption"
date: Mar 11, 2021
---
论文：Show and Tell: A Neural Image Caption Generator

这篇文章的idea来源于nlp的机器翻译。机器翻译中的encoder-decoder结构是由两个RNN构成的，这篇文章把encoder的RNN换成了CNN。

这个想法感觉很简单啊，看来这篇CVPR2015是开山之作？

# 一、优化函数
$$
\theta^*=\argmax_\theta\sum_{(I,S)}logp(S|I;\theta)
$$
其中theta是模型参数，I是输入图像，S是正确的描述。

$$
logp(S|I)=\sum_{t=0}^NlogP(S_t|I,S_0,...,S_{t-1})
$$
使用联合分布计算概率，其中N是句子的长度。

从上面这个需要优化的公式来看，用RNN是很自然的，可以用hidden state来表示前t-1个单词，即：
$$
h_{t+1}=f(h_t,x_t)
$$
其中，f是LSTM。

# 二、LSTM
LSTM结构如下图。
<center>
<img src="../imgs/lstm.png">
</center>

LSTM由输入门i，输出门o，遗忘门f构成。
比如输入门：
$$
i_t =\sigma (W_{ix}x_t+W_{im}m_{t-1})
$$
遗忘门和输出门类似。$\sigma(\cdot)$是sigmoid函数。

$$
c_t = f_t\odot c_{t-1}+i_t\odot h(W_{cx}x_t+W_{cm}m_{t-1})
$$
$$
m_t = o_t\odot c_t
$$
$$
p_{t+1}= Softmax(m_t)
$$
c为cell，m为memory，p为预测的概率。


然后是总体的流程：
$$
x_{-1}=CNN(I)
$$
$$
x_t=W_eS_t, t\in {0,...,N-1}
$$
$$
p_{t+1}=LSTM(x_t)
$$
<center>
<img src="../imgs/lstm2.png">
</center>

每个word是一个one-hot vector， 维度和dictionary的大小一样。

注意：图片只在开始的时候输入了一次。如果每个state都输入一下反而效果不好。

Loss:
$$
L(I,S)=-\sum_{t=1}^Nlogp_t(S_t)
$$