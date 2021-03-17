---
title: "Intrinsic Relationship Reasoning"
date: Mar 17, 2021
---
论文：Intrinsic Relationship Reasoning for Small Object Detection

# 一、相关方法介绍
核心问题：
1. 时间信息衰减
2. 捕捉语义特征

有一种方法：先检测，然后用super resolution network提高分辨率，再进行分类。但是缺点是计算量大。

idea是：人类在判断小物体的时候有时候是根据它所在的群体，比如一群大雁。因此找到物体间的内部语义关系说不定可以提高小目标检测。

这篇文章基于Graph Convolutional Network提出了一种context reasoning的方法。包含3个模块：
1. semantic module
2. spatial layout module
3. context reasoning module

下图展示了SR网络和本文的网络的结构。
<center>
<img src="../imgs/reasoning.png">
</center>

# 二、具体结构

<center>
<img src="../imgs/reasoning2.png">
</center>