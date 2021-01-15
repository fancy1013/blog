---
title: "初读PointNet++"
date: Dec 3,2020
---



## 零、摘要

**PointNet的局限性**：不能捕获 由度量空间的点引起的 局部结构。

补充：

> **度量空间**（metric space）是一种具有度量函数（metric function）或者叫做距离函数（distance function）的集合，此函数定义集合内所有元素间的距离，被称为集合上的metric。



本文提出了一种 在嵌套着的输入点集上 **递归调用PointNet** 的层次神经网络。



通过**利用 metric space distance**，该网络可以随着 contextual scale的增加而学习局部特征。



通过进一步观察发现，点集的采样密度经常是不同的，这导致了使用单一采样密度训练的网络 的性能下降。于是提出了 **novel set learning layers** ，适应性地从多种规模来结合特征。



实验表明，该网络（命名为**PointNet++**）能够高效与鲁棒地学习深度点集的特征，并且在3D点云的benchmark上的实验结果远优于之前最先进的网络。



# 一、引言

**点云特点**：

1. 由欧氏距离定义
2. 其中一点相对于其他点的顺序应当不变
3. 不同位置，采集到的点云在密度等属性上不同



PointNet先学习每个点的空间编码，再整合成点云的整体特征。因此这种设计没有抓住局部特征。

与之相对，CNN的神经元的感受野是能够随层次增加而增加的，这导致了更好的泛化能力。



**PointNet++的总体思想**：首先通过底层空间的距离度量将点集划分为重叠的局部区域。接着，与CNN的思想类似，先局部提取特征，再逐步提高层次。重复这一步骤直到完全提取了整个点云的特征。



**PointNet++的设计必须解决两个问题**：

1. 如何生成划分的点集
2. 如何通过一个局部特征学习器（local feature learner）提取出局部特征

这两个问题是相关的。因为需要设计出通用的划分方法，才能使得局部学习器的参数可以通用。（就像CNN的**卷积核**一样，我想作者是这个意思）



**解决方法**：

1. 局部特征学习器：**PointNet**
2. 点集划分：**local neighborhood ball**（局部相邻球（？），参数包括质心和规模），以及**最远点采样**（ farthest point sampling，FPS），并且**感受野根据输入数据与度量改变**。



**困难之处：**

1. 决定local neighbourhood ball的合适的规模。因为在实际数据集中，采样密度会经常改变。与CNN相反（kernel越小越准确），如果neighbourhood太小，采集到的点很少，对于提取特征就不够了。



**不过最终还是解决了这个困难。**原文如是说道：

> PointNet++ leverages neighborhoods at multiple scales to achieve both robustness and detail capture.



# 二、问题重述

假设离散度量空间：
$$
X=(M,d),M\subseteq R^n
$$
M为n维点集，d为欧式距离。

并且：M在欧氏空间的密度并不均匀。

目标：

学习函数$f$，实现分类/分割任务。



# 三、方法

层次结构由许多*set abstraction level*组成。每一级由3个关键层组成：*Sampling layer*, *Grouping layer*, *PointNet layer*（采样层，组合层，PointNet层）。

采样层从输入中选取一些能够定义局部区域质心的点。

组合层找到质心附近的区域并组合。

PointNet层使用一个小型PointNet将局部区域特征编码为特征向量。



## (1) 网络结构

### 采样层 Sampling layer

**最远点采样（FPS）：**对于n个给定点，挑选m个点做降采样。从j=1开始开始循环至j=m：距离前j-1个点的距离是最大，即为第j个点。

**为什么要最远点采样？：**与随机抽样相比，对于给定相同数量的质心，FPS可以更好地覆盖整个点集。（简单而有效的想法）

> In contrast to CNNs that scan the vector space agnostic of data distribution, our sampling strategy generates
>
> receptive fifields in a data dependent manner.



### 组合层 Grouping layer

输入：N ×（d+c）的点集合，N_prime × d 的质心坐标 

输出：N_prime × K × (d+C)，K是质心附近采样点的数量

注意，不同的质心附近可能采集的点数量不一样，即K不一样，但是Pointnet layer层可以**让K为固定的数**。

两种算法：**ball query， KNN**

**那么这两种有什么区别呢？搜索后发现如下：**

> kd-tree基于欧氏距离的特性：![[公式]](https://www.zhihu.com/equation?tex=%5CVert+x+-+y+%5CVert+%5Cge+%5CVert+x_i+-+y_i+%5CVert)
> balltree基于更一般的距离特性：![[公式]](https://www.zhihu.com/equation?tex=%5CVert+x+-+y+%5CVert+%2B+%5CVert+y+-+z+%5CVert+%5Cge+%5CVert+x+-+z+%5CVert)
>
> 因此：
> kd-tree只能用于欧氏距离，并且处理高维数据效果不佳。
> balltree在kd-tree能够处理的数据范围内要慢于kd-tree。



### PointNet layer

输入：N_prime × K × (d+C)

输出：N_prime × (d+C_prime)



## （2）非均匀采样密度下的鲁棒特征学习

为适应点云可能出现局部稀疏的问题，使用  **density adaptive PointNet layers，密度自适应PointNet层。**

PointNet++的每个级别提取多个不同规模的局部特征，并根据局部点密度智能地组合它们。

两种密度自适应方法：**MSG，MRG**

![](postimage\msr.png)

### **Multi-scale grouping (MSG)**

训练网络学习一个优化的策略来结合多尺度特征。 这是通过random dropout每个实例的随机概率输入点来完成的，我们称之为*random input dropout*。

> For each point, we randomly drop a point with probability *θ*



### **Multi-resolution grouping (MRG)**

MSG的计算量较大。

获得两个特征，一个特征（左边）：从较低级别总结每个次区域的特征，另一个特征（右边）：使用single PointNet直接处理本地区域中的所有原始点。

密度较大时，第一个特征更可靠。密度较小时，第二个特征更可靠。



总的来说，MRG更好，因为计算量较少。



## （3） **Point Feature Propagation for Set Segmentation 用于点分割的点特征传播**

在做分割时，需要获得所有点的特征。一种解决方案是在所有级别中始终将所有点采样为质心，但这会导致较高的计算成本。 另一种方法是将特征从子采样点传播到原始点。

通过插值法实现。

在许多插值选择中，我们使用基于k个最近邻的逆距离加权平均值(如方程。 默认情况下，我们使用p=2，k=3。

公式如下：

<center>
<img src="../imgs/formula1.png" alt="formula1" style="height: 300px;margin-left:20pt;"/>
</center>

segmentation的示意图：

<center>
<img src="../imgs/pointnet++segmentation.png" alt="segmentation" style="height: 300px;margin-left:20pt;"/>
</center>