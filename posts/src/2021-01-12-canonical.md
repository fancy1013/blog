---
title: "Canonical Voting：更好的投票机制"
date: Jan 12,2021
---

### 来自mentor的论文：**Canonical Voting: Towards Robust Oriented Bounding Box Detection in 3D Scenes**



# 一、介绍

首先，估计Local Canonical Coordinates和box scales

然后，设计了canonical voting algorithm找到object orientations 和centers，并且 进行**LCC checking with back projection**.



那么什么是Local Canonical Coordinates呢？从文中看来我不是很清楚，因此做一些查询。

## Canonical transformation

In [Hamiltonian mechanics](https://en.wikipedia.org/wiki/Hamiltonian_mechanics), a **canonical transformation** is a change of [canonical coordinates](https://en.wikipedia.org/wiki/Canonical_coordinates) (**q**, **p**, *t*) → (**Q**, **P**, *t*) that preserves the form of [Hamilton's equations](https://en.wikipedia.org/wiki/Hamilton's_equations). 

## Canonical Coordinates

In [mathematics](https://en.wikipedia.org/wiki/Mathematics) and [classical mechanics](https://en.wikipedia.org/wiki/Classical_mechanics), **canonical coordinates** are sets of [coordinates](https://en.wikipedia.org/wiki/Coordinates) on [phase space](https://en.wikipedia.org/wiki/Phase_space) which can be used to describe a physical system at any given point in time. Canonical coordinates are used in the [Hamiltonian formulation](https://en.wikipedia.org/wiki/Hamiltonian_mechanics) of [classical mechanics](https://en.wikipedia.org/wiki/Classical_mechanics).



> 疑问1：还是没有看懂。所以Local Canonical Coordinates是什么？

# 二、方法

## （1）Pipeline

首先，regress Local Canonical Coordinates, scales and objectness for each scene point

然后，提出canonical voting algorithm,生成投票

最后，用LCC back projection checking module去除false positive，生成bouding boxes。



<center>
<img src="../imgs/canonical_pipeline.png" alt="canonical_pipeline" style="height: 300px;margin-left:20pt;"/>
</center>



## （2） **Learning to Regress Local Canonical Coordinates**

>  疑问2：为什么选取y轴做 gravity axis?

旋转角只考虑heading angle around the gravity axis。



<center>
<img src="../imgs/formula3.png" alt="formula3" style="height: 300px;margin-left:20pt;"/>
</center>

s是bouding box scale，P_hat 是LCC坐标，P是世界坐标，t是bounding box centers。

**公式（1）可以进行LCC坐标与世界坐标的转化。**



然后，用**Minkowski convolution with ResNet structure**来学习s和p_hat。

损失函数：

<center>
<img src="../imgs/formula4.png" alt="formula4" style="height: 300px;margin-left:20pt;"/>
</center>



> 疑问3：损失函数里为什么只对scale进行一个单独的差值运算？

理解公式（2）：用来拟合LCC坐标和bounding box scale，而不拟合旋转角α和object center。因为LCC坐标具有旋转不变性，但direct offset没有。

> 疑问4：这里的direct offset是什么？



### **Why Local Canonical Coordinates?**

<center>
<img src="../imgs/canonical_p4.png" alt="canonical_p4" style="height: 300px;margin-left:20pt;"/>
</center>

(这个小鸭子好可爱)

LCC可以随旋转而不改变学习到的特征。



### Object Symmetry

由于只有一个旋转角，对于一些高度对称性的物体，LCC表示会导致很大的error。

因此，对于对称的形状，进行旋转然后学习。

对于二重对称（例如矩形），旋转0度或180度。

对于四重对称（例如正方形），旋转0度，90度，180度，270度。

对于无限重对称（例如圆形），旋转36个不同的角度。

计算每一个角度的loss然后取最小的loss。



## （3）**Canonical Voting with Objectness**

提出 canonical voting algorithm，产生vote map，表明存在物体的概率。

每个点有个额外的**objectness score**，o∈[0,1]，1表示有实例物体，0表示没有。使用Cross Entropy loss。



在有了s，p_hat，o之后，可以进行vote，投票出bounding box center。

为了计算投票数，将连续的欧式空间离散化为**H×D×W**的格子。H，D，W取决于事先预定好的间隔tao和输入的点云规模。

G_obj：**H×D×W**

G_rot：**H×D×W**

G_scale：**H×D×W×3**

> 疑问5：这里的H，D，W是什么意思？

具体算法如下图。

<center>
<img src="../imgs/alg1.png" alt="alg1" style="height: 300px;margin-left:20pt;"/>
</center>


算法1说明：

第4行，生成可能的offset，未考虑旋转。

第6行，旋转K次，找到可能的orientation。（实验中k=120）

第7行，根据公式（1）计算center。

第8行，找到center附近的8个格子（2×2×2），记为N

第9、10、11行，投票计数，用trilinear interpolation

第14行，投票归一化。



## （4） **LCC Checking with Back Projection for Bounding Box Generation**

最后一步了。

找到G_obj中最大的值然后生成bounding boxes。其中，false positive被LCC checking with back projection给过滤掉。

LCC checking with back projection的算法流程：

<center>
<img src="../imgs/alg2.png" alt="alg2" style="height: 300px;margin-left:20pt;"/>
</center>

> 疑问6：11行的cnt代表什么？可能是count，用来数被检测到的点的数量。

第25行，检测是否有足够多的点被检测到，以及误差足够小，然后才能确认为bounding box。

（算法2还没有理解清楚，看完一遍再回来看看。）

## （5）推广到多分类问题

给每个点增加一个class score属性，用cross entropy loss。

然后和算法2类似。



