---
title: "PointPillars:点云目标检测"
date: Jan 30, 2021
---
为了将这篇paper复现出来，需要对其流程以及参数都更加熟悉。

# 一、算法流程

分为三个部分：Pillar Feature Net，Backbone（2D CNN），Detection Head（SSD）。

## 第一部分：Pillar Feature Net

**这一部分将点云转化为pseudo-image。**

设点云中的一个点为l=(x,y,z,r)。

第一步将点云离散化为成xy平面上的许多网格，形成一系列pillars，每个面积为B。

然后将每个pillar的点数据增强，即**将l=(x,y,z,r)加上(x_c,y_c,z_c,x_p,y_p)这五个维度，变为：l=(x,y,z,r,x_c,y_c,z_c,x_p,y_p),维度D=9**。其中c下标表示与pillar中的所有点的距离的算术平均，p下标表示pillar的中心坐标偏移量。


由于点云是稀疏的，为了获得较好的tensor（D，P，N），需要做一些处理。（P表示Sample的数量，N表示Pillar的数量。）**对于比较密的，进行随机采样。对比很稀疏的，用0填充。**

然后，使用一个简化版的PointNet。*一个线性层+一个Batch Norm+一个ReLU，生成（C，P，N）大小的tensor。然后用一个max操作，获得（C，P）大小的tensor。注意，线性层可以用1✖️1的卷积核，效果好。*

编码后，这些特征分散到原来的pillar的位置，创建一个（C，H，W）的伪图像。H和W是画布的高度和宽度。

**第一部分总结**：点云离散为pillar -> 每个点数据增强 -> 对于疏密进行不同处理 -> Pointnet解码 ->生成伪图像。

## 第二部分：Backbone

<center>
<img src="../imgs/pointpillars1.jpg">
</center>
结构如图。

有两个子网络：一个自上而下的网络，以越来越小的空间分辨率产生特征，另一个网络执行自上而下特征的上采样和拼接。

自上而下的网络用一系列Block（S，L，F）描述。每个Block以步长S移动。每个Block有L个3✖️3的2维卷积层，输出F个通道，并接着BatchNorm 和 ReLU。层中的第一个卷积的步长为S/S_in。剩下的步长为1。（？）

从每个自上而下的网络的最终特征由上采样和拼接聚合。首先，上采样为（S_in，S_out，F），用一个2维卷积核。接着，BatchNorm和ReLU。最后的输出是来自不同步长的拼接。

**第二部分总结**：几个不同步长的卷积核进行卷积 -> 上采样 -> 拼接。

## 第三部分：Detection Head

使用SSD。

# 二、实现细节
实现细节分为两部分，Network和Loss。

## 第一部分：Network
所有权重以正态分布随机初始化。

encoder network的输出特征维度C=64.

车、行人、骑车人的第一个Block不同。（S=2 for car， S=1 for pedestrian/cyclist）。

2个网络都有3个Block。Block1(S, 4, C), Block2(2S, 6, 2C), and Block3(4S, 6, 4C)。每个Block上采样： Up1(S, S, 2C), Up2(2S, S, 2C) and Up3(4S, S, 2C)。然后合成6C个特征。

## 第二部分：Loss
ground truth boxes 和 anchors定义：（x,y,z,w,l,h,theta）
第一个loss：localization loss

<center>
<img src="../imgs/pointpillars2.jpg">
</center>

第二个loss：softmax classification loss on the
discretized directions（L_dir）

第三个loss：object classification loss使用了focal loss。
<center>
<img src="../imgs/pointpillars3.jpg">
</center>

上面这个式子为最终的loss。其中N_pos为positive anchors的数量，beta_loc =2 , beta_cls = 1, beta_dir = 0.2。

优化器：Adam。初始learning rate=2*10^-4，decay = 0.8 every 15 epochs。

一共训练160 epochs。

batch size = 2 for validation set， 4 for test submission。

# 三、实验设置
分为三个部分：数据集，参数设置，数据增强。

## 第一部分：数据集
kitti数据集，既包含雷达点云也包含图像。

原始的数据集是7481个training， 7518个testing。

实验中，将training分为3712个trianing，3769个validation。在test submission中，784个来自validation， 在剩下的6733 samples训练。

训练三个网络分别识别车，行人，骑车人。

kitti数据集的label说明：
<center>
<img src="../imgs/kitti.png">
</center>


## 第二部分：设置
xy resolution: 0.16

max number of pillars(P): 12000

max number of points per pillar(N):100

每一个class anchor用width，length，height，z center来描述，两个朝向：0度，90度。

Anchors are matched to ground truth using the 2D
IoU with the following rules. A positive match is either the highest with a ground truth box, or above the positive match threshold, while a negative match is below the negative threshold. All other anchors are ignored in the loss.

At inference time we apply axis aligned non maximum
suppression (NMS) with an overlap threshold of 0.5 IoU.This provides similar performance compared to rotational NMS, but is much faster.

### Car：
The x, y, z range is [(0, 70.4), (-40, 40), (-3, 1)]
meters respectively. The car anchor has width, length, and height of (1.6, 3.9, 1.5) m with a z center of -1 m. Matching uses positive and negative thresholds of 0.6 and 0.45.

### Pedestrian & Cyclist：
The x, y, z range of [(0, 48), (-20,20), (-2.5, 0.5)] meters respectively. The pedestrian anchor has width, length, and height of (0.6, 0.8, 1.73) meters with a z center of -0.6 meters, while the cyclist anchor has width, length, and height of (0.6, 1.76, 1.73) meters with a z center of -0.6 meters. Matching uses positive and negative thresholds of 0.5 and 0.35.

## 第三部分：Data Augmentation

明天再说。