---
title: "PointGroup, for 3D instance segmentation"
date: Mar 10, 2021
---
论文：PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation

instance segmentation总的来说有两种思路，一种是detection的思路，自顶向下，直接检测物体；一种是segmentation的思路，自底向上，先做语义分割，再做物体的分割。

# 一、Overview
<center>
<img src="../imgs/pointgroup.png">
</center>
三个部分：backbone，clustering，scorenet。

# 二、Backbone
输入n个points，每个包含坐标与颜色。

用U-net作为Backbone，对每个点提取pointwise的F个特征，接着分为两个branch去处理。

Semantic Segmentation Branch提取label，Offset Prediction Branch提取offset vector。更多的细节在论文与代码中。

下图为Scannet中offset vector的大小，可以发现大多数点相对于质心的距离是比较小的。
<center>
<img src="../imgs/centroiddistance.png">
</center>

# 三、Clustering
Clustering算法如下。
<center>
<img src="../imgs/clustering.png">
</center>
这个算法和那篇Multi object tracking里的clustering算法很像。

这篇文章的一个创新点就是把这个Clustering算法用在“dual point”，即同时对semantic label和offset vector使用，从两个方面去train，因为前者容易misgroup，后者对物体边缘的点效果不好。

文章后面提到做了ablation study，对dual point对有效性做了一个证明，结果表格如下。
<center>
<img src="../imgs/dualpoint.png">
</center>

# 四、ScoreNet
输入：m个cluster

输出：对每个cluster的打分

下图为ScoreNet的结构。
<center>
<img src="../imgs/scorenet.png">
</center>