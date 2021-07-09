# Preparation for MMLab

Yanjie Ze, 8 July 2021

明天就要面试MMLab了，可能是此生以来面试到的最强的大牛（之前和John Hopcroft合影不算）。今天准备思路是：再看看自己简历上写的啥，深入思考一下做的东西。

说来搞笑，这个简历是我很久以前写的了，这之后又有许多新东西也没加上去，就已经经历了很多次面试。先不修改了，等到大二的这个暑假结束再总结一下。



## 1 MVIG做的什么

### 1.复现了PointPillars算法。

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708171406973.png" alt="image-20210708171406973" style="zoom:50%;" />

其中，SSD算法是一个single-stage的算法。（SSD不太熟悉，建议略过）



### 2.Multi-virtual view fusion

主要是把3D mesh给render成2D 图像，再反project回2D。<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708173403825.png" alt="image-20210708173403825" style="zoom:50%;" />



## 2 RL做了什么

### 1.Image Caption Generator

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708175336249.png" alt="image-20210708175336249" style="zoom:50%;" />

inference时候有两种方法：Sampling和BeamSearch

BeamSearch如下图所示：

<img src="/Users/yanjieze/Downloads/v2-a760198d6b851fc38c8d21830d1f27c9_r.png" alt="v2-a760198d6b851fc38c8d21830d1f27c9_r" style="zoom:90%;" />

### 2. QMIX算法在Visual Dialog的实现

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708193616158.png" alt="image-20210708193616158" style="zoom:50%;" />

