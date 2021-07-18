# Implementation: Virtual Multi-view Fusion for 3D Semantic Segmentation

Yanjie Ze, July 2021

Website: http://yanjieze.xyz

# 1 Method Overview

如下图所示，主要分为training和inference。

- 在training的时候，选择view和相机的intrinsic和extrinsic，进行render，获得2D data和ground truth。

- 在inference的时候，进行2D的semantic segmentation。

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210706122231647.png" alt="image-20210706122231647" style="zoom:50%;" />

# 2 Virtual View Selection

相机内参：用更大的FOV。

相机外参：如图2和图4所示，用了好几种增强的方法：

- 位置坐标用uniform sampling，视角是top-down的。
- scale-invariant sampling。

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210706131828955.png" alt="image-20210706131828955" style="zoom:50%;" />

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210706131851160.png" alt="image-20210706131851160" style="zoom:50%;" />

# 3 Multi-view Fusion

## 3.1 2D semantic segmentation model

Feature extractor: **xception65**

Decoder: **DeepLabV3+**

Pretrain: **Classification Model on ImageNet**



## 3.2 3D fusion of 2D semantic features

将点云project到2D图像上，depth相同的点才对应。（depth check）

注意，这比从2D图像进行ray casting要快。

具体过程：

**首先**，根据下面这个公式，用相机内参、外参，将三维的点投影二维上，获得坐标。

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210706220040486.png" alt="image-20210706220040486" style="zoom:50%;" />

以及相机与这个三维点的距离，如下公式。

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210706220237628.png" alt="image-20210706220237628" style="zoom:50%;" />

**然后**，三维点从每个view采集获得feature vector，如下公式。

其中的含义是depth check，

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210706220447689.png" alt="image-20210706220447689" style="zoom:50%;" />

获得feature后，做一个**平均值**（而不是直接取最大值）。因为这个效果比较好。



# 4 实现思路

## 4.1 第一步，写dataloader和renderer

第一个问题，怎么从一个3D mesh来render一个图像？

先参考这篇medium上的文章试一试: [How to render a 3D mesh and convert it to a 2D image using PyTorch3D](https://towardsdatascience.com/how-to-render-3d-files-using-pytorch3d-ef9de72483f8)

后来开始参照pytorch3d的官方文档，尝试写一个renderer。

renderer的具体结构如下:

<img src="/Users/yanjieze/Downloads/architecture_renderer.jpeg" alt="architecture_renderer" style="zoom:69%;" />

后来改用pyrender写了。

可以渲染。目前存在的问题：

1. 渲染的画面不清楚。
2. 渲染的相机位置不知道怎么找。
3. 只有RGB和dpeth的图，其他的（normalized global coordinate image， normal）不知道怎么如何提取出来
4. backface culling？
