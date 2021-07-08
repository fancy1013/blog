# Unsupervised Keypoint GANeration

Yanjie Ze, 2021

UKPGAN，来自论文**UKPGAN: Unsupervised KeyPoint GANeration**，主要介绍了一种可以通过无监督学习的方式获得keypoint的GAN的变体。这篇论文有三个相关的project：KeyPointNet, TopNet, 3DSmoothNet。

我将首先阅读UKPGAN这篇论文，并仔细剖析代码。如果遇到一些阻碍，就再阅读这三个相关的project。

## 1 Model Overview

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210628230234744.png" alt="image-20210628230234744" style="zoom:50%;" />

## 2 Rotational Invariant Feature Extraction





## ? Code

2021.6.28 跑了一下，跑通了一个data point的情况。

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210628232126520.png" alt="image-20210628232126520" style="zoom:50%;" />

2021.6.30 先把keypointnet的数据下载好了

随便改了一下dataflow.py，在跑了在跑了。

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210630220554648.png" alt="image-20210630220554648" style="zoom:50%;" />

2021.7.2 继续跑，又看懂了点代码

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210703124055484.png" alt="image-20210703124055484" style="zoom:50%;" />

跑了一下，reconstruction的效果不太好啊。

# KeypointNet

Keypoint aggregation pipeline

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210630205818382.png" alt="image-20210630205818382" style="zoom:50%;" />

