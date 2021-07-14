#! https://zhuanlan.zhihu.com/p/389653208

# 一文带你搞懂相机内参外参
Yanjie Ze, July 14 2021

## 1 相机内参

<img src="/Users/yanjieze/Downloads/IMG_1786(20210713-113445).PNG" style="zoom:50%;" />

在左图中，我们把相机看作是针孔，现实世界中的点P经过相机的光心O，投影到物理成像平面上，变为点P'。

在右图中，对这个模型进行了一个简化，将其看作是一个相似三角形。

下面我们来对这个模型进行建模。



设$O-x-y-z$为相机坐标系，习惯上我们把z轴指向相机前方，x向右，y向下。O为摄像机的**光心**，也是针孔模型中的针孔。



设真实世界点中的P的坐标为$[X,Y,Z]^T$，成像的点P'的坐标为$[X',Y',Z']^T$， 物理成像平面和光心的距离为$f$（即为焦距）。



根据右图中的三角形相似关系，有：
$$
\frac Zf = -\frac{X}{X'} = -\frac{Y}{Y'}
$$
其中，有负号是因为坐标轴方向，也就表示了成的像是倒立的。

为了表示起来更方便，我们把成像平面从相机的后面对称到前面去，如下图所示。这样，负号就没有了。

<img src="/Users/yanjieze/Downloads/IMG_1788.jpg" style="zoom:50%;" />



在对称后，有：
$$
\frac Zf = \frac{X}{X'} = \frac{Y}{Y'}
$$
整理解出$P'$的坐标：
$$
X'=f \frac XZ
$$

$$
Y' =f \frac YZ
$$

上面两个式子就描述了P点与它所成像的坐标关系，可以看到，$X$对应的$X'$与焦距$f$有关，与距离Z有关。



映射到成像平面上还不够，我们还需要将这个像给放到像素坐标系内。

我们设在物理成像平面上固定着像素平面$o-u-v$。

设$P'$在像素平面坐标系上的坐标是$[u,v]^T$。

**像素坐标系**通常定义方式是：原点o'位于图像的左上角，u轴向右与x轴平行，v轴向下与y轴平行。

我们设像素坐标在u轴上缩放$\alpha$倍，在v轴上缩放了$\beta$倍。同时，原点平移了$[c_x, c_y]^T$。

因此可以得到P'与像素坐标的关系：
$$
u=\alpha X' + c_x
$$

$$
v=\beta Y' +c_y
$$

代入P与P'的关系式可得：
$$
u=\alpha f \frac XZ + c_x = f_x \frac XZ + c_x
$$

$$
v=\beta f \frac YZ + c_y = f_y \frac YZ + c_y
$$

其中，我们用$f_x, f_y$替换了 $\alpha f$ 和 $\beta f$。$f_x,f_y$的单位是像素。

用齐次坐标，把上式写出矩阵的形式：
$$
\begin{pmatrix} u\\v\\1\end{pmatrix} = \frac 1Z \begin{pmatrix} f_x & 0&c_x\\ 0&f_y &c_y\\ 0& 0&1\end{pmatrix}\begin{pmatrix}X\\Y\\Z \end{pmatrix} = \frac 1Z \bold{KP}
$$
也可以把Z写到等式左边去，就变成了：
$$
Z\begin{pmatrix} u\\v\\1\end{pmatrix} =  \begin{pmatrix} f_x & 0&c_x\\ 0&f_y &c_y\\ 0& 0&1\end{pmatrix}\begin{pmatrix}X\\Y\\Z \end{pmatrix} =  \bold{KP}
$$
上式中，$\bold{K}$即为相机的内参矩阵(Intrinsics)。通常来说，相机的内参在出厂之后就是固定的了。



## 2 相机外参

在上面的推导中，我们用的是P在相机坐标系的坐标（也就是以相机为O点），所以我们应该先将世界坐标系中的$P_w$给变换到相机坐标系中的$P$。

相机的位姿由旋转矩阵$\bold{R}$和平移向量$\bold{t}$来描述，因此：
$$
\bold{P=RP_W+t}
$$
再代入之前的内参的式子，得到：
$$
Z\bold{P_{uv}} = \bold{K(RP_w+t)}=\bold{KTR_w}
$$
后面一个等号蕴含了一个齐次坐标到非齐次坐标的转换。

其中，$\bold{R,t}$为相机的外参(Extrinsics)。



## 3 总结

本文介绍了：

1. 从相机坐标系转换到像素坐标系中，相机内参的作用
2. 从世界坐标系转换到相机坐标系中，相机外参的作用



相机内参是这样的一个矩阵：
$$
\begin{pmatrix} f_x & 0&c_x\\ 0&f_y &c_y\\ 0& 0&1\end{pmatrix}
$$
里面的参数一般都是相机出厂就定下来的，可以通过相机标定的方式人为计算出来。



相机外参是旋转矩阵$R$和平移向量$t$构成,一般来说写成：
$$
\begin{pmatrix} \bold{R}& \bold{t}\\ 0 &1\end{pmatrix}
$$
这个矩阵决定了相机的位姿。