---
title: "Paper For MSRA Topic"
date: Apr 12, 2021
---



# 一、基础知识准备



## (1) *Bezier curve*

### 讲解链接

[「贝塞尔曲线」有哪些作用和特点，该如何正确使用？ - DBinary的回答 - 知乎 ](https://www.zhihu.com/question/419155303/answer/1454607426)

[Bezier算法详解-blog](https://www.vectormoon.net/2020/09/25/Bezier/)

### 曲线定义

![](/Users/yanjieze/Library/Application Support/typora-user-images/image-20210412222730363.png)

### Bezier生成曲线算法

![](/Users/yanjieze/Library/Application Support/typora-user-images/image-20210412223042588.png)



### 代码

```python
def draw_curve(p_list):
	"""
	:param p_list: (list of list of int:[[x0, y0], [x1, y1], ...])point set of p
	result: (list of list of int:[[x0, y0], [x1, y1], ...])point on curve
	"""
	result = []
	P = []
	P = p_list.copy()
	r = len(p_list)
	for i in range(0, 20923): #2020/09/23
		t = i/20923
		x, y = de_Casteljau(r, P, t)
		result.append((x, y))
	return result
	
def de_Casteljau(n, pointSet_p, t):
	"""
	:param n: number of control
	:param pointSet_p: (list of list of int:[[x0, y0], [x1, y1], ...])point set of p
	:param t: t
	"""
	while(n):
		for i in range(0, n-1):
			P[i][0] = (1-t)*P[i][0] + t*P[i+1][0]
			P[i][1] = (1-t)*P[i][1] + t*P[i+1][1]
		n -= 1
	P[0][0] = int(P[0][0] + 0.5)
	P[0][1] = int(P[0][1] + 0.5)
	return P[0]
```



## (2) *B-spline*（B样条曲线）



## (3) Inverse Graphics

In the visual domain, inversion of a renderer for the purposes of scene understanding is typically referred to as inverse graphics.



# 二、CVPR2021:*Cloud2Curve: Generation and Vectorization of Parametric Sketches*



## (1) Related Work

这个领域接触得很少，大概扫一眼related work。

**Generative Models**: GAN, VAE, SketchRNN, 还有一篇是Reinforced Adversarial Learning,可以细看一下。



**Parametric representation:** Bezier, B- Splines, Hermite Splines，三种曲线。还有*ICML*, 2018的那篇用的RL。



**Learning parametric curves**: 一般生成Bezier曲线/B样条曲线的方法都是代入t计算。比较新的方法：：BezierEncoder用深度循环模型做的（ECCV 2020）。



# 三、ICLR2018: *A Neural Representation of Sketch Drawings*

这篇文章应该是 **ai绘画** 的第一篇。

## （1）点的形式

$$
(∆x, ∆y, p_1, p_2, p_3).
$$

> The first two elements are the offset distance in the x and y directions of the pen from the previous point. The last 3 elements represents a binary one-hot vector of 3 possible states. The fifirst pen state, *p*1, indicates that the pen is currently touching the paper, and that a line will be drawn connecting the next point with the current point. The second pen state, *p*2, indicates that the pen will be lifted from the paper after the current point, and that no line will be drawn next. The fifinal pen state, *p*3, indicates that the drawing has ended, and subsequent points, including the current point, will not be rendered.

## (2) 模型

 基本模型是Sequence-to-Sequence Variational Autoencoder (VAE)，但是是一个**双向RNN**，即如下图左边所示，encoder先把input正着输入一遍，再倒着输入一遍。



![](/Users/yanjieze/Library/Application Support/typora-user-images/image-20210413221003563.png)

这是latent vector的表达式。

![image-20210413223346437](/Users/yanjieze/Library/Application Support/typora-user-images/image-20210413223346437.png)



## （3）训练

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210414002308030.png" style="zoom:67%;" />



## （4）实验

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210414002513628.png" style="zoom:67%;" />



# 四、ICML2018: *Synthesizing Programs for Images using Reinforced Adversarial Learning*

这是他们的[视频](https://www.youtube.com/watch?v=iSyvwAwa7vk)。

this is the first demonstration of an end-to-end, unsupervised and adversarial inverse graphics agent on challenging real world

下图是SPIRAL结构。

![image-20210414005203819](/Users/yanjieze/Library/Application Support/typora-user-images/image-20210414005203819.png)



# 五、ECCV2020: *B´ezierSketch: A generative model for scalable vector sketches*

