---
title: "概率论&线性代数"
date: Apr 11,2021
---
对于概率论和线性代数的复习。

# 一、贝叶斯公式
$$
P(\Theta|X) = \frac{P(X|\Theta)P(\Theta)}{P(X)}
$$

解释：

$P(\Theta)$,先验分布。

$P(X|\Theta)$,在确定了参数的情况下，试验数据的概率分布。实际上这就是对实际观测数据的一种描述。

$P(\Theta|X)$，后验分布。

$P(X)$，边缘概率。

# 二、MLE和MAP

知识点复习链接：[聊一聊机器学习的MLE和MAP：最大似然估计和最大后验估计 - 夏飞的文章 - 知乎](https://zhuanlan.zhihu.com/p/32480810)

### 例题：一枚硬币，扔了一亿次都是正面朝上，再扔一次反面朝上的概率是多少？

[解答链接](https://blog.csdn.net/qq_29884019/article/details/100008617)

# 三、矩阵求逆：LU分解

知识点复习链接：[矩阵分解—1-LU分解 - rocketman的文章 - 知乎](https://zhuanlan.zhihu.com/p/54943042)

<center>
<img src="../imgs/LU_decomposition.jpeg">
</center>


<center>
<img src="../imgs/LU_decomposition2.png">
</center>

<center>
<img src="../imgs/LU_decomposition3.jpeg">
</center>

# 四、正交矩阵与正定矩阵

正交矩阵：
$$
QQ^T = I
$$

<center>
<img src="../imgs/zheng_matrix.png">
</center>

正定矩阵：
$$
当且仅当对于所有的非零实系数向量z, z^TMz>0
$$

# 五、线性方程组有解/无解/唯一解的条件
增广矩阵：增广矩阵，又称广置矩阵，是在线性代数中系数矩阵的右边添上线性方程组等号右边的常数列得到的矩阵。
1. 无解：系数矩阵秩 < 增广矩阵的秩
2. 唯一解：系数矩阵秩 = 增广矩阵的秩 = 列数 n
3. 无穷解：系数矩阵秩 = 增广矩阵的秩 < 列数 n

# 六、总结
太多不会了，这周又要面试字节的AIlab又要面试MSRA，估计是没了。