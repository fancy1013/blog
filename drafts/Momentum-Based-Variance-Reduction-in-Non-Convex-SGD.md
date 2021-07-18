# Momentum-Based Variance Reduction in Non-Convex SGD

Yanjie Ze, June 2021

## 0 Introduction







## 1 Motivation

two potential issues of **SVRG**:

1. Non-adaptive learning rates
2. Reliance on giant batch sizes to construct variance reduced gradients throughout the use of low-noise gradients calculated at a "checkpoint"

In this paper, we address both of these issues.

Present a new algorithm called **STOchastic Recursive Momemtum**.



**Affect**: Achieve variance reduction through the use of a variant of the momentum term.



SAG:

![preview](https://pic3.zhimg.com/a8c0d0bf11ce31035dbe59dce32ce446_r.jpg)

SVRG:

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210709203323412.png" alt="image-20210709203323412" style="zoom:50%;" />

我的个人理解：

SVRG的缺陷主要在于两点。第一，giant batch。第二，learning rate是固定的。



## 2 Setting

We can access a stream of independent random variables:
$$
\xi_1,...,\xi_T \in \Xi
$$


A sample function $f$ that satisfies:
$$
\forall t,\bold{x},\ \mathbb{E}[f(\bold{x},\xi_t)|\bold{x}] = F(\bold{x})
$$
Where $F(x)$ is the oracle function we can not access directly.



The noise of the gradients is bounded by $\sigma^2$:
$$
\mathbb{E}[||\nabla f(\bold{x},\xi_t)-\nabla F(\bold{x})||^2]\leq \sigma^2
$$


Define:
$$
F^\star = \inf_x F(\bold{x})
$$

$$
F^\star > -\infty
$$

Assume our function $f$ is L-smooth and G-Lipschitz:
$$
\forall x, ||\nabla f(\bold{x})||\leq G
$$

$$
\forall x\ and\ y, ||\nabla f(\bold{x})-\nabla f(\bold{y})||\leq L||\bold{x}-\bold{y}||
$$



## 3 Notation

Gradient direction:
$$
\bold{d_t}=(1-a)\bold{d_{t-1}} +a\nabla f(\bold{x_t},\xi_t)+(1-a)(\nabla f(\bold{x_t},\xi_t)-\nabla f(\bold{x_{t-1}, \xi_t}))
$$
Update formula:
$$
\bold{x_{t+1}}=\bold{x_t}-\eta \bold{d_t}
$$


Error term:
$$
\bold{\epsilon_t}=\bold{d_t} -\nabla F(\bold{x_t})
$$


Variables in Theorem 1:
$$
k=\frac{bG^{\frac 23}}{L}
$$

$$
c=28L^2+G^2/(7Lk^3)=L^2(28+1/(7b^3))
$$

$$
w=max((4Lk)^3, 2G^2,(\frac{ck}{4L})^3)=G^2max((4b)^3,2,(28b+\frac 1 {7b^2})^3/64)
$$

$$
M=\frac 8k (F(\bold{x_1})-F^\star)+\frac{w^{1/3}\sigma^2}{4L^2k^2}+\frac{k^2 c^2}{2L^2}\ln(T+2)
$$

Variables in Algorithm STORM:
$$
\eta_t \leftarrow \frac{k}{(w+\sum_{i=1}^tG_t^2)^\frac 13}
$$

$$
a_{t+1}\leftarrow c\eta_t^2
$$

$$
G_{t+1} \leftarrow ||\nabla f(\bold{x_{t+1}, \eta_{t+1}})||
$$

$$
\bold{d_{t+1}}\leftarrow \nabla f(\bold{x_{t+1}},\xi_{t+1})+(1-a_{t+1})(\bold d_t - \nabla f(\bold x_t, \xi_{t+1}))
$$



## 4 Background: Momentum and Variance Reduction

$$
\bold{d_t}=(1-a)\bold{d_{t-1}}+a\nabla f(\bold{x_t},\xi_t)
$$

$$
\bold{x_{t+1}}=\bold{x_{t}}-\eta \bold{d_t}
$$

Where $a$ is small, i.e. $a=0.1$

**However, it's still unclear if the actual convergence rate can be improved by the momentum.**



Hence, instead of showing that momentum in SGD works in the same way as in the noiseless case, we show that **a variant of momentum can provably reduce the variance of the gradients**.
$$
\bold{d_t}=(1-a)\bold{d_{t-1}} +a\nabla f(\bold{x_t},\xi_t)+(1-a)(\nabla f(\bold{x_t},\xi_t)-\nabla f(\bold{x_{t-1}, \xi_t}))
$$

$$
\bold{x_{t+1}}=\bold{x_t}-\eta \bold{d_t}
$$

The only difference is a new term:
$$
(1-a)(\nabla f(\bold{x_t},\xi_t)-\nabla f(\bold{x_{t-1}, \xi_t}))
$$


## 5 Algorithm: Storm

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210709223815738.png" alt="image-20210709223815738" style="zoom:50%;" />

## 5 Theorem 1

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210709230921061.png" alt="image-20210709230921061" style="zoom:50%;" />
$$
k=\frac{bG^{\frac 23}}{L}
$$

$$
c=28L^2+G^2/(7Lk^3)=L^2(28+1/(7b^3))
$$

$$
w=max((4Lk)^3, 2G^2,(\frac{ck}{4L})^3)=G^2max((4b)^3,2,(28b+\frac 1 {7b^2})^3/64)
$$

$$
M=\frac 8k (F(\bold{x_1})-F^\star)+\frac{w^{1/3}\sigma^2}{4L^2k^2}+\frac{k^2 c^2}{2L^2}\ln(T+2)
$$

**Explanation:**

If there is no noise, which means $\sigma =0$, then convergence rate is:
$$
O(\frac {\ln T}{\sqrt{T}})
$$
If there is noise (SGD), which means $\sigma \neq 0 $, then convergence rate is:
$$
O(\frac{2\sigma^{1/3}}{T^{1/3}})
$$
In SGD case, this matches the optimal rate, which was obtained by SVRG-based algorithms that require a **mega batch**.



注意到，在第一项中，当G趋于0时，k趋于0，M趋于无穷，似乎第一项是趋于无穷的。但是，并不是这样。根据G-Lipschitz条件可得：
$$
F(\bold{x_1})-F^\star = O(G)\ and\ \sigma = O(G)
$$
因此the numerators of M actually go to zero at least as fast as the denominator



注意到，当L=0时，no critical point，因为gradient都是相同的。



总的来说，M可以看作是一个$O(\log T)$的项。

 



## 6 Lyapunov potential function

In the theory of ordinary differential equations (ODEs), **Lyapunov functions** are scalar functions that may be used to prove the stability of an equilibrium of an ODE. 

typical form:
$$
\Phi_t = F(\bold{x_t})
$$
Our form:
$$
\Phi_t = F(\bold{x_t}) + z_t ||\epsilon_t||^2
$$
Where $z_t \propto \eta^{-1}_{t-1} $ and $\epsilon$ is the error term.



## 7 Proof of Theorem 1

First we introduce several lemmas.

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210710224701286.png" alt="image-20210710224701286" style="zoom:50%;" />

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210710224717379.png" style="zoom:50%;" />



<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210710233624045.png" alt="image-20210710233624045" style="zoom:50%;" />

Consider a Lyapunov function of the form:
$$
\Phi_t = F(\bold{x_t}) + \frac 1 {32L^2\eta_{t-1}}||\epsilon_t||^2
$$
We will upper bound  $\Phi_{t+1}-\Phi_t$ for each t, which will allow us to bound $\Phi_T$ in terms of $\Phi_1$ by summing over t.



###  $\mathbb{E}[\eta_t^{-1}||\bold{\epsilon_{t+1}}||^2 - \eta_{t-1}^{-1}||\bold{\epsilon_t||^2}]$

Use Lemma 2, we first consider $\mathbb{E}[\eta_t^{-1}||\bold{\epsilon_{t+1}}||^2 - \eta_{t-1}^{-1}||\bold{\epsilon_t||^2}]$:
$$
\begin{align}
\mathbb{E}&[\eta_t^{-1}||\bold{\epsilon_{t+1}}||^2 - \eta_{t-1}^{-1}||\bold{\epsilon_t||^2}]\\


\leq & \mathbb{E}\left[2c^2\eta_t^3G_{t+1}^2+\
(\eta_t^{-1}(1-a_{t+1})(1+4L^2\eta_t^2)-\eta_{t-1}^{-1})||\bold{\epsilon}||^2 + 4L^2\eta_t||\nabla F(\bold{x_t})||^2\right]
\end{align}
$$


There are three terms in the right side, and we denote them as $A_t, B_t, C_t$.

$$
A_t = 2c^2\eta_t^3G_{t+1}^2
$$

$$
B_t = \
(\eta_t^{-1}(1-a_{t+1})(1+4L^2\eta_t^2)-\eta_{t-1}^{-1})||\bold{\epsilon}||^2
$$

$$
C_t = 4L^2\eta_t||\nabla F(\bold{x_t})||^2
$$

Then let us focus on these terms individually.



For $A_t$:
$$
\sum_{t=1}^T A_t = \sum_{t=1}^T 2c^2\eta_t^3G_{t+1}^2  \leq 2k^3c^2\ln (T+2)
 \ (using\ Lemma\ 4)
$$



For $B_t$:
$$
B_t \leq (\eta_t^{-1} - \eta_{t-1}^{-1}+\eta_t (4L^2 -c))||\bold{\epsilon_t}||^2
$$

$$
\frac 1{\eta_t} - \frac 1{\eta_{t-1}} \leq \frac {G^2}{7Lk^3}\eta_t
$$

$$
\eta_t (4L^2 - c)\leq -24L^2\eta_t - G^2\eta_t / (7Lk^3)
$$

$$
Thus, B_t \leq -24L^2\eta_t||\bold{\epsilon_t}||^2
$$

For $C_t$:

We haven't done something on $C_t$ yet.



Putting all this together, we can get:
$$
\frac{1}{32L^2}\sum_{t=1}^T\left(\frac{||\epsilon_{t+1}||^2}{\eta_t} - \frac{||\epsilon_t||^2}{\eta_{t-1}}\right)\leq \frac{k^3c^2}{16L^2}\ln (T+2) + \sum_{t=1}^T\left[ \frac{\eta_t}{8}||\nabla F(x_t) ||^2 - \frac{3\eta_t}{4}||\epsilon_t||^2 \right]
$$

### $$ \mathbb{E}[\Phi_{t+1} - \Phi_t] $$

Now we are ready to analyze the potential $\Phi_t$.

Since $\eta_t \leq \frac{1}{4L}$, we can use Lemma 1 to obtain:
$$
\mathbb{E}[\Phi_{t+1} - \Phi_t] \leq \mathbb{E}\left[-\frac{\eta_t}{4} ||\nabla F(x_t)||^2  + \frac{3\eta_t}{4}||\epsilon_t||^2 + \frac{1}{32L^2\eta_t}||\epsilon_{t+1}||^2  -\frac{1}{32L^2\eta_{t-1}}||\epsilon_t||^2 \right]
$$


Summing over t and using the formula in the last part, we can get:
$$
\mathbb{E}[\Phi_{T+1} - \Phi_1]\leq \mathbb{E}\left[  \frac{k^3c^2}{16L^2}ln(T+2) - \sum_{t=1}^T \frac {\eta_t} 8 ||\nabla F(x_t)||^2 \right]
$$


Reordering the terms, we have:
$$
\mathbb{E}\left[ \sum_{t=1}^T\eta_t ||\nabla F(x_t) ||^2 \right] \leq 8(F(x_1) - F^\star) + \frac{w^\frac 13 
\sigma^2 }{(4L^2k) }+ \frac{k^3 c^2}{(2L^2)}\ln (T+2)
$$



### $$ \mathbb{E}\left[ \sum_{t=1}^{T}||\nabla F(x_t)||^2 \right]$$

Now, we relate $\mathbb{E}\left[ \sum_{t=1}^T\eta_t ||\nabla F(x_t) ||^2 \right]$ to $\mathbb{E}\left[ \sum_{t=1}^{T}||\nabla F(x_t)||^2 \right]$.



First, since $\eta_t$ is decreasing, 
$$
\mathbb{E}\left[ \sum_{t=1}^{T}\eta_t ||\nabla F(x_t)||^2 \right] \geq \mathbb{E}\left[ \eta_T\sum_{t=1}^{T}||\nabla F(x_t)||^2 \right]
$$


Now, from Cauchy-Schwarz inequality, for any random variables $A$ and $B$ we have:
$$
\mathbb{E}[A^2]\mathbb{E}[B^2]\geq \mathbb{E}[AB]^2
$$
Hence, setting:
$$
A =\sqrt{\eta_T \sum_{t=1}^{T-1}||\nabla F(x_t)||^2}
$$

$$
B = \sqrt \frac{1}{\eta_T}
$$

We obtain:
$$
\mathbb{E}\left[\eta_T \sum_{t=1}^{T-1}||\nabla F(x_t)||^2\right]\mathbb{E}\left[\frac{1}{\eta_T}\right]\geq \mathbb{E}\left[\sqrt {\sum_{t=1}^{T-1}||\nabla F(x_t)||^2} \right]^2
$$


To simplify the result, we set:
$$
M = \frac{1}{k}\left[  8(F(x_1) - F^\star) + \frac{w^\frac 13 
\sigma^2 }{(4L^2k) }+ \frac{k^3 c^2}{(2L^2)}\ln (T+2)\right]
$$
Then we get:
$$
\mathbb{E}\left[\sqrt {\sum_{t=1}^{T-1}||\nabla F(x_t)||^2} \right]^2 \leq  \mathbb{E}\left[ M\left (w + \sum_{t=1}^T G_t^2\right)^{\frac 13}\right]
$$


Define $\zeta = \nabla f(x_t, \xi_t) - \nabla F(x_t)$, so that:
$$
\mathbb{E}[||\zeta_t||^2] \leq \sigma^2
$$
Then, we have:
$$
G_t^2 = ||\nabla F(x_t) + \zeta _t ||^2 \leq 2 ||\nabla F(x_t)||^2 + 2||\zeta_t||^2
$$
And another formula:
$$
(a+b)^\frac{1}{3} \leq a^{\frac13} + b^\frac 13
$$


Plug them in, we obtain:
$$
\mathbb{E}\left[\sqrt {\sum_{t=1}^{T-1}||\nabla F(x_t)||^2} \right]^2 \leq  M(w+2T\sigma^2)^\frac{1}{3} + 2^\frac 13 M\left( \mathbb{E}\left[\sqrt {\sum_{t=1}^{T-1}||\nabla F(x_t)||^2} \right] \right)^\frac 23
$$



To simplify this inequality, we define:
$$
X = \sqrt{\sum_{t=1}^T ||\nabla F(x_t)||^2}
$$
Then the above can be written as:
$$
(\mathbb{E}\left[X \right])^2 \leq  M(w+2T\sigma^2)^\frac{1}{3} + 2^\frac 13 M\left( \mathbb{E}\left[X \right] \right)^\frac 23
$$
This  means that

either
$$
(\mathbb{E}\left[X \right])^2 \leq  M(w+2T\sigma^2)^\frac{1}{3}
$$
or 
$$
(\mathbb{E}\left[X \right])^2 \leq  2^\frac 13 M\left( \mathbb{E}\left[X \right] \right)^\frac 23
$$



Thus, we can solve $\mathbb{E}[X]$:
$$
\mathbb{E}[X]\leq\sqrt{2M}(w+2T\sigma^2)^\frac 16 + 2M^\frac 34
$$


By Cauchy-Schwarz, we have:
$$
\sum_{t=1}^T ||\nabla F(x_t)||/T \leq X/ \sqrt T
$$
And also, 
$$
(a+b)^\frac{1}{3} \leq a^{\frac13} + b^\frac 13
$$
Thus:
$$
\mathbb{E}\left[ \sum_{t=1}^{T} \frac{||\nabla F(x_t)||}{T}\right] \leq \frac{w^\frac 16 \sqrt{2M} + 2M^{\frac 34}}{\sqrt T} + \frac{2\sigma^\frac 13}{T^\frac 13}
$$

