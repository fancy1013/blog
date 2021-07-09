# Notes for Bolei Zhou's RL Course

Yanjie Ze, 2021

# Lecture 2 MDP

## Bellman Expectation Backup

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708194212678.png" alt="image-20210708194212678" style="zoom:50%;" /> 

## Bellman Optimality Function

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708194328375.png" alt="image-20210708194328375" style="zoom:50%;" />

## Value Iteration

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708194714908.png" alt="image-20210708194714908" style="zoom:50%;" />

## Policy Iteration

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708195414426.png" alt="image-20210708195414426" style="zoom:50%;" />



## Difference between PI and VI

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708194952155.png" alt="image-20210708194952155" style="zoom:50%;" />

## Summary for prediction and control

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708195048282.png" alt="image-20210708195048282" style="zoom:50%;" />





# Lecture 3 Model-free prediction and control

## Model-free RL

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708195608725.png" alt="image-20210708195608725" style="zoom:50%;" />

## Monte-Carlo Policy Evaluation

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708195712127.png" alt="image-20210708195712127" style="zoom:50%;" />

## Incremental Mean

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708200059221.png" alt="image-20210708200059221" style="zoom:50%;" />

## Advantages of MC over DP

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708200421364.png" alt="image-20210708200421364" style="zoom:50%;" />

## Temporal Difference Learning

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708200503914.png" alt="image-20210708200503914" style="zoom:50%;" />

与MC相比，TD不需要跑完一个episode就可以更新。

## Advantages of TD over MC

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708200710567.png" alt="image-20210708200710567" style="zoom:50%;" />

## Comparison of TD and MC

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708201023678.png" alt="image-20210708201023678" style="zoom:50%;" />

## n-step TD

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708201054677.png" alt="image-20210708201054677" style="zoom:50%;" />

## 各种算法的总结

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708201352628.png" alt="image-20210708201352628" style="zoom:50%;" />

## Monte Carlo with epsilon greedy 

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708202106552.png" alt="image-20210708202106552" style="zoom:50%;" />

## Sarsa

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708202216918.png" alt="image-20210708202216918" style="zoom:50%;" />

## n-step Sarsa

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708202420820.png" alt="image-20210708202420820" style="zoom:50%;" />

## on-policy vs off-policy

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708202449514.png" alt="image-20210708202449514" style="zoom:50%;" />

## off-policy learning

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708202703865.png" alt="image-20210708202703865" style="zoom:50%;" />

## Comparison of Sarsa and Q-learning

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708202931324.png" alt="image-20210708202931324" style="zoom:50%;" />

# Lecture 4 价值函数近似

## Types of Value function approximation

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708204954538.png" alt="image-20210708204954538" style="zoom:50%;" />

## Linear Value Function Approximation

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708214104549.png" alt="image-20210708214104549" style="zoom:50%;" />

## Incremental VFA Prediction Algorithms

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708214413956.png" alt="image-20210708214413956" style="zoom:50%;" />

## Monte Carlo Prediction with VFA

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708214633304.png" alt="image-20210708214633304" style="zoom:50%;" />

## TD Prediction with VFA

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708214707905.png" alt="image-20210708214707905" style="zoom:50%;" />

## Incremental Control Algorithm

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708214943649.png" alt="image-20210708214943649" style="zoom:50%;" />

## Convergence of VFA

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708215428310.png" alt="image-20210708215428310" style="zoom:50%;" />

## 死亡三角

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708215645369.png" alt="image-20210708215645369" style="zoom:50%;" />

## Convergence Summary

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708215831533.png" style="zoom:50%;" />

## DQN

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708225255280.png" alt="image-20210708225255280" style="zoom:50%;" />

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708225346379.png" alt="image-20210708225346379" style="zoom:50%;" />

# Lecture 5 策略优化基础

## Value-based RL versus Policy-based RL

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708230504137.png" alt="image-20210708230504137" style="zoom:50%;" />

## Advantages of Policy-based RL

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708232036271.png" alt="image-20210708232036271" style="zoom:50%;" />

## 两种策略函数

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708232100303.png" alt="image-20210708232100303" style="zoom:50%;" />

## 优化目标

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708232711106.png" alt="image-20210708232711106" style="zoom:50%;" />

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708232830723.png" alt="image-20210708232830723" style="zoom:50%;" />

## 黑箱优化方法之CEM

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708233550950.png" alt="image-20210708233550950" style="zoom:50%;" />



## 黑箱优化方法之近似梯度

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708233948168.png" alt="image-20210708233948168" style="zoom:50%;" />



## 梯度的新形式 

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708234304387.png" style="zoom:50%;" />

## Policy例子

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708234549158.png" alt="image-20210708234549158" style="zoom:50%;" />

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708234631762.png" alt="image-20210708234631762" style="zoom:50%;" />

## Policy Gradient

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708234928404.png" alt="image-20210708234928404" style="zoom:50%;" />

<img src="/Users/yanjieze/Library/Application Support/typora-user-images/image-20210708235001301.png" alt="image-20210708235001301" style="zoom:50%;" />

