# A Breif Summary of Information Source

2020.2.26  Yanjie Ze

> An explanation is a piece of information displayed to users, explaining why a particular item is recommended.



比如：

1. textual sentence

2. topical word cloud

3. visually explainable recommendation

4. a list of freinds who also like

5. statistical histograms/pie charts



常见的几种如下图所示：

![](/Users/yanjieze/Documents/code/YanjieZe.github.io/imgs/recom.png)



## (1)Relevant User or Item Explanation

relevant user explanation：展示其他user的评价情况（打分）来作为推荐理由。基于此，发展出social friend推荐。



relevant item explanation：通过以前使用过的产品，推荐相似的物品。



相似物品这种解释比较intuitive。



## (2)Feature-based Explanation

和content-based比较接近。



一种是对item的feature进行分析。尽量提供和user比较匹配的特征。比如把电影用很多tag标记，user有prefer的tag，两者进行匹配，利用tag进行解释。



还有一种是user demographic feature，比如age，gender，residence location。利用user自身的特征与相似人群匹配，利用这些feature进行解释。



## (3)Opinion-based Explanation

分为aspect-level和sentence-level的，主要讲aspect-level。



这种方法和feature-based很像，但是不同之处在于：aspect是不能直接获得的，是推荐系统学出来的（比如从reviews）。



从review提取aspect-sentiment，用到sentiment analysis。



## (4)Sentence Explanation

主要分为template-based和generation-based。



template-based的要学feature，填入template。



NLG的方法：LSTM，adversarial seq2seq等。



许多生成exlpanation的方法依赖于user的review来训练，但是review的噪声是很大的。因此处理的方法有：hierarchical sequence-to-sequence model（auto-denoising），提取有效数据，等等。

## (5)Visual Explanation

可以通过对图片的某一部分高亮来表示user感兴趣的部分。



比如用attention来获得user关注的区域。



下图是具体例子。

![](/Users/yanjieze/Documents/code/YanjieZe.github.io/imgs/recom2.png)

> In general, the research on visually explainable recommendation is still at its initial stage.



## (6)Social Explanation

比如facebook会根据共同好友推荐新好友，或推荐好友们都喜欢的东西。



但是，这与user自身的喜好不一定有关联。结合user自身与好友的喜好可以推荐出novelty的东西。



## (Final)个人小结

从模型的逻辑上分，model-intrinsic  or model-agnostic。可类比成人类做决定，是先有理由再做决定（model-intrinsic），还是先做决定再想理由（model-agnostic）。

从内容的形式上分，有文字的和图片的，文字的这种比较常见，图片的做的不多。（个人感觉用attention关注图片上的某一区域来进行推荐有一点牵强）

对于推荐文字的内容可以细分，比如：

1. 直接用NLG模型，从所有用户的review中生成explanation。
2. 人为发现数据中的feature然后制定，进行explain（可以填入template）
3. 用网络学习feature，然后用sentiment analysis之类的nlp分析，然后explain（可以填入template）
4. 把用户社交圈的喜好倾向作为依据进行explain。
5. 根据用户个人喜欢（preference）作为推荐依据，生成explain。（最基本的一种）

这其中有的可以是model- intrinsic，有的可以是model- agnostic，没有特意强调。