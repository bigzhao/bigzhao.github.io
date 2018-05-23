---
layout:     post
title:      "CS224N NLP for DL 笔记（in progress）"

date:       2018-05-22 17:08:00
author:     "Bigzhao"
header-img: "img/post-bg-js-module.jpg"
---
# CS224n DL FOR NLP 笔记 （in progress）
## 第一讲
第一讲主要围绕以下几个问题进行
### 1. 什么是NLP？人类语言的特性是什么？
- NLP是交叉学科：计算机技术 x AI x 语言学
- Goal: 计算机能够“理解”自然语言
- NLP编码的离散化带来稀疏性问题

### 2.DL是什么？
- 是机器学习的子领域
- 机器学习：人工设计特征，机器负责优化处一个令人满意的权重。（机器在这里只负责优化部分numerical optimization）
- 表征学习 representation learning：自动学习好的特征和模型

### 3.为什么DL4NLP困难？
- 语言的表示、学习、语义
- 人类的语言是模糊不清的
- 语言的解释需要上下文及要结合语义来解释

### 4.APP (从易到难)
- 拼写检查、关键词搜索
- 情感分析
- 机器翻译、机器问答系统



## 第二讲 词语的表达
传统用词表来表示词语：词语是词表的one-hot向量

之后JR Faith大牛提出

“You should know a word by company it keeps” -》“词的意义从共现的词来体现” J. R. Firth甚至建议，如果你能把单词放到正确的上下文中去，才说明你掌握了它的意义。

这是现代统计自然语言处理最成功的思想之一.

### 什么是Word2veb
- Goat:通过中心词来预测上下文
- P(context|Wt) = ...
- loss function： J = 1-p(w-t|wt) w-t是除了wt以外的所有上下文词语

#### Word2Veb相关的两个算法：
1. Skip-grams（SG): 通过中心词预测上下文
2. Continues bags of word (CBOW): 通过上下文预测中心词

### Detial of Word2veb
- 描述：对于单词t=1...T,预测以窗口半径为m范围内的上下文词语
- 目标函数：给定中心词，最大化预测上下文的概率
![image](http://wx2.sinaimg.cn/large/006Fmjmcly1fgcn8dsnw5j318q08m41y.jpg)
可使用log函数来简化：
![image](http://wx1.sinaimg.cn/large/006Fmjmcly1fgcn9ndo8dj316s092jwj.jpg)

- 预测到的某个上下文条件概率p(wt+j|wt)可由softmax得到：
![image](http://wx2.sinaimg.cn/large/006Fmjmcly1fgcnjx5gssj30tu09u47t.jpg)

### 梯度推导：
![image](http://wx2.sinaimg.cn/large/006Fmjmcly1fgcrcwktprj31es114npd.jpg)
![image](http://wx2.sinaimg.cn/large/006Fmjmcly1fgcrd88be3j31dy110hdt.jpg)
![image](http://wx4.sinaimg.cn/large/006Fmjmcly1fgcreegiemj31fg0zkhdt.jpg)

## 第三讲 高级词向量表示
- word2vec将窗口视作训练单位，每个窗口或者几个窗口都要进行一次参数更新。要知道，很多词串出现的频次是很高的。能不能遍历一遍语料，迅速得到结果呢？
- 在word2veb出现前已有许多方法被提出
- 基于统计共现矩阵的方法 Count-based methed window-based co-occurrence matrix

比如窗口半径为1，在如下句子上统计共现矩阵：

- I like deep learning.
- I like NLP.
- I enjoy flying.

共现矩阵为![image](http://wx4.sinaimg.cn/large/006Fmjmcly1fgdwg5nku6j315s0jyju2.jpg)

### 存在的问题：
- 当出现新词的时候，以前的旧向量连维度都得改变
- 高维度（词表大小）
- 高稀疏性

### 解决方法
SVD分解

![image](http://wx3.sinaimg.cn/large/006Fmjmcly1fgdwqgw3cbj311w0mmdia.jpg)

### 改进
1. 限制高频词的频次，或者干脆停用词
2. 根据与中央词的距离衰减词频权重
3. 用皮尔逊相关系数代替词频

### 效果
方法虽然简单，但效果也还不错：
![image](http://wx3.sinaimg.cn/large/006Fmjmcly1fgdwxjg5oqj30su0puq58.jpg)

### SVD的问题
1. 计算复杂度高：对n×m的矩阵是O(mn2)
2. 不方便处理新词或新文档
3. 与其他DL模型训练套路不同 不能表现词与词之间的高阶属性
4. 而NNLM, HLBL, RNN, Skip-gram/CBOW这类进行预测的模型必须遍历所有的窗口训练，也无法有效利用单词的全局统计信息。但它们显著地提高了上级NLP任务，其捕捉的不仅限于词语相似度。

### 综合两者的优势：GloVe
目标函数：
![image](http://o6gcipdzi.bkt.clouddn.com/glove.png)

这里的Pij是两个词共现的频次，f是一个max函数：

![image](http://wx4.sinaimg.cn/large/006Fmjmcly1fge0jnuvw6j30ig0a074y.jpg)


### 评测方法
如何评测词向量的好坏呢？两种方法：：Intrinsic（内部） vs extrinsic（外部）

Intrinsic：专门设计单独的试验，由人工标注词语或句子相似度，与模型结果对比。好处是是计算速度快，但不知道对实际应用有无帮助。有人花了几年时间提高了在某个数据集上的分数，当将其词向量用于真实任务时并没有多少提高效果，想想真悲哀。

- 类比法：man::women->king::?

Extrinsic：通过对外部实际应用的效果提升来体现。耗时较长，不能排除是否是新的词向量与旧系统的某种契合度产生。需要至少两个subsystems同时证明。这类评测中，往往会用pre-train的向量在外部任务的语料上retrain。

## 第四讲 Word Window分类与神经网络
这节课介绍了根据上下文预测单词分类的问题，与常见神经网络课程套路不同，以间隔最大化为目标函数，推导了对权值矩阵和词向量的梯度；初步展示了与传统机器学习方法不一样的风格。

这一讲推导比较多 不是很懂 详情看
http://www.hankcs.com/nlp/cs224n-word-window-classification-and-neural-networks.html

### Window classification
这是一种根据上下文给单个单词分类的任务，可以用于消歧或命名实体分类。上下文Window的向量可以通过拼接所有窗口中的词向量得到。


## 第五讲 反向传播
这一讲主要是从不同的角度去解释、推导反向传播算法，其中有向电图法还是令我眼前一亮的

首先问题还是 window-based classification，目标函数还是下面这个，也就是间隔最大化目标函数
```
J = max(0, 1- s +sc)
```

现在推演到含有两层隐藏层那个的情况，如下图所示：

![image](http://o6gcipdzi.bkt.clouddn.com/2layers.png)

最后经过推导，总结规律如下：

![image](http://o6gcipdzi.bkt.clouddn.com/guilv.png)

#### 另一种直观方法是电路图法

这里我就跳过简单的例子，直接上复杂例子：

![image](http://wx3.sinaimg.cn/large/006Fmjmcly1fggduhglzgj30pi0cu0x2.jpg)

- 红色的是梯度
- 绿色的是节点值
- 这样搞虽然有点冗余，但是非常直观
- 多层网络也可以这样思考，其实就是多个电路堆叠起来

#### Flow graphy
将上述电路视作有向无环流程图去理解链式法则，比如一条路径：

![image](http://wx3.sinaimg.cn/large/006Fmjmcly1fgge2zd5tcj30id0cr75r.jpg)

复杂的情况：

![image](http://wx2.sinaimg.cn/large/006Fmjmcly1fgge5ww9dzj30ox0eijub.jpg)

## 第四讲 Word Window分类与神经网络
这节课介绍了根据上下文预测单词分类的问题，与常见神经网络课程套路不同，以间隔最大化为目标函数，推导了对权值矩阵和词向量的梯度；初步展示了与传统机器学习方法不一样的风格。

这一讲推导比较多 不是很懂 详情看
http://www.hankcs.com/nlp/cs224n-word-window-classification-and-neural-networks.html

### Window classification
这是一种根据上下文给单个单词分类的任务，可以用于消歧或命名实体分类。上下文Window的向量可以通过拼接所有窗口中的词向量得到。


## 第五讲 反向传播
这一讲主要是从不同的角度去解释、推导反向传播算法，其中有向电图法还是令我眼前一亮的

首先问题还是 window-based classification，目标函数还是下面这个，也就是间隔最大化目标函数
```
J = max(0, 1- s +sc)
```

现在推演到含有两层隐藏层那个的情况，如下图所示：

![image](http://o6gcipdzi.bkt.clouddn.com/2layers.png)

最后经过推导，总结规律如下：

![image](http://o6gcipdzi.bkt.clouddn.com/guilv.png)

#### 另一种直观方法是电路图法

这里我就跳过简单的例子，直接上复杂例子：

![image](http://wx3.sinaimg.cn/large/006Fmjmcly1fggduhglzgj30pi0cu0x2.jpg)

- 红色的是梯度
- 绿色的是节点值
- 这样搞虽然有点冗余，但是非常直观
- 多层网络也可以这样思考，其实就是多个电路堆叠起来

#### Flow graphy
将上述电路视作有向无环流程图去理解链式法则，比如一条路径：

![image](http://wx3.sinaimg.cn/large/006Fmjmcly1fgge2zd5tcj30id0cr75r.jpg)

复杂的情况：

![image](http://wx2.sinaimg.cn/large/006Fmjmcly1fgge5ww9dzj30ox0eijub.jpg)

## 第六讲 依存句法分析 Dependency Parsing
#### 文献一般有两种方法：
1. 上下文无关 短语结构文法，英文术语是：Constituency = phrase structure grammar = context-free grammars (CFGs)。这种短语语法用固定数量的分解句子为短语和单词、分解短语为更短的短语或单词。
2. 依存句法分析 Dependency Parsing: 用单词之间的依存关系来表达语法。如果一个单词修饰另一个单词，则称该单词依赖于另一个单词。

这一讲主要focus在第二种方法

#### 举一个歧义的例子
```
Scientist study whales from space.
```
- 到底是study from space 还是 whales from space

总的来说，一个句子的dependence可以form成一棵树

#### 那么，应该怎么样去做依存句法分析呢？

1. 动态规划
2. 图算法
3. 限制约束
4. “transition-based parsing” or “deterministic dependence parsing” 个人理解是从左到右，逐个单词分析，可以用贪心算法，也可以用分类器来做

#### Transition-based parser
主要有以下三个步骤
1. shift
2. left-arc
3. right-arc

![image](http://o6gcipdzi.bkt.clouddn.com/threesteps.png)

具体做法如下：
![mak](http://o6gcipdzi.bkt.clouddn.com/baseline.png)
那么问题来了，我们应该什么时候才做第一第二第三步呢？
1. 根据规则建立features，然后用分类器做多分类 （特征是稀疏的，因为规则是人定的，不可能每个单词都符合规则，所以是一系列的01特征）
2. beam search （slower but better）

#### 评估指标
1. UAS:不考虑label 只考虑箭头方向
2. LAS：考虑label

计算方式如下图所示：
![image](http://o6gcipdzi.bkt.clouddn.com/metric.png)

#### 基于神经网络的parser
传统机器学习方法存在以下缺点：

1. sparse 特征太过稀疏
2. incomplete 不能够涵盖所有规则特征
3. expensive computation 主要是花费在特征计算上面去

为了克服以下缺点，陈丹奇做了基于NN的parser

hankcs指出无非是传统方法拼接单词、词性、依存标签，新方法拼接它们的向量表示：

![image](http://wx4.sinaimg.cn/large/006Fmjmcly1fgnda5d97hj310c0nk785.jpg)
![image](http://wx3.sinaimg.cn/large/006Fmjmcly1fgndbkj631j31eu0qwgta.jpg)

效果如下：

![image](http://wx4.sinaimg.cn/large/006Fmjmcly1fgnd5teln2j31aw0k0gon.jpg)

看到其实效果提升不大，基本与图算法持平，但是速度上有优势
