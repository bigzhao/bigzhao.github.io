---
layout:     post
title:      "CS224N NLP for DL 笔记（in progress）"

date:       2018-05-22 17:08:00
author:     "Bigzhao"
header-img: "img/post-bg-js-module.jpg"
---
# CS224n DL FOR NLP 笔记 （in progress）
## Content
- [正文](#CS224n-DL-FOR-NLP-笔记)
  - [第一讲 概述](#第一讲)
  - [第二讲 词语的表达](#第二讲-词语的表达)
  - [第三讲 高级词向量表示](#第三讲-高级词向量表示)
  - [第四讲 Word Window分类与神经网络](#第四讲-word-window分类与神经网络)
  - [第五讲 反向传播](#第五讲-反向传播)
  - [第六讲 依存句法分析 Dependency Parsing](#第六讲-依存句法分析-dependency-parsing)
  - [第七讲 tensorflow](#第七讲-tensorflow)
  - [第10讲NMT系统模型和attention ](#第10讲NMT系统模型和attention )
  - [第11讲 深入GRU和机器翻译](#第11讲-深入GRU和机器翻译)
  - [第12讲 语音识别的端对端模型](#第12讲-语音识别的端对端模型)
  - [第13讲 卷积神经网络](#第13讲-卷积神经网络)
  - [第16讲 DMN与问答系统](#第16讲-DMN与问答系统)
  - [第18讲 挑战深度学习与自然语言处理的极限](#第18讲-挑战深度学习与自然语言处理的极限)

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

## 第七讲 tensorflow
这一讲主要由tensorflow的基本介绍和印度小哥的现场码代码组成

TensorFlow是一个图计算的开源类库

最初由Google Brain团队开发，用来进行机器学习研究

“TensorFlow是一个描述机器学习算法和实现机器学习算法的接口”

### 图计算编程模型
- 中心思想是将数值运算以图的形式描述。
- 图的节点是某种运算，支持任意数量的输入和输出
- 图的边是tensor（张量，n维数组），在节点之间流动

例子如下：

![image](http://wx2.sinaimg.cn/large/006Fmjmcly1fgslt0okgfj30ho0n4tes.jpg)

一般来说W,b是变量，x是占位符

### Tensorflow主要概念：
- Variable：
- Placeholder：
- Mathematical operation:

### 主要步骤：
1. 创建权重变量，初始化
2. 创建占位符X，y
3. 构建flow graph

以上图为例，在tensorflow实现图中的计算代码如下:

![image](http://wx3.sinaimg.cn/large/006Fmjmcly1fgsmc9spusj31ek0r8jyq.jpg)
### LOSS的定义
```
prediction = tf.nn.softmax(...)  #Output of neural network
label = tf.placeholder(tf.float32, [100, 10])

cross_entropy = -tf.reduce_sum(label * tf.log(prediction), axis=1)
```

### 计算梯度的方法：
```
train_step = tf.train.GradientDescentOptimizer(learing_rate).minimize(cross_entropy)
```


不同的scope可以用 tf.variable_scope(...)

### Summary:
1. Build a graph
2. Initialize a session: tf.global_variables_initializer()
3. Train with session.run(train_step, feed_dict)

PS:其中feed_dict是对占位符的映射，将真正的数据放进去

## RNN和语言模型
语言模型就是计算一个单词序列（句子）的概率（P(w1,…,wm)）的模型。

### 传统模型
为了简化问题，必须引入马尔科夫假设，句子的概率通常是通过待预测单词之前长度为n的窗口建立条件概率来预测

### RNN
![](http://wx1.sinaimg.cn/large/006Fmjmcly1fgtta8g6cmj30mt0d0400.jpg)

### 训练RNN非常难
- 梯度爆炸/梯度消失

原因：
![](http://o6gcipdzi.bkt.clouddn.com/%E9%93%BE%E5%BC%8F%E6%B3%95%E5%88%99.png)

### 梯度消失实例
http://cs224d.stanford.edu/notebooks/vanishing_grad_example.html

### 防止梯度爆炸
- clipping 截断 大于某个阈值就缩到阈值
- 但这种trick无法推广到梯度消失，因为你不想设置一个最低值硬性规定之前的单词都相同重要地影响当前单词。

### 减缓梯度消失
与其随机初始化参数矩阵，不如初始化为单位矩阵。这样初始效果就是上下文向量和词向量的平均。然后用ReLU激活函数。这样可以在step多了之后，依然使得模型可训练。

### softmax有时候计算复杂度很高
词表太大的话，softmax很费力。一个技巧是，先预测词语的分类（比如按词频分），然后在分类中预测词语。分类越多，困惑度越小，但速度越慢。所以存在一个平衡点：
![](http://wx2.sinaimg.cn/large/006Fmjmcly1fgu8nn5srqj30s20kegyf.jpg)

### 双向RNN
这里箭头表示从左到右或从右到左前向传播，对于每个时刻t的预测，都需要来自双向的特征向量，拼接后进行分类。箭头虽然不同，但参数还是同一套参数
![](http://o6gcipdzi.bkt.clouddn.com/%E5%8F%8C%E5%90%91RNN.png)

多层Bi-RNN

![](http://o6gcipdzi.bkt.clouddn.com/DEEPBIRNN.png)
## 第10讲NMT系统模型和attention
### NMT
NMT = encoder + decoder

![image](http://wx2.sinaimg.cn/large/006Fmjmcly1fgzs95nj6uj31dw0h4q4q.jpg)

![image](http://wx1.sinaimg.cn/large/006Fmjmcly1fgzszvodr6j30p80bp75k.jpg)

### 优势与缺点
NMT的四大优势
1. End-to-end training
为优化同一个损失函数调整所有参数

2. Distributed representation
更好地利用词语、短语之间的相似性

3. Better exploitation of context
利用更多上下文——原文和部分译文的上下文

4. 生成的文本更流畅
可能跟上述优势有关

NMT还避免了传统MT中的黑盒子（reorder之类）模型。

NMT也存在弱点

1. 无法显式利用语义或语法结构（依存句法分析完全用不上了，有些工作正在展开）

2. 无法显式利用指代相消之类的结果

### Attention
朴素encoder-decoder的问题是，只能用固定维度的最后一刻的encoder隐藏层来表示源语言Y，必须将此状态一直传递下去，这是个很麻烦的事情。事实上，早期的NMT在稍长一点的句子上效果就骤降。

这种机制也与人类译员的工作流程类似：不是先把长长的一个句子暗记于心再开始闭着眼睛翻译，而是草草扫一眼全文，然后一边查看原文一边翻译。这种“一边……一边……”其实类似于语料对齐的过程，即找出哪部分原文对应哪部分译文。而NMT中的attention是隐式地做对齐的。

### 打分（权重）
有一种打分机制，以前一刻的decoder状态和某个encoder状态为参数，输出得分：
![image](http://wx4.sinaimg.cn/large/006Fmjmcly1fgzww54hclj30pm0mcmzi.jpg)
![image](http://wx1.sinaimg.cn/large/006Fmjmcly1fgzwx6217oj30o60mkjtq.jpg)

分数转概率，用上熟悉的softmax
![image](http://wx3.sinaimg.cn/large/006Fmjmcly1fh0n92pf50j315c0o8whi.jpg)

### beam search
一直保留前K概率大小的组合

K的值不是越大越好，一般来说5左右


## 第11讲 深入GRU和机器翻译
### GRU
传统RNN和GRU
![image](http://wx3.sinaimg.cn/large/006Fmjmcly1fh26tkuf3mj31ay0r8qat.jpg)
GRU其实是为了让单元间有Adaptive shortcut connections 这样就可以RNN里面严重的梯度消失或者爆炸的问题

自适应=权衡过去/未来（从f函数可以看出来）

GRU = Candidate + Update Gate + Reset Gate

![image](http://wx4.sinaimg.cn/large/006Fmjmcly1fh2ylvadzrj319u0h6agh.jpg)

例子：out前面接不同的动词意思是完全不一样的，例如make out/take out 所以要记住前面那个单词到底是什么才可以更新隐藏层的信息

### LSTM
LSTM比GRU复杂了一点点，但思想是大同小异的

LSTM = Candidate + Output Gate + Forget Gate + Input Gate

- GRU与LSTM
![image](http://wx3.sinaimg.cn/large/006Fmjmcly1fh370tezfyj30ov0a6wil.jpg)

PS：其中LSTM的forget Gate 其实是Dont forget的概率

类似于GRU中的加法，在反向传播的时候允许原封不动地传递残差，也允许不传递残差，总之是自适应的。

有了这些改进，LSTM的记忆可以比RNN持续更长的step（大约100）：


![image](http://wx3.sinaimg.cn/large/006Fmjmcly1fh38axh12vg30hs0a0x6p.gif)

### 训练技巧
- 将递归权值矩阵初始化为正交
- 将其他矩阵初始化为较小的值
- 将forget gate偏置设为1：默认为不遗忘
- 使用自适应的学习率算法：Adam、AdaDelta
- 裁剪梯度的长度为小于1-5
- 在Cell中垂直应用Dropout而不是水平Dropout
- 保持耐心，通常需要训练很长时间

如果想获得额外的2个百分点的效果提升，可以训练多个模型，平均它们的预测。

### MT的Evaluation
主要分为以下三大类
#### 1. manual 手工
特点：慢、贵

#### 2. 讲MT当作一个子组成部分，去验证系统的整体功能
例如问答系统

### 3.Auto Metric：BLUE
核心思想：通过比较标准译文与机翻译文中NGram的重叠比率（0到1之间）来衡量机翻质量。

计算方式：

![image](http://wx1.sinaimg.cn/large/006Fmjmcly1fh3dppisxdj30t80oo79p.jpg)


## 第12讲 语音识别的端对端模型
这节课由师从Hinton就职于英伟达的Navdeep主讲，主题围绕语音识别。

## 第13讲 卷积神经网络
RNN 缺点：
- NN无法利用未来的特征预测当前单词，就算是bi-RNN，也不过是双向重蹈覆辙而已。
- 经常把过多注意力放到最后一个单词上。

CNN 思路：短语拆分，那就计算相邻的ngram，不管它到底是不是真正的短语，统一计算向量。

### 一维卷积定义：
![image](http://o6gcipdzi.bkt.clouddn.com/%E4%B8%80%E7%BB%B4%E5%8D%B7%E7%A7%AF%E5%AE%9A%E4%B9%89.png)

二维卷积图示：
![image](http://wx3.sinaimg.cn/large/006Fmjmcly1fdwcx2zdqag30em0aojsv.gif)

### 池化：
实际上用的比较多的是Max Pooling 目的是将更强更active的n-grams传递给下一层

为了得到多个卷积特征，简单地使用多个卷积核（不需要大小一致），然后把池化结果拼接起来。

另外，有一个关于词向量的技巧。如果任由梯度流入词向量，则词向量会根据分类任务目标而移动，丢失语义上泛化的相似性。解决办法是用两份相同的词向量，称作两个通道（channel）。一个通道可变，一个通道固定。将两个通道的卷积结果输入到max-pool中。

一般使用pre-trained的word vectors比较好，但是假如你的语料库有几个g那直接一起训练效果也应该会不错。

例子：
![image](http://wx2.sinaimg.cn/large/006Fmjmcly1fh7mqfmhm0j319w0kwwip.jpg)

### Dropout
一般用在CNN里面比较多的避免过拟合的方法是Dropout

调参的话一般使用随机调参

## 第16讲 DMN与问答系统
这节课主要讲了Richard的DMN动态记忆网络用于QA系统的应用。
### QA系统
例子如下图所示：
![image](http://wx1.sinaimg.cn/large/006Fmjmcly1fhgvlvr4clj31kw0ksqb8.jpg)

### 观点
将所有NLP任务视作QA问题。模仿人类粗读文章和问题，再带着问题反复阅读文章的行为，利用DMN这个通用框架漂亮地解决了从词性标注、情感分析到机器翻译、QA等一系列任务。

在old-school NLP系统中，必须手工整理一个“知识库”；然后在这个知识库上做规则推断。这节课介绍的DMN完全不同于这种小作坊，它能够直接从问答语料中学习所有必要的知识表达。

DMN还可以在问答中做情感分析、词性标注和机器翻译。

所以构建一个joint model用于通用QA成为终极目标。

### Dynamic Memory Networks
DMN仅仅解决了第一个问题。虽然有些超参数还是得因任务而异，但总算是个通用的架构了。


#### 总体结构
![image](http://wx2.sinaimg.cn/large/006Fmjmcly1fhgwvmstwnj313d0fjn4i.jpg)

左边输入input的每个句子每个单词的词向量，送入input module的GRU中。同样对于Question Module，也是一个GRU，两个GRU可以共享权值。

Question Module计算出一个Question Vector q，根据q应用attention机制，回顾input的不同时刻。根据attention强度的不同，忽略了一些input，而注意到另一些input。这些input进入Episodic Memory Module，注意到问题是关于足球位置的，那么所有与足球及位置的input被送入该模块。该模块每个隐藏状态输入Answer module，softmax得到答案序列。

DMN的sequence能力来自GRU，虽然一开始用的是LSTM，后来发现GRU也能达到相同的效果，而且参数更少。

## 第18讲 挑战深度学习与自然语言处理的极限

最后一课，总结了目前这两个领域中的难题，介绍了一些前沿研究：快16倍的QRNN、自动设计神经网络的NAS等。

### 障碍1：通用架构

没有单个模型能够胜任多个任务，所有模型要么结构不同，要么超参数不同。

暂时有希望的研究：上一级讲提到的DMN

### 障碍2：联合多任务学习

- 不像计算机视觉，只能共享低层参数（Word vector）
- 只在任务相关性很强的时候才会有帮助
- tasks之间如果相关性不强则会影响效果得不偿失

#### 解决方案
在第一课中提到的MetaMind团队提出的A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks。

![image](http://wx4.sinaimg.cn/large/006Fmjmcly1fhj7vrffjlj30oe0lin0a.jpg)

Richard说这个模型简直是个monster。多层LSTM并联，从下往上看在文本颗粒度上是越来越大，在任务上是越来越复杂。由底而上分别是词性标注、CHUNK、句法分析、两个句子的相关性、逻辑蕴涵关系。输入可能是词，也可能是字符ngram。底层任务的隐藏状态有到其他层级的直接路径。相关性encoder的输出卷积一下，供逻辑关系分类器使用。

整个模型使用同一个目标函数。左右对称只是示意可以接受两个句子用于关联分析，其实是同一套参数。



## 障碍3：预测从未见过的词语

## 障碍4：重复的词语表示
输入的词向量是一个词语表示，RNN或LSTM的隐藏状态到softmax的权值又是一个词语表示。这并不合理，为什么同一个单词非要有重复的表示呢？


## 障碍5：问题的输入表示是独立的
问题输入表示应当相关于fact的表示，比如May I cut you？根据场景是凶案现场还是超市排队的不同，表示也应该不同。

### 解决方案
Dynamic CoaaenQon Networks for QuesQon Answering by Caiming Xiong, Victor Zhong, Richard Socher (ICLR 2017)这篇论文提出用Coattention encoder同时接受fact encoder和query encoder的输出，这样就可以根据场景决定问题在问什么了。

## 障碍6：RNN很慢
作为最重要的积木之一，RNN成为很多NLP系统的性能瓶颈。

### 解决方案
Quasi-Recurrent Neural Networks by James Bradbury, Stephen Merity, Caiming Xiong & Richard Socher (ICLR 2017)这篇论文提出综合CNN和RNN的优点，得到一种叫做QRNN的模型

![image](http://wx1.sinaimg.cn/large/006Fmjmcly1fhjgtq414sj31c00uwdll.jpg)

RNN每个时刻需要等前一个时刻计算完毕，所以无法并行化。QRNN先将所有输入相邻两个拼成一行（这是为什么叫Quasi的原因），得到一个大矩阵乘上W，于是可以在时间上并行化。在计算隐藏状态的时候，虽然无法在时间上并行化，但可以在特征维度上并行化。

## 障碍7：架构研究很慢
（跟刚才说的手工设计特征->手工设计架构一样）能不能让深度学习模型自动设计架构呢？

Neural architecture search with reinforcement learning by Zoph and Le, 2016做了一点这样的研究，虽然无法自动设计QRNN、DMN这样的复杂架构，但好歹可以“设计”一些诸如隐藏层数、单元数等超参数

## 总结：NLP受到很多限制
1. 无法做有目的的QA
2. 无法用通用架构联合训练多任务
3. 很难综合多元逻辑与基于记忆的推理
4. 需要很多数据，无法从少量数据展开想象
