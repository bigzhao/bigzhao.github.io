---
layout:     post
title:      "courase machine learning 课程笔记"
date:       2017-09-02 15:20:00
author:     "Bigzhao"
header-img: "img/post-bg-04.jpg"
---

#### 线性回归

- 梯度下降更新的时候要同步更新
- theta 一般初始值都为 0

##### batch gradient descent

事实证明，用于线性回归的代价函数总是一个凸函数convex function 因此这个函数没有局部最优只有全局最优

数据similar scale 能够更快收敛

缩放其实不需要太精确，目的是为了让梯度下降更好地工作

画收敛图来验证梯度下降的效果

纵轴是j（theta）横轴是iteration

如果j（theta）上升或者波动 可能就是learning rate的值调太大了 此时应该减少学习步长

选择learning rate α 可以尝试 0.001，0.003,0.01，0 03,0.1，...,1 ...

#### 多项式回归
特征结合

我们的模型可能不是线性的，用多项式回归可能更好地fit数据

不过多项式回归可能导致不同特征直接的差异性很大，所以应该要scale一下，以便更好地进行梯度下降

#### ormal equation
X*theta =y

利用矩阵知识求得
```
theta = inv（X’*X）*X’*y
```

特征不需要归一化

### 逻辑回归
#### Cost Fuction
通过推导，逻辑回归的梯度方程跟线性回归一样，这是hypothesis不一样
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-1.png)
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-2.png)
#### Advanced Optimization
还有很多其他的优化算法例如共轭梯度法、BFGS，LBFGS等

优点：
1. 不需要手动选学习步长
2. 更快收敛

缺点：
1.	比梯度下降复杂度更高

不推荐自己实现，可使用matlab的库函数fminunc

我们需要写一个CostFunction函数 返回J(theta) 和此时的梯度
```matlab
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```
例子：
```matlab
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
   [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```
**过拟合？ overfitting high variance**

**欠拟合  underfitting high bias**
##### 过拟合的解决方法：
1. 减少特征：
1）手工选择 2） 从模型中选择

PS：舍弃特征=丢弃信息

2. 正则化 Regulization
1） 保留所有特征，但减小了theta （岭回归 l2范数）

#### Regulization
Small value of parameters theta1,theta2...thetan

 - 'simpler' hypothesis
- less prone to overfitting

假如有一个模型需要预测，我们不知道哪些特征是重要的，所以我们才一起缩减全部的theta参数，通过J（theta）乘法函数里面加一则正则化项（惩罚项，一般不对theta0做正则）

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-3.png)
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-4.png)
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-5.png)
PS: λ太大会导致所有θ≈0 underfitting

###### 应用到梯度下降：
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-6.png)

###### 应用到normal equation：
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-7.png)

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-8.png)

此外，正则化可以为我们处理XTX不可逆的情况（例如m < n）

Recall that if m < n, then xtx is non-invertible. However, when we add the term λ⋅L, then + λ⋅L becomes invertible.
### 神经网络
神经元 = 计算单元

activation function 激励函数 （例如sigmoid）

神经网络由不同的神经元组成

###### 三层结构
输入层 隐藏层（多层） 输出层

activation units
###### Forward propagation
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-9.png)
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-10.png)
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-11.png)




神经网络的计算单元跟逻辑回归很像（只不过输入由上一层计算得来）

**神经元实现OR | AND 逻辑:**
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-12.png)

下面是一个利用AND （NOT） AND (NOT) OR 来构建更加复杂的XNOR逻辑的例子：
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-13.png)

##### 神经网络->多分类问题(handwriting recognition) one vs all
例如有4个待分类的类别 标签使用one-hot编码
[1;0;0;0] or [0;1;0;0] [0;0;1;0] [0;0;0;1]
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-14.png)

#### 神经网络的 Cost Function ：
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-15.png)

BP 算法 （反向传播算法） 求出J(theta)的偏导

For training example t =1 to m:
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-16.png)

1.	Set a(1):=x(t)
2.	Perform forward propagation to compute a(l) for l=2,3,…,L
3.	Using y(t) , compute δ(L)=a(L)−y(t)
4.	Compute δ(L−1),δ(L−2),…,δ(2) using δ(l)=((Θ(l))Tδ(l+1)) .∗ a(l) .∗ (1−a(l))
5.	Δ(l)i,j:=Δ(l)i,j+a(l)jδ(l+1)i or with vectorization, Δ(l):=Δ(l)+δ(l+1)(a(l))T
计算完Δ矩阵之后，可以根据以下公式计算J(theta)的偏导数
-	D(l)i,j:=1m(Δ(l)i,j+λΘ(l)i,j) , if j≠0.
-	D(l)i,j:=1mΔ(l)i,j If j=0
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-17.png)
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-18.png)
```
 δ(2)2 = Θ(2)12 *δ(3)1 +Θ(2)22 *δ(3)2 .
```
##### Unrolling Parameters
因为神经网络有多层theta向量，所以要在CostFuction中要展开以便更好地运算

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-19.png)

matlab命令：
-	合并：
```matlab
thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]
deltaVector = [ D1(:); D2(:); D3(:) ]
```
-	展开：
```matlab
Theta1 = reshape(thetaVector(1:110),10,11)
Theta2 = reshape(thetaVector(111:220),10,11)
Theta3 = reshape(thetaVector(221:231),1,11)
```
##### 梯度检验
因为反向传播算法容易出BUG，所以我们需要用梯度检验这一数值算法来检验我们的反向传播算法是否正确计算。
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-20.png)
一旦我们确认反向传播backpropagation是没有错误之后我们就可以关掉gradient checking，因为其复杂度高运行时间长。
-	代码：
```matlab
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;
```
#### breaking symmetry
**theta 的初始值要怎么设定才好呢？**

像逻辑回归一样设为0吗？这样做在逻辑回归里面是可以的，但是在神经网络里面是不行的，会导致对称的现象出现，因此为了breaking symmetry，要随机初始化theta的值

#### 综合
首先要选择一个合适的架构（多少层 多少节点）
* 输入节点：跟特征数一样
* 输出节点：跟分类问题的类别一样
* 隐藏层的层数：一般来说大于1，默认一层，一般隐藏层每层的单元数相等

接下来是训练神经网络
1. 随机初始化theta
2. 实现forward propagation
3. 实现cost function
4. 计算误差，进行back propagation计算偏导数
5. 使用梯度检验验证back prop的正确与否
6. 使用优化算法优化J(theta)

PS: 神经网络的J(theta) 不是convex的，所以我们不能保证一定收敛到全局最优

### 模型选择、特征选择
一般来说训练集要分三部分 训练集60% 交叉验证集20% 测试集%

训练集用来优化J(theta) 验证集用来调参或者特征工程 测试集是评估模型的好坏

总的来说 一定不能让测试集参与到训练的过程 让测试集代表那些模型从未接触过的新数据

-	欠拟合：训练集误差高 验证集误差高
-	过拟合：训练集误差低 验证集误差高（泛化能力差）
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-21.png)

##### learning curve
高偏差 high bias 增加样本是没用的
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-22.png)

高方差 high variance （gap）增加训练样本可能有用
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-23.png)

接下来看看那种方法应对哪种情况
##### Deciding What to Do Next Revisited
Our decision process can be broken down as follows:
-	Getting more training examples: Fixes high variance
-	Trying smaller sets of features: Fixes high variance
-	Adding features: Fixes high bias
-	Adding polynomial features: Fixes high bias
-	Decreasing λ: Fixes high bias
-	Increasing λ: Fixes high variance.
一般来说先撸一个baseline model 以便分析是不是要加特征、减特征、增数据？建造新特征 或者做特征工程的时候可以用来做对比

小型的神经网络 ：计算复杂度小 但是容易欠拟合

大型的神经网络（隐藏层多 计算units多）：计算复杂度高 可能会回出现过拟合
一般来说比较偏向于去使用大型的神经网络+正则项（J（theta）里加Theta矩阵的总平方）

##### Error Analysis
The recommended approach to solving machine learning problems is to:
-	Start with a simple algorithm, implement it quickly, and test it early on your cross validation data.
-	Plot learning curves to decide if more data, more features, etc. are likely to help.
-	Manually examine the errors on examples in the cross validation set and try to spot a trend where most of the errors were made.

对于一些skewed classes的模型 单纯用一些简单的正确率是不能很好地评价的 因此我们引入了召回率和准确率

一般来说召回率和准确率是两个相反的极端：

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-24.png)

阈值很高的时候也就是很有信心才判1的情况下 准确率很高 召回率很低

保守一点 很容易判1的情况就召回率很高 准确率很低

F1 score = 2 * PR / (P+R) F值权衡了准确率和召回率
### SVM（support  vector machine ）
- 也被称为large margin classifier
- 对ourlier 离群点非常敏感

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-25.png)

对于核函数的讲解，吴老师有不同的解释方式

** SVM Kernel**

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-26.png)

对于非线性的划分边界 我们之前的做法是加多项式特征，然而
1. 多项式是否真的是我们需要的特征
2. 当原本特征数量多的时候，例如图像识别，做多项式似乎是不可行的
有别的方法吗？kernel

高斯核函数：

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-26.png)

对于我们要建造的新的三个特征值，我们选中三个参考点计算我们的训练集点与这三个点的相似性
###### 相似性计算公式：
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-27.png)

如果离参考点很近：≈1

如果离参考点很远： ≈0

二维向量的情况下画出来大概是这样：

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-28.png)
当标准差σ变小是峰会变窄，增大时峰会变平缓

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-29.png)
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-30.png)
上面这幅图说明当选们θ0取-0.5 θ1=1 θ2= 1 θ3= 0时，离参考点1、2越近那么就越容易被判为1，因此决策边界如上图红线所示。因此也获得了非线性的决策边界。

但是问题来了？如何挑选参考点？-将自己的训练样本作为参考点 -> 特征多

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-31.png)
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-32.png)

大的方差，模型缓慢，y变化可能也缓慢，所以是更好的泛化性，所以高偏差低误差
为什么叫做最大margin？

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-33.png)

其中，p(i) 是x(i) 在θ向量上的投影

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-34.png)
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-35.png)

观察下图 哪个投影比较大？肯定是右边 所以能够更好地min θ

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-36.png)

SVM需要我们选择C 选择σ 选择核函数（相似性函数）

注意：当训练样本比较少的情况下，应该不适用核函数，因为容易过拟合

核函数：满足默塞尔条件的核函数-原因是能够满足各种高级优化方法从而能够快速求θ
线性核的时候跟逻辑回归没有差别

逻辑回归的cost函数：

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-37.png)
SVM的：

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-38.png)
C= 1/λ 即svm提出了λ 其实意义还是一样 No kernel = linear kernel
**其他核（比较少用）**：多项式核、卡方核

svm与逻辑回归的应用场合

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-39.png)

### unsupervised learning
1. 聚类
2. PCA
##### K-means algorithm
两个输入
1. 簇的个数k
2. 没有label的training set
PS：这个时候training set 是n维的 不用加常数项了

执行的时候分为两个阶段：
1. cluster assignmentc(i) -> 跟踪每个样本属于那个类 可以证明这个是在min 损失函数 通过改变 c(1)...c(m)
2. move centroid  u(i)->跟踪每个聚类的中心 可以证明这个是在min 损失函数 通过改变 u(1)...u(m)

伪代码：

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-40.png)

PS: 有可能某一个类一个点都没有，这时候删除掉这个类是比较好的做法

优化目标：

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-41.png)
用多几次k-means取找全局最优
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-42.png)
怎样选择k？
1. 一般来讲手工来选
2. elbow methow
3. 根据后续操作来选k
##### PCA（主成分分析）
-	作用：降维 压缩数据
-	例如：加入两个特征高度相关，那么就需要降维
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-43.png)

可视化：将高维数据映射到低维然后plot出来

假设数据点是二维的

目的：找一条直线来做映射

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-44.png)


-	优化：找低维度的面，将投影误差的平方最小化

-	前期准备：归一化|标准化|规范化

尽管PCA和线性回归有点像，但是是两个totally different的算法
首先线性回归是找垂直距离，PCA是找直角距离，其次线性回归是预测y，PCA没啥好预测的

	数据预处理：
如果特征变量之间的跨度很大的化，需要归一化

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-45.png)

svd奇异值分解获取u

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-46.png)

Sigma = 1/m * X' * X (假设X ∈ m×n维)

其中U n×n矩阵，就是我们变换后的方向矩阵（取前k个特征即可）：这个相当于是找了k个方向

我们需要获取的是变换后的z(x是n×1维的)：z = Ureduce' * x

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-47.png)
PS：跟k-means算法一样，不需要用到常数项
###### reconstruct
PCA是将高维压缩到低维度，那么我们有时候也需要将压缩完了的数据恢复到高维情况下（例如3维数据压缩到2维，我们想回去看看再三维的情况下哪个映射平面是怎样的）

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-48.png)
因此

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-49.png)
----
接下来继续降维的问题，怎样取选择k（主成分的数量）呢？
尽可能保留数据的差异性

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-50.png)

事实上实际大部分数据的特征都是高度相关的，所以维度下降的幅度大，然而数据的差异性也保持得很好

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-51.png)


PS：利用S矩阵来计算不同得k是否满足上面得原则 k=1、k=2、...
###### PCA的作用：
1. 减轻磁盘储存压力
2. 加速算法学习
3. 压缩到2维或者3维来可视化数据
###### PCA的错误用法
1.	不应该用来作为减少特征从而避免过拟合（应该加正则项才是更适合）
2.	不应该在项目的一开始就将PCA考虑进去，我们应该处理原始数据，直至有实质性的数据支撑我们需要用到PCA时我们才使用（例如收敛过慢、内存开销太大、磁盘开销太大等）
### 异常检测 Anomaly detection
利用已有数据建模P(x) 测试新样本是否存在异常情况（P(x) < ε）

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-52.png)

-	高斯分布:
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-53.png)
-	参数估计：
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-54.png)
-	平均值：
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-55.png)
-	方差：
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-56.png)
-	对于μ和σ的极大似然估计, 假设特征之间独立分布，我们有：

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-57.png)

-	具体步骤：
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-58.png)
-	例子：
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-59.png)
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-60.png)

-	与监督学习很像
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-61.png)

###### 用什么指标来评估比较好呢？
回忆倾斜的监督学习的评估指标我们有
1. True positive、false negative， true negative， false negative
2. precision
3. recall
4. F1 score 2 * PR /(P+R)
**监督学习和异常检测算法的差异和应用场合？**

两种算法的应用场合：

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-62.png)

1. 正样本太少
2. 太多异常的类型，有限的数据集难以处理这么多异常类型
3. 未知的异常
PS：尽管垃圾邮件spam的分类也有很多种类型，但是因为我们能拿到大量的垃圾邮件样本，所以我们将垃圾邮件分类归于监督学习

###### 并不是所有特征都是有用的，那么怎么选择特征？
先把数据分布画出来，看是否满足高斯分布，如果右偏，那么加个log一般就能处理好，让数据分布更像高斯分布

选择那种一出错值就会很明显地变大或者变小的特征，异常时会变得unusual的特征
多元高斯分布

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-63.png)

改变μ，Σ就能获得不同的多元高斯分布

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-64.png)
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-65.png)
什么时候用原来的模型，什么时候用多元高斯模型？

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-66.png)

### 推荐系统
1.	基于内容的推荐（内容有特征 基于内容特征的预测）
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-67.png)
对于每个用户都有自己的一个theta向量 我们可以对每个用户进行线性回归来预测分数

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-68.png)
-	梯度下降
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-69.png)
但是一般来说电影等推荐产品的特征比较难搞

协同过滤 collaboration filter 自行学习需要的特征

假如我们不知道特征值是多少，我们可以通过用户的评分来得到电影的特征值，通过用户的数据来学习得到电影的特征值，如下所示

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-70.png)

其中我们采访（问卷调查）了每个用户，theta向量是指每个用户对浪漫爱情电影和动作电影的喜爱程度（加上了常数项，所以维度为3）

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-71.png)
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-72.png)

协同过滤就是我们拥有一大批用户的数据，其实每个用户都在协同运作，帮助我们学习出更适合的特征

我们可以先随机生成Theta来预测X，再来用X预测更好的theta，一直循环下去，我们的算法最后会收敛到一个比较好的结果

不过这样做实在在傻逼了，所以我们可以将上面两个优化目标合成一个，一次性求出最好的thtea和x

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-73.png)
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-74.png)
PS:不需要加常数项

得到的结果是我们将会获得每个用户对应的theta值和每部电影的特征值，因此即可预测用户还未评分的电影
###### 均值归一化
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-75.png)

避免了如果没有打分的话所有的预测均为0的情况，

-	验证数据量多是否对模型有效的一个方法是绘制学习曲线
-	随机梯度下降-stochastic 用于处理数据量较大的情况
-	与批次梯度下降不同，随机梯度下降只遍历一次数据集（除掉shuffle那次），而不是像批次梯度下降那样每次迭代都要遍历所有数据

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-76.png)

随机梯度下降和批次梯度下降不同之处在于收敛轨迹不一样，随机梯度下降更多时候是在最小值附近迂回而不是准确停留在最小值上

###### mini batch gradient descent 小批量梯度下降
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-77.png)
PS: b =  2 to 100

mini梯度下降可能比随机梯度下降快的关键是——向量化 vectorization求和相乘啥的可以用向量来做

###### 怎么确保我们选的步长是对的，整个模型是在收敛的？
-	批次梯度下降的做法是画出J(theta)的收敛图
-	然而这在数据量大的时候并不那么可行，因为计算一次J要遍历所有的数据
解决方法是:
-	因为每一代只操作一条数据，所以每一代都记录那条数据的J，然后每隔1000代来求个平均值，画出这个收敛的曲线

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-78.png)

**画出来可能是下面这几种情况**
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-79.png)

如果想收敛得比较好一点而不是一直在最小值附近震荡得话，可以在运行得时候随着迭代次数得增加在学习的后期减少learning rate α

###### 在线学习 online learning
-	一次学习一个样本，然后将其丢弃
-	这样预测结果可以随着用户流的偏好改变而改变
-	例如新闻抓取网站 预测点击率
-	特点：数据集不固定，用完即丢

#### MAP-reduce 和并行化
例子:
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-80.png)
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-81.png)

是否我们的训练算法能够将训练集拆开（加权求和）耗时的主要是求和部分,只要求和部分能拆 就能用并行,对于多核的电脑，也可以这样做

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-82.png)

因为是在同一台电脑上，所以网络通信成本也就小了很多

#### photo ocr
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-83.png)

**机器学习pipeline**：一个包含多个组件的系统，一系列组件使用机器学习算法。
**text detection**：跟之前過的行人识别有点像 多了一步展开的步骤 因为文字是一串串的。白色区域是代表区域里含有文字

PS：鉴于文字的特性，我们直接忽略又高又瘦的白色区域

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-84.png)
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-85.png)

将滑动窗的矩形分为两类，一类是能分割的，一类是不能分割的

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-86.png)

#### 人工数据合成
途径：
1.	手工创造数据
2.	数据增加噪声 扭曲
PS：一定要有意义，一般来讲加纯随机噪声意义不大
###### 在增加数据前，先考虑以下问题
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-87.png)

因为一个机器学习的pipeline有很多部件，而我们的时间是宝贵的，因此我们要对不同的component进行ceiling analysis 上限分析，看哪些部件能给我的模型带来最显著的效果。
具体的步骤就是我们手工调节上一步得到的结果，将输入调至最优，看效果的涨幅。

![]( http://o6gcipdzi.bkt.clouddn.com/coursera-88.png)

###总结：
![]( http://o6gcipdzi.bkt.clouddn.com/coursera-89.png)
