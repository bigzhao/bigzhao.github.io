---
layout:     post
title:      "2018中国高校计算机大赛——大数据挑战赛 Rank 20th 代码"

date:       2018-08-11 17:00:00
author:     "Bigzhao"
header-img: "img/post-bg-js-module.jpg"
---

# 2018中国高校计算机大赛——大数据挑战赛 Rank 20th 代码
比赛链接：[摸我](https://www.kesci.com/home/competition/5ab8c36a8643e33f5138cba4)
代码地址: [摸我](https://github.com/bigzhao/Kuaishou_2018_rank20th)

## 简介
简单来说就是活跃用户预测，根据用户之前快手APP信息推断日后会不会使用APP。详细数据说明看链接。

## Preprocessing
这道题需要自己划分数据集。
#### 划分方式

Name | User | Feature | Label
---|---|---|---
feature_1 | 1-16日  | 1-16日 | 17-23日
feature_2 |  1-23日  |  8-23日 | 24-30日
feature_test | 1-30日 |  15-30日 | 31-37日

#### 特征工程
特征包括对各个表的统计特征，例如历史登陆次数、历史操作次数、时间差特征,其中act表挖的比较多，还有就是关于author_id、video_id的统计信息，比如说重复看同一视频/作者次数的最大值、均值等，还有各种action、page的比例等等。

此外就是权重了，因为这是时序题，所以越靠近label区间的用户动作应该越重要。在此我用的是指数衰减权重：
```python
np.exp(cre.day - date)
```
还有其他选手也有使用其他权重计算方式：
```
 1 / (T - (T-X))
```

此外，feature engineering里面包括两个版本的特征提取函数，feature_v1 和 feature_v2主要的特征都是相同的，只是feature_v2增加了时间窗口特征，比如三、五、七天内登陆次数、交互次数等。

## 特征选择
在这里我使用了二进制编码的遗传算法来做特征选择，能给我找出一些不错的特征组合，虽然提升很有限，但是在后面做融合的时候还是能够获得还不错的增益。

## 模型
对于统计特征我主要用了Lightgbm和全连接DNN，XGB和Cat都用过效果不好，ffm也尝试用过，但是效果很差，不知道是不是特征的问题。此外，我还提取了每日的统计信息来做RNN，特征包括当天有没有登陆、看了多少个视频、action的数量及ratio、author_id和video_id的一些统计信息等。RNN模型我尝试了好几种，包括Text-CNN、LSTM、QRNN等，其中Text-CNN效果最好。

## 融合
融合方法为五折stacking，这题的融合收益还是很高的，我统计特征+LGB单模型才0.91056，融合后能做到0.91128。

## 总结
参加这次比赛收获还是很多的，这次参赛经历也是神奇，从初赛A榜20+到换榜后差点进不到复赛到最后复赛20名。总的来说，大赛Q群里面的小伙伴和工作人员个个都skr人才，说话又好听，超喜欢在里面。
