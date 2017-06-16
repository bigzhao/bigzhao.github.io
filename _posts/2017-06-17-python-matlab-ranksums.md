---
layout:     post
title:      "关于python matlab ranksum函数得到结果不一致的问题"
date:       2017-06-17 11:14:00
author:     "Bigzhao"
header-img: "img/post-bg-02.jpg"
---
# python matlab ranksum函数得到结果不一致的问题

今日在利用秩和检验计算两组数据是否差异显著的时候，发现matlab与python的ranksum结果不一致。

首先matlab我使用的是ranksum函数，python使用的是ranksums函数。后查证python的scipy包还有mannwhitneyu()等秩和检验函数。因此有点混淆到底应该使用哪个？

后来自己测试和上网找例子之后发现ranksums和ranksum函数是最接近的，结果都在同一数量级。stackoverflow中有个回答[(戳这里)](https://stackoverflow.com/questions/31709475/what-is-pythons-equivalent-of-matlabs-ranksum)也提到可能是默认的选项不同，不过到底选哪个倒是没有讲的很清楚。

所以说到底并没有什么鸟用。。。。。还是不知道选哪个。。。。。

matlab的ranksum函数官方介绍如下所示：
```
p = ranksum(x,y) returns the p-value of a two-sided Wilcoxon rank sum test. ranksum tests the null hypothesis that data in x and y are samples from continuous distributions with equal medians, against the alternative that they are not. The test assumes that the two samples are independent. x and y can have different lengths.

This test is equivalent to a Mann-Whitney U-test.

example

[p,h] = ranksum(x,y) also returns a logical value indicating the test decision. The result h = 1 indicates a rejection of the null hypothesis, and h = 0 indicates a failure to reject the null hypothesis at the 5% significance level.

example

[p,h,stats] = ranksum(x,y) also returns the structure stats with information about the test statistic.

example

[___] = ranksum(x,y,Name,Value) returns any of the output arguments in the previous syntaxes, for a rank sum test with additional options specified by one or more Name,Value pair arguments.
```
python scipy包的ranksums介绍如下：

```
scipy.stats.ranksums(x, y)[source]

   Compute the Wilcoxon rank-sum statistic for two samples.

   The Wilcoxon rank-sum test tests the null hypothesis that two sets of measurements are drawn from the same distribution. The alternative hypothesis is that values in one sample are more likely to be larger than the values in the other sample.

   This test should be used to compare two samples from continuous distributions. It does not handle ties between measurements in x and y. For tie-handling and an optional continuity correction see scipy.stats.mannwhitneyu.
```
