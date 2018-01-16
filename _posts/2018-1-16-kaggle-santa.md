---
layout:     post
title:      "Santa Gift Matching Challenge"
date:       2018-01-16 18:29:00
author:     "Bigzhao"
header-img: "img/post-bg-05.jpg"
---


# Santa Gift Matching Challenge
In this playground competition, you’re challenged to build a toy matching algorithm that maximizes happiness by pairing kids with toys they want. In the dataset, each kid has 10 preferences for their gift (from 1000) and Santa has 1000 preferred kids for every gift available. What makes this extra difficult is that 0.4% of the kids are twins, and by their parents’ request, require the same gift.

其实就是一个匹配的问题，有两个评估矩阵，评估标准如下（非线性，但可以近似转为线性）：
```math
ANCH = \frac{1}{n_c} \sum_{i=0}^{n_c-1} \frac{ChildHappiness}{MaxChildHappiness},

ANSH = \frac{1}{n_g} \sum_{i=0}^{n_g-1} \frac{GiftHappiness}{MaxGiftHappiness}.

```

```math
z = (x_1+x_2+...+x_n)^3 + (y_1+y_2+...+y_m)^3.
```
利用线性回归，转换为，其中f非常小，所以近似认为只与x有关（孩子的快乐度）：
```math
c \cdot x_i + f \cdot y_j
```

## 最小费用最大流问题
### 1. 最大流性质：

对一个流网络G=(V,E)，其容量函数为c，源点和汇点分别为s和t。G的流f满足下列三个性质： 

- 容量限制：对所有的u，v∈V，要求f(u,v)<=c(u,v)。
- 反对称性：对所有的u，v∈V，要求f(u,v)=-f(v,u)。
- 流守恒性：对所有u∈V-{s,t}，要求∑f(u,v)=0(v∈V)。

### 2. 最小费用最大流：
在一个网络中每段路径都有“容量”和“费用”两个限制的条件下，此类问题的研究试图寻找出：流量从A到B，如何选择路径、分配经过路径的流量，可以在流量最大的前提下，达到所用的费用最小的要求。如n辆卡车要运送物品，从A地到B地。由于每条路段都有不同的路费要缴纳，每条路能容纳的车的数量有限制，最小费用最大流问题指如何分配卡车的出发路径可以达到费用最低，物品又能全部送到。

### 3. Google ortools demo
```py
from ortools.graph import pywrapgraph

min_cost_flow = pywrapgraph.SimpleMinCostFlow()

# Add each arc.
for i in range(0, len(start_nodes)):
    min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i], capacities[i], unit_costs[i])

# Add node supplies.
for i in range(0, len(supplies)):
    min_cost_flow.SetNodeSupply(i, supplies[i])

# Find the minimum cost flow
print('Start solve....')
min_cost_flow.SolveMaxFlowWithMinCost()
res1 = min_cost_flow.MaximumFlow()
print('Maximum flow:', res1)
res2 = min_cost_flow.OptimalCost()
print('Optimal cost:', -res2 / 2000000000)
print('Num arcs:', min_cost_flow.NumArcs())
```

## 匈牙利算法
n*n矩阵求最大匹配（二部图匹配）

算法最基本轮廓：

1. 置边集M为空（初始化，谁和谁都没连着）
2. 选择一个新的原点寻找增广路
3. 重复(2)操作直到找不出增广路径为止（2，3步骤构成一个循环）


### demo
比如由于矩阵太大，一次性算完匈牙利时间复杂度非常高，于是使用分块的思想，局部优化。


```py
def optimize_block(child_block, current_gift_ids):
    gift_block = current_gift_ids[child_block]
    C = np.zeros((block_size, block_size)) # C是代价矩阵
    for i in range(block_size):
        c = child_block[i]
        for j in range(block_size):
            g = gift_ids[gift_block[j]]
            C[i, j] = child_happiness[c][g]
    row_ind, col_ind = linear_sum_assignment(C)
    return (child_block[row_ind], gift_block[col_ind])
```


## mpi4py
MPI的全称是Message Passing Interface，即消息传递接口。

mpi4py是一个构建在MPI之上的Python库，主要使用Cython编写。mpi4py使得Python的数据结构可以方便的在多进程中传递。

由于该次比赛计算资源非常重要，在这次比赛中主要用mpi领多台PC主机（集群）进行协作。

比如由于矩阵太大，一次性算完匈牙利时间复杂度非常高，于是使用分块的思想，那么一般来说一台电脑由于内存等资源有些只能一次优化一个块，但是利用多台PC协作即可每台PC都优化一个块，进而使得优化速度显著加快。

github 项目地址：https://github.com/bigzhao/MPI-Hungarian-method