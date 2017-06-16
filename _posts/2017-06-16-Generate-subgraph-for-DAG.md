---
layout:     post
title:      "Generate subgraph for DAG 有向无环图的子图生成"
date:       2017-06-16 11:27:00
author:     "Bigzhao"
header-img: "img/post-bg-01.jpg"
---
# Generate subgraph for DAG 有向无环图的子图生成
-------------

> About this project：  
  Github地址：[Generate_sub_graph](https://github.com/bigzhao/Generate_sub_graph)


#### 示例:  
输入的DAG：
![DAG](http://o6gcipdzi.bkt.clouddn.com/DAG.png)

## ↓

得到的子图:
![SUB](http://o6gcipdzi.bkt.clouddn.com/SUB.png)



###原理说明
原理很简单，就是有分叉的时候通过随机概率来判定是否应该将该结构转换为OR结构，然后再依次利用队列进行遍历，将遍历到的节点置为1即可.


### 使用方法
- 运行程序即可得到子图的拓扑结构
- 附加的“calculate_number_of_nodes.py”作用是统计每个子图的节点数量，并且根据节点数产生相应的权值。


### 注意事项
- DAG必须是按照模板格式给出节点以及有向边

- 子图生成的结果是输出0|1数组，1代表该节点存在，反之亦然
