---
layout:     post
title:      "数据结构：有向无环图的表示"
date:       2017-02-10 11:22:00
author:     "Bigzhao"
header-img: "img/post-bg-01.jpg"
---
# 数据结构：有向无环图的表示
最近在做workflow的时候有用到怎么去存储一个有向无环图，在百度上看到一个答复感觉很棒
http://blog.chinaunix.net/uid-24774106-id-3505579.html

文中使用先是 malloc 一个内存然后每当超出长度的时候就 realloc 内存（其实感觉跟 python 的 list 差不多）后面发现其实可以用vector，不用实现那么多代码（捂脸），但是自己实现了这个想法的确是work的，而且也省了不少内存

- 我程序的代码片如下：
```cpp
struct node {
	struct node **succ;
	int num_succ = 0;
	int num_pred = 0;
	int pred_len, succ_len;
}; // 省略了很多参数
```
然后先初始化:
```cpp
	nodes[i].succ = (struct node**)malloc(1 * sizeof(struct node*));
	nodes[i].num_succ = 0;
	nodes[i].succ_len = 1;
```
接下来遇到后继的时候判断num_succ 和succ_len 为图方便每次内存扩大两倍（还有更更加使用的策略，参考python的list的扩充策略）
```cpp
	// 检查后继是否越界，如果是则realloc内存,为了方便每次扩大两倍 应该能够满足需求
	while (nodes[current - 1].succ_len <= nodes[current - 1].num_succ) {
		nodes[current - 1].succ_len *= 2;
		new_ptr = realloc(nodes[current - 1].succ, sizeof(struct node *) * (nodes[current - 1].succ_len));
		if (!new_ptr) {
			fprintf(stderr, "succ realloc error!");
			exit(-1);
		}
		nodes[current - 1].succ = (struct node **)new_ptr;
    }
```
上述代码与链接里面相比添加了错误处理机制
### realloc 的注意事项
1. 如果有足够空间用于扩大mem_address指向的内存块，realloc()试图从堆上现存的数据后面的那些字节中获得附加的字节，如果能够满足，自然天下太平。也就是说，如果原先的内存大小后面还有足够的空闲空间用来分配，加上原来的空间大小＝newsize。那么就ok。得到的是一块连续的内存。
2. 如果原先的内存大小后面++没有++足够的空闲空间用来分配，那么从堆中另外找一块newsize大小的内存。并把原来大小内存空间中的内容复制到newsize中。返回新的mem_address指针。（数据被移动了,原来内存free掉）。

*因此，假如还有指针指向原来的内存的话，那么free则会出现double free的情况,另外避免 p = realloc(p,2048); 这种写法。有可能会造成 realloc 分配失败后，p原先所指向的内存地址丢失。*
