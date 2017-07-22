---
layout:     post
title:      "解决虚拟机出现‘锁定文件失败’错误"
date:       2017-07-22 14:41:00
author:     "Bigzhao"
header-img: "img/post-bg-miui6.jpg"
---
# 解决虚拟机出现‘锁定文件失败’错误
----
#### 问题描述
VMWare 虚拟机开机时出现以下提示的错误：

```
锁定文件失败

打不开磁盘“G:\Virtual Machines\Ubuntu 16.04 64bit\Ubuntu 64 位.vmdk”或它所依赖的某个快照磁盘。

模块“Disk”启动失败。

未能启动虚拟机。
```
#### 原因
虚拟机在运行的时候，会锁定文件防止被修改，而如果突然系统崩溃了，虚拟机就来不急把已经锁定的文件解锁，所以你在启动的时候，就会提示无法锁定文件。

#### 解决方法
- 进入你的虚拟机的文件目录下（就是你虚拟机存在哪里）
- 删除掉含有.lck文件的文件夹即可:

![](http://o6gcipdzi.bkt.clouddn.com/lck.png)
