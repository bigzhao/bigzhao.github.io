---
layout:     post
title:      "ubuntu server 16.04 挂载硬盘"
date:       2016-12-03 12:00:00
author:     "Bigzhao"
header-img: "img/post-bg-04.jpg"
---

实验室最近新购了一批主机，带4个硬盘，但是进去只有1个正在使用。
检查发现系统上目前并没有挂载其余硬盘
```
fdisk -l
```
尝试过直接挂载单数出现错误如下：
```
wrong fs type, bad option, bad superblock
```
个人猜测是没有格式化，硬盘格式不对。
执行以下命令格式化 sdx:
```bash
sudo mkfs -t ext4 /dev/sdb
```
说明：
* -t ext4 表示将分区格式化成ext4文件系统类型。
科普模式开启：

++EXT4是第四代扩展文件系统（英语：Fourth extended filesystem，缩写为 ext4）是Linux系统下的日志文件系统，是ext3文件系统的后继版本。++

注意：在格式 化完成后系统有如下提示：

```
This filesystem will be automatically checked every 28 mounts or
180 days, whichever comes first. Use tune2fs -c or -i to override.
```
表示系统为了保证文件系统的完整，每加载28次或每隔180天就要完整地检查文件系统，可以使用命令 tune2fs -c 或 tune2fs -i 来调整默认值 。

1. 显示硬盘挂载情况
```bash
sudo df -l
```
2. 挂载
```
sudo mount -t ext4 /dev/sdb /bigzhao
```
说明：
* 指定硬盘分区文件系统类型为ext4 ，同时将 /dev/sdb 分区挂载到目录 /bigzhao。
3. 自动挂载
改变 /etc/fstab
```
vim /etc/fstab
```
加入下面语句
```
/dev/sdb  /home/tandazhao/sdb(这个是要挂载的目录) ext4 default 0 0
```
重启即可
