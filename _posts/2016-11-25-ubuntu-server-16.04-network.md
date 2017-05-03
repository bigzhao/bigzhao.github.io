---
layout:     post
title:      "ubuntu server 16.04 配置网络"
date:       2016-11-25 12:00:00
author:     "Bigzhao"
header-img: "img/post-bg-03.jpg"
---
# ubuntu server 16.04 配置网络
DELL PowerEdgeT630 装好系统后按传统方法配网络失败，折腾一番才知道是网卡名字不对路
* 查看网卡具体信息(我的网卡名字是eno1，通常是eth0)
```bash
ifconfig -a
```
* 下面这个命令没啥用 看网卡信息的
```bash
lspci -v | grep eth
```
* 先看网卡有没有开
```bash
ifconfig
```
* 没有开的话就开网络
```bash
ifconfig eno1 up
```
```bash
sudo vim /etc/init.d/interfaces
```
* 更改上面那个文件 添加所需要动态或者静态的信息 这里比较特殊 网卡名叫做eno1
```
auto eno1
iface eno1 inet dhcp
```
* 重启网络
```bash
sudo /etc/inid.d/networking restart
```
OK!
