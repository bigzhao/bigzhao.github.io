---
layout:     post
title:      "ubuntu 离线安装makefile"
date:       2017-04-29 12:00:00
author:     "Bigzhao"
header-img: "img/post-bg-02.jpg"
---
## ubuntu 离线安装makefile
1. 到[http://ftp.gnu.org/gnu/make/]() 下载make 安装包 我下的是make-3.81.tar.gz
2. 复制到机器上，解压并进入目录
```sh
tar -xzvf make-3.81.tar.gz
cd make-3.81
```
3. 依次执行以下命令
```
./configure
./make check
./make install
./make clean
```
4. 完成
