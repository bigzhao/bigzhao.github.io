---
layout:     post
title:      "Win10下同时安装python3和python2并解决pip共存问题"
date:       2017-05-15 13:18:00
author:     "Bigzhao"
header-img: "img/post-bg-03.jpg"
---
# Win10下同时安装python3和python2并解决pip共存问题
- 背景：想在win10下安装tensorflow 然而系统只用anaconda装了python2 因此想要再装一个python3

总体步骤如下：
1. [官网](https://www.python.org/)下载python2&python3
2. 安装python2&python3
3. 进入安装目录，将两个python.exe和pythonw.exe 分别改成python2.exe pythonw2.exe和python3.exe pythonw3.exe
4. 接下来打开dos，重装pip
```
python3 -m pip install --upgrade pip --force-reinstall
python2 -m pip install --upgrade pip --force-reinstall
```
![image](http://o6gcipdzi.bkt.clouddn.com/win10python2&3.png)
5. 现在可以通过pip2 -V 和 pip3-V 查看两个版本的pip信息，以后只需运行pip2install XXX和pip3 install XXX即可安装各自的python包。
6. Done
