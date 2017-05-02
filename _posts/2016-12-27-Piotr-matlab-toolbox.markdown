---
layout:     post
title:      "Piotr's matlab toolbox 遇到的问题"
date:       2016-12-27 12:00:00
author:     "Bigzhao"
header-img: "img/post-bg-05.jpg"
---
# Piotr's matlab toolbox 遇到的问题

- 背景：最近想使用计算机视觉大牛 piotor 的工具箱中的acfdetector。
- 下载:
```bash
git clone https://github.com/pdollar/toolbox.git
```
- 在 matlab 中添加路径
```
>>> addpath(genpath('~\toolbox\'))  %这里是你的路径
>>> savepath
```
- 编译
```
>>> toolcompile
```
- 遇到的问题：我测试的代码是这样的
```matlab
I=imread('example.png');
t=load('AcfCaltech+Detector.mat');  
detector=t.detector;  
tic;
bbs=acfDetect(I,detector);
toc;
figure(1); im(I); bbApply('draw',bbs);
```
会报错，错误如下:
```matlab
未定义与 'struct' 类型的输入参数相对应的函数 'acfDetect1'。

出错 acfDetect>acfDetectImg (line 77)
    bb = acfDetect1(P.data{i},Ds{j}.clf,shrink,...

出错 acfDetect (line 41)
if(~multiple), bbs=acfDetectImg(I,detector); else

出错 test (line 6)
bbs=acfDetect(I,detector);
```
我估计是找不到 acfDetect1

所以把原来在 \toolbox\detector\private 里面的 mex 后的 acfDetect1 文件拿出来 放到 \toolbox\detector 即可解决
![example](http://o6gcipdzi.bkt.clouddn.com/acfmatlaberror.png)
