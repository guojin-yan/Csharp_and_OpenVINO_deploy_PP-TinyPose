# 基于Csharp和OpenVINO<sup>TM</sup>部署PP-TinyPose

## 1. 项目介绍

&emsp;该项目基于OpenVINOTM模型推理库，在C#语言下，调用封装的OpenVINO<sup>TM</sup>动态链接库，部署推理PP-TinyPose人体关键点识别模型，实现了在C#平台调用OpenVINO<sup>TM</sup>部署PP-TinyPose人体关键点识别模型。

&emsp;如图所示，PaddlePaddle向我们提供了完整的人体关键点识别解决方案，主要包括行人检测以及关键点检测两部分。人体检测主要是实现行人位置检测，在多人关键点识别任务中，可以做行人区域划分等工作，此处飞桨提供了轻量级PicoDet行人识别模型，用于行人区域识别。关键点识别采用的是基于Lite-HRNet骨干网络的PP-TinyPose模型，并增加了DARK关键点矫正算法，使模型关键点识别更加精准；且该网络至此多bath_size推理，可以实现同时多图片推理运算。

​                               ![image-20221011191258081](doc\image\image-20221011191258081.png)

## 2. 项目编码环境

&emsp;为了防止复现代码出现问题，列出以下代码开发环境，可以根据自己需求设置，注意OpenVINO<sup>TM</sup>一定是2022版本，其他依赖项可以根据自己的设置修改。

-  操作系统：Windows 11

-  OpenVINOTM：2022.2.0

- OpenCV：4.5.5

- Visual Studio：2022

- C#框架：.NET 6.0

- OpenCvSharp：OpenCvSharp4

## 3. 源码下载方式

&emsp;项目所使用的源码均已经在Github和Gitee上开源，

```shell
Github:

git clone https://github.com/guojin-yan/Csharp_and_OpenVINO_deploy_PP-TinyPose

Gitee:

git clone https://gitee.com/guojin-yan/Csharp_and_OpenVINO_deploy_PP-TinyPose.git
```

 