# 本科毕设
基于 yolov3 的口罩检测系统

## 1.论文摘要
<br>
今年以来新型冠状病毒在全球肆虐。为了防止疾病传播国家规定出入公共场所必须佩戴口罩。在人流量较大的区域，靠人工检测人们是否佩戴口罩会给检测者带来一定危险。本文利用深度学习技术实现了一个口罩检测系统，当输入静态图片或者动态视频时，能迅速准确识别出人群中哪些人没有佩戴口罩，并加以标记。本系统主要使用深度学习技术，yolov3 目标检测模型来训练计算机对于口罩的检测。该项目分别自己训练和使用迁移学习两种方法。
<br>

## 2.项目步骤

- 配置 anaconda 环境
- 使用 labelImg 进行图像标注
- 使用 yolov3 训练图像（对 voc 数据集进行修改）
- mAP 性能测试



## 3.yolov3 网络架构

yolov3 框架采用 DenseNet-53
<div align=center>
<img width="584" alt="image" src="https://user-images.githubusercontent.com/22310531/153112387-d4ab6067-593b-4a8b-8a3e-9b3bd4ad75fa.png">
</div>

## 4.mAP 模型测试
<div align=center>
<img width="401" alt="image" src="https://user-images.githubusercontent.com/22310531/153112145-f50ef3a5-fe40-416d-9fdc-265743f04ef8.png">
</div>
  <br>
<div align=center>
  <img width="405" alt="image" src="https://user-images.githubusercontent.com/22310531/153112224-7cc8a167-2a03-4691-b4fb-3278f5aa15d4.png">
</div>


## 5.运行效果

视频测试：
<br>
<div align=center>
<img width="785" alt="image" src="https://user-images.githubusercontent.com/22310531/153112580-ac7c5b58-a62c-45b9-965f-5b9d2c7b0c95.png">
</div>
  <br>
<br>
图像测试：
<br>
<div align=center>
<img width="273" alt="image" src="https://user-images.githubusercontent.com/22310531/153112630-7f35bfdc-bfaf-463e-9777-91d24b79c8c1.png">
</div>


## 6. 运行环境

虽然都是 keras 和 tensorflow 框架，但是不同的 cuda 和 GPU 型号是需要下载不同版本的框架的，下面就是框架的版本，是必须要一一应的。
（1）项目框架版本
<br>
Python ：3.7.4
<br>
Tensorflow :1.14.0
<br>
Numpy 1.16.0
<br>
Keras :2.24
<br>
opencv-python：4.2.0.32

## 7.相关资料

1.论文查看地址：
链接: https://pan.baidu.com/s/19XHaMhE6gpYCeasSXNX_Og 提取码: 8hbr 
--来自百度网盘超级会员v5的分享

2.毕设答辩 PPT
链接: https://pan.baidu.com/s/1SU-IntPSuBNg5b4WVVEDIw 提取码: h7ti 
--来自百度网盘超级会员v5的分享
