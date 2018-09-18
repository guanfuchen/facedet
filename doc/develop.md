# develop

---
## 开发日志
- 预计增加SSH模型。
- Cascadeface开发中的LRN模版参考[alexnet.py](https://github.com/jiecaoyu/pytorch_imagenet/blob/master/networks/model_list/alexnet.py)，其中有LRN模块的实现和[Feature Request: Local response normalization (LRN) ](https://github.com/pytorch/pytorch/issues/653)，pytorch中已经实现该模块[normalization.py](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/normalization.py)。
- 增加MTCNN实现人脸检测和关键点对齐。
- 完成MTCNN三种网络的实现，下一步进行WIDER FACE数据集收集策略实现（12x12，24x24，48x48分辨率的neg，pos和part）TODO。
- PyramidBox，实现该人脸检测模型，可参考论文和示例TF代码[PyramidBox](https://github.com/EricZgw/PyramidBox) TODO。
- 已完成AFLW数据集加载，但是加载较慢，下一步进行优化加载速度 TODO。
- 重新阅读人脸检测算法VJ同时总结实现，可参考[浅析人脸检测之Haar分类器方法](http://www.cnblogs.com/ello/archive/2012/04/28/2475419.htm) TODO。
- [Face detection with OpenCV and deep learning](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/) opencv中实现了ssd用来检测人脸，查看相应的caffe源码。
<<<<<<< HEAD
- [sfd.pytorch](https://github.com/louis-she/sfd.pytorch) S3FD: single shot face detector in pytorch **TODO**，基于S3FD完成人脸检测。
- [face-py-faster-rcnn](https://github.com/playerkk/face-py-faster-rcnn) 实现face faster rcnn模型 TODO。
=======
- [sfd.pytorch](https://github.com/louis-she/sfd.pytorch) S3FD: single shot face detector in pytorch TODO
- [face-py-faster-rcnn](https://github.com/playerkk/face-py-faster-rcnn) 实现face faster rcnn模型，同时参考[Face-Detecion-with-the-Faster-R-CNN-R-FCN](https://github.com/kensun0/Face-Detecion-with-the-Faster-R-CNN-R-FCN) TODO。
>>>>>>> 422dbcb16698b38321e182ac9cffd4c36b295c50


---
## bug日志

---
在pytorch0.4版本中运行dface的训练脚本，发生错误按以下方案修改即可。

```
torch.cat(temp)
torch.stack(temp, dim=0)
```

[zero-dimensional tensor (at position 0) cannot be concatenated](https://github.com/wengong-jin/icml18-jtnn/issues/6)


---
## 参考资料

- [face_detector](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector) 基于OpenCV的深度学习人脸检测代码。
- [dbt_face_detection.cpp](https://github.com/opencv/opencv/blob/master/samples/cpp/dbt_face_detection.cpp) 基于OpenCV实现的VJ人脸检测算法。
- A Convolutional Neural Network Cascade for Face Detection实现级联CNN网络人脸检测架构，参考代码[fast_face_detector](https://github.com/IggyShone/fast_face_detector)，[A-Convolutional-Neural-Network-Cascade-for-Face-Detection](https://github.com/mks0601/A-Convolutional-Neural-Network-Cascade-for-Face-Detection)。
- [dface](https://gitee.com/kuaikuaikim/dface) dface的pytorch实现，其中对MTCNN进行了重新实现，大量参考其中的实现代码，训练机制。
- [HelloFace](https://github.com/becauseofAI/HelloFace) 该仓库收集了常用的face detection论文。
- [人脸检测 CSDN工作笔记](https://blog.csdn.net/app_12062011/article/category/7574422) 该博客系列作者对人脸检测常用算法进行了总结，可以作为中文参考资料索引。