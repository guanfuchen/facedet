# develop

---
## 开发日志
- 预计增加SSH模型。
- Cascadeface开发中的LRN模版参考[alexnet.py](https://github.com/jiecaoyu/pytorch_imagenet/blob/master/networks/model_list/alexnet.py)，其中有LRN模块的实现和[Feature Request: Local response normalization (LRN) ](https://github.com/pytorch/pytorch/issues/653)，pytorch中已经实现该模块[normalization.py](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/normalization.py)。


---
## 参考资料

- [face_detector](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector) 基于OpenCV的深度学习人脸检测代码。
- [dbt_face_detection.cpp](https://github.com/opencv/opencv/blob/master/samples/cpp/dbt_face_detection.cpp) 基于OpenCV实现的VJ人脸检测算法。
- A Convolutional Neural Network Cascade for Face Detection实现级联CNN网络人脸检测架构，参考代码[fast_face_detector](https://github.com/IggyShone/fast_face_detector)，[A-Convolutional-Neural-Network-Cascade-for-Face-Detection](https://github.com/mks0601/A-Convolutional-Neural-Network-Cascade-for-Face-Detection)。