# develop

---
## 开发日志
- 预计增加SSH模型。
- Cascadeface开发中的LRN模版参考[alexnet.py](https://github.com/jiecaoyu/pytorch_imagenet/blob/master/networks/model_list/alexnet.py)，其中有LRN模块的实现和[Feature Request: Local response normalization (LRN) ](https://github.com/pytorch/pytorch/issues/653)，pytorch中已经实现该模块[normalization.py](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/normalization.py)。
- 增加MTCNN实现人脸检测和关键点对齐。
- 完成MTCNN三种网络的实现，下一步进行WIDER FACE数据集收集策略实现（12x12，24x24，48x48分辨率的neg，pos和part）。


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