# facedet

face detection algorithms

这个仓库旨在实现常用的人脸检测算法，主要参考如下：
- [faceboxes](https://github.com/lxg2015/faceboxes) 使用multi scale one shot的CNN网络实现人脸检测。
- [face_classification](https://github.com/oarriaga/face_classification)，实时的人脸检测（OpenCV）和分类（情感和性别）。

---
## 人脸识别

- [MobileFace](https://github.com/becauseofAI/MobileFace)
- [facenet_caffe](https://github.com/lippman1125/facenet_caffe)

---
## 行人检测

- [Is Faster R-CNN Doing Well for Pedestrian Detection?](https://arxiv.org/abs/1607.07032)
- [CityPersons: A Diverse Dataset for Pedestrian Detection](https://arxiv.org/abs/1702.05693)
- Learning Efficient Single-stage Pedestrian Detectors by Asymptotic Localization Fitting，行人检测

---
## 网络

- FaceBoxes，实现问题参考[facebox_understanding](./doc/facebox_understanding.md)，已实现
- CascadeCNN，实现问题参考[cascadeface_understanding](./doc/cascadeface_understanding.md)，未实现
- MTCNN，实现问题参考[mtcnn_understanding](./doc/mtcnn_understanding.md)，未实现。
- S3FD，实现问题可参考[s3fd_understanding](./doc/s3d_understanding.md)，未实现。
- Faster RCNN，参考技术报告[Face Detection with the Faster R-CNN](https://arxiv.org/abs/1606.03473)，和对应源代码[face-py-faster-rcnn](https://github.com/playerkk/face-py-faster-rcnn)。
- SSH，实现代码可参考[SSH](https://github.com/mahyarnajibi/SSH)。


---
## 数据

人脸检测数据集可参考[Face Detection Dataset](https://xuchong.github.io/dataset/facedetection/2016/08/22/face-detetion-dataset.html) 其中主要有WIDER FACE，IJBA-A，MALF，FDDB和AFW数据集。

详细数据集相关实现问题参考[facedet_dataset_understanding](./doc/facedet_dataset_understanding.md)

- WIDER FACE，实现问题参考[wider_face_understanding](./doc/wider_face_understanding.md)
- FDDB，实现问题可参考[fddb_understanding](./doc/fddb_understanding.md)
- AFLW，实现问题可参考[aflw_understanding](./doc/aflw_understanding.md)
- CelebA，大尺度CelebFaces属性，实现问题可参考[celeba_understanding](./doc/celeba_understanding.md)
- IMDb-Face，IMBb人脸噪声数据，可查看[IMDb-Face](https://github.com/fwang91/IMDb-Face)


---
## 用法

**可视化**

[visdom](https://github.com/facebookresearch/visdom)
[开发相关问题](doc/visdom_problem.md)

```bash
# 在tmux或者另一个终端中开启可视化服务器visdom
python -m visdom.server
# 然后在浏览器中查看127.0.0.1:9097
```

**训练**
```bash
# 训练模型
python train.py
```

**校验**
```bash
# 校验模型
python validate.py
```

**测试**
```bash
# 测试模型
python test.py
```

**demo**
```bash
# 读取摄像头实时检测人脸
python demo.py
```

---
## 依赖

- pytorch
- ...
