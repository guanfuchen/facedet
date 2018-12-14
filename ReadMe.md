# facedet

face detection algorithms

---
## 公告

避免大家花费时间去折腾，说明如下，目前该仓库主要还是将[faceboxes](https://github.com/lxg2015/faceboxes)阅读修改，基本跑通但是性能等还未系统测试，大家可以直接参考作者实现。下一步将最近阅读的人脸检测框架尝试统一实现并能获得较好的性能，整体实现思路尝试我写的语义分割框架[semseg](https://github.com/guanfuchen/semseg)的完善思路，有想法的可以一起来学习实现。

---
## Announcement
Avoid everyone spending time to toss, as explained below, the current warehouse is still mainly [faceboxes](https://github.com/lxg2015/faceboxes) read and modify, basically run through but performance has not been systematically tested, you can directly refer to The author realizes that the next step is to try to implement the face detection framework recently and achieve better performance. Try the perfect idea of the semantic segmentation framework [semseg](https://github.com/guanfuchen/semseg), and have ideas. Can be learned together to achieve.

---
## 概述
这个仓库旨在实现常用的人脸检测算法，主要参考如下：
- [faceboxes](https://github.com/lxg2015/faceboxes) 使用multi scale one shot的CNN网络实现人脸检测。
- [face_classification](https://github.com/oarriaga/face_classification)，实时的人脸检测（OpenCV）和分类（情感和性别）。
- [Face-Resources](https://github.com/betars/Face-Resources)，其中有相关人脸模型和数据集资源。
- [awesome-face](https://github.com/polarisZhao/awesome-face)，整理过的人脸检测论文和数据集。
- [mxnet-face](https://github.com/tornadomeet/mxnet-face)，常用的face相关论文的mxnet实现。

---
## 人脸识别

- [MobileFace](https://github.com/becauseofAI/MobileFace)
- [facenet_caffe](https://github.com/lippman1125/facenet_caffe)
- [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch)，使用pytorch实现的经典reid模型。
- [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)
- [Face Alignment in Full Pose Range: A 3D Total Solution](http://cvlab.cse.msu.edu/pdfs/Xiangyu_PAMI16.pdf)

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
