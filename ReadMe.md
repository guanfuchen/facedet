# facedet

face detection algorithms

这个仓库旨在实现常用的人脸检测算法，主要参考如下：
- [faceboxes](https://github.com/lxg2015/faceboxes) 使用multi scale one shot的CNN网络实现人脸检测。

---
## 网络

- FaceBoxes，实现问题参考[facebox_understanding](./doc/facebox_understanding.md)


---
## 数据

人脸检测数据集可参考[Face Detection Dataset](https://xuchong.github.io/dataset/facedetection/2016/08/22/face-detetion-dataset.html) 其中主要有WIDER FACE，IJBA-A，MALF，FDDB和AFW数据集。

详细数据集相关实现问题参考[facedet_dataset_understanding](./doc/facedet_dataset_understanding.md)

- WIDER FACE，实现问题参考[wider_face_understanding](./doc/wider_face_understanding.md)
- FDDB，实现问题可参考[fddb_understanding](./doc/fddb_understanding.md)
- ALFW，实现问题可参考[alfw_understanding](./doc/alfw_understanding.md)


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
