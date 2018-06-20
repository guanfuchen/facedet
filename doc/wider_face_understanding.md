# WIDER FACE数据集

该数据集共有32,203张图像，其中含有393,703个人脸bounding box，数据集含有对模糊、表情、光照、遮挡、位姿和有效等额外标注，具体标注参考数据集中的readme.txt。

```
The format of txt ground truth.
File name
Number of bounding box
x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
```

---
## 数据集结构

WIDER_ROOT=~/Data/WIDER

WIDER_ROOT/wider_face_split文件夹中存放训练集、校准集、测试集相关的bounding box等标注内容。

WIDER_ROOT/WIDER_test/images存放原始测试图像

WIDER_ROOT/WIDER_train/images存放原始训练图像

WIDER_ROOT/WIDER_val/images存放原始校准图像

---
## 参考资料
- WIDER FACE: A Face Detection Benchmark该数据集论文。