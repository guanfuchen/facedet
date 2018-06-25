# mtcnnface

这里主要参考[dface](https://gitee.com/kuaikuaikim/dface)，网络可视化参考caffe版本实现的[mtcnn-caffe](https://github.com/CongWeilin/mtcnn-caffe)和[mtcnn](https://github.com/dlunion/mtcnn)，对照论文和相应的模型的train_val.prototxt以及可视化[netscope](http://ethereon.github.io/netscope/#/editor)实现而成。




---
## 参考资料

- [mtcnn](https://github.com/dlunion/mtcnn)
- [mtcnn-caffe](https://github.com/CongWeilin/mtcnn-caffe)
- [dface](https://gitee.com/kuaikuaikim/dface) 基本参考这个仓库。
- [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) CelebA数据集，人脸alignment数据集。

---
## 网络架构

MTCNN主要有三个网络，分别是PNet，RNet和ONet。

### PNet

PNet网络主要训练cls和box，所以只需要WIDER FACE数据集即可（包含人脸图像的bounding box），同时将图像分为pos，neg，part这三个不同的训练样本进行训练。

PNet网络架构如下所示：

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/mtcnn_pnet.png)

### RNet

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/mtcnn_rnet.png)

### ONet

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/mtcnn_onet.png)
