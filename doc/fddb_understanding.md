# FDDB数据集

FDDB数据集主要从yahoo新闻网站中采集的相关人脸（椭圆形标注），共有2845张图像，5171个人脸。

但是这里的数据集标注形状是椭圆型，需要转换为长方形的bounding box标注形状，这里参考[fddbToSigset.py](https://github.com/biometrics/openbr/blob/master/data/FDDB/fddbToSigset.py)。

也正是因为这里是椭圆形的标注，常常用来评估人脸检测器的性能，可以使用cross validation的ROC曲线和其他人脸检测算法比较性能。

---
## 数据集结构

FDDB_ROOT=~/Data/FDDB

FDDB_ROOT/originalPics存放原始图像

FDDB_ROOT/FDDB-folds存放分成10份的标注数据，其中可以使用任意1份作为校准集，另外9份作为训练集进行cross validation。

FDDB_ROOT/FDDB-folds/FDDB-fold-01.txt，FDDB_ROOT/FDDB-folds/FDDB-fold-02.txt，...，FDDB_ROOT/FDDB-folds/FDDB-fold-10.txt每一行为图像相对于FDDB_ROOT/originalPics的相对路径。

FDDB_ROOT/FDDB-folds/FDDB-fold-01-ellipseList.txt，FDDB_ROOT/FDDB-folds/FDDB-fold-02-ellipseList.txt，...，FDDB_ROOT/FDDB-folds/FDDB-fold-10-ellipseList.txt同时包含了图像相对路径和长轴、短轴、旋转弧度、椭圆中心x、椭圆中心y。

---
## 参考资料

- [Face Detection Data Set and Benchmark Home](http://vis-www.cs.umass.edu/fddb/) FDDB数据集主页。
- FDDB: A Benchmark for Face Detection in Unconstrained Settings，对应论文。