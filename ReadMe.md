# facedet

face detection algorithms

这个仓库旨在实现常用的人脸检测算法，主要参考如下：
- [faceboxes](https://github.com/lxg2015/faceboxes) 使用multi scale one shot的CNN网络实现人脸检测。

---
## 网络

- FaceBoxes


---
## 数据

- WIDER FACE


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

---
## 依赖

- pytorch
- ...
