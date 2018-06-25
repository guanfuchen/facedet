# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()

        # PNet head Net
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1)
        self.relu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv2 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1)
        self.relu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.relu3 = nn.PReLU()

        # PNet task Net
        self.conv4_1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1)  # 人脸分类
        self.conv4_2 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1)  # bounding box回归
        self.conv4_3 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1, stride=1)  # 面部landmark定位


    def forward(self, x):
        x = self.conv1(x)
        # print('x.size():', x.size())
        x = self.relu1(x)
        # print('x.size():', x.size())
        x = self.pool1(x)
        # print('x.size():', x.size())

        x = self.conv2(x)
        # print('x.size():', x.size())
        x = self.relu2(x)
        # print('x.size():', x.size())

        x = self.conv3(x)
        # print('x.size():', x.size())
        x = self.relu3(x)
        # print('x.size():', x.size())

        cls = F.sigmoid(self.conv4_1(x))
        # print('cls.size():', cls.size())
        box = self.conv4_2(x)
        # print('box.size():', box.size())
        return cls, box


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()

        # RNet head Net
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1)
        self.relu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv2 = nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3, stride=1)
        self.relu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2, stride=1)
        self.relu3 = nn.PReLU()

        self.fc4 = nn.Linear(in_features=576, out_features=128)
        self.relu4 = nn.PReLU()

        # RNet task Net
        self.fc5_1 = nn.Linear(in_features=128, out_features=1)  # 人脸分类
        self.fc5_2 = nn.Linear(in_features=128, out_features=4)  # bounding box回归
        self.fc5_3 = nn.Linear(in_features=128, out_features=10)  # 面部landmark定位


    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.conv1(x)
        # print('x.size():', x.size())
        x = self.relu1(x)
        # print('x.size():', x.size())
        x = self.pool1(x)
        # print('x.size():', x.size())

        x = self.conv2(x)
        # print('x.size():', x.size())
        x = self.pool2(x)
        # print('x.size():', x.size())
        x = self.relu2(x)
        # print('x.size():', x.size())

        x = self.conv3(x)
        # print('x.size():', x.size())
        x = self.relu3(x)
        # print('x.size():', x.size())

        x = x.view(batch_size, -1)
        # print('x.size():', x.size())
        x = self.fc4(x)
        # print('x.size():', x.size())
        x = self.relu4(x)
        # print('x.size():', x.size())

        cls = F.sigmoid(self.fc5_1(x))
        # print('cls.size():', cls.size())
        box = self.fc5_2(x)
        # print('box.size():', box.size())
        # landmark = self.fc5_3(x)
        # print('landmark.size():', landmark.size())
        return cls, box


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()

        # RNet head Net
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.relu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.relu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.relu3 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1)
        self.relu4 = nn.PReLU()

        self.fc5 = nn.Linear(in_features=1152, out_features=256)
        self.relu5 = nn.PReLU()

        # RNet task Net
        self.fc6_1 = nn.Linear(in_features=256, out_features=1)  # 人脸分类
        self.fc6_2 = nn.Linear(in_features=256, out_features=4)  # bounding box回归
        self.fc6_3 = nn.Linear(in_features=256, out_features=10)  # 面部landmark定位


    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.conv1(x)
        # print('x.size():', x.size())
        x = self.relu1(x)
        # print('x.size():', x.size())
        x = self.pool1(x)
        # print('x.size():', x.size())

        x = self.conv2(x)
        # print('x.size():', x.size())
        x = self.pool2(x)
        # print('x.size():', x.size())
        x = self.relu2(x)
        # print('x.size():', x.size())

        x = self.conv3(x)
        # print('x.size():', x.size())
        x = self.relu3(x)
        # print('x.size():', x.size())
        x = self.pool3(x)
        # print('x.size():', x.size())

        x = self.conv4(x)
        # print('x.size():', x.size())
        x = self.relu4(x)
        # print('x.size():', x.size())

        x = x.view(batch_size, -1)
        # print('x.size():', x.size())
        x = self.fc5(x)
        # print('x.size():', x.size())
        x = self.relu5(x)
        # print('x.size():', x.size())

        cls = F.sigmoid(self.fc6_1(x))
        # print('cls.size():', cls.size())
        box = self.fc6_2(x)
        # print('box.size():', box.size())
        landmark = self.fc6_3(x)
        # print('landmark.size():', landmark.size())
        return cls, box, landmark
