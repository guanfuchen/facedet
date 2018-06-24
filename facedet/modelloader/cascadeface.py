# -*- coding: utf-8 -*-
import torch
from torch import nn


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class Cascade12Net(nn.Module):
    def __init__(self):
        super(Cascade12Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fc1 = nn.Linear(400, 16)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.conv1(x)
        # print('x.size():', x.size())
        x = self.relu1(x)
        x = self.pool1(x)
        # print('x.size():', x.size())

        x = x.view(batch_size, -1)
        # print('x.size():', x.size())
        x = self.fc1(x)
        x = self.relu2(x)

        x = self.fc2(x)
        # print('x.size():', x.size())
        return x

class Cascade12CalNet(nn.Module):
    def __init__(self):
        super(Cascade12CalNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fc1 = nn.Linear(400, 128)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(128, 45)  # 校准bounding box策略使用s_n，x_n，y_n以及校准公式进行调整，其中s_n（5种），x_n（3种），y_n（3种），5*3*3=45，通过最大的score选取对应的调整值作为最后的bounding box。

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = x.view(batch_size, -1)
        # print('x.size():', x.size())
        x = self.fc1(x)
        x = self.relu2(x)

        x = self.fc2(x)
        # print('x.size():', x.size())
        return x


class Cascade24Net(nn.Module):
    def __init__(self):
        super(Cascade24Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fc1 = nn.Linear(6400, 128)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.conv1(x)
        # print('x.size():', x.size())
        x = self.relu1(x)
        # print('x.size():', x.size())
        x = self.pool1(x)

        x = x.view(batch_size, -1)
        # print('x.size():', x.size())
        x = self.fc1(x)
        x = self.relu2(x)

        x = self.fc2(x)
        # print('x.size():', x.size())
        return x


class Cascade24CalNet(nn.Module):
    def __init__(self):
        super(Cascade24CalNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fc1 = nn.Linear(3200, 64)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(64, 45)  # 校准bounding box策略使用s_n，x_n，y_n以及校准公式进行调整，其中s_n（5种），x_n（3种），y_n（3种），5*3*3=45，通过最大的score选取对应的调整值作为最后的bounding box。

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = x.view(batch_size, -1)
        # print('x.size():', x.size())
        x = self.fc1(x)
        x = self.relu2(x)

        x = self.fc2(x)
        # print('x.size():', x.size())
        return x


class Cascade48Net(nn.Module):
    def __init__(self):
        super(Cascade48Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # self.norm1 = LRN(local_size=5, alpha=5e-5, beta=0.75)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=5e-5, beta=0.75)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # self.norm2 = LRN(local_size=5, alpha=5e-5, beta=0.75)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=5e-5, beta=0.75)

        self.fc1 = nn.Linear(5184, 256)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.conv1(x)
        # print('x.size():', x.size())
        x = self.relu1(x)
        # print('x.size():', x.size())
        x = self.pool1(x)
        # print('x.size():', x.size())

        x = self.norm1(x)
        # print('x.size():', x.size())

        x = self.conv2(x)
        # print('x.size():', x.size())
        x = self.relu2(x)
        # print('x.size():', x.size())
        x = self.pool2(x)
        # print('x.size():', x.size())

        x = self.norm2(x)
        # print('x.size():', x.size())

        x = x.view(batch_size, -1)
        # print('x.size():', x.size())
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        # print('x.size():', x.size())
        return x


class Cascade48CalNet(nn.Module):
    def __init__(self):
        super(Cascade48CalNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # self.norm1 = LRN(local_size=5, alpha=5e-5, beta=0.75)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=5e-5, beta=0.75)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # self.norm2 = LRN(local_size=5, alpha=5e-5, beta=0.75)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=5e-5, beta=0.75)

        self.fc1 = nn.Linear(5184, 256)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(256, 45)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.conv1(x)
        # print('x.size():', x.size())
        x = self.relu1(x)
        # print('x.size():', x.size())
        x = self.pool1(x)
        # print('x.size():', x.size())

        x = self.norm1(x)
        # print('x.size():', x.size())

        x = self.conv2(x)
        # print('x.size():', x.size())
        x = self.relu2(x)
        # print('x.size():', x.size())
        x = self.pool2(x)
        # print('x.size():', x.size())

        x = self.norm2(x)
        # print('x.size():', x.size())

        x = x.view(batch_size, -1)
        # print('x.size():', x.size())
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        # print('x.size():', x.size())
        return x
