#!/usr/bin/python
# -*- coding: UTF-8 -*-

import unittest
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

from context import facedet
from facedet.modelloader import cascadeface

class TestCascadeface(unittest.TestCase):
    def test_cascade12net(self):
        C, H, W = (3, 12, 12)
        net = cascadeface.Cascade12Net()
        x = Variable(torch.randn(1, C, H, W))
        net(x)

    def test_cascade12calnet(self):
        C, H, W = (3, 12, 12)
        net = cascadeface.Cascade12CalNet()
        x = Variable(torch.randn(1, C, H, W))
        net(x)

    def test_cascade24net(self):
        C, H, W = (3, 24, 24)
        net = cascadeface.Cascade24Net()
        x = Variable(torch.randn(1, C, H, W))
        net(x)

    def test_cascade24calnet(self):
        C, H, W = (3, 24, 24)
        net = cascadeface.Cascade24CalNet()
        x = Variable(torch.randn(1, C, H, W))
        net(x)

    def test_cascade48net(self):
        C, H, W = (3, 48, 48)
        net = cascadeface.Cascade48Net()
        x = Variable(torch.randn(1, C, H, W))
        net(x)

    def test_cascade48calnet(self):
        C, H, W = (3, 48, 48)
        net = cascadeface.Cascade48CalNet()
        x = Variable(torch.randn(1, C, H, W))
        net(x)

    def test_speed(self):
        pass


if __name__ == '__main__':
    unittest.main()
