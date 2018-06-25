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
from facedet.modelloader import mtcnnface

class TestMTCNNface(unittest.TestCase):
    def test_mtcnn_pnet(self):
        C, H, W = (3, 12, 12)
        net = mtcnnface.PNet()
        x = Variable(torch.randn(1, C, H, W))
        net(x)

    def test_mtcnn_rnet(self):
        C, H, W = (3, 24, 24)
        net = mtcnnface.RNet()
        x = Variable(torch.randn(1, C, H, W))
        net(x)

    def test_mtcnn_onet(self):
        C, H, W = (3, 48, 48)
        net = mtcnnface.ONet()
        x = Variable(torch.randn(1, C, H, W))
        net(x)

    def test_speed(self):
        pass


if __name__ == '__main__':
    unittest.main()
