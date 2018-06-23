# -*- coding: utf-8 -*-
import torch
import visdom
from torch.autograd import Variable
from torch import optim
from torch.utils import data
import torch.nn.functional as F
import numpy as np
import os
import cv2
import onnx
import onnx_caffe2.backend
import onnx_tf.backend

from facedet.modelloader import facebox
from facedet.dataloader import wider_face_loader


def main():
    num_classses = 2
    net = facebox.FaceBox(num_classes=num_classses)

    net.load_state_dict(torch.load('weight/facebox.pt', map_location=lambda storage, loc: storage))
    net.eval()

    images = Variable(torch.randn(1, 3, 1024, 1024), requires_grad=True)

    torch_out = torch.onnx._export(net, images, "./onnx/facebox.onnx", export_params=True)

    onnx_net = onnx.load("./onnx/facebox.onnx")
    # prepared_backend = onnx_tf.backend.prepare(onnx_net)
    # prepared_backend = onnx_caffe2.backend.prepare(onnx_net)
    # W = {onnx_net.graph.input[0].name: images.data.numpy()}



if __name__ == '__main__':
    main()
