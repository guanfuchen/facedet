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
from facedet.modelloader import facebox

class TestFaceBoxDataEncoder(unittest.TestCase):
    def test_decode(self):

        C, H, W = (3, 1024, 1024)
        net = facebox.FaceBox(num_classes=2)
        facebox_data_coder = facebox.FaceBoxCoder(net)
        facebox_default_boxes = facebox_data_coder.default_boxes
        print('facebox_default_boxes.size():{}'.format(facebox_default_boxes.size()))
        # print('facebox_default_boxes:{}'.format(facebox_default_boxes))
        # locs和confs刚好只有一个batch
        locs = torch.load('../data/loc.pt')
        confs = torch.load('../data/conf.pt')
        # print('locs:', locs)
        # print('confs:', confs)

        loc = locs[0, :, :]
        conf = confs[0, :, :]

        loc_np = loc.data.numpy()
        conf_np = conf.data.numpy()

        print('loc.size():{}'.format(loc.size()))
        print('conf.size():{}'.format(conf.size()))

        boxes, labels, probs = facebox_data_coder.decode(loc, F.softmax(conf).data)
        print('boxes:{}'.format(boxes))
        print('labels:{}'.format(labels))
        print('probs:{}'.format(probs))

        import cv2
        img = cv2.imread('../obama.jpg')
        img_h, img_w, img_c = img.shape

        for box in boxes:
            box_x1 = box[0]*img_w
            box_y1 = box[1]*img_h
            box_x2 = box[2]*img_w
            box_y2 = box[3]*img_h
            cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0))

        cv2.imshow('img', img)
        cv2.waitKey()

    def test_encode(self):
        pass
        C, H, W = (3, 300, 300)
        net = facebox.FaceBox(num_classes=2)
        facebox_data_coder = facebox.FaceBoxCoder(net)

        boxes = torch.from_numpy(np.array([(0.4531,  0.1200,  0.6465,  0.4567)], dtype=np.float32))
        labels = torch.from_numpy(np.array([1], dtype=np.int32))
        loc_targets, conf_targets = facebox_data_coder.encode(boxes, labels)
        # print('loc_targets:{}'.format(loc_targets))
        # print('conf_targets:{}'.format(conf_targets))

    def test_train(self):
        num_classses = 2
        net = facebox.FaceBox(num_classes=num_classses)
        facebox_box_coder = facebox.FaceBoxCoder(net)

        C, H, W = (3, 1024, 1024)
        x = Variable(torch.randn(1, C, H, W))
        boxes = torch.from_numpy(
            np.array([(0, 0, 100, 100), (25, 25, 125, 125), (200, 200, 250, 250), (0, 0, 300, 300)], dtype=np.float32))
        boxes /= torch.Tensor([W, H, W, H]).expand_as(boxes)  # norm to [0-1]
        labels = torch.from_numpy(np.array([1, 1, 1, 1], dtype=np.long))
        loc_targets, cls_targets = facebox_box_coder.encode(boxes, labels)
        loc_targets = loc_targets[None, :]
        cls_targets = cls_targets[None, :]
        # print('loc_targets.size():{}'.format(loc_targets.size()))
        # print('cls_targets.size():{}'.format(cls_targets.size()))

        # optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4)
        optimizer = optim.Adam(net.parameters(), lr=1e-5, weight_decay=5e-4)
        criterion = facebox.FaceBoxLoss(num_classes=num_classses)

        for epoch in range(1):
            loc_preds, cls_preds = net(x)
            # print('loc_preds.size():{}'.format(loc_preds.size()))
            # print('cls_preds.size():{}'.format(cls_preds.size()))
            optimizer.zero_grad()

            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss.backward()
            optimizer.step()

    def test_speed(self):
        pass


if __name__ == '__main__':
    unittest.main()
