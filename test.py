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

from facedet.modelloader import facebox
from facedet.dataloader import wider_face_loader


def test():
    num_classses = 2
    net = facebox.FaceBox(num_classes=num_classses)
    facebox_box_coder = facebox.FaceBoxCoder(net)

    root = os.path.expanduser('~/Data/WIDER')
    train_dataset = wider_face_loader.WiderFaceLoader(root=root, split='train', boxcoder=facebox_box_coder)
    train_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    net.load_state_dict(torch.load('weight/facebox.pt', map_location=lambda storage, loc: storage))
    net.eval()

    for epoch in range(1):

        for train_id, (images, loc_targets, conf_targets) in enumerate(train_dataloader):
            images = Variable(images)
            # images = cv2.imread('obama.jpg')
            # images = cv2.resize(images, (1024, 1024))
            # images = torch.from_numpy(images.transpose((2, 0, 1)))
            # images = images.float().div(255)
            # images = Variable(torch.unsqueeze(images, 0), volatile=True)

            loc_preds, conf_preds = net(images)

            loc = loc_preds[0, :, :]
            conf = conf_preds[0, :, :]

            loc_np = loc.data.numpy()
            conf_np = conf.data.numpy()

            # image_np = images[0, :, :, :].data.numpy()
            # image_np = image_np.transpose((1, 2, 0))
            # print(image_np.dtype)
            # print(image_np.shape)
            # cv2.imshow('img', image_np)
            # cv2.waitKey()


            boxes, labels, probs = facebox_box_coder.decode(loc, F.softmax(conf).data)
            print('boxes:{}'.format(boxes))
            print('labels:{}'.format(labels))
            print('probs:{}'.format(probs))

            print('loc_preds.size():{}'.format(loc_preds.size()))
            print('conf_preds.size():{}'.format(conf_preds.size()))
            # print('loc_targets.size():{}'.format(loc_targets.size()))
            # print('conf_targets.size():{}'.format(conf_targets.size()))
            break

if __name__ == '__main__':
    test()
