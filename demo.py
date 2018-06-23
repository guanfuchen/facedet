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


def demo():
    num_classses = 2
    net = facebox.FaceBox(num_classes=num_classses)
    facebox_box_coder = facebox.FaceBoxCoder(net)

    net.load_state_dict(torch.load('weight/facebox.pt', map_location=lambda storage, loc: storage))
    net.eval()

    cap = cv2.VideoCapture(0)

    while True:
        # images_np = cv2.imread('13_Interview_Interview_2_People_Visible_13_52.jpg')
        retval, images_np = cap.read()
        images = cv2.resize(images_np, (1024, 1024))
        images = torch.from_numpy(images.transpose((2, 0, 1)))
        images = images.float().div(255)
        images = Variable(torch.unsqueeze(images, 0), volatile=True)

        loc_preds, conf_preds = net(images)

        loc = loc_preds[0, :, :]
        conf = conf_preds[0, :, :]


        boxes, labels, probs = facebox_box_coder.decode(loc, F.softmax(conf).data)
        # print('boxes:{}'.format(boxes))
        # print('labels:{}'.format(labels))
        print('probs:{}'.format(probs))

        img_h, img_w, img_c = images_np.shape
        print('images_np.shape:{}'.format(images_np.shape))
        for box_id, box in enumerate(boxes):
            prob = probs[box_id]
            box_x1 = box[0] * img_w
            box_y1 = box[1] * img_h
            box_x2 = box[2] * img_w
            box_y2 = box[3] * img_h
            print('({},{})->({},{})'.format(box_x1, box_y1, box_x2, box_y2))
            cv2.rectangle(images_np, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0))
            cv2.putText(images_np, str(prob), (box_x1, box_y1), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, (0, 255, 0))

        cv2.imshow('images_np', images_np)
        key = cv2.waitKey(1)
        if key==27:
            break

    cap.release()

if __name__ == '__main__':
    demo()
