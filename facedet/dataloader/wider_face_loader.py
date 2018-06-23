# -*- coding: utf-8 -*-
import random

import torch
import os
from torch.utils import data
from torchvision import transforms
from torch.autograd import Variable
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


from facedet.modelloader import facebox, utils

class WiderFaceLoader(data.Dataset):

    def __init__(self, root, split='train', img_size=(1024, 1024), transform=[transforms.ToTensor()], boxcoder=None):
        """
        :param root:
        :param split:
        :param img_size:
        :param transform:
        :param boxcoder:
        """
        self.boxcoder = boxcoder  # type: facebox.FaceBoxCoder
        self.root = root
        self.split = split
        self.img_size = img_size
        self.small_threshold = 10. / self.img_size[0]  # 人脸小的忽略
        self.transform = transform

        self.boxes = []
        self.labels = []
        self.files = []
        self.bbox_fname = None

        if self.split == 'train':
            self.bbox_fname = os.path.join(self.root, 'wider_face_split/wider_face_train_bbx_gt.txt')
        elif self.split == 'test':
            self.bbox_fname = os.path.join(self.root, 'wider_face_split/wider_face_test_filelist.txt')

        with open(self.bbox_fname) as bbox_fp:
            bbox_lines = bbox_fp.readlines()

        # print(bbox_lines)
        bbox_lines_num = len(bbox_lines)
        for bbox_lines_id in range(bbox_lines_num):
            box = []
            label = []
            bbox_line = bbox_lines[bbox_lines_id].strip()

            if 'jpg' not in bbox_line:
                continue

            image_name = bbox_line
            if self.split == 'train':
                image_name = os.path.join(self.root, 'WIDER_train/images',image_name)
            elif self.split == 'test':
                image_name = os.path.join(self.root, 'WIDER_test/images',image_name)

            if not os.path.isfile(image_name):
                # 不存在文件
                continue
            else:
                pass
                # # 造成加载文件较慢
                # img = cv2.imread(image_name)
                # if img is None:
                #     continue
            self.files.append(image_name)

            face_num = int(bbox_lines[bbox_lines_id+1].strip())  # 图片中有多少人脸

            for face_id in range(face_num):
                bbox_line = bbox_lines[bbox_lines_id+2+face_id].strip()
                bbox_line_split = bbox_line.split(' ')
                x = float(bbox_line_split[0])
                y = float(bbox_line_split[1])
                w = float(bbox_line_split[2])
                h = float(bbox_line_split[3])
                box.append([x, y, x+w, y+h])
                label.append(1)
            self.boxes.append(box)
            self.labels.append(label)
        # print('self.boxes:', self.boxes)
        # print('len(self.boxes):', len(self.boxes))

    def __getitem__(self, index):
        loc_target, conf_target = [], []
        img_path = self.files[index]
        # img_path = '/Users/cgf/GitHub/Quick/facedet/obama.jpg'
        # print('img_path:{}'.format(img_path))
        img = cv2.imread(img_path)


        boxes = self.boxes[index]
        boxes = torch.from_numpy(np.array(boxes, dtype=np.float32))

        labels = self.labels[index]
        labels = torch.from_numpy(np.array(labels, dtype=np.long))

        boxes = boxes.clone()
        labels = labels.clone()

        if self.split == 'train':
            img, boxes, labels = self.random_crop(img, boxes, labels)
            img = utils.random_bright(img)
            img, boxes = utils.random_flip(img, boxes)

        img_h, img_w, img_c = img.shape
        boxes /= torch.Tensor([img_h, img_w, img_h, img_w]).expand_as(boxes)

        img = cv2.resize(img, self.img_size)

        for transform_item in self.transform:
            img = transform_item(img)  # ToTensor将HxWxC转换为CxHxW


        if self.boxcoder is not None:
            loc_target, conf_target = self.boxcoder.encode(boxes, labels)

        return img, loc_target, conf_target

    def __len__(self):
        return len(self.files)

    def random_getimg(self):
        idx = random.randrange(0, self.__len__())
        fname = self.files[idx]
        img = cv2.imread(fname)

        boxes = self.boxes[idx]
        boxes = torch.from_numpy(np.array(boxes, dtype=np.float32))

        labels = self.labels[idx]
        labels = torch.from_numpy(np.array(labels, dtype=np.long))

        return img, boxes, labels

    def random_crop(self, im, boxes, labels):
        imh, imw, _ = im.shape
        short_size = min(imw, imh)
        while True:
            mode = random.choice([None, 0.3, 0.5, 0.7, 0.9])
            if mode is None:
                boxes_uniform = boxes / torch.Tensor([imw, imh, imw, imh]).expand_as(boxes)
                boxwh = boxes_uniform[:, 2:] - boxes_uniform[:, :2]
                mask = (boxwh[:, 0] > self.small_threshold) & (boxwh[:, 1] > self.small_threshold)
                if not mask.any():
                    # 所有的bounding box都小于阈值
                    print('default image have none box bigger than small_threshold')
                    im, boxes, labels = self.random_getimg()
                    imh, imw, _ = im.shape
                    short_size = min(imw, imh)
                    continue
                selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))
                selected_labels = labels.index_select(0, mask.nonzero().squeeze(1))
                return im, selected_boxes, selected_labels

            for _ in range(10):
                # 随机选择较小的宽度，随机选择ROI
                w = random.randrange(int(0.3 * short_size), short_size)
                h = w

                x = random.randrange(imw - w)
                y = random.randrange(imh - h)
                roi = torch.Tensor([[x, y, x + w, y + h]])

                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                roi2 = roi.expand(len(center), 4)
                mask = (center > roi2[:, :2]) & (center < roi2[:, 2:])
                mask = mask[:, 0] & mask[:, 1]
                # 其中ROI必须存在标注为目标的中心
                if not mask.any():
                    continue

                selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))
                img = im[y:y + h, x:x + w, :]
                # crop以后坐标系统中的bounding box都进行了变化
                selected_boxes[:, 0].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:, 1].add_(-y).clamp_(min=0, max=h)
                selected_boxes[:, 2].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:, 3].add_(-y).clamp_(min=0, max=h)
                # print('croped')

                boxes_uniform = selected_boxes / torch.Tensor([w, h, w, h]).expand_as(selected_boxes)
                boxwh = boxes_uniform[:, 2:] - boxes_uniform[:, :2]
                mask = (boxwh[:, 0] > self.small_threshold) & (boxwh[:, 1] > self.small_threshold)

                # after crop图像中的人脸bounding box仍然太小
                if not mask.any():
                    print('crop image have none box bigger than small_threshold')
                    im, boxes, labels = self.random_getimg()
                    imh, imw, _ = im.shape
                    short_size = min(imw, imh)
                    continue
                selected_boxes_selected = selected_boxes.index_select(0, mask.nonzero().squeeze(1))
                selected_labels = labels.index_select(0, mask.nonzero().squeeze(1))
                return img, selected_boxes_selected, selected_labels



if __name__ == '__main__':
    root = os.path.expanduser('~/Data/WIDER')
    train_dataset = WiderFaceLoader(root=root)
    train_dataloader = data.DataLoader(train_dataset, batch_size=4)
    for train_id, (images, _, _) in enumerate(train_dataloader):
        pass
        # print('images.shape:', images.shape)
        # print('box.shape:', box.shape)
        image_np = images.numpy()[0, :, :, :]
        image_np = image_np.transpose([1, 2, 0])

        # for box_item in box:
        #     print('box_item:', box_item)
        #     x1 = int(box_item[0])
        #     y1 = int(box_item[1])
        #     x2 = int(box_item[2])
        #     y2 = int(box_item[3])
        #     cv2.rectangle(image_np, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0))
        # cv2.imshow('image_np', image_np)
        # cv2.waitKey()
