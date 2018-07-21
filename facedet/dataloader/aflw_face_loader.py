# -*- coding: utf-8 -*-
import random
import glob
import torch
import os
from torch.utils import data
from torchvision import transforms
from torch.autograd import Variable
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# from sqlalchemy import create_engine, MetaData


from facedet.modelloader import facebox, utils

class AFLWFaceLoader(data.Dataset):

    def __init__(self, root, split='train', img_size=(1024, 1024), transform=[transforms.ToTensor()], boxcoder=None, vis=False):
        """
        :param root:
        :param split:
        :param img_size:
        :param transform:
        :param boxcoder:
        :param vis: just for vis the data
        """
        self.boxcoder = boxcoder  # type: facebox.FaceBoxCoder
        self.root = root
        self.split = split
        self.img_size = img_size
        self.small_threshold = 10. / self.img_size[0]  # 人脸小的忽略
        self.transform = transform
        self.vis = vis

        self.boxes = []
        self.labels = []
        self.files = []

        # data_engine = create_engine('sqlite:///{}/aflw.sqlite'.format(root), echo=True)

        if self.split == 'train':
            self.bbox_fname = os.path.join(self.root, 'alfw.txt')
        elif self.split == 'test':
            self.bbox_fname = os.path.join(self.root, 'alfw.txt')

        with open(self.bbox_fname) as bbox_fp:
            bbox_lines = bbox_fp.readlines()

        # print(bbox_lines)
        bbox_lines_num = len(bbox_lines)
        # for bbox_lines_id in range(bbox_lines_num):
        bbox_lines_id = 0
        # image00035.jpg \t 62 \t 64 \t 348 \t 348
        while bbox_lines_id < bbox_lines_num:
            box = []
            label = []
            bbox_line = bbox_lines[bbox_lines_id].strip()

            image_name = bbox_line.split('\t')[0]

            if self.split == 'train':
                image_name = os.path.join(self.root, 'flickr', image_name)
            elif self.split == 'test':
                image_name = os.path.join(self.root, 'flickr', image_name)

            if not os.path.isfile(image_name):
                # 不存在文件
                continue
            else:
                pass
                # # 造成加载文件较慢
                # img = cv2.imread(image_name)
                # if img is None:
                #     continue
            if image_name not in self.files:
                self.files.append(image_name)

            image_id = self.files.index(image_name)
            # print('image_id:', image_id)

            face_num = 1  # 图片中有多少人脸，该数据集中一行标注一个人脸，可能一个文件有多行

            for face_id in range(face_num):
                # major_axis_radius minor_axis_radius angle center_x center_y 1
                bbox_line_split = bbox_line.split('\t')[1:]
                x = float(bbox_line_split[0])
                y = float(bbox_line_split[1])
                w = float(bbox_line_split[2])
                h = float(bbox_line_split[3])
                if image_id < len(self.boxes):
                    self.boxes[image_id].append([x, y, x + w, y + h])
                    self.labels[image_id].append([1])
                else:
                    # 填充新的boxes和labels的位置
                    self.boxes.append([])
                    self.labels.append([])

                    self.boxes[image_id].append([x, y, x + w, y + h])
                    self.labels[image_id].append([1])


            bbox_lines_id = bbox_lines_id + 1
        # print('self.files:', self.files)
        # print('self.boxes:', self.boxes)
        # print('len(self.boxes):', len(self.boxes))

    def __getitem__(self, index):
        loc_target, conf_target = [], []
        img_path = self.files[index]
        # print('img_path:{}'.format(img_path))
        img = cv2.imread(img_path)


        boxes = self.boxes[index]
        boxes = torch.from_numpy(np.array(boxes, dtype=np.float32))

        labels = self.labels[index]
        labels = torch.from_numpy(np.array(labels, dtype=np.long))

        boxes = boxes.clone()
        labels = labels.clone()

        if not self.vis:
            if self.split == 'train':
                pass
                # img, boxes, labels = self.random_crop(img, boxes, labels)
                # img = utils.random_bright(img)
                # img, boxes = utils.random_flip(img, boxes)

            img = cv2.resize(img, self.img_size)

            for transform_item in self.transform:
                img = transform_item(img)  # ToTensor将HxWxC转换为CxHxW

            boxes /= torch.Tensor([1024.0, 1024.0, 1024.0, 1024.0]).expand_as(boxes)

            if self.boxcoder is not None:
                loc_target, conf_target = self.boxcoder.encode(boxes, labels)

            return img, loc_target, conf_target
        else:
            return img, boxes, labels

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    root = os.path.expanduser('~/Data/AFLW/aflw/data')
    train_dataset = AFLWFaceLoader(root=root, split='test', vis=True)
    train_dataloader = data.DataLoader(train_dataset, batch_size=1)
    for train_id, (images, boxes, labels) in enumerate(train_dataloader):
        pass
        # print('images.shape:', images.shape)
        # print('boxes.shape:', boxes.shape)
        image_np = images.numpy()[0, :, :, :]

        # vis的情况下直接返回图像HxWxC
        image_h, image_w, _ = image_np.shape
        # print('image_h:{}'.format(image_h))
        # print('image_w:{}'.format(image_w))


        boxes_np = boxes.numpy()[0, :, :]

        for box_item in boxes_np:
            # print('box_item:', box_item)
            x1 = int(box_item[0])
            y1 = int(box_item[1])
            x2 = int(box_item[2])
            y2 = int(box_item[3])
            if x1<0 or y1<0 or x2<0 or y2<0:
                # print('(x1,y1)->(x2,y2):({},{})->({},{})'.format(x1, y1, x2, y2))
                pass
            else:
                pass
            cv2.rectangle(image_np, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0))
        cv2.imshow('image_np', image_np)
        cv2.waitKey(int(0.0*1000))
