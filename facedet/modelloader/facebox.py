# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import itertools
import numpy as np

from . import utils


class FaceBoxCoder:
    def __init__(self, facebox_model):
        """

        :type facebox_model: FaceBox
        """
        self.steps = facebox_model.steps
        self.fm_sizes = facebox_model.fm_sizes
        self.fm_num = len(self.fm_sizes)
        self.aspect_ratios = facebox_model.aspect_ratios
        self.box_sizes = facebox_model.box_sizes
        self.density = facebox_model.density
        self.variances = (0.1, 0.2)
        self.default_boxes = self._get_default_boxes()

    def _get_default_boxes(self):
        """
        :return: boxes: (#boxes, 4), 4 is for (cx, cy, h, w) box format
        """
        boxes = []
        for fm_id, fm_size in enumerate(self.fm_sizes):
            for h, w in itertools.product(range(fm_size), repeat=2):
                # print('(h,w):({},{})'.format(h, w))
                cx = (w + 0.5) * self.steps[fm_id]  # steps recover the center to the origin map
                cy = (h + 0.5) * self.steps[fm_id]  # steps recover the center to the origin map
                # print('(cx,cy):({},{})'.format(cx, cy))

                s = self.box_sizes[fm_id]

                # aspect_ratio just save 2, 3 and append 1/2, 1/3
                for aspect_ratio_id, aspect_ratio in enumerate(self.aspect_ratios[fm_id]):
                    if fm_id == 0:
                        for dx, dy in itertools.product(self.density[aspect_ratio_id], repeat=2):
                            boxes.append((cx + dx / 8. *s *aspect_ratio, cy + dy / 8. *s *aspect_ratio , s * aspect_ratio, s * aspect_ratio))  # boxes append (cx, cy, h, w)
                    else:
                        boxes.append((cx, cy, s * aspect_ratio, s * aspect_ratio))  # boxes append (cx, cy, h, w)

        return torch.Tensor(boxes)

    def encode(self, boxes, labels):
        default_boxes = self.default_boxes
        default_boxes_xyxy = utils.change_box_format(default_boxes, 'xywh2xyxy')
        num_obj = boxes.size(0)  # 人脸个数

        iou = utils.box_iou(boxes, default_boxes_xyxy)
        max_iou, max_iou_index = iou.max(1)
        iou, max_index = iou.max(0)
        max_index.squeeze_(0)
        iou.squeeze_(0)

        max_index[max_iou_index] = torch.LongTensor(range(num_obj))
        boxes = boxes[max_index]
        boxes_xywh = utils.change_box_format(boxes, 'xyxy2xywh')

        # iou_np = iou.numpy()
        # boxes_np = boxes.numpy()

        cxcy = (boxes_xywh[:, :2] - default_boxes[:, :2]) / default_boxes[:, 2:] / self.variances[0]
        wh = torch.log(boxes_xywh[:, 2:] / default_boxes[:, 2:]) / self.variances[1]

        inf_flag = wh.abs() > 10000  # 防止其中值过大
        # print('inf_flag.long().sum():{}'.format(inf_flag.long().sum()))

        loc = torch.cat([cxcy, wh], 1)

        conf = labels[max_index]
        conf[iou < 0.35] = 0  # 这里设置thread为0.35
        conf[max_iou_index] = 1

        # np.save('/tmp/conf1.npz', conf.numpy())
        # np.save('/tmp/loc1.npz', loc.numpy())

        return loc, conf


    def decode(self, loc, conf):
        """
        :param loc: 21842*4，cx cy w h
        :param conf: 21842*2
        :return:
        """
        cxcy = loc[:, :2] * self.variances[0] * self.default_boxes[:, 2:] + self.default_boxes[:, :2]
        wh = torch.exp(loc[:, 2:] * self.variances[1]) * self.default_boxes[:, 2:]
        boxes = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1)  # [21824,4]

        max_conf, labels = conf.max(1)  # [21842,1]
        print('labels', labels.long().sum())

        # 无人脸
        if labels.long().sum() is 0:
            sconf, slabel = conf.max(0)
            max_conf[slabel[0:5]] = sconf[0:5]
            labels[slabel[0:5]] = 1

        ids = labels.nonzero().squeeze(1)
        keep = utils.box_nms(boxes[ids], max_conf[ids], threshold=0.5)  # 非极大值抑制存储的人脸

        return boxes[ids][keep], labels[ids][keep], max_conf[ids][keep]


class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), 1)

class StrideConv(nn.Module):
    """
    StrideConv：H，W根据stride进行下采样，H*W->(H/stride)*(W/stride)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param dilation:
        :param groups:
        :param bias:
        """
        super(StrideConv, self).__init__()
        padding = (kernel_size-1)//2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(x)

class StridePool(nn.Module):
    """
    StridePool：H，W根据stride进行下采样，H*W->(H/stride)*(W/stride)
    """
    def __init__(self, kernel_size, stride=None):
        super(StridePool, self).__init__()
        padding = (kernel_size-1)//2
        self.pool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        return self.pool(x)

class Inception(nn.Module):
    """
    Inception：输入128，32，32，输出也是128，32，32
    """
    def __init__(self):
        super(Inception, self).__init__()
        self.conv1 = StrideConv(in_channels=128, out_channels=32, kernel_size=1)

        self.pool1 = StridePool(kernel_size=3, stride=1)
        self.conv2 = StrideConv(in_channels=128, out_channels=32, kernel_size=1)

        self.conv3 = StrideConv(in_channels=128, out_channels=24, kernel_size=1)
        self.conv4 = StrideConv(in_channels=24, out_channels=32, kernel_size=3)

        self.conv5 = StrideConv(in_channels=128, out_channels=24, kernel_size=1)
        self.conv6 = StrideConv(in_channels=24, out_channels=32, kernel_size=3)
        self.conv7 = StrideConv(in_channels=32, out_channels=32, kernel_size=3)

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.pool1(x)
        x2 = self.conv2(x2)

        x3 = self.conv3(x)
        x3 = self.conv4(x3)

        x4 = self.conv5(x)
        x4 = self.conv6(x4)
        x4 = self.conv7(x4)

        return torch.cat((x1, x2, x3, x4), 1)


# class MultiBoxLayer(nn.Module):
#     def __init__(self):
#         super(MultiBoxLayer, self).__init__()
#         self.loc_conv_1 = StrideConv(in_channels=128, out_channels=21*4, kernel_size=3)
#         self.conf_conv_1 = StrideConv(in_channels=128, out_channels=21*2, kernel_size=3)
#
#         self.loc_conv_2 = StrideConv(in_channels=256, out_channels=1*4, kernel_size=3)
#         self.conf_conv_2 = StrideConv(in_channels=256, out_channels=1*2, kernel_size=3)
#
#         self.loc_conv_3 = StrideConv(in_channels=256, out_channels=1*4, kernel_size=3)
#         self.conf_conv_3 = StrideConv(in_channels=256, out_channels=1*2, kernel_size=3)
#
#     def forward(self, x1, x2, x3):
#         locs = []
#         confs = []
#
#         # 第一层
#         batch_size = x1.size(0)
#         loc = self.loc_conv_1(x1)
#         loc = loc.permute(0,2,3,1).contiguous()
#         loc = loc.view(batch_size, -1, 4)
#         locs.append(loc)
#
#         conf = self.conf_conv_1(x1)
#         conf = conf.permute(0,2,3,1).contiguous()
#         conf = conf.view(batch_size, -1, 2)
#         confs.append(conf)
#
#         # 第二层
#         batch_size = x2.size(0)
#         loc = self.loc_conv_2(x2)
#         loc = loc.permute(0, 2, 3, 1).contiguous()
#         loc = loc.view(batch_size, -1, 4)
#         locs.append(loc)
#
#         conf = self.conf_conv_2(x2)
#         conf = conf.permute(0, 2, 3, 1).contiguous()
#         conf = conf.view(batch_size, -1, 2)
#         confs.append(conf)
#
#         # 第三层
#         batch_size = x3.size(0)
#         loc = self.loc_conv_3(x3)
#         loc = loc.permute(0, 2, 3, 1).contiguous()
#         loc = loc.view(batch_size, -1, 4)
#         locs.append(loc)
#
#         conf = self.conf_conv_3(x3)
#         conf = conf.permute(0, 2, 3, 1).contiguous()
#         conf = conf.view(batch_size, -1, 2)
#         confs.append(conf)
#
#         locs = torch.cat(locs, 1)
#         confs = torch.cat(confs, 1)
#
#         return locs, confs




class FaceBoxExtractor(nn.Module):
    def __init__(self):
        super(FaceBoxExtractor, self).__init__()
        self.conv1 = StrideConv(in_channels=3, out_channels=24, kernel_size=7, stride=4)
        self.crelu1 = CReLU()
        self.pool1 = StridePool(kernel_size=3, stride=2)

        self.conv2 = StrideConv(in_channels=24*2, out_channels=64, kernel_size=5, stride=2)
        self.crelu2 = CReLU()
        self.pool2 = StridePool(kernel_size=3, stride=2)

        self.inception1 = Inception()
        self.inception2 = Inception()
        self.inception3 = Inception()

        self.conv3_1 = StrideConv(in_channels=128, out_channels=128, kernel_size=1, stride=1)
        self.conv3_2 = StrideConv(in_channels=128, out_channels=256, kernel_size=3, stride=2)

        self.conv4_1 = StrideConv(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.conv4_2 = StrideConv(in_channels=128, out_channels=256, kernel_size=3, stride=2)

        # self.multibox = MultiBoxLayer()

    def forward(self, x):
        xs = []
        x = self.conv1(x)
        x = self.crelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.crelu2(x)
        x = self.pool2(x)

        x = self.inception1(x)
        x = self.inception2(x)
        x1 = self.inception3(x)
        xs.append(x1)
        x = x1
        # print('x1.size():', x1.size())

        x = self.conv3_1(x)
        x2 = self.conv3_2(x)
        xs.append(x2)
        x = x2
        # print('x2.size():', x2.size())

        x = self.conv4_1(x)
        x3 = self.conv4_2(x)
        xs.append(x3)
        x = x3
        # print('x3.size():', x3.size())


        return xs
        # locs, confs = self.multibox(x1, x2, x3)

        # return locs, confs


class FaceBox(nn.Module):
    steps = (32 / 1024.0, 64 / 1024., 128 / 1024.) # steps for recover to the origin image size
    fm_sizes = (32, 16, 8)  # feature map size
    aspect_ratios = ((1, 2, 4), (1,), (1,))  # aspect ratio
    box_sizes = (32 / 1024., 256 / 1024., 512 / 1024.)  # box size
    density = ((-3, -1, 1, 3), (-1, 1), (0,))

    def __init__(self, num_classes=2):
        super(FaceBox, self).__init__()
        self.num_classes = num_classes  # 人脸和非人脸，所以共有两类目标需要检测

        self.num_anchors = (21, 1, 1)
        self.in_channels = (128, 256, 256)

        self.extractor = FaceBoxExtractor()

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.loc_layers.append(StrideConv(self.in_channels[i], self.num_anchors[i] * 4, kernel_size=3))
            self.conf_layers.append(StrideConv(self.in_channels[i], self.num_anchors[i] * self.num_classes, kernel_size=3))

    def forward(self, x):
        loc_preds = []
        conf_preds = []

        xs = self.extractor(x)

        for i, x in enumerate(xs):
            loc_pred = self.loc_layers[i](x)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_pred = loc_pred.view(loc_pred.size(0), -1, 4)
            loc_preds.append(loc_pred)

            conf_pred = self.conf_layers[i](x)
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous()
            conf_pred = conf_pred.view(conf_pred.size(0), -1, 2)
            conf_preds.append(conf_pred)

        loc_preds = torch.cat(loc_preds, 1)
        conf_preds = torch.cat(conf_preds, 1)
        return loc_preds, conf_preds


class FaceBoxLoss(nn.Module):
    def __init__(self, num_classes=2):
        super(FaceBoxLoss, self).__init__()
        self.num_classes = num_classes

    def cross_entropy_loss(self, x, y):
        x = x.detach()
        y = y.detach()
        xmax = x.data.max()
        # xmax = xmax.detach()
        log_sum_exp = torch.log(torch.sum(torch.exp(x - xmax), 1, keepdim=True)) + xmax
        return log_sum_exp - x.gather(1, y.view(-1, 1))

    def hard_negative_mining(self, conf_loss, pos):
        """
        conf_loss [N*21482,]
        pos [N,21482]
        return negative indice
        """
        batch_size, num_boxes = pos.size()
        conf_loss[pos.view(-1, 1)] = 0  # 去掉正样本,the rest are neg conf_loss
        conf_loss = conf_loss.view(batch_size, -1)

        _, idx = conf_loss.sort(1, descending=True)
        _, rank = idx.sort(1)

        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(3 * num_pos, max=num_boxes - 1)

        neg = rank < num_neg.expand_as(rank)
        return neg

    def forward(self, loc_preds, loc_targets, conf_preds, conf_targets):
        """
        loc_preds[batch,21842,4]
        loc_targets[batch,21842,4]
        conf_preds[batch,21842,2]
        conf_targets[batch,21842]
        """
        batch_size, num_boxes, _ = loc_preds.size()
        pos = conf_targets > 0
        num_pos = pos.float().sum(1, keepdim=True)
        num_matched_boxes = pos.data.long().sum()
        if num_matched_boxes == 0:
            return Variable(torch.Tensor([0]), requires_grad=True)
        pos_mask1 = pos.unsqueeze(2).expand_as(loc_preds)
        pos_loc_preds = loc_preds[pos_mask1].view(-1, 4)
        pos_loc_targets = loc_targets[pos_mask1].view(-1, 4)
        loc_loss = F.smooth_l1_loss(pos_loc_preds, pos_loc_targets, size_average=False)

        conf_loss = self.cross_entropy_loss(conf_preds.view(-1, self.num_classes), conf_targets.view(-1, 1))
        neg = self.hard_negative_mining(conf_loss, pos)
        pos_mask = pos.unsqueeze(2).expand_as(conf_preds)
        neg_mask = neg.unsqueeze(2).expand_as(conf_preds)
        mask = (pos_mask + neg_mask).gt(0)
        pos_and_neg = (pos + neg).gt(0)
        preds = conf_preds[mask].view(-1, self.num_classes)
        targets = conf_targets[pos_and_neg]
        conf_loss = F.cross_entropy(preds, targets, size_average=False)
        N = num_pos.data.sum()
        loc_loss /= N
        conf_loss /= N

        print('loc_loss:%f conf_loss:%f, pos_num:%d' % (loc_loss.data[0], conf_loss.data[0], N))
        return loc_loss + conf_loss


if __name__ == '__main__':
    C, H, W = (3, 1024, 1024)
    x = Variable(torch.randn(1, C, H, W))
    net = FaceBox()
    loc_preds, conf_preds = net(x)
    print('loc_preds.size():', loc_preds.size())
    print('conf_preds.size():', conf_preds.size())
