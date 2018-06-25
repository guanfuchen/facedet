# -*- coding: utf-8 -*-
import torch
import visdom
from torch.autograd import Variable
from torch import optim
from torch.utils import data
import torch.nn.functional as F
import numpy as np
import os

from facedet.modelloader import facebox
from facedet.dataloader import wider_face_loader


def train():
    vis = visdom.Visdom()

    num_classses = 2
    net = facebox.FaceBox(num_classes=num_classses)
    if os.path.exists('weight/facebox.pt'):
        net.load_state_dict(torch.load('weight/facebox.pt', map_location=lambda storage, loc: storage))
    facebox_box_coder = facebox.FaceBoxCoder(net)

    root = os.path.expanduser('~/Data/WIDER')
    train_dataset = wider_face_loader.WiderFaceLoader(root=root, boxcoder=facebox_box_coder)
    train_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    # optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=1e-5, weight_decay=1e-4)
    criterion = facebox.FaceBoxLoss(num_classes=num_classses)

    for epoch in range(100):

        loss_epoch = 0
        loss_avg_epoch = 0
        data_count = 0

        for train_id, (images, loc_targets, conf_targets) in enumerate(train_dataloader):
            # data_count = train_id+1
            images = Variable(images)
            loc_preds, conf_preds = net(images)
            # print('loc_preds.size():{}'.format(loc_preds.size()))
            # print('conf_preds.size():{}'.format(conf_preds.size()))
            # print('loc_targets.size():{}'.format(loc_targets.size()))
            # print('conf_targets.size():{}'.format(conf_targets.size()))
            optimizer.zero_grad()
            loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)

            loss_numpy = loss.data.numpy()
            loss_numpy = np.expand_dims(loss_numpy, axis=0)

            if not np.isinf(loss_numpy.sum()):
                loss_epoch += loss_numpy
                data_count += 1
            else:
                data_count = 0
                loss_epoch = 0


            loss.backward()
            optimizer.step()

            # print('loss_numpy:', loss_numpy)
            # print('loss_epoch:', loss_epoch)
            # print('loss_numpy:{},loss_epoch:{}'.format(loss_numpy, loss_epoch))

            if not np.isinf(loss_numpy.sum()):
                win = 'loss'
                win_res = vis.line(X=np.ones(1) * train_id, Y=loss_numpy, win=win, update='append')
                if win_res != win:
                    vis.line(X=np.ones(1) * train_id, Y=loss_numpy, win=win)

            # 50个batch显示一次作为平均值
            if data_count==30:
                loss_avg_epoch = loss_epoch / (30 * 1.0)
                loss_avg_epoch = np.expand_dims(loss_avg_epoch, axis=0)
                print('loss_avg_epoch:', loss_avg_epoch)

                win = 'loss_epoch'
                win_res = vis.line(X=np.ones(1) * (epoch * 30 + train_id / 30), Y=loss_avg_epoch, win=win, update='append')
                if win_res != win:
                    vis.line(X=np.ones(1) * (epoch * 30 + train_id / 30), Y=loss_avg_epoch, win=win)

                data_count = 0
                loss_epoch = 0

        # loss_avg_epoch = loss_epoch / (data_count * 1.0)
        # print('loss_avg_epoch:', loss_avg_epoch)

        # 关闭清空一个周期的loss
        win = 'loss'
        vis.close(win)

        if not os.path.exists('weight/'):
            os.mkdir('weight')
        print('saving model ...')
        torch.save(net.state_dict(),'weight/facebox.pt')


if __name__ == '__main__':
    train()
