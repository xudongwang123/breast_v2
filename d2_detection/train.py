#coding=utf-8

import os
import math
import torch
import argparse
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as patches
import sys
sys.path.append('../')

# from darknet import Darknet
from yolo_v3 import Darknet
from dataload import Dataset2D, Dataset2D_breast
from cfg import *
from utils.utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=136, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate of train')
parser.add_argument('--img_size', type=int, default=512, help='size of each image dimension')

parser.add_argument('--model_config_path', type=str, default='../cfg/yolov3_breast.cfg', help='path to model config file')
parser.add_argument('--weight', type=str, default='', help='path to weight file')
parser.add_argument('--data_config_path', type=str, default='../cfg/voc.data', help='path to dataset configure')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')

parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
parser.add_argument('--cuda_device', type=str, default='0', help='whether to use cuda if avaliable')
parser.add_argument('--num_workers', type=int, default=1, help='number of threads to use during batch generation')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint_breast', help='directory where model checkpoints are saved')
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")

opt = parser.parse_args()

print opt
cuda = torch.cuda.is_available() and opt.use_cuda

#获取数据配置信息
data_config = parse_data_cfg(opt.data_config_path)
train_path = data_config['train']

#获取网络训练超参
hyperparams = parse_model_cfg(opt.model_config_path)[0]
momentum = float(hyperparams['momentum'])
decay = float(hyperparams['decay'])

#初始化模型
model = Darknet(opt.model_config_path)
model.apply(weights_init_normal)
# model.save_weights('checkpoint/0.weights')
# model.load_weights(opt.weight)

if cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda_device
    torch.cuda.set_device(int(opt.cuda_device))
    model = model.cuda()
model.train()


file_2d = '/home/wxd/dataset/DATASET_BREAST'
dataloader = DataLoader(Dataset2D_breast(file_2d, train='train', net_shape=(opt.img_size, opt.img_size)), batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate/opt.batch_size, momentum=momentum, dampening=0, weight_decay=decay*opt.batch_size)
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def adjust_learning_rate(optimizer, epoch):
    lr = opt.learning_rate / opt.batch_size * (math.pow(0.1, epoch//30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def patch(coord, color):
    x, y, w, h = coord
    rect = patches.Rectangle((x-w/2, y-h/2), w, h, linewidth=1, edgecolor=color, facecolor='none')
    plt.gca().add_patch(rect)


def show(imgs, targets, net_shape):
    print imgs.shape, targets.shape
    for img, target in zip(imgs, targets):
        print img.shape, target.shape, img.max(), img.min()
        for label in target:
            if label.sum() == 0:
                break
            label = [label[0], int(label[1] * net_shape), int(label[2] * net_shape), int(label[3] * net_shape),
                     int(label[4] * net_shape)]
            patch(label[1:], 'r')
        img = np.transpose(img, (1, 2, 0)) * 255
        img = np.uint8(img)
        plt.imshow(img)
        plt.show()


def show_gray(imgs, targets, net_shape):
    print imgs.shape, targets.shape
    for img, target in zip(imgs, targets):
        print img.shape, target.shape, img.max(), img.min()
        for label in target:
            if label.sum() == 0:
                break
            label = [int(label[1] * net_shape), int(label[2] * net_shape), int(label[3] * net_shape),
                     int(label[4] * net_shape)]
            patch(label, 'r')
        img = img[0] * 255
        img = np.uint8(img)
        plt.imshow(img, 'gray')
        plt.show()


def train():
    for epoch in range(1, opt.epoch):
        # adjust_learning_rate(optimizer, epoch)
        # dataloader = DataLoader(
        #     Dataset2D(train_path, train=True, net_shape=(opt.img_size, opt.img_size), shuffle=True, seen=model.seen,
        #               batch_size=opt.batch_size),
        #     batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        # print optimizer
        for i, (imgs, targets, net_shape) in enumerate(tqdm(dataloader)):
            # show(imgs, targets, float(net_shape[0]))
            # show_gray(imgs, targets, float(net_shape[0]))
            print model.seen
            model.modify_shape(float(net_shape[0]))
            imgs = Variable(imgs.type(FloatTensor))
            targets = Variable(targets.type(FloatTensor), requires_grad=False)
            optimizer.zero_grad()
            loss = model(imgs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10000)
            optimizer.step()

            model.seen += imgs.size(0)
            print(
                    "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f]"
                    % (
                        epoch,
                        opt.epoch,
                        i,
                        len(dataloader),
                        model.losses["x"],
                        model.losses["y"],
                        model.losses["w"],
                        model.losses["h"],
                        model.losses["conf"],
                        model.losses["cls"],
                        loss
                    )
            )
        if epoch % opt.checkpoint_interval == 0:
            if not os.path.exists(opt.checkpoint_dir):
                os.makedirs(opt.checkpoint_dir)
            model.save_weights('{}/{}.weights'.format(opt.checkpoint_dir, epoch))


if __name__ == '__main__':
    train()
