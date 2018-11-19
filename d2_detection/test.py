#coding=utf-8
#coding=utf-8

import os
import time
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys
sys.path.append('../')

import ast
# from darknet import Darknet
from yolo_v3 import Darknet
from dataload import Dataset2D, Dataset2D_breast, Dataset2D_breast_old
from utils.utils import *
from cfg import parse_data_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='../data/samples', help='path to dataset')
parser.add_argument('--config_path', type=str, default='../cfg/yolov3_breast.cfg', help='path to model config file')
parser.add_argument('--weight', type=str, default='', help='path to weight file')
parser.add_argument('--class_path', type=str, default='../cfg/voc.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
parser.add_argument('--save_path', type=str, default='./output', help='path to save results')
parser.add_argument('--num_workers', type=int, default=1, help='number of threads to use during batch generation')
parser.add_argument('--cuda_device', type=str, default='0', help='whether to use cuda if avaliable')
parser.add_argument('--train', type=ast.literal_eval, default=True, help='whether to use train path or validate path')


parser.add_argument('--data_config_path', type=str, default='../cfg/voc.data', help='path to dataset configure')
opt = parser.parse_args()
print opt

data_config = parse_data_cfg(opt.data_config_path)
train_path = data_config['train']
valid_path = data_config['valid']
if opt.train:
    data_path = train_path
else:
    data_path = valid_path

voc2coco_label = {0:4, 1:1, 2:14, 3:8, 4:39, 5:5, 6:2, 7:15, 8:56, 9:19, 10:60, 11:16, 12:17, 13:3, 14:0, 15:58, 16:18, 17:57, 18:6, 19:62}

def patch(coord, color):
    x, y, w, h = coord
    rect = patches.Rectangle((x-w/2, y-h/2), w, h, linewidth=1, edgecolor=color, facecolor='none')
    plt.gca().add_patch(rect)


def show(imgs, targets):
    print imgs.shape, targets.shape
    for img, target in zip(imgs, targets):
        print img.shape, target.shape, img.max(), img.min()
        for label in target:
            if label.sum() == 0:
                break
            label = [label[0], int(label[1] * opt.img_size), int(label[2] * opt.img_size), int(label[3] * opt.img_size),
                     int(label[4] * opt.img_size)]
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


def test():
    cuda = torch.cuda.is_available() and opt.use_cuda
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    model = Darknet(opt.config_path)
    if opt.weight:
        model.load_weights(opt.weight)
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda_device
        torch.cuda.set_device(int(opt.cuda_device))
        model = model.cuda()
    model.eval()
    file_2d = '/home/wxd/dataset/DATASET_BREAST'
    dataloader = DataLoader(Dataset2D_breast(file_2d, train='test', net_shape=(opt.img_size, opt.img_size)),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    #
    # file_2d = '/home/wxd/dataset/DATASET_BREAST/preprocess_2d.txt'
    # dataloader = DataLoader(Dataset2D_breast_old(file_2d, train=False, net_shape=(opt.img_size, opt.img_size)),
    #                         batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # dataloader = DataLoader(
    #     Dataset2D(data_path, train=False, net_shape=(opt.img_size, opt.img_size), shuffle=False, seen=model.seen),
    #     batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    correct = 0.0
    GT = 1e-10
    proposal = 1e-10
    for i, (imgs, targets, _) in enumerate(dataloader):
        # show(imgs, targets)
        # show_gray(imgs, targets, opt.img_size)
        imgs = imgs.type(FloatTensor)
        targets = targets.numpy()
        if cuda:
            imgs = imgs.cuda()
        imgs = Variable(imgs)
        output = model(imgs)

        all_boxes = non_max_suppresion(output, opt.conf_thres, opt.nms_thres)

        for j in range(len(imgs)):
            proposal += len(all_boxes[j])
            flag = [0 for _ in range(len(all_boxes[j]))]    #该框是否已经为一个ground Truth负责
            for k in range(len(targets[j])):
                if targets[j][k].sum() == 0:
                    break
                GT += 1
                for z, (x, y, w, h, conf, class_pred, class_conf) in enumerate(all_boxes[j]):
                    bbox = [x, y, w, h]
                    gt = targets[j][k][1:] * opt.img_size
                    class_target = targets[j][k][0]
                    # print bbox, gt, bbox_iou(bbox, gt, x1y1x2y2=False)
                    if flag[z] == 0 and class_pred == class_target and bbox_iou(bbox, gt, x1y1x2y2=False) > 0.5:
                        correct += 1
                        flag[z] = 1
                        break
        print 'recall:{:.4f}    precision:{:.4f}'.format(correct/GT, correct/proposal)


def modify_data(num=1):
    train_path_tmp = '/home/wxd/dataset/VOCdevkit/train.txt'
    with open(train_path_tmp, 'r') as fp:
        lines = fp.readlines()[:num]
    with open(train_path_tmp.replace('train.txt', 'train_first10.txt'), 'w') as fp:
        fp.writelines(lines)


def count_data():
    label = [0 for _ in range(20)]
    train_path_tmp = '/home/wxd/dataset/VOCdevkit/train_first10.txt'
    with open(train_path_tmp, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.replace('JPEGImages', 'labels').replace('.png', '.txt').replace('.jpg', '.txt').strip()
        targets = np.loadtxt(line).reshape(-1, 5)  # [class, x, y, w, h]

        for cls, x, y, w, h in targets:
            label[int(cls)] += 1
    print label


def get_one_class(class_num):
    train_path_tmp = '/home/wxd/dataset/VOCdevkit/2007_test.txt'
    one_class = '/home/wxd/dataset/VOCdevkit/2007_test_{}.txt'.format(class_num)
    with open(train_path_tmp, 'r') as fp:
        lines = fp.readlines()

    fp = open(one_class, 'w')
    for line in lines:
        label = line.replace('JPEGImages', 'labels').replace('.png', '.txt').replace('.jpg', '.txt').strip()
        targets = np.loadtxt(label).reshape(-1, 5)  # [class, x, y, w, h]
        for cls, x, y, w, h in targets:
            if cls == class_num:
                fp.write(line)
                break
    fp.close()


if __name__ == '__main__':
    test()
    # get_mAP()
    # modify_data(500)
    # count_data()
    # get_one_class(14)





