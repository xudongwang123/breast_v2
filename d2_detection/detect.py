#coding=utf-8

import ast
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

from yolo_v3 import Darknet
# from darknet import Darknet
from dataload import Dataset2D, Dataset2D_breast
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



def detect():
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
    # dataloader = DataLoader(
    #     Dataset2D(data_path, train=False, net_shape=(opt.img_size, opt.img_size), shuffle=False, seen=model.seen),
    #     batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    file_2d = '/home/wxd/dataset/DATASET_BREAST'
    dataloader = DataLoader(Dataset2D_breast(file_2d, train='valid', net_shape=(opt.img_size, opt.img_size)),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    classes = load_classes(opt.class_path)

    for i, (input_imgs, targets, _) in enumerate(dataloader):
        # show(input_imgs, targets)
        input_imgs = input_imgs.type(FloatTensor)
        if cuda:
            input_imgs = input_imgs.cuda()
        input_imgs = Variable(input_imgs)

        t1 = time.time()
        output = model(input_imgs)

        all_boxes = non_max_suppresion(output, opt.conf_thres, opt.nms_thres)
        t2 = time.time()
        print t2-t1, 's'

        for j in range(len(input_imgs)):
            plt.figure()
            # img = np.uint8(np.transpose(np.array(input_imgs[j])*255, (1, 2, 0)))
            # plt.imshow(img)
            img = input_imgs[j][0]*255
            plt.imshow(img, cmap='gray')

            for x, y, w, h, conf, class_pred, class_conf in all_boxes[j]:
                rect = patches.Rectangle((x - w / 2, y - h / 2), w, h, linewidth=1, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
                # plt.text(x - w / 2, y - h / 2, s=classes[int(class_pred)], color='white', verticalalignment='top',
                #          bbox={'color': 'r', 'pad': 0})
            #plt.savefig('{0}/{1}.png'.format(opt.save_path, j), bbox_inches='tight', pad_inches=0.0)


            for cls, x, y, w, h in targets[j]:
                if x == 0:
                    break
                x, y, w, h = x*opt.img_size, y*opt.img_size, w*opt.img_size, h*opt.img_size
                rect = patches.Rectangle((x - w / 2, y - h / 2), w, h, linewidth=1, edgecolor='g', facecolor='none')
                plt.gca().add_patch(rect)
            # plt.savefig('{0}/{1}.png'.format(opt.save_path, j), bbox_inches='tight', pad_inches=0.0)

            plt.show()


if __name__ == '__main__':
    detect()





