#coding=utf-8

import sys
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import label as Label
from skimage.measure import regionprops as Region
from torch.utils.data import Dataset, DataLoader
import random

sys.path.append('../')
from utils.augmentation import resize_fill, get_true_label, transform_data, resize, random_flip


class Dataset2D(Dataset):
    def __init__(self, list_path, train=True, net_shape=(416, 416), shuffle=False, seen=0, batch_size=8, crop=False, jitter=0.3, hue=0.1, saturation=1.5, exposure=1.5):
        with open(list_path, 'r') as fp:
            self.img_files = fp.readlines()

        if shuffle:
            random.shuffle(self.img_files)

        self.net_shape = net_shape
        self.train = train
        self.max_objects = 50
        self.seen = seen
        self.batch_sizes = batch_size * 10
        self.crop = crop
        self.jitter = jitter
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure

    def get_different_scale(self):
        if self.seen < 4000*64:
            wh = 13*32                          # 416
        elif self.seen < 8000*64:
            wh = (random.randint(0, 3) + 13)*32  # 416, 480
        elif self.seen < 12000*64:
            wh = (random.randint(0, 5) + 12)*32  # 384, ..., 544
        elif self.seen < 16000*64:
            wh = (random.randint(0, 7) + 11)*32  # 352, ..., 576
        else: # self.seen < 20000*64:
            wh = (random.randint(0, 9) + 10)*32  # 320, ..., 608
        return (wh, wh)

    def __getitem__(self, index):
        img_path = self.img_files[index].strip()
        # print img_path
        label_path = img_path.replace('JPEGImages', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')

        if self.train:
            #图片金字塔
            if index % self.batch_sizes == 0:#每64张图片选择一次图片尺寸
                self.net_shape = self.get_different_scale()
            input_img, target = transform_data(img_path, label_path, self.net_shape, self.crop, self.jitter, self.hue, self.saturation, self.exposure)
            input_img = np.transpose(np.array(input_img), (2, 0, 1)) / 255.0  # 像素值归一化

        else:
            img = Image.open(img_path).convert('RGB')
            img_w, img_h = img.size

            #按长边等比例缩放，并填充
            img = resize_fill(img, self.net_shape)
            input_img = np.transpose(np.array(img, dtype=np.float32), (2, 0, 1)) / 255.0  # 像素值归一化
            target = np.loadtxt(label_path).reshape(-1, 5)    #[class, x, y, w, h]
            target = get_true_label(target, (img_w, img_h), self.net_shape)

        #选取人这一类作为预测
        tmp = []
        for t in target:
            if t[0] == 14:
                tmp.append(t)
        target = np.array(tmp)

        if len(target) >= self.max_objects:
            target = target[:self.max_objects]
        else:
            stuff = np.zeros((self.max_objects-len(target), 5))
            target = np.concatenate((target, stuff), axis=0)

        if self.train:
            return input_img, target, self.net_shape[0]
        else:
            return input_img, target#, self.net_shape[0]

    def __len__(self):
        return len(self.img_files)


class Dataset2D_breast(Dataset):
    def __init__(self, main_dir, train='train', net_shape=(512, 512)):
        super(Dataset2D_breast, self).__init__()
        self.net_shape = net_shape
        self.train = train
        self.max_objects = 20  # 假设一张图片中最多有20个病变，主要为了batch设置
        if train == 'train':
            self.files = glob.glob(main_dir + '/train/*_clean.npy')
            #将大的GT多次复制
            big_bbox = []
            for file in self.files:
                label = np.load(file.replace('clean', 'bbox'))
                label = yxyx2xywh(label)
                for lab in label:
                    x, y, w, h = lab
                    if w*h>8000:
                        big_bbox.append(file)
                    break
            for file in big_bbox:
                for i in range(6):
                    self.files.append(file)

        elif train == 'valid':
            self.files = glob.glob(main_dir + '/val/*_clean.npy')
        elif train == 'test':
            self.files = glob.glob(main_dir + '/test/*_clean.npy')
        else:
            print 'invalid phase.'
            self.files = []

    def __getitem__(self, idx):
        img = np.load(self.files[idx])
        label = np.load(self.files[idx].replace('clean', 'bbox'))
        label = yxyx2xywh(label)
        dw, dh = img.shape[1:]

        if len(label) != 0:
            label = label.astype(np.float32)
            label[:, 0] = label[:, 0] / dw
            label[:, 1] = label[:, 1] / dh
            label[:, 2] = label[:, 2] / dw
            label[:, 3] = label[:, 3] / dh
            #add class label
            stuff = np.ones((len(label), 1))
            label = np.concatenate((stuff, label), axis=1)
        else:
            label = np.array([[0, 0, 0, 0, 0]])

        if self.train == 'train':
            img = Image.fromarray(img[0])
            img = img.resize(self.net_shape, Image.ANTIALIAS)
            flag = np.random.randint(0, 2)
            if flag == 1:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                label[:, 1] = 1 - label[:, 1]
            img = np.array(img).reshape(1, self.net_shape[0], self.net_shape[1])
        else:
            img = Image.fromarray(img[0])
            img = img.resize(self.net_shape, Image.ANTIALIAS)
            img = np.array(img).reshape(1, self.net_shape[0], self.net_shape[1])

        img = img / 255.0  # 像素值归一化

        if len(label) > self.max_objects:
            label = label[:self.max_objects]
        else:
            stuff = np.zeros((self.max_objects - len(label), 5))
            label = np.concatenate((label, stuff), axis=0)
        return img, label, self.net_shape[0]

    def __len__(self):
        return len(self.files)


def yxyx2xywh(bboxes):
    new_bboxes = []
    for bbox in bboxes:
        y1, x1, y2, x2 = bbox
        w, h = x2 - x1, y2 - y1
        x, y = x1 + w/2, y1 + h/2
        new_bboxes.append([x, y, w, h])
    return np.array(new_bboxes)

def xyxy2xywh(bboxes):
    new_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        x, y = x1 + w/2, y1 + h/2
        new_bboxes.append([x, y, w, h])
    return np.array(new_bboxes)


class Dataset2D_breast_old(Dataset):
    def __init__(self, list_path, train=True, net_shape=(512, 512)):
        super(Dataset2D_breast_old, self).__init__()
        self.net_shape = net_shape
        self.train = train
        self.max_objects = 20  # 假设一张图片中最多有20个病变，主要为了batch设置
        with open(list_path, 'r') as fp:
                files = fp.readlines()

        files.sort()
        length = len(files)

        #前80%作为训练，后20%作为测试
        if train:
            self.files = files[:4*length/5]
        else:
            self.files = files[4*length/5:]
            # false_label = ['/home/wxd/dataset/DATASET_BREAST/preprocess_3d/zhangaiyun-2016-3-17-4_clean.npy', \
            #                '/home/wxd/dataset/DATASET_BREAST/preprocess_3d/zhanghongyan-2016-01-08-4_clean.npy', \
            #                '/home/wxd/dataset/DATASET_BREAST/preprocess_3d/zhangjing-2016-11-18-4_clean.npy', \
            #                '/home/wxd/dataset/DATASET_BREAST/preprocess_3d/zhangneng-2017-11-16-4_clean.npy', \
            #                '/home/wxd/dataset/DATASET_BREAST/preprocess_3d/zhangneng-2018-2-23-4_clean.npy']
            # for file_index in self.files:
            #     file = file_index.split(' ')
            #     file, index = file[0], int(file[1])
            #     if file in false_label:
            #         self.files.remove(file_index)

    def __getitem__(self, idx):
        while True:
            file = self.files[idx]

            file = file.split(' ')
            file, index = file[0], int(file[1])

            dcm = np.load(file)[index]
            seg = np.load(file.replace('_clean', '_seg'))[index]
            # seg[seg>0] = 255
            dcm[dcm<0] = 0
            # print dcm.max(), dcm.min(), seg.max(), seg.min()

            dcm = Image.fromarray(dcm.astype(np.uint8))
            seg = Image.fromarray(seg.astype(np.uint8))
            #print file, index
            #数据增强
            #print np.sum(seg)
            if self.train:
                # dcm.show()
                # seg.show()
                dcm, seg = resize(dcm, seg, self.net_shape)
                dcm, seg = random_flip(dcm, seg)
                dcm = np.array(dcm).reshape((1, self.net_shape[0], self.net_shape[1]))
                seg = np.array(seg)
                # dcm = resize_fill(dcm, self.net_shape, mode='L', fill_color=0)
                # seg = resize_fill(seg, self.net_shape, mode='L', fill_color=0)
                # dcm = np.array(dcm).reshape((1, self.net_shape[0], self.net_shape[1]))
                # seg = np.array(seg)
                #dcm, seg = random_rotated(dcm, seg)
                #dcm, seg = change_scale(dcm, seg, self.shape)
                #dcm, seg = random_scale(dcm, seg)
                #dcm, seg = random_crop(dcm, seg, shape=self.shape)
                #dcm = randomGaussian(dcm)
                #dcm = randomColor(dcm)
                #dcm, seg = random_flip(dcm, seg)
            else:
                dcm, seg = resize(dcm, seg, self.net_shape)
                dcm = np.array(dcm).reshape((1, self.net_shape[0], self.net_shape[1]))
                seg = np.array(seg)
                # dcm = resize_fill(dcm, self.net_shape, mode='L', fill_color=0)
                # seg = resize_fill(seg, self.net_shape, mode='L', fill_color=0)
                # dcm = np.array(dcm).reshape((1, self.net_shape[0], self.net_shape[1]))
                # seg = np.array(seg)
                #dcm, seg = change_scale(dcm, seg, self.shape)
                #dcm = center_crop(dcm, crop_size=self.shape)
                #seg = center_crop(seg, crop_size=self.shape)
                #dcm, seg = random_scale(dcm, seg)
                #dcm, seg = random_crop(dcm, seg, shape=self.shape)

            dcm = dcm / 255.0  # 像素值归一化
            dw = 1.0 / self.net_shape[0]
            dh = 1.0 / self.net_shape[1]
            label_lists = []

            seg_1 = (seg == 1) #肿块
            bboxs = cal_label(seg_1)
            for bbox in bboxs:
                w, h = bbox[3]-bbox[1], bbox[2]-bbox[0]
                x, y = bbox[1]+w/2, bbox[0]+h/2
                if w*h < 120 or w/h > 10 or h/w>10:
                    continue
                tmp = [0, x, y, w, h]       #用1表示肿块
                tmp = [tmp[0], tmp[1]*dw, tmp[2]*dh, tmp[3]*dw, tmp[4]*dh]      #归一化
                label_lists.append(tmp)

            seg_2 = (seg == 2) #结节
            bboxs = cal_label(seg_2)
            for bbox in bboxs:
                w, h = bbox[3] - bbox[1], bbox[2] - bbox[0]
                x, y = bbox[1] + w / 2, bbox[0] + h / 2
                if w*h < 120 or w/h > 10 or h/w>10:
                    continue
                tmp = [1, x, y, w, h]       #用2表示结节
                tmp = [tmp[0], tmp[1] * dw, tmp[2] * dh, tmp[3] * dw, tmp[4] * dh]  #归一化
                label_lists.append(tmp)
            if len(label_lists) != 0:
                break
            idx = random.randint(0, len(self.files)-1)

        if len(label_lists) > self.max_objects:
            label_lists = label_lists[:self.max_objects]
        else:
            label_lists = np.array(label_lists)
            stuff = np.zeros((self.max_objects - len(label_lists), 5))
            label_lists = np.concatenate((label_lists, stuff), axis=0)
        return dcm, label_lists, self.net_shape[0]

    def __len__(self):
        return len(self.files)

def cal_label(segs):
    '''
        按照连通域划分并返回其所在的方框
    :param segs:多张图片的分割label,0:表示背景；1：表示label
    :return: 表示长方体的位置信息（z1, y1, x1, z2, y2, x2）序列
    '''
    regions = Label(segs) #分割连通域，默认8联通
    props_list = Region(regions) #将连通域用方框框起来，返回对角坐标
    bboxs = []
    for props in props_list:
        bbox = np.asarray(props.bbox, dtype='int')
        bboxs.append(bbox)
    return bboxs

def patch(coord, color):
    x, y, w, h = coord
    rect = patches.Rectangle((x-w/2, y-h/2), w, h, linewidth=1, edgecolor=color, facecolor='none')
    plt.gca().add_patch(rect)


if __name__ == '__main__':
    file_2d = '/home/wxd/dataset/DATASET_BREAST'

    shape = (512, 512)
    dataset = Dataset2D_breast(file_2d, train='valid', net_shape=shape)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # print len(data_loader)
    for imgs, labels, _ in data_loader:
        for img, label in zip(imgs, labels):
            for bbox in label:
                if bbox[-1] == 0:
                    break
                bbox = [int(bbox[1] * shape[0]), int(bbox[2] * shape[1]), int(bbox[3] * shape[0]),
                         int(bbox[4] * shape[1])]
                patch(bbox, 'r')
            plt.imshow(img[0]*256, cmap='gray')
            plt.show()
