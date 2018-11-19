#coding=utf-8

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import defaultdict

#self package
from cfg import parse_model_cfg
from utils.utils import *


class Darknet(nn.Module):
    def __init__(self, cfg_file):
        super(Darknet, self).__init__()
        self.blocks = parse_model_cfg(cfg_file)   #对网络配置文件解析
        self.models = self.create_network(self.blocks)  #构建网络结构

        self.loss_names = ['loss', 'x', 'y', 'w', 'h', 'conf', 'cls']
        self.header = torch.IntTensor([0, 0, 0, 0, 0])
        self.seen = 0

    def forward(self, x, targets=None):
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        for i, (block, model) in enumerate(zip(self.blocks, self.models)):
            if block['type'] in ['convolutional', 'upsample', 'maxpool']:
                x = model(x)
            elif block['type'] == 'route':
                layer_k = [int(k.strip()) for k in block['layers'].split(',')]
                x = torch.cat([layer_outputs[k] for k in layer_k], 1)
            elif block['type'] == 'shortcut':
                layer_i = int(block['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif block['type'] == 'yolo':
                if targets is not None:
                    x = model[0](x, targets)
                else:
                    x = model(x)
                output.append(x)
            layer_outputs.append(x)
            # if block['type'] != 'yolo':
            #     print i, x[0][0][0][0].data.cpu()
        if targets is not None:
            losses = select_backward_loss(output, targets)
            for name, loss in zip(self.loss_names, losses):
                self.losses[name] += loss

        return losses[0] if targets is not None else torch.cat(output, 1)

    def create_network(self, blocks):
        hyperparams = blocks.pop(0)
        out_filters = [int(hyperparams['channels'])]

        models = nn.ModuleList()
        for i, block in enumerate(blocks):
            model = nn.Sequential()
            if block['type'] == 'convolutional':
                BN = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size - 1) / 2 if is_pad else 0
                activation = block['activation']
                model.add_module('conv{0}'.format(i),
                                 nn.Conv2d(out_filters[-1], filters, kernel_size, stride, pad, bias=not BN))
                if BN:
                    model.add_module('bn{0}'.format(i), nn.BatchNorm2d(filters))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(i), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(i), nn.ReLU(inplace=True))

            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                filters = out_filters[-1]
                if stride > 1:
                    max_pool = nn.MaxPool2d(pool_size, stride)
                else:
                    print'maxpool stride {0} does not set.'.format(stride)
                model.add_module('maxpool_{0}'.format(i), max_pool)

            elif block['type'] == 'avgpool':
                filters = out_filters[-1]
                print'avgpool stride does not set.'

            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                filters = out_filters[-1]
                model.add_module('upsample_{0}'.format(i), nn.Upsample(scale_factor=stride, mode='nearest'))

            elif block['type'] == 'route':
                layers = [int(x.strip()) for x in block['layers'].split(',')]
                filters = sum(out_filters[i] for i in layers)
                model.add_module('route_{0}'.format(i), EmptyLayer())

            elif block['type'] == 'shortcut':
                filters = out_filters[int(block['from'])]
                model.add_module('shortcut_{0}'.format(i), EmptyLayer())

            elif block['type'] == 'yolo':
                filters = out_filters[-1]
                anchor_idxs = [int(x.strip()) for x in block['mask'].split(',')]
                anchors = [float(x.strip()) for x in block['anchors'].split(',')]
                anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in anchor_idxs]
                #anchors = np.array(anchors)*int(hyperparams['width'])

                num_classes = int(block['classes'])
                img_height = int(hyperparams['height'])
                #定义检测层
                yolo_layer = YOLOLayer(anchors, num_classes, img_height)
                model.add_module('yolo_{}'.format(i), yolo_layer)
            else:
                filters = out_filters[-1]
                print'block no settings.'

            out_filters.append(filters)
            models.append(model)
        return models

    def load_weights(self, weight_file):
        fp = open(weight_file, 'rb')
        self.header = torch.IntTensor(np.fromfile(fp, count=4, dtype=np.int32))
        self.seen = self.header[3]
        # print self.header
        weights = np.fromfile(fp, dtype=np.float32)
        fp.close()
        start = 0
        for i, (block, model) in enumerate(zip(self.blocks, self.models)):
            if block['type'] == 'convolutional':
                conv_layer = model[0]
                if block['batch_normalize']:
                    bn_layer = model[1]
                    start = load_conv_bn(weights, start, conv_layer, bn_layer)
                else:
                    start = load_conv(weights, start, conv_layer)

    def save_weights(self, save_file, cutoff=-1):
        fp = open(save_file, 'wb')
        self.header[3] = self.seen
        self.header.numpy().tofile(fp)
        for i, (block, model) in enumerate(zip(self.blocks[:cutoff], self.models[:cutoff])):
            if block['type'] == 'convolutional':
                conv_layer = model[0]
                if block['batch_normalize']:
                    bn_layer = model[1]
                    save_conv_bn(fp, conv_layer, bn_layer)
                else:
                    save_conv(fp, conv_layer)

        fp.close()


# for route and shortcut
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.img_size = img_size

        self.num_anchors = len(anchors)
        self.bbox_attrs = 5 + num_classes   #(x, y, w, h, c, num_classes)

        self.noobject_scale = 1.0
        self.object_scale = 1.0
        self.coord_scale = 5.0
        self.thresh = 0.5

    def forward(self, x, target=None):
        nB = x.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nG = x.size(2)
        stride = self.img_size / float(nG)     #与原图相比， feature map缩小了stride倍

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()    #(nB, nA, nG, nG, 5+num_classes)

        #Get outputs
        x = torch.sigmoid(prediction[..., 0])   #center x relative to Grid
        y = torch.sigmoid(prediction[..., 1])   #center y relative to Grid
        w = prediction[..., 2]                  #w = log((Wt/stride)/scaled_anchor)，其中stride为原图到feature map上的缩放比, anchor为feature map上的
        h = prediction[..., 3]                  #h = log((Ht/stride)/scaled_anchor)，其中stride为原图到feature map上的缩放比
        pred_conf = torch.sigmoid(prediction[..., 4])   #confidence
        pred_cls = prediction[..., 5:] # class

        #Calculate offsets for each gird
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
        grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])   #获得feature map上anchor的大小
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        #Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x    #在feature map上的位置和大小
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        if target is not None:
            return (
                pred_boxes, x, y, w, h, pred_conf, pred_cls, scaled_anchors, nA, nB, nC, nG
            )
        else:
            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes)
                ),
                -1
            )
            return output


def select_backward_loss(output, targets, thresh=0.4):
    ByteTensor = torch.cuda.ByteTensor if output[0][1].is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if output[0][1].is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if output[0][1].is_cuda else torch.LongTensor

    conf_mask = None
    mask = None

    tx = None
    ty = None
    tw = None
    th = None
    tconf = None
    tcls = None

    px = None
    py = None
    pw = None
    ph = None
    pconf = None
    pcls = None

    anchor_range = [0]
    for i, (pred_box, x, y, w, h, pred_conf, pred_cls, scaled_anchors, nA, nB, nC, nG) in enumerate(output):
        if conf_mask is None:
            conf_mask = torch.ones(nB, nA, nG, nG).view(-1)
            mask = torch.zeros(nB, nA, nG, nG).view(-1)
            tx = torch.zeros(nB, nA, nG, nG).view(-1)
            ty = torch.zeros(nB, nA, nG, nG).view(-1)
            tw = torch.zeros(nB, nA, nG, nG).view(-1)
            th = torch.zeros(nB, nA, nG, nG).view(-1)
            tconf = torch.zeros(nB, nA, nG, nG).view(-1)
            tcls = torch.zeros(nB, nA, nG, nG).view(-1)

            px = x.view(-1)
            py = y.view(-1)
            pw = w.view(-1)
            ph = h.view(-1)
            pconf = pred_conf.view(-1)
            pcls = pred_cls.view(-1, nC)

        else:
            conf_mask = torch.cat((conf_mask, torch.ones(nB, nA, nG, nG).view(-1)), 0)
            mask = torch.cat((mask, torch.zeros(nB, nA, nG, nG).view(-1)), 0)
            tx = torch.cat((tx, torch.zeros(nB, nA, nG, nG).view(-1)), 0)
            ty = torch.cat((ty, torch.zeros(nB, nA, nG, nG).view(-1)), 0)
            tw = torch.cat((tw, torch.zeros(nB, nA, nG, nG).view(-1)), 0)
            th = torch.cat((th, torch.zeros(nB, nA, nG, nG).view(-1)), 0)
            tconf = torch.cat((tconf, torch.zeros(nB, nA, nG, nG).view(-1)), 0)
            tcls = torch.cat((tcls, torch.zeros(nB, nA, nG, nG).view(-1)), 0)

            px = torch.cat((px, x.view(-1)), 0)
            py = torch.cat((py, y.view(-1)), 0)
            pw = torch.cat((pw, w.view(-1)), 0)
            ph = torch.cat((ph, h.view(-1)), 0)
            pconf = torch.cat((pconf, pred_conf.view(-1)), 0)
            pcls = torch.cat((pcls, pred_cls.view(-1, nC)), 0)
        anchor_range.append(nB*nA*nG*nG+anchor_range[-1])
    tx = tx.type(FloatTensor)
    ty = ty.type(FloatTensor)
    tw = tw.type(FloatTensor)
    th = th.type(FloatTensor)
    tconf = tconf.type(FloatTensor)
    tcls = tcls.type(LongTensor)

    nCorrect = 0
    nGT = 0
    for b in range(targets.size(0)):
        for t in range(targets.size(1)):
            if targets[b][t].sum() == 0:
                break
            nGT += 1

            best_iou = -1
            best_n = [-1, -1]   #尺度，anchor
            best_gt_box = None
            best_grid = None
            best_index = None
            best_anchor = None
            for i, (pred_box, x, y, w, h, pred_conf, pred_cls, scaled_anchors, nA, nB, nC, nG) in enumerate(output):
                gx = targets[b][t][1] * nG  # ground truth 在feature map上的位置和大小
                gy = targets[b][t][2] * nG
                gw = targets[b][t][3] * nG
                gh = targets[b][t][4] * nG

                gi = int(gx)
                gj = int(gy)
                gt_box = [0, 0, gw, gh]

                for n in range(nA):
                    aw, ah = scaled_anchors[n]
                    anchor_box = [0, 0, aw, ah]
                    iou = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)

                    if iou > best_iou:
                        best_iou = iou
                        best_n = [i, n]
                        best_gt_box = [gx, gy, gw, gh]
                        best_grid = [gj, gi]
                        best_index = anchor_range[i] + b*nA*nG*nG + n*nG*nG + gj*nG + gi
                        best_anchor = [aw, ah]
                    if iou > thresh:
                        conf_mask[anchor_range[i] + b*nA*nG*nG + n*nG*nG + gj*nG + gi] = 0

                    if iou > 0.7:
                        index = anchor_range[i] + b*nA*nG*nG + n*nG*nG + gj*nG + gi
                        mask[index] = 1
                        conf_mask[index] = 1
                        tx[index] = gx - gi  # 相对于每一个方格，中心点所占的百分比
                        ty[index] = gy - gj

                        tw[index] = math.log(gw / scaled_anchors[n][0])
                        th[index] = math.log(gh / scaled_anchors[n][1])

                        tconf[index] = 1
                        tcls[index] = targets[b][t][0]

            pred_box = output[best_n[0]][0][b][best_n[1]][best_grid[0]][best_grid[1]]
            mask[best_index] = 1
            conf_mask[best_index] = 1
            tx[best_index] = best_gt_box[0] - best_grid[1]  # 相对于每一个方格，中心点所占的百分比
            ty[best_index] = best_gt_box[1] - best_grid[0]

            tw[best_index] = math.log(best_gt_box[2] / best_anchor[0])
            th[best_index] = math.log(best_gt_box[3] / best_anchor[1])
            tconf[best_index] = 1
            tcls[best_index] = targets[b][t][0]

            iou = bbox_iou(best_gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(output[best_n[0]][6][b, best_n[1], best_grid[0], best_grid[1]])
            target_label = int(targets[b, t, 0])
            score = output[best_n[0]][5][b, best_n[1], best_grid[0], best_grid[1]]

            if iou > 0.75 and pred_label == target_label and score > 0.8:  # 0.5作为预测正确的阈值
                nCorrect += 1

    mask = Variable(mask.type(ByteTensor))
    conf_mask = Variable(conf_mask.type(ByteTensor))
    conf_mask_true = mask
    conf_mask_false = conf_mask - mask

    #hard mining
    pconf_false = pconf[conf_mask_false]
    pconf_false, _ = pconf_false.sort(descending=True)

    positive_sample = mask.sum()
    negetive_sample = positive_sample * 3

    if (pconf_false>0.8).sum() > negetive_sample:
        pconf_false = pconf_false[pconf_false>0.8]
    else:
        pconf_false = pconf_false[:negetive_sample]
    length = len(pconf_false)
    tconf_false = tconf[conf_mask_false][:length]

    coord_scale = 1
    noobject_scale = 1
    loss_x = nn.MSELoss(size_average=True)(px[mask], tx[mask]) * coord_scale
    loss_y = nn.MSELoss(size_average=True)(py[mask], ty[mask]) * coord_scale
    loss_w = nn.MSELoss(size_average=True)(pw[mask], tw[mask]) * coord_scale
    loss_h = nn.MSELoss(size_average=True)(ph[mask], th[mask]) * coord_scale

    loss_conf = nn.MSELoss(size_average=True)(pconf[conf_mask_true], tconf[conf_mask_true]) + \
                nn.MSELoss(size_average=True)(pconf_false, tconf_false) * noobject_scale

    loss_cls = nn.CrossEntropyLoss(size_average=True)(pcls[mask], tcls[mask])

    loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

    print nCorrect, nGT, len(pconf_false)
    print nn.MSELoss(size_average=True)(pconf[conf_mask_true], tconf[conf_mask_true]), nn.MSELoss(size_average=True)(pconf_false, tconf_false)*noobject_scale

    return loss, loss_x, loss_y, loss_w, loss_h, loss_conf, loss_cls
