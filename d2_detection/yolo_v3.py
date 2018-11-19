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

        self.net_shape = 416
        self.loss_names = ['loss', 'x', 'y', 'w', 'h', 'conf', 'cls']
        self.header = np.array([0, 0, 0, 0], dtype=np.int32)
        self.seen = 0

    def modify_shape(self, net_shape):
        for model in self.models:
            if isinstance(model[0], YOLOLayer):
                model[0].img_size = net_shape

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
                    for name, loss in zip(self.loss_names, x):
                        self.losses[name] += loss
                    x = x[0]
                else:
                    x = model(x)
                output.append(x)
            # if block['type'] != 'yolo':
            #     print(i, x[0][0][0][0].data.cpu())
            layer_outputs.append(x)
        return sum(output) if targets is not None else torch.cat(output, 1)

    def create_network(self, blocks):
        hyperparams = blocks.pop(0)
        self.net_shape = int(hyperparams['height'])
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
                # anchors = np.array(anchors)*int(hyperparams['width'])

                num_classes = int(block['classes'])
                #定义检测层
                yolo_layer = YOLOLayer(anchors, num_classes, self.net_shape)
                model.add_module('yolo_{}'.format(i), yolo_layer)
            else:
                filters = out_filters[-1]
                print'block no settings.'

            out_filters.append(filters)
            models.append(model)
        return models

    def load_weights(self, weight_file, cutoff=-1):
        fp = open(weight_file, 'rb')
        # self.header = np.fromfile(fp, count=5, dtype=np.int64)
        self.header = np.fromfile(fp, count=4, dtype=np.int32)
        self.seen = self.header[3]
        # print self.header
        weights = np.fromfile(fp, dtype=np.float32)
        fp.close()
        start = 0
        for i, (block, model) in enumerate(zip(self.blocks[:cutoff], self.models[:cutoff])):
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
        self.header.tofile(fp)
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

        self.thresh = 0.7

    def forward(self, x, target=None):
        # print x
        nB = x.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nG = x.size(2)
        stride_xy = self.img_size / float(nG)     #与原图相比， feature map缩小了stride倍
        stride = 1.0

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
            # print self.anchors
            # print h.data.reshape(-1)[-1], h.data.shape
            # print anchor_w, anchor_h
            # print pred_boxes.reshape(-1, 4), pred_boxes.shape
            # pred_boxes[..., 2] = torch.exp(w.data) * anchor_w * stride
            # pred_boxes[..., 3] = torch.exp(h.data) * anchor_h * stride
            coord_mask, obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes.cpu().data,
                                                                                        pred_conf.cpu().data,
                                                                                        pred_cls.cpu().data,
                                                                                        target.cpu().data,
                                                                                        scaled_anchors.cpu().data,
                                                                                        nA, nG,
                                                                                        self.thresh, self.img_size, self.num_classes)
            x = x.view(-1)
            y = y.view(-1)
            w = w.view(-1)
            h = h.view(-1)
            pred_conf = pred_conf.view(-1)
            pred_cls = pred_cls.view(-1, nC)

            tx = tx.type(FloatTensor).view(-1)
            ty = ty.type(FloatTensor).view(-1)
            tw = tw.type(FloatTensor).view(-1)
            th = th.type(FloatTensor).view(-1)
            tconf = tconf.type(FloatTensor).view(-1)
            tcls = tcls.type(FloatTensor).view(-1, nC)

            coord_mask = coord_mask.type(FloatTensor).view(-1)
            obj_mask = obj_mask.type(FloatTensor).view(-1)
            noobj_mask = noobj_mask.type(FloatTensor).view(-1)


            # mask = mask.type(ByteTensor)
            # conf_mask = conf_mask.type(ByteTensor)
            # conf_mask_true = mask
            # conf_mask_false = conf_mask - mask

            # hard mining
            # pconf_false = pred_conf[conf_mask_false]
            # pconf_false, _ = pconf_false.sort(descending=True)
            # positive_sample = mask.sum()
            # negetive_sample = positive_sample * 3
            # if (pconf_false > 0.8).sum() > negetive_sample:
            #     pconf_false = pconf_false[pconf_false > 0.8]
            # else:
            #     pconf_false = pconf_false[:negetive_sample]
            # length = len(pconf_false)
            # tconf_false = tconf[conf_mask_false][:length]
            #


            loss_x = nn.MSELoss(size_average=False)(x*coord_mask, tx*coord_mask) / nB
            loss_y = nn.MSELoss(size_average=False)(y*coord_mask, ty*coord_mask) / nB
            loss_w = nn.MSELoss(size_average=False)(w*coord_mask, tw*coord_mask) / nB
            loss_h = nn.MSELoss(size_average=False)(h*coord_mask, th*coord_mask) / nB
            conf_mask = obj_mask + noobj_mask
            loss_conf = nn.BCELoss(size_average=False)(pred_conf * conf_mask, tconf * conf_mask) / nB
            loss_cls = nn.BCEWithLogitsLoss(size_average=False)(pred_cls, tcls) / nB
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            # print (loss_x + loss_y + loss_w + loss_h).data.cpu(), loss_conf.data.cpu(), loss_cls.data.cpu()

            # loss_conf = nn.MSELoss(size_average=True)(pred_conf[conf_mask_true], tconf[conf_mask_true]) + \
            #             nn.MSELoss(size_average=True)(pconf_false, tconf_false)

            #nn.BCELoss(size_average=False)(pred_conf*noobj_mask, tconf*noobj_mask) / nB

            # loss_cls = nn.CrossEntropyLoss(size_average=True)(pred_cls[mask], tcls[mask])


            return (
                loss,
                loss_x,
                loss_y,
                loss_w,
                loss_h,
                loss_conf,
                loss_cls,
            )
        else:
            pred_boxes[..., 0] = (x.data + grid_x) * stride_xy  # 在feature map上的位置和大小
            pred_boxes[..., 1] = (y.data + grid_y) * stride_xy

            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes)
                ),
                -1
            )
            return output


def build_targets(pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, grid_size, thresh, img_size, num_classes):
    nB = target.size(0)
    nA = num_anchors
    nG = grid_size

    noobj_mask = torch.ones(nB, nA, nG, nG)
    obj_mask = torch.zeros(nB, nA, nG, nG)
    coord_mask = torch.zeros(nB, nA, nG, nG)
    tx = torch.zeros(nB, nA, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    tconf = torch.zeros(nB, nA, nG, nG)
    tcls = torch.zeros(nB, nA, nG, nG, num_classes)

    nGT = 0
    nCorrect = 0
    for b in xrange(nB):
        # select negative samples

        cur_pred_boxes = pred_boxes[b]
        cur_pred_boxes = cur_pred_boxes.view(-1, 4).t()
        cur_ious = torch.zeros(nA*nG*nG)
        for t in xrange(target.size(1)):
            if target[b][t].sum() == 0:
                break
            gx = target[b][t][1] * nG  # ground truth 在feature map上的位置和大小
            gy = target[b][t][2] * nG
            gw = target[b][t][3] * img_size
            gh = target[b][t][4] * img_size
            cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).repeat(nA*nG*nG, 1).t()
            cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        ignore_idx = (cur_ious > thresh).view(nA, nG, nG)
        noobj_mask[b][ignore_idx] = 0
        # print ignore_idx
        # print noobj_mask[b].sum()

        for t in xrange(target.size(1)):
            if target[b][t].sum() == 0:
                break
            nGT += 1

            gx = target[b][t][1] * nG  # ground truth 在feature map上的位置和大小
            gy = target[b][t][2] * nG
            gw = target[b][t][3] * img_size
            gh = target[b][t][4] * img_size

            gi = int(gx)
            gj = int(gy)
            gt_box = [0, 0, gw, gh]

            best_iou = -1
            best_n = None
            for n in xrange(nA):
                w, h = anchors[n]
                anchor_box = [0, 0, w, h]
                iou = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n
                # if iou > thresh:
                #     noobj_mask[b, n, gj, gi] = 0
            # print best_n

            pred_box = pred_boxes[b][best_n][gj][gi]

            coord_mask[b][best_n][gj][gi] = 2.0 - target[b][t][3]*target[b][t][4]
            obj_mask[b][best_n][gj][gi] = 1
            noobj_mask[b][best_n][gj][gi] = 0
            tx[b][best_n][gj][gi] = gx - gi      #相对于每一个方格，中心点所占的百分比
            ty[b][best_n][gj][gi] = gy - gj
            tw[b][best_n][gj][gi] = math.log(gw / anchors[best_n][0])
            th[b][best_n][gj][gi] = math.log(gh / anchors[best_n][1])

            gt_box = [gx, gy, gw, gh]
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            tconf[b][best_n][gj][gi] = 1
            tcls[b][best_n][gj][gi][int(target[b][t][0])] = 1

            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            target_label = int(target[b, t, 0])
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.75 and pred_label == target_label and score > 0.8:
                nCorrect += 1
    print nCorrect, nGT
    return coord_mask, obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls
