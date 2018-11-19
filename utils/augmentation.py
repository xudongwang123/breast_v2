#coding=utf-8
import os
import random
import numpy as np
from PIL import Image


def resize_fill(img, new_size, mode='RGB', fill_color=(127, 127, 127)):
    '''
    :param img: Image类型
    :param img_size: （w, h）
    :return: 将长边resize为目标长度，并把短边同比例resize，不够的部分使用定值填充
    '''
    im_w, im_h = img.size
    net_w, net_h = new_size
    if float(net_w) / float(im_w) < float(net_h) / float(im_h):
        new_w = net_w
        new_h = (im_h * net_w) // im_w
    else:
        new_w = (im_w * net_h) // im_h
        new_h = net_h
    resized = img.resize((new_w, new_h), Image.ANTIALIAS)
    fill_Image = Image.new(mode, (net_w, net_h), fill_color)
    fill_Image.paste(resized, \
                  ((net_w - new_w) // 2, (net_h - new_h) // 2, \
                   (net_w + new_w) // 2, (net_h + new_h) // 2))
    return fill_Image


def get_true_label(label, img_shape, net_shape):
    img_w, img_h = img_shape
    net_w, net_h = net_shape
    new_label = []

    if float(net_w) / float(img_w) < float(net_h) / float(img_h):
        new_w = net_w
        new_h = (img_h * net_w) // img_w
    else:
        new_w = (img_w * net_h) // img_h
        new_h = net_h

    for i in range(len(label)):
        cls, x, y, w, h = label[i]
        x, y, w, h = x*new_w, y*new_h, w*new_w, h*new_h
        x, y = x+(net_w-new_w)/2, y+(net_h-new_h)/2
        x, y, w, h = x/net_w, y/net_h, w/net_w, h/net_h
        new_label.append([cls, x, y, w, h])

    return np.array(new_label)


def transform_data(img_path, label_path, net_shape, crop, jitter, hue, saturation, exposure):
    #数据增强
    img = Image.open(img_path).convert('RGB')

    if crop:         # marvis version
        img,flip,dx,dy,sx,sy = data_augmentation_crop(img, net_shape, jitter, hue, saturation, exposure)
    else:            # original version
        img,flip,dx,dy,sx,sy = data_augmentation_nocrop(img, net_shape, jitter, hue, saturation, exposure)
    label = fill_truth_detection(label_path, crop, flip, -dx, -dy, sx, sy)
    return img, label


def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):
        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)


def data_augmentation_crop(img, shape, jitter, hue, saturation, exposure):
    oh = img.height
    ow = img.width

    dw = int(ow * jitter)
    dh = int(oh * jitter)

    pleft = np.random.randint(-dw, dw)
    pright = np.random.randint(-dw, dw)
    ptop = np.random.randint(-dh, dh)
    pbot = np.random.randint(-dh, dh)

    swidth = ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = ow / float(swidth)
    sy = oh / float(sheight)

    flip = np.random.randint(2)

    cropbb = np.array([pleft, ptop, pleft + swidth - 1, ptop + sheight - 1])
    # following two lines are old method. out of image boundary is filled with black (0,0,0)
    # cropped = img.crop( cropbb )
    # sized = cropped.resize(shape)

    nw, nh = cropbb[2] - cropbb[0], cropbb[3] - cropbb[1]
    # get the real image part
    cropbb[0] = -min(cropbb[0], 0)
    cropbb[1] = -min(cropbb[1], 0)
    cropbb[2] = min(cropbb[2], ow)
    cropbb[3] = min(cropbb[3], oh)
    cropped = img.crop(cropbb)

    # calculate the position to paste
    bb = (pleft if pleft > 0 else 0, ptop if ptop > 0 else 0)
    new_img = Image.new("RGB", (nw, nh), (127, 127, 127))
    new_img.paste(cropped, bb)

    sized = new_img.resize(shape)
    del cropped, new_img

    dx = (float(pleft) / ow) * sx
    dy = (float(ptop) / oh) * sy

    if flip:
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)
    # for compatibility to nocrop version (like original version)
    return img, flip, dx, dy, sx, sy


def data_augmentation_nocrop(img, shape, jitter, hue, sat, exp):
    net_w, net_h = shape
    img_w, img_h = img.width, img.height

    # determine the amount of scaling and cropping
    dw = jitter * img_w
    dh = jitter * img_h

    new_ar = (img_w + np.random.uniform(-dw, dw)) / (img_h + np.random.uniform(-dh, dh))
    # scale = np.random.uniform(0.25, 2)
    scale = 1.

    if (new_ar < 1):
        new_h = int(scale * net_h)
        new_w = int(net_h * new_ar)
    else:
        new_w = int(scale * net_w)
        new_h = int(net_w / new_ar)

    dx = int(np.random.uniform(0, net_w - new_w))
    dy = int(np.random.uniform(0, net_h - new_h))
    sx, sy = float(new_w) / net_w, float(new_h) / net_h

    # apply scaling and shifting
    new_img = image_scale_and_shift(img, new_w, new_h, net_w, net_h, dx, dy)

    # randomly distort hsv space
    new_img = random_distort_image(new_img, hue, sat, exp)

    # randomly flip
    flip = np.random.randint(2)
    if flip:
        new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)

    dx, dy = dx / float(net_w), dy / float(net_h)
    return new_img, flip, dx, dy, sx, sy

def image_scale_and_shift(img, new_w, new_h, net_w, net_h, dx, dy):
    scaled = img.resize((new_w, new_h))
    # find to be cropped area
    sx, sy = -dx if dx < 0 else 0, -dy if dy < 0 else 0
    ex, ey = new_w if sx+new_w<=net_w else net_w-sx, new_h if sy+new_h<=net_h else net_h-sy
    scaled = scaled.crop((sx, sy, ex, ey))

    # find the paste position
    sx, sy = dx if dx > 0 else 0, dy if dy > 0 else 0
    assert sx+scaled.width<=net_w and sy+scaled.height<=net_h
    new_img = Image.new("RGB", (net_w, net_h), (127, 127, 127))
    new_img.paste(scaled, (sx, sy))
    del scaled
    return new_img


def fill_truth_detection(labpath, crop, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.zeros((max_boxes, 5))
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))
        cc = 0
        for i in range(bs.shape[0]):
            if bs[i][0] != 14:
                continue
            # print('person')
            x1 = bs[i][1] - bs[i][3] / 2
            y1 = bs[i][2] - bs[i][4] / 2
            x2 = bs[i][1] + bs[i][3] / 2
            y2 = bs[i][2] + bs[i][4] / 2

            x1 = min(0.999, max(0, x1 * sx - dx))
            y1 = min(0.999, max(0, y1 * sy - dy))
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))
            # print(bs[i])
            # print(sy, dy)
            # print(x1, y1, x2, y2, y1 * sy - dy, y2 * sy - dy)
            bs[i][1] = (x1 + x2) / 2  # center x
            bs[i][2] = (y1 + y2) / 2  # center y
            bs[i][3] = (x2 - x1)  # width
            bs[i][4] = (y2 - y1)  # height

            if flip:
                bs[i][1] = 0.999 - bs[i][1]

            # print(bs[i][3], bs[i][4], crop, bs[i][3] / bs[i][4] > 20, bs[i][4] / bs[i][3] > 20)
            # when crop is applied, we should check the cropped width/height ratio
            if bs[i][3] < 0.002 or bs[i][4] < 0.002 or \
                    (crop and (bs[i][3] / bs[i][4] > 20 or bs[i][4] / bs[i][3] > 20)):
                continue
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1, 5))
    return label


def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size//5, 5) # to avoid single truth problem
        # print(len(truths))
        # tmp = []
        # for truth in truths:
        #     if truth[0]== 14:
        #         tmp.append(truth)
        # print(len(tmp))
        return truths
    else:
        return np.array([])

def random_distort_image(im, hue, saturation, exposure):
    dhue = np.random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res


def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)

    def change_hue(x):
        x += hue * 255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x

    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    # constrain_image(im)
    return im


def rand_scale(s):
    scale = np.random.uniform(1, s)
    if np.random.randint(2):
        return scale
    return 1. / scale


def augment_hsv(img):
    img_hsv = img.convert('HSV')
    H, S, V = img_hsv.split()
    H, S, V = np.array(H, dtype=np.float32), np.array(S, dtype=np.float32), np.array(V, dtype=np.float32)

    #H, S, V分别增强正负50%
    fraction = 0.5
    hue = (random.random() * 2 - 1) * fraction + 1  # 产生范围[0.5, 1.5]
    H *= hue
    if hue > 1:
        np.clip(H, a_min=0, a_max=255, out=H)

    sat = (random.random() * 2 - 1) * fraction + 1 #产生范围[0.5, 1.5]
    S *= sat
    if sat > 1:
        np.clip(S, a_min=0, a_max=255, out=S)

    val = (random.random() * 2 - 1) * fraction + 1 #产生范围[0.5, 1.5]
    V *= val
    if val > 1:
        np.clip(V, a_min=0, a_max=255, out=V)
    H, S, V = H.astype(np.uint8), S.astype(np.uint8), V.astype(np.uint8)
    img = Image.merge(mode='HSV', bands=[H, S, V])
    img = img.convert('RGB')

    return img


def random_affine(img, target):
    pass


def resize_padding(img, new_size, mode='RGB', padding_color=(127, 127, 127)):
    '''
    :param img: Image类型
    :param new_size: （w, h）
    :return: 将长边resize为目标长度，并把短边同比例resize，不够的部分使用定值填充
            填充好的图片、缩放比、填充宽、填充高
    '''
    im_w, im_h = img.size
    net_w, net_h = new_size
    if float(net_w) / float(im_w) < float(net_h) / float(im_h):
        new_w = net_w
        new_h = (im_h * net_w) // im_w
    else:
        new_w = (im_w * net_h) // im_h
        new_h = net_h
    resized = img.resize((new_w, new_h), Image.ANTIALIAS)
    padding_Image = Image.new(mode, (net_w, net_h), padding_color)
    padding_Image.paste(resized, \
                  ((net_w - new_w) // 2, (net_h - new_h) // 2, \
                   (net_w + new_w) // 2, (net_h + new_h) // 2))
    return padding_Image


# def get_true_label(label, img_shape, net_shape):
#     img_w, img_h = img_shape
#     net_w, net_h = net_shape
#
#     if float(net_w) / float(img_w) < float(net_h) / float(img_h):
#         new_w = net_w
#         new_h = (img_h * net_w) // img_w
#     else:
#         new_w = (img_w * net_h) // img_h
#         new_h = net_h
#
#     new_label = label.copy()
#     new_label[:, 1] = label[:, 1] * new_w + (net_w - new_w) / 2
#     new_label[:, 2] = label[:, 2] * new_h + (net_h - new_h) / 2
#     new_label[:, 1] = label[:, 1] / net_w
#     new_label[:, 2] = label[:, 2] / net_h
#     new_label[:, 3] = label[:, 3] * new_w / net_w
#     new_label[:, 4] = label[:, 4] * new_h / net_h
#
#     return new_label
#
#
# def get_true_label(label, img_shape, net_shape):
#     img_w, img_h = img_shape
#     net_w, net_h = net_shape
#     new_label = []
#
#     if float(net_w) / float(img_w) < float(net_h) / float(img_h):
#         new_w = net_w
#         new_h = (img_h * net_w) // img_w
#     else:
#         new_w = (img_w * net_h) // img_h
#         new_h = net_h
#
#     for i in range(len(label)):
#         cls, x, y, w, h = label[i]
#         x, y, w, h = x*new_w, y*new_h, w*new_w, h*new_h
#         x, y = x+(net_w-new_w)/2, y+(net_h-new_h)/2
#         x, y, w, h = x/net_w, y/net_h, w/net_w, h/net_h
#         new_label.append([cls, x, y, w, h])
#
#     return np.array(new_label)
#
#
def resize(img, seg, shape=(512,512)):
    # if not isinstance(img, np.array):
    #
    # img = Image.fromarray(img)
    img = img.resize(shape, Image.ANTIALIAS)
    #
    # seg = Image.fromarray(seg)
    seg = seg.resize(shape, Image.ANTIALIAS)
    return img, seg

def random_flip(data, target):
    '''
        上下或者左右随机翻转
    '''
    flag = np.random.randint(0, 2)
    if flag == 1:
        data = data.transpose(Image.FLIP_LEFT_RIGHT)
        target = target.transpose(Image.FLIP_LEFT_RIGHT)
    elif flag == 2:
        data = data.transpose(Image.FLIP_TOP_BOTTOM)
        target = target.transpose(Image.FLIP_TOP_BOTTOM)
    return data, target