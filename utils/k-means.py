#coding=utf-8
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from utils import bbox_iou

centers = 12
files = '/home/wxd/dataset/VOCdevkit/train.txt'

if __name__ == '__main__':

    with open(files, 'r') as fp:
        img_files = fp.readlines()
    label_files = [path.replace('JPEGImages', 'labels').replace('.png', '.txt').replace('.jpg', '.txt').strip() for path in img_files]

    X = None
    for img_file, label_file in zip(img_files, label_files):
        targets = np.loadtxt(label_file).reshape(-1, 5)
        img = Image.open(img_file.strip())
        img_w, img_h = img.size
        if X is None:
            X = targets[:, -2:]
        else:
            targets[:, -2] = targets[:, -2] * img_w
            targets[:, -1] = targets[:, -1] * img_h
            X = np.concatenate((X, targets[:, -2:]), 0)
    kmeans_model = KMeans(n_clusters=centers, random_state=0).fit(X)
    cluster_centers = kmeans_model.cluster_centers_
    cluster_labels = kmeans_model.labels_

    IOU = 0.0
    for i in cluster_labels:
        cluster_center = cluster_centers[i]
        cluster_center = [0, 0, cluster_center[0], cluster_center[1]]
        x = X[i]
        x = [0, 0, x[0], x[1]]
        iou = bbox_iou(x, cluster_center, x1y1x2y2=False)
        IOU += iou
    avg_iou = IOU/len(X)

    cluster_centers = sorted(cluster_centers, key = lambda wh: wh[0]*wh[1])

    print avg_iou
    print np.array(cluster_centers).astype(np.int32).reshape(-1)



