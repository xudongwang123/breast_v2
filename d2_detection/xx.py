#coding=utf-8
import glob
import os

datadir = '/home/wxd/dataset/DATASET_BREAST/'
train = 'train/*clean*'
val = 'val/*clean*'

train_files = glob.glob(datadir+ train)
val_files = glob.glob(datadir+val)

count = 0
for t in train_files:
    t = t.split('/')[-1]
    for v in val_files:
        v = v.split('/')[-1]
        print t, v
        if t == v:
            count += 1

print count