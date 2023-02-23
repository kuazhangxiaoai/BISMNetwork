import glob
import hashlib
import json
import os
import random
import shutil
import time

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, dataloader

from tqdm import tqdm

from utils.general import letterbox
from utils.augmentation import mosaic,augment_hsv, mixup

def create_dataloader(path, img_size, batch_size, augment=True, shuffle=True, train=True,num_worker=8):
    loader = DataLoader(
        dataset=BISMDataset(path,train,img_size, batch_size, augment),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_worker,
        pin_memory=False,
        collate_fn=collate_fn
    )
    return loader

def get_image_label_path(img_path, suffix='.txt'):
    labelpath = img_path.replace('images', 'labels').replace('.png', suffix)
    return img_path, labelpath

def load_image_label(img_path, label_path):
    img = cv2.imread(img_path)
    labels = []
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split('\n')[0]
            x, y = int(line.split(',')[0]), int(line.split(',')[1])
            labels.append([x, y])
    return img, np.array(labels)

def collate_fn(batch):
    image, labels = zip(*batch)
    torch_labels = []
    maxn = 0
    for i, label in enumerate(labels):
        label[0] = i
        if maxn < label.shape[0]:
            maxn = label.shape[0]
    for i, label in enumerate(labels):
        n0 = label.shape[0]
        n1 = maxn - n0
        if n1 > 0:
            sn = torch.FloatTensor([-1.0] * n1)
            torch_labels.append(torch.cat([label, sn], 0))
        else:
            torch_labels.append(label)
    return torch.stack(image,0), torch.stack(torch_labels,0)


class BISMDataset(Dataset):
    def __init__(self, path, train, img_size, batch_size=8, augment=None):
        super(BISMDataset, self).__init__()
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment
        self.path = path
        self.train = train
        self.imgfiles, self.labelfiles = glob.glob(path + '/images/*.*'), glob.glob(path + '/labels/*.*')
        assert len(self.imgfiles) == len(self.labelfiles)
        self.nLength = len(self.imgfiles)
        self.indices = range(0, self.nLength)

    def __len__(self):
        return self.nLength

    def __getitem__(self, index):
        index = self.indices[index]
        img_path, labelpath= get_image_label_path(self.imgfiles[index])
        assert labelpath in self.labelfiles
        img, labels = load_image_label(img_path, labelpath)  # whole image and labels
        img, ylabel, h, w = letterbox(img, labels, img_size=self.img_size)  # cutted image and label
        ylabels = [ylabel]

        if self.augment['mosaic']['enable'] and self.train :
            img, ylabels = self.mosaic_augment(index, img, ylabel)
        if self.augment['hsv_augment']['enable'] and self.train:
            augment_hsv(
                img,
                self.augment['hsv_augment']['h'],
                self.augment['hsv_augment']['s'],
                self.augment['hsv_augment']['v'],
            )
        if self.augment['mixup']['enable'] and self.train:
            index_mixup = random.choice(self.indices)
            img_path_mixup, label_path_mixup = get_image_label_path(self.imgfiles[index_mixup])
            assert label_path_mixup in self.labelfiles
            img_mixup, labels_mixup = load_image_label(img_path_mixup, label_path_mixup)
            img_mixup, label_mixup, _, _ = letterbox(img_mixup, labels_mixup)
            img, ylabels = mixup(img, ylabels, img_mixup, [label_mixup], index)

        if self.train and self.augment['enable']:
            self.augment['mosaic']['enable'] = random.choices([True, False],[self.augment['mosaic']['posible'], 1 - self.augment['mosaic']['posible']])
            self.augment['hsv_augment']['enable'] = random.choices([True, False],[self.augment['hsv_augment']['posible'], 1 - self.augment['hsv_augment']['posible']])
            self.augment['mixup']['enable'] = random.choices([True, False],[self.augment['mixup']['posible'], 1 - self.augment['mixup']['posible']])
        else:
            self.augment['mosaic']['enable'] = False
            self.augment['hsv_augment']['enable'] = False
            self.augment['mixup']['enable'] = False

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        ylabels.insert(0, 0)
        ylabels = np.ascontiguousarray(np.array(ylabels).astype(np.float32))
        return torch.from_numpy(img), torch.from_numpy(ylabels)

    def mosaic_augment(self, index, source_img, source_ylabel):
        mosaic_imgs, mosaic_ylabels = [], []
        #img_path, labelpath = get_image_label_path(self.imgfiles[index])
        #assert labelpath in self.labelfiles
        #img, labels = load_image_label(img_path, labelpath)
        #source_img, source_ylabel, h, w = letterbox(img, labels, img_size=self.img_size)

        for i in range(0, 4):
            mosaic_index = random.choice(self.indices)
            mosaic_img_path, mosaic_label_path = get_image_label_path(self.imgfiles[mosaic_index])
            mosaic_img, mosaic_label = load_image_label(mosaic_img_path, mosaic_label_path)
            mosaic_img, mosaic_label, mh, mw = letterbox(mosaic_img,
                                                           mosaic_label,
                                                           img_size=self.img_size,
                                                           resize=True,
                                                           padding=True)
            mosaic_imgs.append(mosaic_img)
            mosaic_ylabels.append(mosaic_label)

        img, ylabels = mosaic(mosaic_imgs, mosaic_ylabels, source_img, source_ylabel,(1024, 64, 3), index)
        return img, ylabels

if __name__ == '__main__':
    #dataset = BISMDataLoader('/home/yanggang/PyCharmWorkspace/BISMNetwork/data/bismskyline')
    augment = {
        'enable': True,
        'mosaic': {
            'enable': True,
            'posible': 0.4,
        },
        'mixup': {
            'enable': True,
            'posible': 0.4,
        },
        'hsv_augment':{
            'enable':True,
            'posible':0.5,
            'h': 0.015,
            's': 0.7,
            'v': 0.5,
        }
    }
    loader = create_dataloader(
        path='/home/yanggang/PyCharmWorkspace/BISMNetwork/data/bismskyline',
        img_size=(1024, 64),
        batch_size=8,
        augment=augment,
        shuffle=True,
        train=True,
        worker=1
    )

    for i ,(img, label) in enumerate(loader):
        print(f'{label}')
