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
from utils.dataset import BISMValDataset,create_dataloader

if __name__ == '__main__':
    val_dataloader = create_dataloader(
        path='/home/yanggang/PyCharmWorkspace/BISMNetwork/data/bismskyline/val',
        img_size=[1024, 64],
        batch_size=1,
        augment=False,
        train=False
    )

    for i, (img, labels, pr) in enumerate(val_dataloader):
        print(img.shape)
        print(labels.shape)
        print(pr)