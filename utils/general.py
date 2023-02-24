import math
import random

import cv2
import numpy as np
import glob
from pathlib import Path
import torch
import logging
import os
import re

def set_logging(name=None, verbose=True):
    # Sets level and returns logger
    for h in logging.root.handlers:
        logging.root.removeHandler(h)  # remove all handlers associated with the root logger object
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    logging.basicConfig(format="%(message)s", level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


LOGGER = set_logging(__name__)  # define globally (used in train.py, val.py, detect.py, etc.)

def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def letterbox(image, labels, img_size=(1024, 64), color=(114,114,114), resize=True, padding=True):
    h, w, c = image.shape
    h0, w0 = img_size[0], img_size[1]
    padding_width = w0 // 2
    new_image = cv2.copyMakeBorder(image, 0, 0, padding_width, padding_width, cv2.BORDER_CONSTANT, color)
    labels[:, 0] += padding_width
    i = random.randint(0, labels.shape[0] - 1)
    xl,yl = labels[i]
    img = new_image[:, xl-padding_width: xl + padding_width, : ]

    if h < img_size[0] and padding:
        padding_top, padding_bottom = round((img_size[0] - h)/2 - 0.1),round((img_size[0] - h)/2 + 0.1)
        im = cv2.copyMakeBorder(img, padding_top, padding_bottom, 0, 0, cv2.BORDER_CONSTANT, color)
        yl += padding_top
    elif resize:
        #cutting_top, cutting_bottom = round(h - img_size[0] - 0.1), round(h - img_size[0] + 0.1)
        #im = img[cutting_top: cutting_top + img_size[0], :, :]
        #yl -= cutting_top
        r = h / h0
        im = cv2.resize(img, dsize=(w0, h0), interpolation=cv2.INTER_LINEAR)
        yl = yl / r
    else:
        cut_top, cut_bottom = round((h - img_size[0]) / 2 - 0.1), round((h - img_size[0]) / 2 + 0.1)
        im = img[cut_top: cut_top + img_size[0], :, :]
        yl = yl - cut_top

    return im,  yl, h, w

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def load_ckpt(model, ckpt_path, device):
    assert os.path.exists(ckpt_path)
    ckpt = torch.load(ckpt_path)
    model_param = ckpt['models'].float().state_dict()
    model_param = intersect_dicts(model_param, model.state_dict(), exclude=[])
    model.load_state_dict(model_param,strict=False)
    return ckpt, model
