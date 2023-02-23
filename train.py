import glob
import hashlib
import json
import os
import random
import shutil
import time
import argparse

from pathlib import Path

import cv2
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, lr_scheduler
from torch.cuda import amp

from utils.callback import Callbacks
from utils.dataset import create_dataloader
from utils.loss import ComputeLoss
from utils.general import load_ckpt, one_cycle
from utils.torch_utils import ModelEMA, de_parallel
from models.model import build_model

from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[0]

def train(opt, callback=Callbacks):
    with open(opt.config, 'r') as file:
        opt = json.load(file)
    train_cfg = opt['train_cfg']
    train_loader = create_dataloader(**opt['train_dataset'])
    nb = len(train_loader)

    print("creating models ...")
    m = build_model(opt['model'])

    epochs = train_cfg['epoch']
    start_epoch = 0
    batch_size = opt['train_dataset']['batch_size']
    device = train_cfg['device']
    weights = train_cfg['weights']

    m = m.to(device)
    stride, obj_pw, obj_weight, predy_weight = \
        train_cfg['stride'],train_cfg['obj_pw'], train_cfg['obj_weight'], train_cfg['predy_weight']
    compute_loss = ComputeLoss(m, stride,device,obj_pw, obj_weight, predy_weight)

    if train_cfg['pretrained'] is not None:
        ckpt, m = load_ckpt(m, train_cfg['pretrained'], device)

    #optimizer
    accumulate = max(round(64 / batch_size), 1)
    weight_decay = train_cfg["optimizer"]['weight_decay'] * accumulate * batch_size / 64
    param_bn, param_weight, param_bias = [], [], []
    for v in m.modules():
        if hasattr(v,'bias') and isinstance(v.bias, nn.Parameter):
            param_bias.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            param_bn.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            param_weight.append(v.weight)

    if train_cfg["optimizer"]["type"] == "Adam":
        optimizer = Adam(param_bn, lr=train_cfg["optimizer"]["lr"], betas=(train_cfg["optimizer"]['momentum'], 0.999))
    if train_cfg["optimizer"]["type"] == "SGD":
        optimizer = SGD(param_bn, lr=train_cfg["optimizer"]["lr"], momentum=train_cfg["optimizer"]['momentum'], nesterov=True)

    optimizer.add_param_group({'params': param_weight, 'weight_decay': weight_decay})  # add g1 with weight_decay
    optimizer.add_param_group({'params': param_bias})  # add g2 (biases)

    print(f"optimier is generated:\n   type :{type(optimizer).__name__} \n "
          f"  weights :{len(param_weight)} \n   bias : {len(param_bias)} \n   weight decay: {weight_decay} ")
    del param_bn, param_weight, param_bias

    #scheduler
    lf = one_cycle(1, train_cfg['scheduler']['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=True) if device == 'cuda' else amp.GradScaler(enabled=False)

    #ema
    ema = ModelEMA(m)

    start_epoch, best_fitness = 0, 0.0
    if train_cfg['pretrained']:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if train_cfg['resume']:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:
            #LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt

    print("start training ...")
    last_opt_step = -1
    for epoch in range(start_epoch, epochs):
        m.train()
        mloss = torch.zeros(2, device=device)  # mean losses
        pbar = tqdm(enumerate(train_loader), total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        optimizer.zero_grad()
        for i, (imgs, labels) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255
            labels = labels.to(device)

            #forward
            with amp.autocast(enabled=True):
                pred_obj, pred_y = m(imgs)
                loss, loss_items = compute_loss(pred_obj, pred_y, labels)

            if (loss is None) or (loss_items is None):
                continue

            loss *= 2.0
            # Backward
            scaler.scale(loss).backward()

            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(m)
                last_opt_step = ni
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%10s' * 2 + '%10.4g' * 4) % (
                f'{epoch}/{epochs - 1}', mem, *mloss, labels.shape[-1], imgs.shape[-1]))
            #print("batch end")
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()
        final_epoch = (epoch + 1 == epochs)
        #save model
        if not final_epoch:
            ckpt = {
                'epoch': epoch,
                'best_fitness': best_fitness,
                'model': deepcopy(de_parallel(m)).half(),
                'ema': deepcopy(ema.ema).half(),
                'updates': ema.updates,
                'optimizer': optimizer.state_dict()
            }
            torch.save(ckpt, os.path.join(train_cfg["save_dir"], "last.pt"))



def main(option, callback=Callbacks()):
    train(option, callback)

def parser_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join(ROOT, 'configs','bismnetwork_cspdarknet_pafpn_skyline.json'),
                        help='build models')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parser_option()
    main(opt)