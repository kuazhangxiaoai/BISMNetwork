import os
import numpy as np
import torch
import torch.nn as nn

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    def __init__(self, model, stride, device, obj_pw, obj_weight, predy_weight):
        self.model = model
        self.stride = stride
        self.device = device
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(obj_pw))
        self.MSEypred = nn.MSELoss()
        self.obj_weight = obj_weight
        self.predy_weight = predy_weight

    def __call__(self, pred_obj, pred_y, labels):
        tobj, ty = self.build_targets(pred_obj, pred_y, labels)
        if (not tobj.any()) or (not ty.any()):
            return None, None
        ly,lobj = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        bs = pred_obj.shape[0]
        obj_loss = self.BCEobj(pred_obj, tobj)
        predy_loss = self.MSEypred(pred_y, ty)

        lobj += (self.obj_weight * obj_loss)
        ly += (self.predy_weight * predy_loss)
        loss = ( lobj +  ly) * bs
        loss_items = torch.cat((lobj, ly)).detach()
        return loss, loss_items

    def build_targets(self, pred_obj,pred_y, labels):
        target_obj,target_predy = torch.zeros_like(pred_obj), torch.zeros_like(pred_y)
        grid_size = pred_obj.shape[1]
        for i, label in enumerate(labels):
            b = label[0]
            ys = label[1:] / self.stride
            ys = ys[ys > 0]
            yi = ys.long()
            n = torch.BoolTensor([False]*grid_size)
            n = n.cuda() if self.device == 'cuda' else n.cpu()

            n[yi] = True

            #print(target_obj[b.long()])
            try:
                target_obj[b.long(), n] = 1.0
                target_predy[b.long(), n] = (ys - yi.float()).half()
                return target_obj, target_predy
            except:
                return torch.zeros_like(target_obj), torch.zeros_like(target_predy)



