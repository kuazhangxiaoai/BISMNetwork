import torch
import torch.nn as nn
from models.backbones.csp_darknet import CSPDarknet
from models.necks.pafpn import PAFPN
from models.heads.BISM_skyline_head import DenseHead
class Model(nn.Module):
    def __init__(self, backbone, neck, head):
        super(Model, self).__init__()
        m = eval(backbone['type']) if isinstance(backbone['type'], str) else backbone
        self.backbone = m(**backbone['cfg'])

        m = eval(neck['type']) if isinstance(neck['type'], str) else neck
        self.neck = m(**neck['cfg'])

        m = eval(head['type']) if isinstance(head['type'], str) else head
        self.head = m(**head['cfg'])


    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(*x)
        obj,y = self.head(*x)
        return obj, y

def build_model(modelcfg):
    return Model(**modelcfg)
