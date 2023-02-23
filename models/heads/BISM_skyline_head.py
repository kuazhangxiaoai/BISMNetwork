import numpy as np
import torch
import torch.nn as nn

from models.common import Conv, Concat


class DenseHead(nn.Module):
    def __init__(self, ch):
        super(DenseHead, self).__init__()
        self.flaten = nn.Flatten()
        self.concat = Concat()
        self.convModule0 = nn.Sequential(Conv(ch[0], 256), Conv(256 , 2))
        self.convModule1 = nn.Sequential(Conv(ch[1], 256), Conv(256 , 2))
        self.convModule2 = nn.Sequential(Conv(ch[2], 256), Conv(256 , 2))

        self.linear0 = nn.Sequential(nn.Linear(2688, 4096), nn.Dropout())
        self.linear1 = nn.Sequential(nn.Linear(4096, 1024), nn.Dropout())
        self.linear_obj_pred = nn.Linear(1024, 128)
        self.linear_vert_pred= nn.Linear(1024, 128)
        self.outlayer = Concat(dimension=2)

    def forward(self, x0, x1, x2):
        x0 = self.flaten(self.convModule0(x0))
        x1 = self.flaten(self.convModule1(x1))
        x2 = self.flaten(self.convModule2(x2))
        x = self.concat([x0, x1, x2])
        x = self.linear1(self.linear0(x))
        obj_pred = self.linear_obj_pred(x)
        y_pred = self.linear_vert_pred(x)

        return obj_pred, y_pred

if __name__ == '__main__':
    from models.backbones.csp_darknet import CSPDarknet
    from models.necks.pafpn import PAFPN

    anchors = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326]
    ]


    backbone = CSPDarknet(0.67, 0.75, ch=3, nc=1, anchors=anchors)
    pafpn = PAFPN(0.67, 0.75, ch=[192, 384, 768], nc=1, anchors=anchors)
    head  = DenseHead(ch=[192, 384, 768])
    x = torch.randn([8, 3, 1024, 64])
    x = backbone(x)
    x = pafpn(*x)
    obj,y_pred = head(*x)
    print('ending')

