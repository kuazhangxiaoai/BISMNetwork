import torch
import torch.nn as nn

from models.common import Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Concat, Contract
from models.common import MixConv2d, Focus, CrossConv,BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, Expand
from utils.general import make_divisible

config = [
    {'from': 2,      'number': 1,  'module': Conv,        'args': [512, 1, 1],          'in': 2 },
    {'from': -1,     'number': 1,  'module': nn.Upsample, 'args': [None, 2, 'nearest'], 'in': None},
    {'from':[-1, 1], 'number': 1,  'module': Concat,      'args': [1],                  'in': 1 },#from backbone output 0
    {'from': -1,     'number': 3,  'module': C3,          'args': [512],                'in': None},

    {'from': -1,     'number': 1,  'module': Conv,        'args': [256, 1, 1],          'in': None},
    {'from': -1,     'number': 1,  'module': nn.Upsample, 'args': [None, 2, 'nearest'], 'in': None},
    {'from':[-1, 0], 'number': 1,  'module': Concat,      'args': [1],                  'in': 0 },
    {'from': -1,     'number': 3,  'module': C3,          'args': [256],                'in': None},

    {'from': -1,      'number': 1,  'module': Conv,        'args': [256, 3, 2],         'in': None},
    {'from':[-1, 4],  'number': 1,  'module': Concat,      'args': [1],                 'in': None},
    {'from': -1,      'number': 3,  'module': C3,          'args': [512],               'in': None},

    {'from': -1,      'number': 1,  'module': Conv,        'args': [256, 3, 2],         'in': None},
    {'from':[-1, 0],  'number': 1,  'module': Concat,      'args': [1],                 'in': None},
    {'from': -1,      'number': 3,  'module': C3,          'args': [1024],              'in': None}
]

def makeConv(conf_json, x_ch, f_ch, gd, gw):
    if conf_json['in'] is not None:
        c1 = x_ch[conf_json['in']]
    else:
        c1 = f_ch[conf_json['from']]
    c2 = make_divisible(conf_json['args'][0] * gw, 8)
    f_ch.append(c2)
    args = [c1, c2, *conf_json['args'][1:]]
    return Conv(*args)

def makeUpsample(conf_json, f_ch):
    f_ch.append(f_ch[-1])
    return nn.Upsample(*conf_json['args'])

def makeConcat(conf_json, x_ch, f_ch):
    if conf_json['in'] is not None:
        ch1, ch2 = f_ch[-1], x_ch[conf_json['in']]
    else:
        ch1, ch2 = f_ch[-1], f_ch[conf_json['from'][1]]
    f_ch.append(ch1 + ch2)
    return Concat()

def makeC3(conf_json, f_ch, gd, gw):
    n = conf_json['number']
    n = n_ = max(round(n * gd), 1)

    c1, c2 = f_ch[-1],  conf_json['args'][0]
    c2 = make_divisible(c2 * gw, 8)
    args = [c1, c2, *conf_json['args'][1:], False]
    args.insert(2, n)
    n = 1
    f_ch.append(c2)
    return nn.Sequential(*(C3(*args) for _ in range(n))) if n > 1 else C3(*args)

class PAFPN(nn.Module):
    def __init__(self, depth_multiple, width_multiple, ch, nc=1):
        super().__init__()
        self.depth = depth_multiple
        self.width = width_multiple
        self.nc = nc
        self.ch = ch
        self.fch = []

        self.ConvModule0 = makeConv(conf_json=config[0], x_ch=ch, f_ch=self.fch, gd=self.depth, gw=self.width)
        self.Upsample1   = makeUpsample(conf_json=config[1], f_ch=self.fch)
        self.Concat2     = makeConcat(conf_json=config[2], x_ch=ch , f_ch=self.fch)
        self.C3Module3   = makeC3(conf_json=config[3], f_ch=self.fch, gd=self.depth, gw=self.width)

        self.ConvModule4 = makeConv(conf_json=config[4], x_ch=ch, f_ch=self.fch, gd=self.depth, gw=self.width)
        self.Upsample5   = makeUpsample(conf_json=config[5], f_ch=self.fch)
        self.Concat6     = makeConcat(conf_json=config[6], x_ch=ch , f_ch=self.fch)
        self.C3Module7   = makeC3(conf_json=config[7], f_ch=self.fch, gd=self.depth, gw=self.width)

        self.ConvModule8 = makeConv(conf_json=config[8], x_ch=ch, f_ch=self.fch, gd=self.depth, gw=self.width)
        self.Concat9 = makeConcat(conf_json=config[9], x_ch=ch, f_ch=self.fch)
        self.C3Module10 = makeC3(conf_json=config[10], f_ch=self.fch, gd=self.depth, gw=self.width)

        self.ConvModule11 = makeConv(conf_json=config[11], x_ch=ch, f_ch=self.fch, gd=self.depth, gw=self.width)
        self.Concat12 = makeConcat(conf_json=config[12], x_ch=ch, f_ch=self.fch)
        self.C3Module13 = makeC3(conf_json=config[13], f_ch=self.fch, gd=self.depth, gw=self.width)

    def forward(self, x0, x1, x2):
        y,outs = [],[]
        x = self.ConvModule0(x2)
        y.append(x)
        x = self.Upsample1(x)
        x = self.Concat2([x1, x])
        x = self.C3Module3(x)

        x = self.ConvModule4(x)
        y.append(x)
        x = self.Upsample5(x)
        x = self.Concat6([x0, x])
        x = self.C3Module7(x)
        outs.append(x)

        x = self.ConvModule8(x)
        x = self.Concat9([y[-1], x])
        x = self.C3Module10(x)
        outs.append(x)

        x = self.ConvModule11(x)
        x = self.Concat12([x, y[0]])
        x = self.C3Module13(x)
        outs.append(x)
        return outs


if __name__ == '__main__':
    from models.backbones.csp_darknet import CSPDarknet

    anchors = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326]
    ]
    x = torch.randn(4,3, 512, 64)
    backbone = CSPDarknet(0.67, 0.75, ch=3, nc=1)
    pafpn = PAFPN(0.67, 0.75, ch=[192,384,768], nc=1)
    x = backbone(x)
    x = pafpn(*x)
    print('ending')
