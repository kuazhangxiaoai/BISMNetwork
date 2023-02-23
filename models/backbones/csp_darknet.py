import torch
import torch.nn as nn

from models.common import Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Concat, Contract
from models.common import MixConv2d, Focus, CrossConv,BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, Expand
from utils.general import make_divisible

config = [
    {'from': -1, 'number': 1, 'module': Conv, 'args': [64, 6, 2, 2], 'output': False},
    {'from': -1, 'number': 1, 'module': Conv, 'args': [128, 3, 2],   'output': False},
    {'from': -1, 'number': 3, 'module': C3,   'args': [128],         'output': False},
    {'from': -1, 'number': 1, 'module': Conv, 'args': [256, 3, 2],   'output': False},
    {'from': -1, 'number': 6, 'module': C3,   'args': [256],         'output': True },
    {'from': -1, 'number': 1, 'module': Conv, 'args': [512, 3, 2],   'output': False},
    {'from': -1, 'number': 9, 'module': C3,   'args': [512],         'output': True },
    {'from': -1, 'number': 1, 'module': Conv, 'args': [1024, 3, 2],  'output': False},
    {'from': -1, 'number': 3, 'module': C3,   'args': [1024],        'output': False},
    {'from': -1, 'number': 9, 'module': SPPF, 'args': [1024, 5],     'output': True }
]

def parse_model(blocks, ch, nc, depth, width):  # model_dict, input_channels(3)
    #LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    nc, gd, gw =  nc, depth, width

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, block in enumerate(blocks):  # from, number, module, args
        f, n, m,args, out = block['from'], block['number'], block['module'],block['args'], block['output']
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np, m_.out = i, f, t, np, out  # attach index, 'from' index, type, number params
        #LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

class CSPDarknet(nn.Module):
    def __init__(self, depth_multiple, width_multiple, ch=3, nc=1):
        super().__init__()
        self.depth = depth_multiple
        self.width = width_multiple
        self.nc = nc
        self.ch = ch
        self.model, self.save = parse_model(config, [ch], nc, depth_multiple, width_multiple)

    def forward(self, x):
        """
        Args:
            x (tensor): (b, 3, height, width), RGB

        Return：
            x (list[P3_out, ...]): tensor.Size(b, self.na, h_i, w_i, c), self.na means the number of anchors scales
        """
        outputs = []
        for m in self.model:
            #if m.f != -1:  # if not from previous layer
            #    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            if m.out:
                outputs.append(x)
        return outputs

if __name__ == '__main__':
    anchors = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326]
    ]
    x = torch.randn([4, 3, 1024, 64])

    Model = CSPDarknet(depth_multiple=0.67, width_multiple=0.75,ch=3, nc=1)
    x = Model(x)
    print(x.shape)








