
"""SSDLite detector relative blocks"""
import torch
from torch import nn
from math import sqrt
from itertools import product
from ._utils import InvertedResidual, SeperableConv2d, _make_divisible


def create_priorbox(cfg):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.

    Args:
        cfg (dict): SSDLite structure config.

    """

    # number of priors for feature map location (either 4 or 6)
    mean = []
    for k, f in enumerate(cfg['feature_maps_size']):
        for i, j in product(range(f), repeat=2):
            f_k = cfg['input_size'] / cfg['steps'][k]
            # unit center x,y
            cx = (j + 0.5) / f_k
            cy = (i + 0.5) / f_k

            # aspect_ratio: 1
            # rel size: min_size
            s_k = cfg['min_sizes'][k]/cfg['input_size']
            mean += [cx, cy, s_k, s_k]

            # aspect_ratio: 1
            # rel size: sqrt(s_k * s_(k+1))
            s_k_prime = sqrt(s_k * (cfg['max_sizes'][k]/cfg['input_size']))
            mean += [cx, cy, s_k_prime, s_k_prime]

            # rest of aspect ratios
            for ar in cfg['aspect_ratios'][k]:
                mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
    # back to torch land
    output = torch.Tensor(mean).view(-1, 4)
    if cfg['clip']:
        output.clamp_(max=1, min=0)
    return output


class SSDLite(nn.Module):
    """SSDlite detect head which only has 3000 anchors

    Args:
        num_classes (int): number of prediction classes.
        width_mult (:obj:`float`, optional): scale ratio of channel width.
        div_nearest (:obj:`int`, optional): multiple of channels.

    Attributes:
        cfg (dict): default prior boxex config.
    """

    def __init__(self, num_classes, width_mult=1.0, div_nearest=8):
        super().__init__()

        self.num_classes = num_classes

        def div(x):
            return _make_divisible(x * width_mult, div_nearest)

        self.extras = nn.ModuleList([
            InvertedResidual(div(1280), div(512), stride=2, expand_ratio=0.2),
            InvertedResidual(div(512), div(256), stride=2, expand_ratio=0.25),
            InvertedResidual(div(256), div(256), stride=2, expand_ratio=0.5),
            InvertedResidual(div(256), div(64), stride=2, expand_ratio=0.25)
        ])
        self.regression_headers = nn.ModuleList([
            SeperableConv2d(in_channels=div(576), out_channels=6 * 4,
                            kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=div(
                1280), out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=div(512), out_channels=6 * 4,
                            kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=div(256), out_channels=6 * 4,
                            kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=div(256), out_channels=6 * 4,
                            kernel_size=3, padding=1, onnx_compatible=False),
            nn.Conv2d(in_channels=div(64),
                      out_channels=6 * 4, kernel_size=1),
        ])
        self.classification_headers = nn.ModuleList([
            SeperableConv2d(in_channels=div(576),
                            out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=div(1280),
                            out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=div(512),
                            out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=div(256),
                            out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=div(256),
                            out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=div(64),
                      out_channels=6 * num_classes, kernel_size=1),
        ])

    def forward(self, layers):
        loc = []
        conf = []

        for (x, l, c) in zip(layers, self.regression_headers, self.classification_headers):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([i.view(i.size(0), -1) for i in loc], 1)
        conf = torch.cat([i.view(i.size(0), -1) for i in conf], 1)

        if self.training:
            return loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes)
        else:
            return
