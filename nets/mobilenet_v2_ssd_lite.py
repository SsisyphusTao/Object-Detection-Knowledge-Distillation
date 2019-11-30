import torch
import torch.nn as nn
import math
from .prior_box import PriorBox
from .l2norm import L2Norm
from .detection import Detect
# Modified from https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py.
# In this version, Relu6 is replaced with Relu to make it ONNX compatible.
# BatchNorm Layer is optional to make it easy do batch norm confusion.

voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [19, 10, 5, 3, 2, 1],
    'min_dim': 300,
    'steps': [16, 32, 64, 100, 150, 300],
    'min_sizes': [60, 105, 150, 195, 240, 285],
    'max_sizes': [105, 150, 195, 240, 285, 330],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}


def conv_bn(inp, oup, stride, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            ReLU(inplace=True)
        )


def conv_1x1_bn(inp, oup, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            ReLU(inplace=True)
        )

def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        nn.BatchNorm2d(in_channels),
        ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_batch_norm=True, onnx_compatible=False):
        super(InvertedResidual, self).__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )
        else:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
wr = 1.0
extras = nn.ModuleList([
    InvertedResidual(round(1280*wr), round(512*wr), stride=2, expand_ratio=0.2),
    InvertedResidual(round(512*wr), round(256*wr), stride=2, expand_ratio=0.25),
    InvertedResidual(round(256*wr), round(256*wr), stride=2, expand_ratio=0.5),
    InvertedResidual(round(256*wr), round(64*wr), stride=2, expand_ratio=0.25)
])

regression_headers = nn.ModuleList([
    SeperableConv2d(in_channels=round(576 * wr), out_channels=6 * 4,
                    kernel_size=3, padding=1, onnx_compatible=False),
    SeperableConv2d(in_channels=round(1280*wr), out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
    SeperableConv2d(in_channels=round(512*wr), out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
    SeperableConv2d(in_channels=round(256*wr), out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
    SeperableConv2d(in_channels=round(256*wr), out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
    nn.Conv2d(in_channels=round(64*wr), out_channels=6 * 4, kernel_size=1),
])

classification_headers = nn.ModuleList([
    SeperableConv2d(in_channels=round(576 * wr), out_channels=6 * 21, kernel_size=3, padding=1),
    SeperableConv2d(in_channels=round(1280*wr), out_channels=6 * 21, kernel_size=3, padding=1),
    SeperableConv2d(in_channels=round(512*wr), out_channels=6 * 21, kernel_size=3, padding=1),
    SeperableConv2d(in_channels=round(256*wr), out_channels=6 * 21, kernel_size=3, padding=1),
    SeperableConv2d(in_channels=round(256*wr), out_channels=6 * 21, kernel_size=3, padding=1),
    nn.Conv2d(in_channels=round(64*wr), out_channels=6 * 21, kernel_size=1),
])

class MobileNetV2(nn.Module):
    def __init__(self, loc, conf, extras, mode, n_class=21, input_size=300, width_mult=wr, dropout_ratio=0.2,
                 use_batch_norm=True, onnx_compatible=False):
        super(MobileNetV2, self).__init__()
        self.num_classes = n_class
        self.mode =mode
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        # assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, onnx_compatible=onnx_compatible)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s,
                                               expand_ratio=t, use_batch_norm=use_batch_norm,
                                               onnx_compatible=onnx_compatible))
                else:
                    self.features.append(block(input_channel, output_channel, 1,
                                               expand_ratio=t, use_batch_norm=use_batch_norm,
                                               onnx_compatible=onnx_compatible))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel,
                                         use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible))
        self.regression_headers = loc
        self.classification_headers = conf
        self.extras = extras
        self.base_net = nn.ModuleList(self.features)

        self.detect = Detect()
        self.priorbox = PriorBox(voc)
        self.adaptation = nn.Conv2d(192, 512, 1, 1, 0)
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self._initialize_weights()

    def forward(self, x):
        sources = []
        loc = []
        conf = []

        for i in range(7):
            x = self.base_net[i](x)

        for i in range(3):
            x = self.base_net[7].conv[i](x)
        adp = x
        for i in range(3, len(self.base_net[14].conv)):
            x = self.base_net[7].conv[i](x)

        for i in range(8, 14):
            x = self.base_net[i](x)

        for i in range(3):
            x = self.base_net[14].conv[i](x)
        sources.append(x)
        for i in range(3, len(self.base_net[14].conv)):
            x = self.base_net[14].conv[i](x)
      
        for i in range(15, len(self.base_net)):
            x = self.base_net[i](x)
        sources.append(x)
        
        for block in self.extras:
            x = block(x)
            sources.append(x)

        for (x, l, c) in zip(sources, self.regression_headers, self.classification_headers):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.mode == 'test':
            output = self.detect.apply(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                nn.Softmax(dim=-1)(conf.view(conf.size(0), -1,
                                self.num_classes)),                # conf preds
                self.priors.type(type(x.data)).cuda()                  # default boxes
            )
        elif self.mode == 'train':
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors,
                self.adaptation(adp)
            )
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def create_mobilenetv2_ssd_lite(mode):
    return MobileNetV2(regression_headers, classification_headers, extras, mode)