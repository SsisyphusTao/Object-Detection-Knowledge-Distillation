import torch.nn as nn
import math
import torch
from .prior_box import PriorBox
from .l2norm import L2Norm
from .detection import Detect
from .vgg import voc

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ssd_block(nn.Module):
    def __init__(self, i, m, o, s):
        super().__init__()
        self.block = nn.Sequential(
        nn.Conv2d(i, m, 1, 1, 0),
        nn.BatchNorm2d(m),
        nn.ReLU6(inplace=True),

        nn.Conv2d(m, m, 3, s, s-1, groups=m),
        nn.BatchNorm2d(m),
        nn.ReLU6(inplace=True),

        nn.Conv2d(m, o, 1, 1, 0),
        nn.BatchNorm2d(o),
        nn.ReLU6(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

mbox = [4, 6, 6, 6, 4, 4]
loc = [
    nn.Conv2d(192, mbox[0] * 4, kernel_size=3, padding=1),
    nn.Conv2d(384, mbox[1] * 4, kernel_size=3, padding=1),
    nn.Conv2d(512, mbox[2] * 4, kernel_size=3, padding=1),
    nn.Conv2d(256, mbox[3] * 4, kernel_size=3, padding=1),
    nn.Conv2d(256, mbox[4] * 4, kernel_size=3, padding=1),
    nn.Conv2d(128, mbox[5] * 4, kernel_size=3, padding=1),
]
conf = [
    nn.Conv2d(192, mbox[0] * 21, kernel_size=3, padding=1),
    nn.Conv2d(384, mbox[1] * 21, kernel_size=3, padding=1),
    nn.Conv2d(512, mbox[2] * 21, kernel_size=3, padding=1),
    nn.Conv2d(256, mbox[3] * 21, kernel_size=3, padding=1),
    nn.Conv2d(256, mbox[4] * 21, kernel_size=3, padding=1),
    nn.Conv2d(128, mbox[5] * 21, kernel_size=3, padding=1),
]

class MobileNetV2(nn.Module):
    def __init__(self, loc, conf, mode, n_class=21, input_size=300, width_mult=1.):
        super(MobileNetV2, self).__init__()
        self.num_classes = n_class
        self.mode =mode
        block = InvertedResidual
        input_channel = 32
        last_channel = 384
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            #150->75
            [6, 24, 2, 2],
            #75->38
            [6, 32, 3, 2],
            #38->19
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            #19->10
            # [6, 160, 3, 2],
            # [6, 320, 1, 1],
        ]

        custom_layers_config = [
            # t, c, n, s
            #300
            [1, 16, 1, 1],
            #300->150
            [6, 24, 1, 2],
            #150->75
            [6, 32, 2, 2],
            #75->38
            [6, 64, 3, 2],
            #-------------------------->first feature layer
            #38->19
            [6, 96, 4, 2],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
            #-------------------------->second feature layer
        ]

        # building first layer
        # assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                # if i == 0 and c == 64:
                #     self.features.append(nn.Conv2d(32, 512, 1, 1, 0)) #7
                #     self.features.append(nn.ReLU6(inplace=True)) #8
                #     self.features.append(nn.Conv2d(512, 512, 3, 2, 1, groups=512)) #9
                #     self.features.append(nn.ReLU6(inplace=True)) #10
                #     self.features.append(nn.Conv2d(512, 64, 1, 1, 0)) #11
                # elif i == 0:
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.loc = nn.ModuleList(loc)
        self.conf = nn.ModuleList(conf)
        # make it nn.Sequential
        self.features = nn.ModuleList(self.features)

        self.ssd=nn.ModuleList(
        #19->10
        [ssd_block(384,256,512,2),
        #10->5
        ssd_block(512,128,256,2),
        #5->3
        ssd_block(256,128,256,1),
        #3->1
        ssd_block(256,64,128,1)]
        )

        self.priorbox = PriorBox(voc)
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self.L2Norm = L2Norm(192, 20)

        # # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, n_class),
        # )
        self.detect = Detect()
        self.softmax = nn.Softmax(dim=-1)
        self.adaptation = nn.Sequential(nn.Conv2d(192, 512, 1, 1, 0))
        self._initialize_weights()

    def forward(self, x):
        sources = []
        loc = []
        conf = []

        for i in range(7):
            x = self.features[i](x)

        for i in range(3):
            x = self.features[7].conv[i](x)

        #adaption layer
        s = self.L2Norm(x)
        sources.append(s)

        for i in range(3, len(self.features[7].conv)):
            x = self.features[7].conv[i](x)

        for i in range(8, len(self.features)):
            x = self.features[i](x)
        sources.append(x)
        
        for block in self.ssd:
            x = block(x)
            sources.append(x)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.mode == 'test':
            output = self.detect.apply(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                self.num_classes)),                # conf preds
                self.priors.type(type(x.data)).cuda()                  # default boxes
            )
        elif self.mode == 'train':
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors,
                self.adaptation(s)
            )
        # x = self.features(x)
        # x = x.mean(3).mean(2)
        # x = self.classifier(x)
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
            # elif isinstance(m, nn.Linear):
            #     n = m.weight.size(1)
            #     m.weight.data.normal_(0, 0.01)
            #     m.bias.data.zero_()
    def load_weights(self, base_file):
        self.load_state_dict(torch.load(base_file,
                                        map_location=lambda storage, loc: storage))

def mobilenetv2_module(mode):
    return MobileNetV2(loc, conf, mode)