"""Some layers and blocks for model construction"""
from torch import nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = [
    'load_state_dict_from_url',
    '_make_divisible',
    'SeperableConv2d',
    'ConvBNReLU',
    'InvertedResidual',
    'SSDLite'
]


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                  groups=in_channels, stride=stride, padding=padding),
        nn.BatchNorm2d(in_channels),
        ReLU(),
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels, kernel_size=1),
    )


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                      padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim,
                          kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride,
                       groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SSDLite(nn.Module):
    """SSDlite detect head which only has 3000 anchors"""

    def __init__(self, num_classes, width_mult=1.0, div_nearest=8):
        super().__init__()

        def div(x): return _make_divisible(x * width_mult, div_nearest)

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
