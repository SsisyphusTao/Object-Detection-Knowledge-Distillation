"""No docstring :)"""
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from odkd.models.backbone.vgg import vgg16_bn, vgg19_bn
from odkd.models.backbone.mobilenetv2 import mobilenet_v2


__all__ = [
    'vgg16_bn',
    'vgg19_bn',
    'mobilenet_v2'
]
