import torch
import torch.nn

from odkd.models.backbone import (
    vgg16_bn,
    # vgg19_bn
)

x = torch.randn(1, 3, 300, 300, dtype=torch.float32)


def test_vgg16_bn():
    vgg16 = vgg16_bn()
    y = vgg16(x)
    print(y.shape)
