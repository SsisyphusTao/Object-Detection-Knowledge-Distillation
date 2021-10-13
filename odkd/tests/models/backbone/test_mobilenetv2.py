import torch

from odkd.models.backbone import mobilenet_v2

x = torch.randn(1, 3, 300, 300, dtype=torch.float32)


def test_mobilenet_v2():
    mobilenetv2 = mobilenet_v2()
    y = mobilenetv2(x)
    print(y.shape)
