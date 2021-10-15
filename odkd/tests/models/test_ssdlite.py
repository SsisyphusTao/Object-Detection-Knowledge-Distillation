import torch
from odkd.models.ssdlite import (
    SSDLite,
    create_priorbox
)
from odkd.tests import (
    config
)

# steps: [16, 32, 64, 100, 150, 300]
x = [
    torch.randn(1, 576, 19, 19, dtype=torch.float32),
    torch.randn(1, 1280, 10, 10, dtype=torch.float32),
    torch.randn(1, 512, 5, 5, dtype=torch.float32),
    torch.randn(1, 256, 3, 3, dtype=torch.float32),
    torch.randn(1, 256, 2, 2, dtype=torch.float32),
    torch.randn(1, 64, 1, 1, dtype=torch.float32)
]
num_priors = sum(map(lambda x: x*x, config['SSDLite']['feature_maps_size'])) * len(
    config['SSDLite']['aspect_ratios'])


def test_create_priorbox():
    priors = create_priorbox(config['SSDLite'])
    print(priors.shape)
    assert priors.shape == torch.Size([num_priors, 4])


def test_ssdlite():
    head = SSDLite(80)
    location, config = head(x)
    print(location.shape, config.shape)
    assert location.shape[1] == config.shape[1] == num_priors
