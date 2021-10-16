import torch
from odkd.models.ssdlite import (
    Detect,
    SSDLite,
    create_priorbox
)
from odkd.tests import (
    config
)

cfg = config['SSDLite']
# steps: [16, 32, 64, 100, 150, 300]
x = [
    torch.randn(1, 576, 19, 19, dtype=torch.float32),
    torch.randn(1, 1280, 10, 10, dtype=torch.float32),
    torch.randn(1, 512, 5, 5, dtype=torch.float32),
    torch.randn(1, 256, 3, 3, dtype=torch.float32),
    torch.randn(1, 256, 2, 2, dtype=torch.float32),
    torch.randn(1, 64, 1, 1, dtype=torch.float32)
]
num_priors = sum(map(lambda x: x*x, cfg['feature_maps_size'])) * len(
    cfg['aspect_ratios'])
location = torch.randn(1, num_priors, 4, dtype=torch.float32)
confidence = torch.randn(1, num_priors, cfg['num_classes'], dtype=torch.float32)
priors = torch.randn(num_priors, 4, dtype=torch.float32)

def test_create_priorbox():
    _priors = create_priorbox(cfg)
    print(_priors.shape)
    assert _priors.shape == priors.shape

def test_detect():
    detect = Detect(cfg, priors)
    output = detect(location, confidence)
    print(output.shape)
    assert output.shape == torch.Size([1, cfg['num_classes'], cfg['topK'], 5])

def test_ssdlite():
    head = SSDLite(cfg)
    _location, _confidence, _priors = head(x)
    print(_location.shape, _confidence.shape, _priors.shape)
    assert _location.shape == location.shape
    assert _confidence.shape == confidence.shape
    assert _priors.shape == priors.shape
