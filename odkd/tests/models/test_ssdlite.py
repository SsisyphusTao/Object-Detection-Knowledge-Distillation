import torch
from odkd.models.ssdlite import (
    Detect,
    ssd_lite
)
from odkd.tests import (
    config,
    test_batch_size,
    ssd_input,
    location,
    confidence,
    priors,

)


def test_detect():
    cfg = config['SSDLite']
    detect = Detect(cfg, priors)
    output = detect(location, confidence)
    print(output.shape)
    assert output.shape == torch.Size(
        [test_batch_size, cfg['num_classes'], cfg['topK'], 5])


def test_vgg16_ssdlite():
    ssdlite = ssd_lite('vgg16', config)
    test_location, test_confidence, test_priors = ssdlite(ssd_input)
    print(test_location.shape, test_confidence.shape, test_priors.shape)
    assert test_location.shape == location.shape
    assert test_confidence.shape == confidence.shape
    assert test_priors.shape == priors.shape


def test_mobilenetv2_ssdlite():
    ssdlite = ssd_lite('mobilenetv2', config)
    test_location, test_confidence, test_priors = ssdlite(ssd_input)
    print(test_location.shape, test_confidence.shape, test_priors.shape)
    assert test_location.shape == location.shape
    assert test_confidence.shape == confidence.shape
    assert test_priors.shape == priors.shape
