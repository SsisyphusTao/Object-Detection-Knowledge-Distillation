import torch
from odkd.models.ssdlite import Detect, create_priorbox
from odkd.interface import create_ssdlite


def test_create_priorbox(config, num_priors):
    priorbox = create_priorbox(config['input_size'], config['feature_maps_size'],
                               config['steps'], config['max_sizes'], config['min_sizes'],
                               config['aspect_ratios'], config['clip'])
    print(priorbox.shape)
    assert priorbox.shape == torch.Size([num_priors, 4])


def test_detect(config, priors, localization, confidence):
    detect = Detect(config['num_classes'],
                    config['topK'],
                    config['variance'],
                    config['conf_thresh'],
                    config['nms_thresh'],
                    priors)
    output = detect(localization, confidence)
    print(output.shape)
    assert output.shape == torch.Size(
        [config['batch_size'], config['topK'], 6])


def test_vgg16_ssdlite(config, input_tensor, priors, localization, confidence):
    config['priors'] = priors
    ssdlite = create_ssdlite('vgg16', config)
    test_localization, test_confidence = ssdlite(input_tensor)
    print(test_localization.shape, test_confidence.shape)
    assert test_localization.shape == localization.shape
    assert test_confidence.shape == confidence.shape


def test_mobilenetv2_ssdlite(config, input_tensor, priors, localization, confidence):
    config['priors'] = priors
    ssdlite = create_ssdlite('mobilenetv2', config)
    test_localization, test_confidence = ssdlite(input_tensor)
    print(test_localization.shape, test_confidence.shape)
    assert test_localization.shape == localization.shape
    assert test_confidence.shape == confidence.shape
