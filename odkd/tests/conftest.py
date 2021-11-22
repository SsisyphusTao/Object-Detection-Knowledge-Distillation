import os

import pytest
import torch

from odkd.utils import (
    Config
)
from odkd.interface import create_dataloader, create_augmentation, create_ssdlite, create_priorbox


@pytest.fixture(scope='session')
def config():
    cfg = Config(argv=['odkd/utils/template.yml'])
    cfg['dataset_path'] = os.path.dirname(os.path.realpath(__file__)) + '/data'
    cfg['batch_size'] = 2
    cfg['epochs'] = 2
    cfg['cuda'] = False
    return cfg


@pytest.fixture(scope='session')
def input_tensor(config):
    return torch.randn(config['batch_size'], 3, config['input_size'], config['input_size'], dtype=torch.float32)


@pytest.fixture(scope='session')
def num_priors(config):
    return sum(map(lambda x: x*x, config['feature_maps_size'])) * len(config['aspect_ratios'])


@pytest.fixture(scope='session')
def localization(config, num_priors):
    return torch.randn(config['batch_size'], num_priors, 4, dtype=torch.float32)


@pytest.fixture(scope='session')
def confidence(config, num_priors):
    return torch.randn(
        config['batch_size'], num_priors, config['num_classes'], dtype=torch.float32)


@pytest.fixture(scope='session')
def priors(config):
    return create_priorbox(config['input_size'], config['feature_maps_size'],
                           config['steps'], config['max_sizes'], config['min_sizes'],
                           config['aspect_ratios'], config['clip'])


@pytest.fixture(scope='session')
def ssdlite(config, priors):
    config['priors'] = priors
    return create_ssdlite('mobilenetv2', config)


@pytest.fixture(scope='session')
def dataloader(config, augmentation):
    config['augmentation'] = augmentation
    return create_dataloader(config, image_set='train')


@pytest.fixture(scope='session')
def targets(config, num_priors):
    return torch.rand(config['batch_size'], num_priors, 5, dtype=torch.float32)


@pytest.fixture(scope='session')
def predictions(localization, confidence):
    return localization, confidence


@pytest.fixture(scope='session')
def augmentation(config, priors):
    config['priors'] = priors
    return create_augmentation(config)
