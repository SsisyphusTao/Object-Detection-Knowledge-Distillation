import os

import pytest
import torch
from PIL import Image

from odkd.utils import (
    Config
)
from odkd.data import create_dataloader
from odkd.models.ssdlite import ssd_lite


@pytest.fixture(scope='session')
def config():
    cfg = Config()
    cfg.parse_args(['-c', 'default_training_config.yml'])
    cfg['dataset_path'] = os.path.dirname(os.path.realpath(__file__)) + '/data'
    cfg['batch_size'] = 2
    return cfg


@pytest.fixture(scope='session')
def input_tensor(config):
    return torch.randn(config['batch_size'], 3, config['input_size'], config['input_size'], dtype=torch.float32)


@pytest.fixture(scope='session')
def num_priors(config):
    return sum(map(lambda x: x*x, config['feature_maps_size'])) * len(config['aspect_ratios'])


@pytest.fixture(scope='session')
def location(config, num_priors):
    return torch.randn(config['batch_size'], num_priors, 4, dtype=torch.float32)


@pytest.fixture(scope='session')
def confidence(config, num_priors):
    return torch.randn(
        config['batch_size'], num_priors, config['num_classes'], dtype=torch.float32)


@pytest.fixture(scope='session')
def priors(num_priors):
    return torch.randn(num_priors, 4, dtype=torch.float32)


@pytest.fixture(scope='session')
def ssdlite(config):
    return ssd_lite('mobilenetv2', config)


@pytest.fixture(scope='session')
def dataloader(config):
    return create_dataloader(config)


@pytest.fixture(scope='session')
def targets(config, num_priors):
    return torch.rand(config['batch_size'], num_priors, 5, dtype=torch.float32)


@pytest.fixture(scope='session')
def predictions(location, confidence, priors):
    return location, confidence, priors
