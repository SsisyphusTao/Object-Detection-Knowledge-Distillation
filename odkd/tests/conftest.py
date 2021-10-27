import pytest
import torch
from odkd.utils import (
    Config
)


@pytest.fixture(scope='session')
def config():
    cfg = Config()
    cfg.parse_args(['-c', 'default_training_config.yml'])
    return cfg


@pytest.fixture(scope='session')
def input_tensor(config):
    return torch.randn(
        config['batch_size'], 3, config['input_size'], config['input_size'], dtype=torch.float32)


@pytest.fixture(scope='session')
def num_priors(config):
    return sum(map(lambda x: x*x, config['feature_maps_size'])) * len(
        config['aspect_ratios'])


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
