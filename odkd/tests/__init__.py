import torch
from odkd.utils import (
    Config
)

config = Config()
config.parse_args(['-c', 'default_training_config.yml'])
test_batch_size = 2

ssd_input = torch.randn(
    test_batch_size, 3, config['SSDLite']['input_size'], config['SSDLite']['input_size'], dtype=torch.float32)

num_priors = sum(map(lambda x: x*x, config['SSDLite']['feature_maps_size'])) * len(
    config['SSDLite']['aspect_ratios'])
location = torch.randn(test_batch_size, num_priors, 4, dtype=torch.float32)
confidence = torch.randn(
    test_batch_size, num_priors, config['SSDLite']['num_classes'], dtype=torch.float32)
priors = torch.randn(num_priors, 4, dtype=torch.float32)
