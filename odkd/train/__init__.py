import os
import torch

from odkd.train.train import SSDTrainer
from odkd.train.eval import Evaluator
from odkd.interface import create_ssdlite, create_priorbox

__all__ = [
    'create_trainer',
]


def create_trainer(config):
    if config['model'].lower() == 'ssdlite':
        trainer = SSDTrainer
    else:
        raise ValueError('Unsupport model %s' % config['model'])
    return trainer(config)


def create_evaluator(config):
    if config['model'].lower() == 'ssdlite':
        config['priors'] = create_priorbox(**config)
        model = create_ssdlite(config['student_backbone'], config)
    else:
        raise ValueError('Unsupport model %s' % config['model'])
    model.load_state_dict(torch.load(config['last']))
    if config['cuda']:
        model = model.cuda()

    return Evaluator(model, config)
