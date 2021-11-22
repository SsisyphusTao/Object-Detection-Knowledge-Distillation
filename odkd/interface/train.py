import torch


optimizer_factory = {
    'sgd': (torch.optim.SGD, {'lr': 'initial_learning_rate', 'momentum': 'momentum', 'weight_decay': 'weight_decay'}),
    'adam': (torch.optim.Adam, {'lr': 'initial_learning_rate'})
}

scheduler_factory = {
    'steps': (torch.optim.lr_scheduler.MultiStepLR, {'milestones': 'milestones'}),
    'cosine': (torch.optim.lr_scheduler.CosineAnnealingLR, {'T_max': 'epochs'})
}


def create_optimizer(config, **kwargs):
    if config['optim'].lower() == 'sgd':
        optimizer = torch.optim.SGD
    elif config['optim'].lower() == 'adam':
        optimizer = torch.optim.Adam

    args = config[optimizer.__init__.__code__.co_varnames[1:optimizer.__init__.__code__.co_argcount]]
    args.update(kwargs)
    return optimizer(**args)


def create_scheduler(config, **kwargs):
    if config['scheduler'].lower() == 'steps':
        scheduler = torch.optim.lr_scheduler.MultiStepLR
    elif config['scheduler'].lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

    args = config[scheduler.__init__.__code__.co_varnames[1:scheduler.__init__.__code__.co_argcount]]
    args.update(kwargs)
    return scheduler(**args)

