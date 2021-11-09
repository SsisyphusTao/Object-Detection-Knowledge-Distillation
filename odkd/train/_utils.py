import torch


optimizer_factory = {
    'sgd': (torch.optim.SGD, {'lr': 'initial_learning_rate', 'momentum': 'momentum', 'weight_decay': 'weight_decay'}),
    'adam': (torch.optim.Adam, {'lr': 'initial_learning_rate'})
}

scheduler_factory = {
    'steps': (torch.optim.lr_scheduler.MultiStepLR, {'milestones': 'milestones'}),
    'cosine': (torch.optim.lr_scheduler.CosineAnnealingLR, {'T_max': 'epochs'})
}


def create_optimizer(config):
    optimizer, args = optimizer_factory[config['optimizer'].lower()]
    for i, j in args.items():
        args[i] = config[j] if isinstance(j, str) else j
    return lambda x: optimizer(x, **args)


def create_scheduler(config):
    scheduler, args = scheduler_factory[config['scheduler'].lower()]
    for i, j in args.items():
        args[i] =  config[j] if isinstance(j, str) else j
    return lambda x: scheduler(x, **args)
