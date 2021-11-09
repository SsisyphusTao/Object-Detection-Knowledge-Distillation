import torch

optimizer_factory = {
    'sgd': [torch.optim.SGD, {'momentum': 0.9, 'weight_decay': 5e-4}],
    'adam': [torch.optim.Adam, {}]
}


def create_optimizer(config):
    optimizer, args = optimizer_factory[config['optimizer'].lower()]
    for i in args:
        args[i] = config[i]
    args['lr'] = config['initial_learning_rate']
    return lambda x: optimizer(x, **args)
