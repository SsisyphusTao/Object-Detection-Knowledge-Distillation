from odkd.data.transforms.ssd import SSDAugmentation
from odkd.data.transforms.base import BaseTransform
from odkd.data.voc import voc_transform


def create_augmentation(config, **kwargs):
    if config['model'].lower() in ['ssd', 'ssdlite']:
        func = SSDAugmentation
    else:
        raise ValueError('Unsupport model: %s' % config['model'])

    args = config[func.__init__.__code__.co_varnames[1:func.__init__.__code__.co_argcount]]
    args.update(kwargs)
    if config['dataset'] == 'voc':
        args['preprocess'] = voc_transform
    else:
        raise ValueError('Unsupport dataset %s' % config['dataset'])

    return func(**args)


def create_transform(config, **kwargs):
    transform = BaseTransform
    args = config[transform.__init__.__code__.co_varnames[1:transform.__init__.__code__.co_argcount]]
    args.update(kwargs)
    if config['dataset'] == 'voc':
        args['preprocess'] = voc_transform
    else:
        raise ValueError('Unsupport dataset %s' % config['dataset'])

    return transform(**args)
