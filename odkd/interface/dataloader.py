from odkd.data.voc import create_voc_dataloader


def create_dataloader(config, **kwargs):
    if config['dataset'].lower() == 'voc':
        dataloader = create_voc_dataloader
    else:
        raise ValueError('Unsupport dataset %s' % config['dataset'])

    args = config[dataloader.__code__.co_varnames[:dataloader.__code__.co_argcount]]
    args.update(kwargs)
    return dataloader(**args)
