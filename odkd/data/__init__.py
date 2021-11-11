from odkd.data.ssd_augments import SSDAugmentation
from odkd.data.voc import voc_transform, create_voc_dataloader

__all__ = [
    'create_dataloader'
]

dataset_factory = {
    'voc': create_voc_dataloader
}

augmentation_factory = {
    'ssdlite': [SSDAugmentation, {'voc': voc_transform}]
}


def create_augmentation(config):
    augmentation, transform = augmentation_factory[config['detection']]
    return augmentation(config['input_size'],
                        config['mean'],
                        config['overlap_thresh'],
                        config['variance'],
                        config['priors'],
                        transform[config['dataset']])


def create_dataloader(config):
    dataloader = dataset_factory[config['dataset']]
    return dataloader(config['dataset_path'],
                      config['batch_size'],
                      config['workers'],
                      config['augmentation'],
                      config['local_rank'],
                      config['world_size'])
