from odkd.data.voc import create_voc_dataloader

__all__ = [
    'create_dataloader'
]

dataset_factory = {
    'voc': create_voc_dataloader
}


def create_dataloader(config, image_set='train'):
    dataloader = dataset_factory[config['dataset']]
    return dataloader(image_set,
                      config['dataset_path'],
                      config['batch_size'],
                      config['workers'],
                      config['augmentation'],
                      config['local_rank'],
                      config['world_size'])
