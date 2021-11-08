from odkd.data.voc import create_voc_dataloader

__all__ = [
    'create_dataloader'
]

dataset_factory = {
    'voc': create_voc_dataloader
}


def create_dataloader(config, augmentation):
    dataloader = dataset_factory[config['dataset']]
    return dataloader(config['dataset_path'], config['batch_size'], config['workers'], augmentation)
