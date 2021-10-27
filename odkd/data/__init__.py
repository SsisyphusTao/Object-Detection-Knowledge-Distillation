from odkd.data.voc import create_voc_dataloader

__all__ = [
    'create_dataloader'
]

dataset_factory = {
    'voc': create_voc_dataloader
}


def create_dataloader(cfg, augmentation):
    return dataset_factory[cfg['dataset']](cfg, augmentation)
