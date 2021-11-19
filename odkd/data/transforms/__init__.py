from odkd.data.transforms.ssd import SSDAugmentation
from odkd.data.voc import voc_transform


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
