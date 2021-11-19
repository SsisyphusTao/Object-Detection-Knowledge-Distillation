from odkd.data import create_dataloader
from odkd.data.transforms import create_augmentation

def test_create_augmentation(config, priors):
    config['priors'] = priors
    create_augmentation(config)


def test_create_dataloader(config, augmentation):
    config['augmentation'] = augmentation
    dataloader = create_dataloader(config)
    for images, targets in dataloader:
        print(images.shape)
        assert images.shape == (
            config['batch_size'], 3, config['input_size'], config['input_size'])
        assert len(targets) == config['batch_size']
        break
