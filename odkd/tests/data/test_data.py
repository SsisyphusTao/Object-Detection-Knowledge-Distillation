from odkd.data.ssd_augments import SSDAugmentation
from odkd.data import create_dataloader
from odkd.data.voc import voc_transform

def test_create_dataloader(config, priors):
    augment = SSDAugmentation(config, [voc_transform, priors])
    dataloader = create_dataloader(config, augment)
    for images, targets in dataloader:
        print(images.shape)
        assert images.shape == (config['batch_size'], 3, config['input_size'], config['input_size'])
        assert len(targets) == config['batch_size']
        break
