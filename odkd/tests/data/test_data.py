from odkd.data import (
    create_dataloader
)


def test_create_dataloader(config):
    dataloader = create_dataloader(config)
    for images, targets in dataloader:
        print(images.shape)
        assert images.shape == (config['batch_size'], 3, config['input_size'], config['input_size'])
        assert len(targets) == config['batch_size']
        break
