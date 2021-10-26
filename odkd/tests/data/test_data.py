from odkd.data import (
    create_dataloader
)
from odkd.tests import (
    config
)


def test_create_dataloader():
    dataloader = create_dataloader(config['Training'])
    for images, targets in dataloader:
        print(images.shape)
        assert images.shape == (config['Training']['batch_size'], 3, 300, 300)
        assert len(targets) == config['Training']['batch_size']
        break
