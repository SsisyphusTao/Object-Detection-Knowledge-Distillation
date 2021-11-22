from odkd.interface import create_dataloader, create_augmentation

def test_create_augmentation(config, priors):
    config['priors'] = priors
    create_augmentation(config)


def test_create_dataloader(config, augmentation):
    config['augmentation'] = augmentation
    dataloader = create_dataloader(image_set='train', config=config)
    for images, targets in dataloader:
        print(images.shape)
        assert images.shape == (
            config['batch_size'], 3, config['input_size'], config['input_size'])
        assert len(targets) == config['batch_size']
        break
