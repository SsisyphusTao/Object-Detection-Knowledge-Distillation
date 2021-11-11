from odkd.train.train import SSDTrainer


__all__ = [
    'create_trainer',
    'SSDTrainer'
]

trainer_factory = {
    'ssdlite': SSDTrainer
}


def create_trainer(config):
    trainer = trainer_factory[config['detection']]
    return trainer(config)
