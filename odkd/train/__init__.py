from odkd.train.train import SSDTrainer


__all__ = [
    'SSDTrainer'
]

trainer_factory = {
    'ssd': SSDTrainer
}


def create_ssd_trainer(config):
    trainer = trainer_factory[config['detection']]
    return trainer()
