from odkd.train import (
    SSDTrainer
)


def test_ssdtrainer(config):
    trainer = SSDTrainer(config)
    trainer.train_one_epoch(0)
