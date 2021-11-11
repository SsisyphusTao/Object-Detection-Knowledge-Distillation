from odkd.train import (
    create_trainer
)


def test_ssdtrainer(config):
    trainer = create_trainer(config)
    trainer.start()
