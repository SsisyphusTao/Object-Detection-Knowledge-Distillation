import torch
from odkd.utils import (
    Config
)


class Trainer():
    def __init__(self, config: Config) -> None:
        self.config = config
        self.config.check()

    def start(self):
        pass
