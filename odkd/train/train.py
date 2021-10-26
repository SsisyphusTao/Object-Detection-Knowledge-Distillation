from odkd.utils import (
    Config
)


class Trainer():
    def __init__(self, config: Config) -> None:
        self._config = config
        self._config.check()

    def start(self):
        pass
