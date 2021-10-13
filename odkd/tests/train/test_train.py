import pytest
from odkd.utils import (
    Config
)
from odkd.train import (
    Trainer
)


@pytest.mark.run(order=1)
@pytest.mark.dependency(depends=['test_config'], scope='session')
def test_train():
    config = Config()
    config.parse_args(['-c', 'default_training_config.yml'])
    Trainer(config)
