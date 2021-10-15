import pytest
from odkd.train import (
    Trainer
)
from odkd.tests import (
    config
)


@pytest.mark.run(order=1)
@pytest.mark.dependency(depends=['test_config'], scope='session')
def test_train():
    Trainer(config)
