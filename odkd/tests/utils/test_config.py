"""No docstring :)"""
import pytest
from odkd.utils import (
    Config
)

@pytest.mark.run(order=0)
@pytest.mark.dependency(name='test_config')
def test_config():
    test_arguments = {
        'MODE': 'Test'
    }
    config = Config(test_arguments)
    config.parse_args(['-c', 'default_training_config.yml'])
    config['TestPass'] = True
    config.check()
    config.print()
