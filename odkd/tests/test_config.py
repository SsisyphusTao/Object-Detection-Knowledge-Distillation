from odkd.utils import (
    Config
)


def test_config():
    default_parameters = {
        'MODE': 'Test'
    }
    config = Config(default_parameters)
    config['TestPass'] = True
    config.print()
