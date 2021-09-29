from odkd.utils import (
    Config
)


def test_config():
    default_parameters = {
        'MODE': 'Test'
    }
    config = Config(default_parameters)
    config.parse_args(['-c','default_training_config.json'])
    config['TestPass'] = True
    config.print()
