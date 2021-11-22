from odkd.utils import (
    Config
)


def test_config():
    test_arguments = {
        'MODE': 'Test'
    }
    config = Config(test_arguments, ['default_training_config.yml'])
    config['TestPass'] = True
    config.print()
