from odkd.utils import (
    Config
)


def test_config():
    test_arguments = {
        'MODE': 'Test'
    }
    config = Config(test_arguments)
    config.parse_args(['default_training_config.yml'])
    config['TestPass'] = True
    config.print()
