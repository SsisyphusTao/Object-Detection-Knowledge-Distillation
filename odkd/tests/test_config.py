from odkd.utils import (
    Config
)


def test_config():
    test_arguments = {
        'MODE': 'Test'
    }
    config = Config(test_arguments)
    config.parse_args(['-c', 'default_training_config.json'])
    config['TestPass'] = True
    config.print()
    assert config.check() == set()
    del config['epochs']
    assert config.check() == set(['epochs'])
