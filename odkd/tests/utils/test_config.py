from odkd.utils import (
    Config
)


def test_config():
    test_arguments = {
        'MODE': 'Test'
    }
    config = Config(test_arguments, ['odkd/utils/template.yml'])
    config['TestPass'] = True
    config.print()
