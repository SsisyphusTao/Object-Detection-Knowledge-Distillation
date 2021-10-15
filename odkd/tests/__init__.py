from odkd.utils import (
    Config
)

config = Config()
config.parse_args(['-c', 'default_training_config.yml'])
