import sys

from odkd.utils import Config
from odkd.train import create_trainer


def run_train():
    try:
        config = Config()
        config.parse_args()
        trainer = create_trainer(config)
        trainer.start()
    except KeyboardInterrupt:
        if config['local_rank'] in [-1, 0]:
            print('\nStopped by user. Saved in %s' % config['save_dir'])
        sys.exit(0)
