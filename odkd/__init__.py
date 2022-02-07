import sys

from odkd.utils import Config
from odkd.train import create_evaluator, create_trainer


def run_train():
    try:
        config = Config()
        trainer = create_trainer(config)
        trainer.start()
    except KeyboardInterrupt:
        if config['local_rank'] in [-1, 0]:
            print('\nStopped by user. Saved in %s' % config['save_dir'])
        sys.exit(0)


def run_eval():
    try:
        config = Config()
        evaluator = create_evaluator(config)
        evaluator.eval_once()
    except KeyboardInterrupt:
        if config['local_rank'] in [-1, 0]:
            print('\nStopped by user.')
        sys.exit(0)
