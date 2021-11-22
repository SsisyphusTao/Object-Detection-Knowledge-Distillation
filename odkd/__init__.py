import sys

from odkd.utils import Config
from odkd.train import create_trainer
from odkd.train.eval import Evaluator
from odkd.data.transforms.base import BaseTransform


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
        config['augmentation'] = BaseTransform(300, (127, 127, 127))
        trainer = Evaluator(config)
        trainer.evaluate_one_batch()
    except KeyboardInterrupt:
        if config['local_rank'] in [-1, 0]:
            print('\nStopped by user.')
        sys.exit(0)
