from odkd.train.eval import Evaluator
from odkd.data.transforms.base import BaseTransform


def test_eval(config):
    e = Evaluator(config)
    config['augmentation'] = BaseTransform(300, (127, 127, 127))
    e.evaluate_one_batch()
