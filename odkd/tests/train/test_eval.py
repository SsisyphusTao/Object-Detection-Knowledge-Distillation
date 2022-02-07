from odkd.train.eval import Evaluator
from odkd.interface import create_transform


def test_eval(config, ssdlite):
    config['augmentation'] = create_transform(config)
    e = Evaluator(ssdlite, config)
    e.eval_once()
