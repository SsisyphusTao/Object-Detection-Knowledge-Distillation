import argparse
import json


class Config(dict):
    """This is a config dict for determing every details through training.

    There is two ways to set the config, by dict parameter or a json file.
    The priority of them is json file > dict parameter.

    """

    def __init__(
        self,
        default_parameters: dict
    ):
        """Example function with types documented in the docstring.

        `PEP 484`_ type annotations are supported. If attribute, parameter, and
        return types are annotated according to `PEP 484`_, they do not need to be
        included in the docstring:

        Args:
            default_parameters (dict): The first parameter.
        """
        super().__init__()

        self.update(default_parameters)



    def parse_args(self, argv=None):
        parser = argparse.ArgumentParser(
            description='Object Detection Knowledge Distillation.')
        parser.add_argument(
            '--train_config', '-c', default='', type=str, help='JSON config to determine training parameters')
        parser.add_argument('--local_rank', default=0, type=int,
                            help='Used for multi-process training. Can either be manually set or automatically set by using \'python -m multiproc\'.')
        args = parser.parse_args(argv)

        if args.train_config:
            with open(args.train_config, 'r') as f:
                self.update(json.load(f))

    def print(self):
        """ Print all content values with a pretty way."""
        print(json.dumps(self, sort_keys=False, indent=4))

    def check(self):
        pass
