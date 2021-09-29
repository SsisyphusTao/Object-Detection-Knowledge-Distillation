import argparse
import json


class Config(dict):
    """This is a config dict for determing every details through training.

    There is two ways to set the config, by dict parameter or a json file.
    The priority of them is json file > dict parameter.

    Args:
        required_arguments (dict): Minimum required arguments for training start.

    Attributes:
        required (set): Keys of minimum required arguments.

    """

    def __init__(self, required_arguments: dict):
        super().__init__()
        self.update(required_arguments)

        self.required = set(['dataset_type', 'dataset_path',
                             'batch_size', 'epochs', 'initial_learning_rate',
                             'eval_period', 'distillation', 'target_net'])

    def parse_args(self, argv=None):
        """Take command line arguments and updating parameters.

        Args:
            argv (list): FOR_TEST.

        """
        parser = argparse.ArgumentParser(
            description='Object Detection Knowledge Distillation.')
        parser.add_argument(
            '--train_config', '-c', default='', type=str, help='JSON config to determine training parameters')
        parser.add_argument('--local_rank', default=0, type=int,
                            help='Used for multi-process training. Can either be manually set or automatically set by using \'python -m multiproc\'.')
        args = parser.parse_args(argv)

        if args.train_config:
            with open(args.train_config, 'r') as f:
                # Loading parameters from config file, this will overwrite the same parameter.
                self.update(json.load(f))

    def print(self):
        """ Print all content values with a pretty way."""
        print(json.dumps(self, sort_keys=False, indent=4))

    def check(self):
        """ Check if all required arguments are defined

        Returns:
            list: Return the missing arguments.

        """
        return self.required.difference(set(self.keys()))
