"""This moudule contains config class for training."""
import os
import argparse
import yaml
import torch


class Config(dict):
    """This is a config dict for determing every details through training.

    There is two ways to set the config, by dict parameter or a yaml file.
    The priority of them is yaml file > dict parameter.

    Args:
        arguments (dict): Arguments for training.

    Attributes:
        required (set): Keys of minimum required arguments.

    """

    def __init__(self, arguments: dict = ()):
        super().__init__()
        self.update(arguments)

        self._required = {'dataset', 'dataset_path', 'batch_size', 'epochs',
                          'initial_learning_rate', 'period', 'distillation', 'target_net'}

    def parse_args(self, argv=None):
        """Take command line arguments and updating parameters.

        Args:
            argv (list): take commend line parameters from list

        """

        # Training with cpu is not supported and not recommended.
        # assert torch.cuda.is_available(), 'CUDA is not available.'

        parser = argparse.ArgumentParser(
            description='Object Detection Knowledge Distillation.')
        parser.add_argument('--train_config', '-c',
                            default='', type=str,
                            help='YAML config to determine training parameters')
        parser.add_argument('--local_rank',
                            default=0, type=int,
                            help='Used for multi-process training.')
        args = parser.parse_args(argv)

        if args.train_config:
            with open(args.train_config, 'r', encoding='utf-8') as config_file:
                # Loading parameters from config file, this will overwrite the same parameter.
                self.update(yaml.safe_load(config_file))
        self['local_rank'] = args.local_rank
        self['distributed'] = False
        if 'WORLD_SIZE' in os.environ:
            self['distributed'] = (
                int(os.environ['WORLD_SIZE']) > 1 and torch.distributed.is_available())

    def print(self):
        """ Print all content values with a pretty way."""
        print(yaml.dump(dict(self), sort_keys=False, default_flow_style=False))

    def check(self):
        """ Check if all required arguments are defined

        Returns:
            list: Return the missing arguments.

        """
        missing = set(self._required).difference(set(self.keys()))
        if missing:
            raise KeyError(missing)

