import argparse
import yaml


class Config(dict):
    """This is a config dict for determing every details through training.

    There is two ways to set the config, by dict parameter or a yaml file.
    The priority of them is yaml file > dict parameter.

    Args:
        required_arguments (dict): Minimum required arguments for training start.

    Attributes:
        required (set): Keys of minimum required arguments.

    """

    def __init__(self, required_arguments: dict):
        super().__init__()
        self.update(required_arguments)

        self.required = {'Dataset': {'dataset_type', 'dataset_path'},
                         'Training': {'batch_size', 'epochs', 'initial_learning_rate'},
                         'Evaluation': {'period'},
                         'Task': {'distillation', 'target_net'}}

    def parse_args(self, argv=None):
        """Take command line arguments and updating parameters.

        Args:
            argv (list): FOR_TEST.

        """
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

    def print(self):
        """ Print all content values with a pretty way."""
        print(yaml.dump(self, sort_keys=False, default_flow_style=False))

    def check(self):
        """ Check if all required arguments are defined

        Returns:
            list: Return the missing arguments.

        """
        missing = set(self.required).difference(set(self.keys()))
        if missing:
            raise KeyError(missing)

        for i in self.required:
            missing = self.required[i].difference(set(self[i].keys()))
            if missing:
                raise KeyError(missing)
