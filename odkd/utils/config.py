"""This moudule contains config class for training."""
import os
import sys
import shutil
import argparse
import yaml

DEFAULT = {
    'SSDLITE': {
        'input_size': 300,
        'num_classes': 21,
        'feature_maps_size': [19, 10, 5, 3, 2, 1],
        'steps': [16, 32, 64, 100, 150, 300],
        'min_sizes': [60, 105, 150, 195, 240, 285],
        'max_sizes': [105, 150, 195, 240, 285, 330],
        'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        'variance': [0.1, 0.2],
        'clip': True,
        'topK': 200,
        'conf_thresh': 0.01,
        'nms_thresh': 0.45,
        'overlap_thresh': 0.5,
        'neg_pos': 3
    },

    'SSD_DIST_LOSS': {
        'temperature': 1.0,
        'negative_weight': 1.5,
        'positive_weight': 1.0,
        'regression_margin': 0,
        'regression_weight': 1.0,
        'hint_weight': 0.5,
        'u': 0.5
    },

    'VOC_TRANS': {
        'mean': [104, 117, 123]
    },

    'SGD': {
        'momentum': 0.9,
        'weight_decay': 5e-4
    },

    'STEPS': {
        'milestones': [120, 180]
    }
}


class Config(dict):
    """This is a config dict for determing every details through training.

    There is two ways to set the config, by dict parameter or a yaml file.
    The priority of them is yaml file > dict parameter.

    Args:
        arguments (dict): Arguments for training.
        argv (list): take commend line parameters from list

    """

    def __init__(self, arguments: dict = (), argv=None):
        super().__init__()
        for i in DEFAULT.values():
            self.update(i)
        self.update(arguments)
        self.parse_args(argv)

    def parse_args(self, argv=None):
        """Take command line arguments and updating parameters.

        Args:
            argv (list): take commend line parameters from list

        """

        # Training with cpu is not supported and not recommended.
        # assert torch.cuda.is_available(), 'CUDA is not available.'

        parser = argparse.ArgumentParser(
            description='Object Detection Knowledge Distillation.')
        parser.add_argument('train_config', type=str,
                            help='YAML config to determine training parameters.')
        parser.add_argument('--template', '-t', action='store_true',
                            default=False, help='Create config template.')
        parser.add_argument('--local_rank',
                            default=-1, type=int,
                            help='Used for multi-process training.')
        args = parser.parse_args(argv)

        if args.template:
            template_path = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), 'template.yml')
            if '.yml' in args.train_config:
                shutil.copyfile(template_path, args.train_config)
                print('Created config template: %s' % args.train_config)
            else:
                shutil.copyfile(template_path, os.path.join(
                    args.train_config, 'config_template.yml'))
                print('Created config template: %s' % os.path.join(
                    args.train_config, 'config_template.yml'))
            sys.exit(0)

        if args.train_config:
            with open(args.train_config, 'r', encoding='utf-8') as config_file:
                # Loading parameters from config file, this will overwrite the same parameter.
                self.update(yaml.safe_load(config_file))
        self['local_rank'] = args.local_rank
        self['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
        if self['local_rank'] != -1:
            self['cuda'] = True

    def print(self):
        """ Print all content values with a pretty way."""
        print(yaml.dump(self, sort_keys=False, default_flow_style=False))

    def dump(self, path):
        with open(os.path.join(path, 'config.yaml'), 'w', encoding='utf-8') as f:
            for i, j in self.items():
                if isinstance(j, (int, str, float, list, tuple, dict)):
                    f.write(yaml.dump({i: j}))

    def __getitem__(self, __k):
        if isinstance(__k, tuple):
            return {k: self[k] for k in __k if k in self}
        else:
            return super().__getitem__(__k)
