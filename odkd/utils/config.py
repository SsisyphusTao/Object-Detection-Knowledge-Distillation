import argparse
import json


class Config(dict):
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(
        self,
        default_parameters: dict
    ):
        super().__init__()
        parser = argparse.ArgumentParser(
            description='Object Detection Knowledge Distillation.')
        parser.add_argument(
            '--train_config', '-c', default='', type=str, help='JSON config to determine training parameters')
        args = parser.parse_args()

        self.update(default_parameters)

        if args.train_config:
            with open(args.train_config, 'r') as f:
                self.update(json.load(f))

    def print(self):
        """ Print all content values with a pretty way."""
        print(json.dumps(self, sort_keys=False, indent=4))
