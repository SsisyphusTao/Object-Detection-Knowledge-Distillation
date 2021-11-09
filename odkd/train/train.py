from abc import ABC, abstractmethod
import time
import torch
import os
from os import makedirs, path

from odkd.utils import Config
from odkd.data import create_augmentation, create_dataloader
from odkd.models.ssdlite import create_priorbox, ssd_lite
from ._utils import create_optimizer
from .loss import MultiBoxLoss, NetwithLoss, ObjectDistillationLoss, NetwithDistillatedLoss


class Trainer(ABC):
    """Base train class, parsing config to pipeline.

    Args:
        config (dict): All parameters needed for training

    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.parse_config()
        self.prepare()

    @abstractmethod
    def parse_config(self):
        assert hasattr(self, 'dataloader')
        assert hasattr(self, 'optimizer')
        assert hasattr(self, 'compute_loss')
        assert hasattr(self, 'model')

    def prepare(self):
        root = os.path.expanduser(self.config['checkpoints_path'])
        if not path.exists(root):
            makedirs(root)
        self.config['uid'] = os.listdir(root).__len__().__str__()
        uid = path.join(root, self.config['uid'])
        self.config['teacher_path'] = os.path.join(uid, 'teacher')
        self.config['student_path'] = os.path.join(uid, 'student')
        if not path.exists(uid):
            makedirs(uid)
            makedirs(self.config['teacher_path'])
            makedirs(self.config['student_path'])
            self.config.dump(uid)

    def train_one_epoch(self, epoch):
        mloss = 0
        for i, (images, targets) in enumerate(self.dataloader):
            # forward
            self.optimizer.zero_grad()
            loss = self.compute_loss(images, targets)
            loss.backward()
            self.optimizer.step()
            mloss = (mloss * i + loss.item()) / (i + 1)  # update mean losses
            print(('[%s], %s Total:%.4f, iter:%03d') % (time.asctime(time.localtime(
                time.time())), 'Epoch:[%g/%g]' % (epoch+1, self.config['epochs']), mloss, i))

    def train(self):
        for i in range(self.config['epochs']):
            self.train_one_epoch(i)
            torch.save(self.model.state_dict(),
                       path.join(self.config['student_path'], '%s_%s_%03d.pth' % (self.config['student_backbone'], self.config['detection'], i+1)))


class SSDTrainer(Trainer):
    """Specifying the pipeline of SSD training or distillation"""

    def parse_config(self):
        self.config['priors'] = create_priorbox(**self.config)
        self.config['augmentation'] = create_augmentation(self.config)
        self.dataloader = create_dataloader(self.config)
        optimizer = create_optimizer(self.config)

        self.dist_model = ssd_lite(
            self.config['teacher_backbone'], self.config)
        self.model = ssd_lite(self.config['student_backbone'], self.config)
        self.optimizer = optimizer(self.model.parameters())

        if self.config['distillation']:
            loss = ObjectDistillationLoss(self.config)
            self.compute_loss = NetwithDistillatedLoss(
                loss, self.model, self.dist_model)
        else:
            loss = MultiBoxLoss(self.config)
            self.compute_loss = NetwithLoss(loss, self.model)

        super().parse_config()
