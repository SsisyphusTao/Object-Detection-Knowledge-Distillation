from abc import ABC, abstractmethod
import time

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

    @abstractmethod
    def parse_config(self):
        assert hasattr(self, 'dataloader')
        assert hasattr(self, 'optimizer')
        assert hasattr(self, 'compute_loss')

    def train_one_epoch(self, epoch):
        mloss = 0
        for i, (images, targets) in enumerate(self.dataloader):
            # forward
            self.optimizer.zero_grad()
            loss = self.compute_loss(images, targets)
            loss.backward()
            self.optimizer.step()
            mloss = (mloss * i + loss.item()) / (i + 1)  # update mean losses
            return ('[%s], %s Total:%.4f, iter:%03d') % (time.asctime(time.localtime(
                time.time())), 'Epoch:[%g/%g]' % (epoch, self.config['epochs']), mloss, i)


class SSDTrainer(Trainer):
    """Specifying the pipeline of SSD training or distillation"""

    def parse_config(self):
        self.config['priors'] = create_priorbox(**self.config)
        self.config['augmentation'] = create_augmentation(self.config)
        self.dataloader = create_dataloader(self.config)
        optimizer = create_optimizer(self.config)

        dist_model = ssd_lite(self.config['teacher_backbone'], self.config)
        model = ssd_lite(self.config['student_backbone'], self.config)

        if self.config['distillation']:
            loss = ObjectDistillationLoss(self.config)
            self.compute_loss = NetwithDistillatedLoss(loss, model, dist_model)
        else:
            loss = MultiBoxLoss(self.config)
            self.compute_loss = NetwithLoss(loss, model)
        self.optimizer = optimizer(self.compute_loss.model.parameters())
        super().parse_config()
