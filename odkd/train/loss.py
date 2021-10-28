"""Loss implementation of paper "Learning efficient object detection models with knowledge distillation" Chen, G. et al. (2017),
see https://proceedings.neurips.cc/paper/2017/file/e1e32e235eee1f970470a3a6658dfdd5-Paper.pdf for more details."""
import torch
from torch import nn
from odkd.utils.box_utils import log_sum_exp


# def weighted_KL_div(ps, qt, pos_w, neg_w):
#     eps = 1e-10
#     ps = ps + eps
#     qt = qt + eps
#     log_p = qt * torch.log(ps)
#     log_p[:, 0] *= neg_w
#     log_p[:, 1:] *= pos_w
#     return -torch.sum(log_p)


# def bounded_regression_loss(Rs, Rt, gt, m, v=0.5):
#     loss = F.mse_loss(Rs, gt)
#     if loss + m > F.mse_loss(Rt, gt):
#         return loss * v
#     else:
#         loss.fill_(0)
#         return loss


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, config):
        super().__init__()
        self.num_classes = config['num_classes']
        self.negpos_ratio = config['neg_pos']

        self.localization_loss = nn.SmoothL1Loss(reduction='sum')
        self.confidence_loss = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            predT (tuple): teacher's predictions
            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc, conf, _ = predictions
        loc_gt = targets[..., :-1]
        conf_gt = targets[..., -1:].long()

        pos = conf_gt > 0  # positive label from ground truth
        num_pos = pos.sum(dim=1, keepdim=True)
        pos_idx = pos.expand_as(loc)

        # select out the box should be positive from predictions
        loc = loc[pos_idx]
        loc_gt = loc_gt[pos_idx]
        loss_loc = self.localization_loss(loc, loc_gt)

        # Compute max conf across batch for hard negative mining
        loss_conf = log_sum_exp(conf) - conf.gather(-1, conf_gt)
        loss_conf[pos] = 0  # filter out pos boxes for now
        _, loss_idx = loss_conf.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.expand_as(conf)
        neg_idx = neg.expand_as(conf)
        conf = conf[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        conf_gt = conf_gt[(pos+neg).gt(0)]
        loss_conf = self.confidence_loss(conf, conf_gt)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        return (loss_loc+loss_conf)/num_pos.sum()


class NetwithLoss(nn.Module):
    """Combine loss with model for promoting efficiency when data distribution.

    Args:
        cfg: (dict) global training config
        model: (nn.Module) model for training

    Return:
        loss (torch.Tensor)

    """

    def __init__(self, cfg, model):
        super().__init__()
        self.criterion = MultiBoxLoss(cfg)
        self.model = model

    def forward(self, images, targets):
        predictions = self.model(images)
        return self.criterion(predictions, targets)
