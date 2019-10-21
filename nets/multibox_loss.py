# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .box_utils import match, log_sum_exp

cfg = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

def weighted_KL_div(ps, qt, pos_w, neg_w):
    eps = 1e-10
    ps = ps + eps
    qt = qt + eps
    log_p = qt * torch.log(ps)
    log_p[:,0] *= neg_w
    log_p[:,1:] *= pos_w
    return -torch.sum(log_p)
    
def bounded_regression_loss(Rs, Rt, gt, m, v=0.5):
    loss = F.mse_loss(Rs, gt)
    if loss + m > F.mse_loss(Rt, gt):
        return loss * v 
    else:
        loss.fill_(0)
        return loss

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

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True, neg_w=1.5, pos_w=1.0, Temperature=1., reg_m=0., lmda=1.):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

        self.neg_w = neg_w
        self.pos_w = pos_w
        self.reg_m = reg_m
        self.T = Temperature
        # self.u = u
        self.lmda = lmda

    def forward(self, predictions, predT, targets, u=1.):
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
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0) # batch_size
        priors = priors[:loc_data.size(1), :] # priorboxes' location
        num_priors = (priors.size(0))
        num_classes = self.num_classes
        self.u = u
        #predicions of teachers
        locT, confT = predT
        confT = torch.cat((confT, conf_data[:, 2916:, :]), 1)

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4) # grond truch
        conf_t = torch.LongTensor(num, num_priors) # ground truch
        for idx in range(num):
            truths = targets[idx][:, :-1].data.cuda()
            labels = targets[idx][:, -1].data.cuda()
            defaults = priors.data.cuda()
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx) # use priorbox origin positon to match the groud truch to find out which result is to match the grond truth

        # wrap targets
        with torch.no_grad():
            if self.use_gpu:
                loc_t = loc_t.cuda(non_blocking=True)
                conf_t = conf_t.cuda(non_blocking=True) # still not quite sure what this means

        pos = conf_t > 0 # positive label from ground truth
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)--------------------------------------------------------------------------------------------
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4) # select out the box should be positive from predictions
        loc_t = loc_t[pos_idx].view(-1, 4) # same as above
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        #Regression with Teacher Bounds
        # locT_p = locT[pos_idx].view(-1, 4) #same, select out teacher's boxes should be positive(the location prediciton of teacher)
        # loss_br = bounded_regression_loss(loc_p, locT_p, loc_t, self.reg_m)
        loss_reg = loss_l# + loss_br

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        # Conf_loss-----------------------------------------------------------------------------------------------------------------
        # Hard Negative Mining
        # loss_c[pos] = 0  # filter out pos boxes for now
        loss_c[pos.view(-1, 1)] = 0
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)] # gt means greater than(>)

        # modified original code here: add softmax before cross_entropy
        conf_p = F.softmax(conf_p/self.T, dim=1)
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        #soft loss from teacher
        confT_p = confT[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        confT_p = F.softmax(confT_p/self.T, dim=1)
        loss_soft = weighted_KL_div(conf_p, confT_p, self.pos_w, self.neg_w)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        
        N = num_pos.data.sum()
        # loss_l /= N
        # loss_c /= N
        loss_cls = self.u * loss_c + (1 - self.u) * loss_soft
        loss_ssd = (loss_cls + self.lmda * loss_reg) / N

        return loss_ssd