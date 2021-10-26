
"""SSDLite detector relative blocks"""
import torch
from torch import nn
from torchvision.ops import nms

from math import sqrt
from itertools import product

from odkd.models._utils import InvertedResidual, SeperableConv2d, _make_divisible
from odkd.models.backbone import mobilenet_v2, vgg16_bn
from odkd.models.box_utils import decode


class Detect(nn.Module):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.

    Args:
        cfg: (dict) SSD config including inference parameters
        prior_data: (tensor) Prior boxes and variances from priorbox layers
            Shape: [1,num_priors,4]

    Return:
        (tensor) Predictions of bounding boxes and scores of each classes
            Shape: [1,num_classes,topK,5]
    """

    def __init__(self, cfg, prior_data):
        super().__init__()
        self.num_classes = cfg['num_classes']
        self.top_k = cfg['topK']
        self.variance = cfg['variance']
        self.conf_thresh = cfg['conf_thresh']
        self.nms_thresh = cfg['nms_thresh']
        self.prior_data = prior_data
        self.num_priors = self.prior_data.size(0)

    def forward(self, loc_data, conf_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
        """
        num = loc_data.size(0)  # batch size

        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, self.num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            # here change the offset to percentage of the scale
            decoded_boxes = decode(loc_data[i], self.prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                idx = nms(boxes, scores, self.nms_thresh)[:self.top_k]
                output[i, cl, :idx.size(0)] = \
                    torch.cat((scores[idx].unsqueeze(1),
                               boxes[idx]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


class SSDLite(nn.Module):
    """SSDlite detect head which only has 3000 anchors

    Args:
        cfg (int): SSDLite config
        width_mult (:obj:`float`, optional): scale ratio of channel width
        div_nearest (:obj:`int`, optional): multiple of channels

    Attributes:
        cfg (dict): default prior boxex config.
    """

    def __init__(self, features, cfg, width_mult=1.0, div_nearest=8):
        super().__init__()

        self.num_classes = cfg['num_classes']
        self.feature_maps_size = cfg['feature_maps_size']
        self.input_size = cfg['input_size']
        self.steps = cfg['steps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']

        self.features = nn.ModuleList(features)

        def div(x):
            return _make_divisible(x * width_mult, div_nearest)

        self.add_extras(div)
        self.priors = self.create_priorbox()
        self.detect = Detect(cfg, self.priors)

    def create_priorbox(self):
        """Compute priorbox coordinates in center-offset form for each source
        feature map.

        Args:
            cfg (dict): SSDLite config including prior boxes parameters

        Return:
            Prior boxes

        """

        # number of priors for feature map location (either 4 or 6)
        mean = []
        for k, f in enumerate(self.feature_maps_size):
            for i, j in product(range(f), repeat=2):
                f_k = self.input_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.input_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.input_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

    def add_extras(self, div):
        idx = 0
        layer_config = []
        x = torch.randn(2, 3, self.input_size, self.input_size)
        for idx, feature in enumerate(self.features):
            x = feature(x)
            _, c, h, w = x.shape
            layer_config.append((idx, c, h, w))

        while not h == w == 1:
            idx += 1
            block = InvertedResidual(div(c), div(
                c//2), stride=2, expand_ratio=0.5)
            self.features.append(block)
            x = block(x)
            _, c, h, w = x.shape
            layer_config.append((idx, c, h, w))

        self.sources_idx = [None] * len(self.feature_maps_size)
        head_channels = [None] * len(self.feature_maps_size)
        for i in layer_config:
            for n, j in enumerate(self.feature_maps_size):
                if i[2] == i[3] == j:
                    self.sources_idx[n] = i[0]
                    head_channels[n] = i[1]
                    break

        self.regression_headers = nn.ModuleList([
            SeperableConv2d(in_channels=div(head_channels[0]), out_channels=6 * 4,
                            kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=div(
                head_channels[1]), out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=div(head_channels[2]), out_channels=6 * 4,
                            kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=div(head_channels[3]), out_channels=6 * 4,
                            kernel_size=3, padding=1, onnx_compatible=False),
            SeperableConv2d(in_channels=div(head_channels[4]), out_channels=6 * 4,
                            kernel_size=3, padding=1, onnx_compatible=False),
            nn.Conv2d(in_channels=div(head_channels[5]),
                      out_channels=6 * 4, kernel_size=1),
        ])
        self.classification_headers = nn.ModuleList([
            SeperableConv2d(in_channels=div(head_channels[0]),
                            out_channels=6 * self.num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=div(head_channels[1]),
                            out_channels=6 * self.num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=div(head_channels[2]),
                            out_channels=6 * self.num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=div(head_channels[3]),
                            out_channels=6 * self.num_classes, kernel_size=3, padding=1),
            SeperableConv2d(in_channels=div(head_channels[4]),
                            out_channels=6 * self.num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=div(head_channels[5]),
                      out_channels=6 * self.num_classes, kernel_size=1),
        ])

    def forward(self, x):
        sources = []
        loc = []
        conf = []

        for idx, block in enumerate(self.features):
            x = block(x)
            if idx == self.sources_idx[len(sources)]:
                sources.append(x)

        for (f, l, c) in zip(sources, self.regression_headers, self.classification_headers):
            loc.append(l(f).permute(0, 2, 3, 1).contiguous())
            conf.append(c(f).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([i.view(i.size(0), -1) for i in loc], 1)
        conf = torch.cat([i.view(i.size(0), -1) for i in conf], 1)

        if self.training:
            return loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes), self.priors
        else:
            return self.detect(loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes))


def ssd_lite(model_name, config):
    backbones = {
        'mobilenetv2': mobilenet_v2().features,
        'vgg16': vgg16_bn().features,
        'vgg19': vgg16_bn().features
    }
    return SSDLite(backbones[model_name], config['SSDLite'])
