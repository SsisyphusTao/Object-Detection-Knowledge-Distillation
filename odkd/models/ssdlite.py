
"""SSDLite detector relative blocks"""
import torch
from torch import nn
from torchvision.ops import batched_nms

from math import sqrt
from itertools import product

from odkd.models._utils import InvertedResidual, SeperableConv2d, _make_divisible
from odkd.utils.box_utils import decode


def create_priorbox(input_size,
                    feature_maps_size,
                    steps, max_sizes,
                    min_sizes,
                    aspect_ratios,
                    clip, **kwargs):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.

    Args:
        input_size (int): Size of input Image
        feature_maps_size (list): Size of each feature maps in pyramid
        steps (list) : Strides of each feature maps compared with input
        max_sizes (list): Max sizes of boxes in each feature maps
        min_sizes (list): Min sizes of boxes in each feature maps
        aspect_ratios (list): Ratios of box sides from min to max
        clip (bool): Whether allow boxes outside the image
        kwargs (dict): other argss

    Return:
        Prior boxes

    """
    del kwargs
    # number of priors for feature map location (either 4 or 6)
    mean = []
    for k, f in enumerate(feature_maps_size):
        for i, j in product(range(f), repeat=2):
            f_k = input_size / steps[k]
            # unit center x,y
            cx = (j + 0.5) / f_k
            cy = (i + 0.5) / f_k

            # aspect_ratio: 1
            # rel size: min_size
            s_k = min_sizes[k]/input_size
            mean += [cx, cy, s_k, s_k]

            # aspect_ratio: 1
            # rel size: sqrt(s_k * s_(k+1))
            s_k_prime = sqrt(s_k * (max_sizes[k]/input_size))
            mean += [cx, cy, s_k_prime, s_k_prime]

            # rest of aspect ratios
            for ar in aspect_ratios[k]:
                mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
    # back to torch land
    output = torch.Tensor(mean).view(-1, 4)
    if clip:
        output.clamp_(max=1, min=0)
    return output


class Detect(nn.Module):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.

    Args:
        num_classes: (dict) Number of classes
        topK: (int) TopK objects for NMS
        variance: (list) Variance of Prior boxes for encoding/decoding
        conf_thresh: (float) Confidence threshold
        nms_thresh: (float) NMS threshold
        prior_data: (tensor) Prior boxes and variances from priorbox layers
            Shape: [1,num_priors,4]

    Return:
        (tensor) Predictions of bounding boxes and scores of each classes
            Shape: [1,num_classes,topK,5]
    """

    def __init__(self, num_classes, topK, variance, conf_thresh, nms_thresh, prior_data):
        super().__init__()
        self.num_classes = num_classes
        self.top_k = topK
        self.variance = variance
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.prior_data = nn.Parameter(prior_data)
        self.num_priors = self.prior_data.size(0)

    def forward(self, loc_data, conf_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
        """

        output = []
        conf_data = torch.nn.functional.softmax(conf_data, -1)[..., 1:]
        scores_data, cls_data = conf_data.max(-1)

        for boxes, scores, idxs in zip(loc_data, scores_data, cls_data):
            decoded_boxes = decode(boxes, self.prior_data, self.variance)
            pos = scores.gt(self.conf_thresh)
            out_idx = batched_nms(
                decoded_boxes[pos], scores[pos], idxs[pos], self.nms_thresh)[:self.top_k]

            result = torch.zeros(self.top_k, 6).to(boxes.device)
            result[:out_idx.size(0)] = torch.cat((decoded_boxes[pos][out_idx],
                                                  scores[pos][out_idx].unsqueeze(
                                                      1),
                                                  idxs[pos][out_idx].unsqueeze(1)), 1)

            output.append(result)

        return torch.stack(output)


class SSDLite(nn.Module):
    """SSDlite detect head which only has 3000 anchors

    Args:
        features (Tensor): Image features from backbone
        input_size (int): Size of input Image
        num_classes (int): Number of classes
        feature_maps_size (list): Size of each feature maps in pyramid
        topK (int): TopK objects for NMS
        variance (list): Variance of Prior boxes for encoding/decoding
        conf_thresh (float): Confidence threshold
        nms_thresh (float): NMS threshold
        priors (Tensor): Prior boxes and variances from priorbox layers
        width_mult (:obj:`float`, optional): scale ratio of channel width
        div_nearest (:obj:`int`, optional): multiple of channels

    Attributes:
        cfg (dict): default prior boxex config.
    """

    def __init__(self,
                 features,
                 input_size,
                 num_classes,
                 feature_maps_size,
                 topK,
                 variance,
                 conf_thresh,
                 nms_thresh,
                 priors,
                 width_mult=1.0,
                 div_nearest=8):
        super().__init__()

        self.num_classes = num_classes
        self.feature_maps_size = feature_maps_size
        self.input_size = input_size

        self.features = nn.ModuleList(features)

        def div(x):
            return _make_divisible(x * width_mult, div_nearest)

        self.add_extras(div)

        self.detect = Detect(num_classes, topK, variance,
                             conf_thresh, nms_thresh, priors)

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
            return loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes)
        else:
            return self.detect(loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes))
