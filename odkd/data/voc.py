import torch
from torch.utils.data import DataLoader
from torchvision.datasets.voc import VOCDetection

import os
import numpy as np


VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))


def voc_transform(image, *target):
    """Transforms the PIL Image and dict target to required format.

    Args:
        image: PIL Image from torchvision dataset
        target: dict format annotations from torchvision dataset

    Returns:
        [image, boxes, labels]

    """
    image = np.asarray(image)
    target = target[0]
    res = []

    width = int(target['annotation']['size']['width'])
    height = int(target['annotation']['size']['height'])
    for obj in target['annotation']['object']:
        name = obj['name']
        bbox = obj['bndbox']

        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox[pt]) - 1
            # scale height or width
            cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
            bndbox.append(cur_pt)
        label_idx = class_to_ind[name]
        bndbox.append(label_idx)
        res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
    # [[xmin, ymin, xmax, ymax, label_ind], ... ]
    res = np.array(res)
    return image, res[:, :4], res[:, 4]


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    images, targets = [], []
    for sample in batch:
        images.append(torch.Tensor(sample[0]))
        targets.append(torch.Tensor(sample[1]))
    try:
        targets = torch.stack(targets, 0)
    except RuntimeError:
        pass
    return torch.stack(images, 0).permute(0, 3, 1, 2), targets


def create_voc_dataloader(image_set,
                          dataset_path,
                          batch_size,
                          num_worker,
                          augmentation,
                          local_rank=-1,
                          world_size=1):
    """Create a voc dataloader with custom data augmentation.
    """
    try:
        dataset = VOCDetection(
            dataset_path, image_set=image_set, transforms=augmentation)
    except RuntimeError:
        dataset = VOCDetection(dataset_path, image_set=image_set,
                               transform=augmentation, download=True)
    nw = min([os.cpu_count() // world_size, batch_size if batch_size
             > 1 else 0, num_worker])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset) if local_rank != -1 else None
    return DataLoader(dataset, batch_size,
                      num_workers=nw,
                      sampler=sampler,
                      collate_fn=detection_collate,
                      pin_memory=True)
