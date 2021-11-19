import numpy as np
import cv2

from odkd.data.voc import voc_transform


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x /= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image, boxes, labels = voc_transform(image, boxes, labels)
        return base_transform(image, self.size, self.mean), np.hstack((boxes, np.expand_dims(labels, axis=1)))