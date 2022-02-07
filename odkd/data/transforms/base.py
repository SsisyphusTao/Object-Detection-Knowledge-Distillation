import numpy as np
import cv2


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x /= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, input_size, mean, preprocess):
        self.size = input_size
        self.mean = np.array(mean, dtype=np.float32)
        self.preprocess = preprocess

    def __call__(self, image, boxes=None, labels=None):
        image, boxes, labels = self.preprocess(image, boxes, labels)
        return base_transform(image, self.size, self.mean), np.hstack((boxes, np.expand_dims(labels, axis=1)))
