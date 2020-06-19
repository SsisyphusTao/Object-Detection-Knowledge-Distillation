import torch
import cv2 as cv
import numpy as np
def getsingleimg(path, mean=[127., 127., 127.], std=1.0):
    img = cv.imread(path)
    img = show = cv.resize(img, (300,300))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    img = img.astype(np.float32)
    img -= np.array(mean, dtype=np.float32)
    img /= std

    img = torch.from_numpy(img).permute(2, 0, 1)
    return img.unsqueeze(0).float().cuda(), show
