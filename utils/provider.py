import torch
import numpy as np
import math

from torch.utils.data import Dataset

def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap

class DrawGaussian(Dataset):
    def __init__(self, bboxes, labels):
        super().__init__()
        self.x = ((bboxes[..., 0] + bboxes[..., 2]) / 2)
        self.y = ((bboxes[..., 1] + bboxes[..., 3]) / 2)
        self.w = ((bboxes[..., 2] - bboxes[..., 0]))
        self.h = ((bboxes[..., 3] - bboxes[..., 1]))
        self.l = labels
        self.size = (128, 72)
        self.max_objs = self.l.shape[1]

    def __len__(self):
        return self.l.shape[0]

    def __getitem__(self, index):
        hm = np.zeros((2, self.size[1], self.size[0]), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)

        x = self.x[index]*self.size[0]
        y = self.y[index]*self.size[1]
        w = self.w[index]*self.size[0]
        h = self.h[index]*self.size[1]
        l = self.l[index]

        # if not 0 in l:
        #     l[l==1] += 1

        reg_mask = l.gt(0).type(torch.uint8).numpy()
        for k in np.where(reg_mask>0)[0].tolist():
            # if l[k] == 0:
            #   self.images[index][int((y[k]-h[k]/2)*360):int((y[k]+h[k]/2)*360), 
            #                     int((x[k]-w[k]/2)*640):int((x[k]+w[k]/2)*640)] = \
            #   self.rois[index][int((y[k]-h[k]/2)*360):int((y[k]+h[k]/2)*360), 
            #                   int((x[k]-w[k]/2)*640):int((x[k]+w[k]/2)*640)]
            # else:
              radius = gaussian_radius((math.ceil(h[k]), math.ceil(w[k])))
              radius = max(0, int(radius))
              x_int = int(x[k])
              y_int = int(y[k])
              draw_umich_gaussian(hm[l[k]-1], (x_int, y_int), radius)
              wh[k] = w[k], h[k]
              ind[k] = y_int * self.size[0] + x_int
              reg[k] = x[k]-x_int, y[k]-y_int

        return hm, reg_mask, ind, wh, reg
