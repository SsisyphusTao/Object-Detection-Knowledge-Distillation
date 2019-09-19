import torch
from nets import mobilenetv2_module
import cv2 as cv
import numpy as np

mobilenetv2_test = mobilenetv2_module()

# for n, block in enumerate(mobilenetv2_test.features):
#     # if n == 8 or n ==16:
#         print(n)
#         print(block)
#         print('---------------------------------')

img = cv.imread('/home/tao/Pictures/juvenile-penguin.jpg')
img = show = cv.resize(img, (300,300))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = torch.from_numpy(img).permute(2, 0, 1)

mobilenetv2_test.eval()
mobilenetv2_test = mobilenetv2_test.cuda()
torch.backends.cudnn.benchmark = True
x = img.unsqueeze(0).float().cuda()
r = mobilenetv2_test(x).data#.numpy()[0]
for j in range(1, r.size(1)):
    dets = r[0, j, :]
    mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
    dets = torch.masked_select(dets, mask).view(-1, 5)
    if dets.size(0) == 0:
        continue
    boxes = dets[:, 1:].numpy()[0]
    boxes *= 300
    boxes = boxes.astype(int)
    cv.rectangle(show, (boxes[0],boxes[1]), 
    (boxes[2], 
     boxes[3]), 255)    
cv.imshow('sdf', show)
cv.waitKey()