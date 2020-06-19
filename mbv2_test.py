import torch
from nets import create_mobilenetv2_ssd_lite
import cv2 as cv
import numpy as np
import time
from penguin import getsingleimg
from data.voc0712 import VOC_CLASSES

mobilenetv2_test = create_mobilenetv2_ssd_lite('test')
missing, unexpected = mobilenetv2_test.load_state_dict({k.replace('module.',''):v 
for k,v in torch.load('models/mb2-ssd-lite-mp-0_686.pth').items()}, strict=False)
if missing:
    print('Missing:', missing)
if unexpected:
    print('Unexpected:', unexpected)

mobilenetv2_test.eval()
mobilenetv2_test = mobilenetv2_test.cuda()

x, show = getsingleimg('sheep-on-green-grass.jpg')
r = mobilenetv2_test(x).data#.numpy()[0]
for j in range(1, r.size(1)):
    dets = r[0, j, :]
    mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
    dets = torch.masked_select(dets, mask).view(-1, 5)
    if dets.size(0) == 0:
        continue
    boxes = dets[:, 1:].numpy()[0]
    print(VOC_CLASSES[j-1]+': '+str(dets[:,0].numpy()[0]))
    # print(dets)
    # if dets[:, 0].numpy()[0] < 0.9:
    #     continue
    boxes *= 300
    boxes = boxes.astype(int)
    try:
        cv.rectangle(show, (boxes[0],boxes[1]), 
        (boxes[2], 
        boxes[3]), 255)
    except:
        continue
cv.imwrite('output.jpg', show)