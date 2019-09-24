import torch
from nets import vgg_module
import cv2 as cv
import numpy as np
import time
from penguin import getsingleimg

x, show = getsingleimg()
vgg_test = vgg_module('test')
# print(vgg_test)

# vgg_test.load_weights('./models/student_vgg_5000.pth')
vgg_test.load_state_dict({k.replace('module.',''):v 
for k,v in torch.load('./models/student_vgg_10000.pth').items()})
vgg_test.eval()
vgg_test = vgg_test.cuda()
torch.backends.cudnn.benchmark = True
a = time.time()
r = vgg_test(x).data#.numpy()[0]
print(time.time()-a)
for j in range(1, r.size(1)):
    dets = r[0, j, :]
    mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t() # expand -> copy value; t -> Transpose; all these two is making score to the same dim of location
    dets = torch.masked_select(dets, mask).view(-1, 5)  # select out valid boxes and separate them
    if dets.size(0) == 0:
        continue
    boxes = dets[:, 1:].numpy()[0]
    print(dets[:, 0].numpy()[0])
    boxes *= 300
    boxes = boxes.astype(int)
    cv.rectangle(show, (boxes[0],boxes[1]), 
    (boxes[2], 
     boxes[3]), 255)
cv.imshow('sdf', show)
cv.waitKey()