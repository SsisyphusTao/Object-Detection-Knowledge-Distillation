import torch
import cv2 as cv

def getsingleimg():
    img = cv.imread('/home/tao/Pictures/juvenile-penguin.jpg')
    img = show = cv.resize(img, (300,300))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img.unsqueeze(0).float().cuda(), show
