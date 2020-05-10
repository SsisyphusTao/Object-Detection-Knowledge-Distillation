# Knowledge Distillation on SSD

This is an implementation of Chen, G. et al. (2017) [‘Learning efficient object detection models with knowledge distillation’](http://papers.nips.cc/paper/6676-learning-efficient-object-detection-models-with-knowledge-distillation.pdf) with **SSD** structure.

## Overall Structure

![structure](structure.png)

## Introduction

For saving time, I combined codes from two existing repositories, [amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch#installation) and [qfgaohao/pytorch-ssd](https://github.com/amdegroot/ssd.pytorch#installation). Trained models also can be downloaded from thier repositories.

**Training sets:** VOC2007 trainval & VOC2012 trainval  
**Testing sets:** VOC2007 test

||Backbone|mAP|URL
-|-|-|-
Teacher net|VGG16|77.43%|[https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth](https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth)
Student net|MobilenetV2(SSD lite)|68.6%|[https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth](https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth)

## Dependencies

- Python 3.6+  
- PyTorch 1.3+

##  Usages

Some path variables in `train.py` and `eval.py` need to be changed.

## Result                  
>AP for aeroplane = 0.7220          
AP for bicycle = 0.7858        
AP for bird = 0.6641     
AP for boat = 0.5638     
AP for bottle = 0.3708       
AP for bus = 0.8112    
AP for car = 0.7529    
AP for cat = 0.8375    
AP for chair = 0.5493      
AP for cow = 0.6364    
AP for diningtable = 0.7238            
AP for dog = 0.8006    
AP for horse = 0.8431      
AP for motorbike = 0.8294          
AP for person = 0.7211       
AP for pottedplant = 0.4489            
AP for sheep = 0.6518      
AP for sofa = 0.7865     
AP for train = 0.8330      
AP for tvmonitor = 0.6654          
Mean AP = 0.6999

It's pity that I deleted this model by accident...

## TODO

1. add a `.sh` script to help start training easily.
2. retrain the result model.
