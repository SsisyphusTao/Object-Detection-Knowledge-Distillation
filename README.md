# Knowledge Distillation on SSD

This is an implementation of Chen, G. et al. (2017) [‘Learning efficient object detection models with knowledge distillation’](http://papers.nips.cc/paper/6676-learning-efficient-object-detection-models-with-knowledge-distillation.pdf) with **SSD** structure.

## Overall Structure

![structure](structure.png)

## Introduction

For saving time, I combined codes from two existing repositories, [amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch#installation) and [qfgaohao/pytorch-ssd](https://github.com/amdegroot/ssd.pytorch#installation). Trained models also can be downloaded from thier repositories.

**Training sets:** VOC2007 trainval & VOC2012 trainval  
**Testing sets:** VOC2007 test  
(you can use scripts data/VOC2007.sh and data/VOC2012.sh to get them easily)

||Backbone|mAP|URL
-|-|-|-
Teacher net|VGG16|77.43%|[https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth](https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth)
Student net|MobilenetV2(SSD lite)|68.6%|[https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth](https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth)

## Requirements

- Python 3.6+  
- PyTorch 1.3+
- CUDA 10.1+

##  Usages

This repository is just a sample, so you can easily understand how to use it by a quick look of files `train.py` and `eval.py`.  
Basically, if you only want to reproduce the model, you need to change nothing, first download the two models above and put them into `models/`. Then, start training with
>`python train.py`  

to fine tune a vgg-ssd model. Next, use this fine-tuned model to teacher mobilenetv2-ssdlite by 
>`python train.py --teacher_model=$PATH_OF_VGGSSD`

Similarly, evaluate the mobilenetv2 model by
>`python eval.py --trained_model=$PATH_OF_MBV2SSD`

## Result                  
|Classes|Baseline|Promotion|
|-|-|-|
AP for aeroplane | 0.6988    | 0.7220          
AP for bicycle | 0.7788      | 0.7858        
AP for bird | 0.6376         | 0.6641     
AP for boat | 0.5545         | 0.5638     
AP for bottle | 0.3573       | 0.3708       
AP for bus | 0.8001          | 0.8112    
AP for car | 0.7410          | 0.7529    
AP for cat | 0.8240          | 0.8375    
AP for chair | 0.5369        | 0.5493      
AP for cow | 0.6193          | 0.6364    
AP for diningtable | 0.7301  | 0.7238            
AP for dog | 0.7848          | 0.8006    
AP for horse | 0.8236        | 0.8431      
AP for motorbike | 0.8144    | 0.8294          
AP for person | 0.7162       | 0.7211       
AP for pottedplant | 0.4197  | 0.4489            
AP for sheep | 0.6265        | 0.6518      
AP for sofa | 0.7864         | 0.7865     
AP for train | 0.8313        | 0.8330      
AP for tvmonitor | 0.6538    | 0.6654          
Mean AP | 0.6868             | 0.6999

It's the best model I have got but I deleted it becasue of trusting I could get a better one soon...

<!--## For more details
if you have any interest with how these work, [here]() is an article describing the priciples and where I modified in Chinese. You can also post an issue.-->
## TODO

1. provide more details.
2. modify the illogical part.