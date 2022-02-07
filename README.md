# Object Detection Knowledge Distillation(ODKD)

![version](https://img.shields.io/badge/version-beta_0.0.1-brightgreen.svg)
![coverage](https://codecov.io/github/SsisyphusTao/SSD-Knowledge-Distillation/coverage.svg?branch=dev)

The function of this branch is not complete. For [ssd](https://github.com/SsisyphusTao/SSD-Knowledge-Distillation/tree/mbv2-lite) and [yolov5](https://github.com/SsisyphusTao/Object-Detection-Knowledge-Distillation/tree/yolov5) distillation, checking other branches. 

Release edition is coming Soon...

## Update

1. The first edition is the refactor of branch [mbv2-lite](https://github.com/SsisyphusTao/SSD-Knowledge-Distillation/tree/mbv2-lite), which is an implementation of Chen, G. et al. (2017) [‘Learning efficient object detection models with knowledge distillation’](http://papers.nips.cc/paper/6676-learning-efficient-object-detection-models-with-knowledge-distillation.pdf) with **SSD-lite** structure.

2. Replace part of code with pytorch api which has same functionality.

3. Very friendly beginner guidance.

4. System Architecture

![odkd](http://assets.processon.com/chart_image/6198ae621efad406f87a16ec.png)

## Useage
```
$ python setup.py install --user

$ odkd-train ./training_config.yml -t

$ odkd-train training_config.yml
or
$ python -m torch.distributed.launch --nproc_per_node=2 `which odkd-train` training_config.yml

$ odkd-eval ${CHECKPOINTS_PATH}/${RUN_INDEX}/config.yml
```

## TODO

- Evaluation Module

- LOG Module

- Coco dataset support

- Yolov5 distillation
