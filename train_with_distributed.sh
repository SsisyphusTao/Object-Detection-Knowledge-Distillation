#! /bin/bash
rm log.txt
time \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
            --nproc_per_node=4 train.py \
            --batch_size=32 \
            --lr=1e-3 \
            --teacher_model=models/teacher_vgg_7325.pth \
            --dataset_root /ai/ailab/Share/TaoData/voc/VOCdevkit \
            >> log.txt