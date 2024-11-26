#!/bin/bash

MUPVIT_MAIN=~/Downloads/mup-vit/main.py
PYTHON=torchrun
EPOCH=300
CLIP=100
N_WORKERS=80

for LR in 0.001 0.003 0.005
do
    for WD in 0.0001 0.0003 0.0005
    do
        NUMEXPR_MAX_THREADS=124 $PYTHON $MUPVIT_MAIN /data/ImageNet/ --workers $N_WORKERS --multiprocessing-distributed --epochs $EPOCH --log-epoch 90 150 --batch-size 1024 --torchvision-inception-crop --mlp-head --schedule-free --lr $LR --weight-decay $WD --grad-clip-norm $CLIP --report-to wandb --name schedule-free-lr-${LR}-wd-${WD}-no-clip
    done
done
