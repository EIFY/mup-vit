#!/bin/bash

MUPVIT_MAIN=~/Downloads/mup-vit/main.py
PYTHON=torchrun
N_WORKERS=80
EPOCH=300

for LR in 0.001 0.003 0.009
do
    NUMEXPR_MAX_THREADS=124 $PYTHON $MUPVIT_MAIN /data/ImageNet/ --workers $N_WORKERS --multiprocessing-distributed --epochs $EPOCH --log-epoch 90 150 --batch-size 1024 --torchvision-inception-crop --mlp-head --schedule-free --lr $LR --report-to wandb --name schedule-free-lr-${LR}
done
