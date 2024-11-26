#!/bin/bash

MUPVIT_MAIN=~/Downloads/mup-vit/main.py
PYTHON=torchrun
N_WORKERS=80
N_THREADS=124
PREFETCH=4
EPOCH=300
BETA2=0.9955159689799007
WD=0.08121616522670176
PWP=0.75
WARMUP=2799  # int(HPARAMS['warmup_factor'] * workload.step_hint * 0.75) = int(0.02 * 186_666 * 0.75) = 2799
CLIP=100

for LR in 0.00125 0.0025 0.005
do
    NUMEXPR_MAX_THREADS=$N_THREADS $PYTHON $MUPVIT_MAIN /data/ImageNet/ --workers $N_WORKERS --prefetch-factor $PREFETCH --multiprocessing-distributed --epochs $EPOCH --log-epoch 90 150 --batch-size 1024 --torchvision-inception-crop --mlp-head --warmup $WARMUP --schedule-free --grad-clip-norm $CLIP --lr $LR --no-decoupled-weight-decay --weight-decay $WD --beta2 $BETA2 --polynomial-weighting-power $PWP --report-to wandb --name schedule-free-algoperf-lr-${LR}
done
