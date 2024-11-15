#!/bin/bash

MUPVIT_MAIN=~/Downloads/mup-vit/main.py
PYTHON=torchrun
N_WORKERS=80
N_THREADS=124

for EPOCH in 90 150 300
do
    NUMEXPR_MAX_THREADS=$N_THREADS $PYTHON $MUPVIT_MAIN /data/ImageNet/ --workers $N_WORKERS --multiprocessing-distributed --epochs $EPOCH --batch-size 1024 --torchvision-inception-crop --mlp-head --report-to wandb --name better-baseline-${EPOCH}ep
done

for EPOCH in 90 150 300
do
    NUMEXPR_MAX_THREADS=$N_THREADS $PYTHON $MUPVIT_MAIN /data/ImageNet/ --workers $N_WORKERS --multiprocessing-distributed --epochs $EPOCH --batch-size 1024 --torchvision-inception-crop --mlp-head --no-randaug --mixup-alpha 0.0 --report-to wandb --name no-randaug-no-mixup-${EPOCH}ep
done

for EPOCH in 90 150 300
do
    NUMEXPR_MAX_THREADS=$N_THREADS $PYTHON $MUPVIT_MAIN /data/ImageNet/ --workers $N_WORKERS --multiprocessing-distributed --epochs $EPOCH --batch-size 1024 --torchvision-inception-crop --mlp-head --posemb learn --report-to wandb --name learned-posemb-${EPOCH}ep
done

for EPOCH in 90 150 300
do
    NUMEXPR_MAX_THREADS=$N_THREADS $PYTHON $MUPVIT_MAIN /data/ImageNet/ --workers $N_WORKERS --multiprocessing-distributed --epochs $EPOCH --batch-size 4096 --torchvision-inception-crop --mlp-head --report-to wandb --name batch-size-4096-${EPOCH}ep
done

for EPOCH in 90 150 300
do
    NUMEXPR_MAX_THREADS=$N_THREADS $PYTHON $MUPVIT_MAIN /data/ImageNet/ --workers $N_WORKERS --multiprocessing-distributed --epochs $EPOCH --batch-size 1024 --torchvision-inception-crop --mlp-head --pool-type tok --report-to wandb --name token-pool-${EPOCH}ep
done

for EPOCH in 90 150 300
do
    NUMEXPR_MAX_THREADS=$N_THREADS $PYTHON $MUPVIT_MAIN /data/ImageNet/ --workers $N_WORKERS --multiprocessing-distributed --epochs $EPOCH --batch-size 1024 --torchvision-inception-crop --report-to wandb --name linear-head-${EPOCH}ep
done
