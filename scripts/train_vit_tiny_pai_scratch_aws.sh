#!/bin/bash
# PerforatedAI from scratch on ViT-Tiny/16, single T4 (AWS g4dn.xlarge: 4 vCPUs, 16 GB RAM).
# Same fixed 20% ImageNet subset as the fine-tuning ablation, but no --pai-init:
#   ./train_vit_tiny_pai_scratch_aws.sh 11       dendrites on the MLP of encoder layer 11
#   ./train_vit_tiny_pai_scratch_aws.sh control  same wrapping, tracker and schedule, no dendrites
# Any encoder layer index works as the arm. Run inside tmux; one run takes a day or more.
#
# Batch 512, not 1024: PAI runs are uncompiled and cannot accumulate gradients, and batch 1024
# does not fit in 16 GB of GPU memory (peak at 512 is about 9.4 GB). The LR is scaled linearly
# from the 1024-batch 1e-3 used by the 20% from-scratch baseline.
# The 20% subset is about 500 steps per epoch at batch 512: validate (and score for PAI) once per
# epoch and warm up for 8 epochs, as the baseline did (2000 steps at batch 1024).

set -euo pipefail

ARM=${1:?usage: $0 <layer index|control>}
if [ "$ARM" = control ]; then
    PAI_ARGS=(--pai-layers 11 --pai-control)
else
    PAI_ARGS=(--pai-layers "$ARM")
fi

EPOCHS=${EPOCHS:-150}
MAX_DENDRITES=${MAX_DENDRITES:-2}
# Optional: WD (main.py --wd), FIXED_SWITCH (switch to dendrites after N epochs, PAI fixed mode),
# TAG (appended to the run name to tell experiments apart).
EXTRA=()
if [ -n "${WD:-}" ]; then EXTRA+=(--wd "$WD"); fi
if [ -n "${FIXED_SWITCH:-}" ]; then EXTRA+=(--pai-fixed-switch "$FIXED_SWITCH"); fi
PAI_TARGET=${PAI_TARGET:-ln_mlp}
# Always name the target: older runs without a target tag used 'mlp'.
TARGET_TAG="_${PAI_TARGET}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGENET=${IMAGENET:-$HOME/data/imagenet}
LOG_ROOT=${LOG_ROOT:-$HOME/runs}
RUN_NAME="vit_tiny_pai_scratch_${ARM}${TARGET_TAG}${TAG:+_$TAG}_20pct_$(date +%Y%m%d_%H%M)"

# PerforatedAI credentials (PAIEMAIL/PAITOKEN).
if [ -f "$HOME/.pai_env" ]; then source "$HOME/.pai_env"; fi
: "${PAIEMAIL:?PAIEMAIL is not set; create ~/.pai_env}"
: "${PAITOKEN:?PAITOKEN is not set; create ~/.pai_env}"

test -d "$IMAGENET/train"
test -d "$IMAGENET/val"
mkdir -p "$LOG_ROOT"

# main.py reads WANDB_API_TOKEN from $REPO/.env.
export WANDB_DIR="$LOG_ROOT/$RUN_NAME"
export WANDB_ENTITY="IdlF26Group40"
export WANDB_PROJECT="ViT-perforated"
mkdir -p "$WANDB_DIR"

# Avoid CPU thread oversubscription with data-loading workers.
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# uv may not be on PATH in a non-login tmux shell.
UV=$(command -v uv || echo "$HOME/.local/bin/uv")

cd "$REPO"
"$UV" run python -u main.py "$IMAGENET" \
    --gpu 0 \
    --train-fraction 0.2 \
    --subset-seed 42 \
    --hidden-dim 192 \
    --num-layers 12 \
    --num-heads 3 \
    --patch-size 16 \
    --batch-size 512 \
    --accum-freq 1 \
    --amp \
    --mlp-head \
    --torchvision-inception-crop \
    --epochs "$EPOCHS" \
    --lr 5e-4 \
    --pai-warmup 4000 \
    --log-steps 500 \
    --workers 4 \
    "${PAI_ARGS[@]}" \
    --pai-target "$PAI_TARGET" \
    --max-dendrites "$MAX_DENDRITES" \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    --logs "$LOG_ROOT" \
    --name "$RUN_NAME" \
    --report-to wandb
