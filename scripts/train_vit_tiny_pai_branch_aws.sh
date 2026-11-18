#!/bin/bash
# PerforatedAI arms branched from one neuron-phase checkpoint of a from-scratch PAI run
# (train_vit_tiny_pai_scratch_aws.sh), single T4 (AWS g4dn.xlarge). Two ways to branch:
#
#   PAI_RESUME=ckpt ./train_vit_tiny_pai_branch_aws.sh 7
#       ckpt is the run's own main.py checkpoint from before its first switch. Continues that run
#       (step, PAI tracker, optimizer, LR scheduler) with dendrites on layer 7, so PAI decides the
#       switch itself. EPOCHS is the total including the shared epochs (default 300).
#   PAI_INIT=plain.pth.tar ./train_vit_tiny_pai_branch_aws.sh 7
#       plain checkpoint from scripts/export_plain_checkpoint.py. Fresh tracker; dendrite arms are
#       forced into a dendrite phase immediately. EPOCHS counts from 0 (default 300).
#
# ARM "control" keeps layer 11's wrapping without ever adding dendrites. Same data, batch and LR
# as the from-scratch run. EPOCHS is only a safety cap; PAI ends the run itself.

set -euo pipefail

ARM=${1:?usage: PAI_RESUME=ckpt|PAI_INIT=ckpt $0 <layer index|control>}
HISTORY=()
if [ -n "${PAI_RESUME:-}" ]; then
    START=(--pai-resume "$(realpath "$PAI_RESUME")")
    KIND=resume
    # Copy the source run's W&B history up to the resume step. main.py names the W&B run after
    # the log directory, which holds the checkpoint under checkpoints/ or archive/<snapshot>/.
    CKPT=$(realpath "$PAI_RESUME")
    SRC_DIR=${CKPT%%/archive/*}
    [ "$SRC_DIR" = "$CKPT" ] && SRC_DIR=${CKPT%%/checkpoints/*}
    [ "$SRC_DIR" != "$CKPT" ] && HISTORY=(--wandb-history-from "$(basename "$SRC_DIR")")
elif [ -n "${PAI_INIT:-}" ]; then
    START=(--pai-init "$(realpath "$PAI_INIT")")
    KIND=branch
else
    echo "set PAI_RESUME (pre-switch main.py checkpoint) or PAI_INIT (plain checkpoint)" >&2
    exit 1
fi
if [ "$ARM" = control ]; then
    PAI_ARGS=(--pai-layers 11 --pai-control)
else
    PAI_ARGS=(--pai-layers "$ARM")
fi

EPOCHS=${EPOCHS:-300}
MAX_DENDRITES=${MAX_DENDRITES:-2}
# CAND_CLIP: clip candidate dendrite gradients to this norm (PAI's suggestion for NaN candidates).
EXTRA=()
if [ -n "${CAND_CLIP:-}" ]; then EXTRA+=(--pai-candidate-grad-clipping "$CAND_CLIP"); fi
# WD: weight decay (main.py's --wd, divided by the LR). PAI notes weight decay can hurt dendrite learning.
if [ -n "${WD:-}" ]; then EXTRA+=(--wd "$WD"); fi
PAI_TARGET=${PAI_TARGET:-ln_mlp}
# Always name the target: older runs without a target tag used 'mlp'.
TARGET_TAG="_${PAI_TARGET}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGENET=${IMAGENET:-$HOME/data/imagenet}
LOG_ROOT=${LOG_ROOT:-$HOME/runs}
RUN_NAME="vit_tiny_pai_${KIND}_${ARM}${TARGET_TAG}_20pct_$(date +%Y%m%d_%H%M)"

if [ -f "$HOME/.pai_env" ]; then source "$HOME/.pai_env"; fi
: "${PAIEMAIL:?PAIEMAIL is not set; create ~/.pai_env}"
: "${PAITOKEN:?PAITOKEN is not set; create ~/.pai_env}"

test -d "$IMAGENET/train"
test -d "$IMAGENET/val"
test -r "${START[1]}"
mkdir -p "$LOG_ROOT"

export WANDB_DIR="$LOG_ROOT/$RUN_NAME"
export WANDB_ENTITY="IdlF26Group40"
export WANDB_PROJECT="ViT-perforated"
mkdir -p "$WANDB_DIR"

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
    --log-steps 500 \
    --workers 4 \
    "${START[@]}" \
    ${HISTORY[@]+"${HISTORY[@]}"} \
    "${PAI_ARGS[@]}" \
    --pai-target "$PAI_TARGET" \
    --max-dendrites "$MAX_DENDRITES" \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    --logs "$LOG_ROOT" \
    --name "$RUN_NAME" \
    --report-to wandb
