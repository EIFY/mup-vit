#!/bin/bash
#SBATCH --job-name=vit-pai-resume
#SBATCH --account=bjto-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/bjto/lhann/runs/slurm-%j.out

# Continue a from-scratch PAI run's neuron phase from a pre-switch checkpoint (--pai-resume) on
# one Delta GPU, with dendrites on encoder layer(s) ARM. Default: the shared step-61000 snapshot of
# vit_tiny_pai_scratch_11_20pct_20261111_0433, the starting point of the AWS resume-* runs.
#   sbatch scripts/delta_pai_resume.sbatch                       # layer 11, ln_mlp, FP32 + TF32
#   sbatch --export=ALL,ARM=7 scripts/delta_pai_resume.sbatch    # another layer
#   sbatch --export=ALL,ARM=6-11 scripts/delta_pai_resume.sbatch # layers 6..11 (a range, inclusive)
#   sbatch --export=ALL,ARM=control scripts/delta_pai_resume.sbatch          # layer 11, no dendrites
#   sbatch --export=ALL,ARM=6-11,CONTROL=1 scripts/delta_pai_resume.sbatch   # layers 6..11, no dendrites
# Precision: FP32 with TF32 tensor cores by default (no fp16 AMP); AMP=1 restores --amp.
# PAI_MODE: pb (default, Perforated Backpropagation / cascade correlation) or gd (dendrites trained
# by gradient descent with the neurons, PAI's open-source mode).
# DENDRITE_LOSS=covariance (pb only): perforatedbp's CovarianceDendriteLoss instead of 4.0.0's
# default, whose PB scores are always 0; adds _covloss to the run name.
# EPOCHS is the total including the shared epochs and only a safety cap; PAI ends the run.
# SCRATCH=1: train from random initialization instead of branching from RESUME, with a linear LR
# warmup over PAI_WARMUP steps (default 4000). The run is named vit_tiny_pai_scratch_<ARM>_...
# and keeps its last pre-switch state in <run>/archive/pre_switch/, which later arms can branch
# from with RESUME=<run dir>/archive/pre_switch/checkpoint.pth.tar. Pass SCRATCH=1 with
# CONTINUE_RUN too.
# CONTINUE_RUN=<run name>: continue that run after a timeout from its last checkpoint (main.py
# --resume; dendrites allowed, but it must have been saved in a neuron phase), in the same log
# directory. W&B gets a new run, <run name>_cont<job id>, with the original run's history up to
# the checkpoint copied in. Pass the original ARM/CONTROL/PAI_MODE/DENDRITE_LOSS/AMP/MAX_DENDRITES too:
#   sbatch --export=ALL,DENDRITE_LOSS=covariance,CONTINUE_RUN=vit_tiny_pai_resume_11_..._1419 \
#       scripts/delta_pai_resume.sbatch

set -euo pipefail
umask 002

ARM=${ARM:-11}
PAI_TARGET=${PAI_TARGET:-ln_mlp}
EPOCHS=${EPOCHS:-300}
MAX_DENDRITES=${MAX_DENDRITES:-2}
PROJECT=/projects/bjto/lhann
SHARED=/projects/bjto/shared
REPO=$PROJECT/ViT-perforated
PARQUET=$SHARED/imagenet_1k_resized_256/data
RESUME=${RESUME:-$SHARED/checkpoints/vit_tiny_pai_scratch_11_20pct_20261111_0433/archive/1111_234822_step61000/checkpoint.pth.tar}
LOG_ROOT=$PROJECT/runs
LOCAL=${LOCAL:-/tmp/$USER/imagenet_$SLURM_JOB_ID}
# The source run's W&B id is its log directory name (<run>/archive/<snapshot>/ or <run>/checkpoints/).
SRC_DIR=${RESUME%%/archive/*}
[ "$SRC_DIR" = "$RESUME" ] && SRC_DIR=${RESUME%%/checkpoints/*}
SRC_RUN=$(basename "$SRC_DIR")

# ARM: one layer (11), an inclusive range (6-11), or control (= layer 11 with CONTROL=1).
CONTROL=${CONTROL:-0}
ARM_TAG=$ARM
if [ "$ARM" = control ]; then
    LAYERS=(11)
    CONTROL=1
elif [[ $ARM =~ ^([0-9]+)-([0-9]+)$ ]]; then
    LAYERS=($(seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"))
    [ ${#LAYERS[@]} -gt 0 ] || { echo "ARM=$ARM is an empty range" >&2; exit 1; }
elif [[ $ARM =~ ^[0-9]+$ ]]; then
    LAYERS=("$ARM")
else
    echo "ARM must be a layer (11), a range (6-11) or control" >&2; exit 1
fi
PAI_ARGS=(--pai-layers "${LAYERS[@]}")
if [ "$CONTROL" = 1 ]; then
    PAI_ARGS+=(--pai-control)
    [ "$ARM" = control ] || ARM_TAG=${ARM}_control
fi
PAI_MODE=${PAI_MODE:-pb}
MODE_ARGS=()
case "$PAI_MODE" in
    pb) ;;
    gd) MODE_ARGS=(--pai-gd) ;;
    *) echo "PAI_MODE must be pb or gd" >&2; exit 1 ;;
esac
LOSS_TAG=
case "${DENDRITE_LOSS:-default}" in
    default) ;;
    covariance) MODE_ARGS+=(--pai-dendrite-loss covariance); LOSS_TAG=_covloss ;;
    *) echo "DENDRITE_LOSS must be default or covariance" >&2; exit 1 ;;
esac
PRECISION=(--tf32)
PREC_TAG=fp32
if [ "${AMP:-0}" = 1 ]; then PRECISION=(--amp); PREC_TAG=amp; fi
SCRATCH=${SCRATCH:-0}
KIND=resume
[ "$SCRATCH" = 1 ] && KIND=scratch
RUN_PREFIX="vit_tiny_pai_${KIND}_${ARM_TAG}_${PAI_TARGET}_${PAI_MODE}${LOSS_TAG}_${PREC_TAG}_20pct_"
if [ -n "${CONTINUE_RUN:-}" ]; then
    case "$CONTINUE_RUN" in
        "$RUN_PREFIX"*) ;;
        *) echo "CONTINUE_RUN=$CONTINUE_RUN does not match these settings ($RUN_PREFIX*)" >&2; exit 1 ;;
    esac
    RUN_NAME=$CONTINUE_RUN
    CKPT_DIR=$LOG_ROOT/$RUN_NAME/checkpoints
    # The continued run overwrites checkpoint.pth.tar at its first validation; keep the start point.
    CONTINUE_FROM=$CKPT_DIR/continue_job${SLURM_JOB_ID}.pth.tar
    cp "$CKPT_DIR/checkpoint.pth.tar" "$CONTINUE_FROM"
    STEP=$("$REPO/.venv/bin/python" -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['step'])" "$CONTINUE_FROM")
    WANDB_RUN=${RUN_NAME}_cont${SLURM_JOB_ID}
    echo "Continuing $RUN_NAME from step $STEP as W&B run $WANDB_RUN"
    START_ARGS=(--resume "$CONTINUE_FROM" --wandb-name "$WANDB_RUN" --wandb-history-from "$RUN_NAME")
    # Own W&B dir: the original's wandb-resume.json (left by the timeout) would make
    # resume='auto' reach for the old run.
    WANDB_DIR=$LOG_ROOT/$RUN_NAME/$WANDB_RUN
else
    RUN_NAME="${RUN_PREFIX}$(date +%Y%m%d_%H%M)"
    if [ "$SCRATCH" = 1 ]; then
        START_ARGS=(--pai-warmup "${PAI_WARMUP:-4000}")
    else
        START_ARGS=(--pai-resume "$RESUME" --wandb-history-from "$SRC_RUN")
    fi
    WANDB_DIR=$LOG_ROOT/$RUN_NAME
fi

source "$PROJECT/.pai_env"
: "${PAIEMAIL:?PAIEMAIL is not set in $PROJECT/.pai_env}"
: "${PAITOKEN:?PAITOKEN is not set in $PROJECT/.pai_env}"
if [ -z "${CONTINUE_RUN:-}" ] && [ "$SCRATCH" != 1 ]; then
    test -r "$RESUME"
    test -r "$(dirname "$RESUME")/pai/best_model.pt"   # PAI reloads it at the first switch
fi
mkdir -p "$LOG_ROOT"

# Unpack the parquet shards to node-local disk: 1.3M small files would be slow on /projects.
mkdir -p "$LOCAL"
avail_gb=$(df -BG --output=avail "$LOCAL" | tail -1 | tr -dc 0-9)
if [ "$avail_gb" -lt 40 ]; then
    echo "only ${avail_gb} GB free on $(df --output=target "$LOCAL" | tail -1); need ~30 GB" >&2
    exit 1
fi
trap 'rm -rf "$LOCAL"' EXIT
echo "Unpacking ImageNet to $LOCAL ($avail_gb GB free)..."
start=$SECONDS
# pyarrow lives in a small separate venv ($PROJECT/.venv-data) so the training lockfile stays unchanged.
"$PROJECT/.venv-data/bin/python" "$REPO/scripts/parquet_to_imagefolder.py" "$PARQUET" "$LOCAL" --workers "$SLURM_CPUS_PER_TASK"
echo "Unpacked in $((SECONDS - start)) s"

export WANDB_DIR
export WANDB_ENTITY="IdlF26Group40"
export WANDB_PROJECT="ViT-perforated"
mkdir -p "$WANDB_DIR"
export OMP_NUM_THREADS=1

cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/python -u main.py "$LOCAL" \
    --gpu 0 \
    --train-fraction 0.2 \
    --subset-seed 42 \
    --hidden-dim 192 \
    --num-layers 12 \
    --num-heads 3 \
    --patch-size 16 \
    --batch-size 512 \
    --accum-freq 1 \
    "${PRECISION[@]}" \
    --mlp-head \
    --torchvision-inception-crop \
    --epochs "$EPOCHS" \
    --lr 5e-4 \
    --log-steps 500 \
    --workers $((SLURM_CPUS_PER_TASK - 2)) \
    "${START_ARGS[@]}" \
    "${PAI_ARGS[@]}" \
    ${MODE_ARGS[@]+"${MODE_ARGS[@]}"} \
    --pai-target "$PAI_TARGET" \
    --max-dendrites "$MAX_DENDRITES" \
    --logs "$LOG_ROOT" \
    --name "$RUN_NAME" \
    --report-to wandb
