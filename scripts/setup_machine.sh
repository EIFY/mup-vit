#!/bin/bash
# One-time setup of a GPU machine for this repo. Targets: RTX 3090, RTX 4090, L40S, and NCSA Delta
# (A40/A100). Starts no training run.
#
#   ./scripts/setup_machine.sh
#
# On a Slurm cluster, run it on the login node (it has internet for the downloads). The GPU smoke
# test is skipped there; to run it, call the script again inside a GPU job (finished steps are
# skipped):
#   srun --gpus=1 --cpus-per-task=8 --time=1:00:00 -p <partition> -A <account> --pty ./scripts/setup_machine.sh
#
# By default the dataset is only downloaded, as parquet shards, and each training job unpacks
# them to node-local disk itself (as scripts/delta_pai_resume.sbatch does), with:
#   $DATA_VENV/bin/python scripts/parquet_to_imagefolder.py $PARQUET_DIR <local dir> --workers <n>
# UNPACK=1 unpacks them once here instead, e.g. on a personal machine whose disk is already local.
#
# Steps (each is skipped when already done, so re-running is safe):
#   1. install uv
#   2. training venv from uv.lock (torch 2.6.0+cu124), and a small pyarrow venv for unpacking
#   3. credentials: check that .env holds the W&B and PerforatedAI keys (creates it if missing)
#   4. dataset: download evanarlian/imagenet_1k_resized_256 (train + val parquet, ~22 GB);
#      with UNPACK=1 also unpack it to ImageFolder (~25 GB more)
#   5. smoke test (needs a GPU): GPU, credentials, perforation, a 30-step PAI run, and the
#      unpacked dataset if there is one
#
# Settings (environment variables):
#   VENV=path           training venv (default: <repo>/.venv)
#   DATA_VENV=path      pyarrow venv used to unpack the shards (default: <repo>/.venv-data)
#   DATA_DIR=path       download location (default: the repo's parent directory); shards end
#                       up in $DATA_DIR/imagenet_1k_resized_256/data
#   PARQUET_DIR=path    shards already on this machine (a folder with train-*/val-*.parquet, e.g.
#                       Delta's shared copy): only checked, nothing is downloaded or modified
#   UNPACK=1            also unpack the shards to $IMAGENET
#   IMAGENET=path       where UNPACK=1 unpacks to (default: $DATA_DIR/imagenet)
#   DOWNLOAD_DATA=0     skip the download (step 4)
#   DATA_INCLUDE="..."  download only shards matching these patterns, e.g. for a quick test:
#                       DATA_INCLUDE="data/train-00000-* data/val-*"
#   SKIP_SMOKE=1        skip step 5

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV=${VENV:-$REPO/.venv}
DATA_VENV=${DATA_VENV:-$REPO/.venv-data}
DATA_DIR=${DATA_DIR:-$(dirname "$REPO")}
IMAGENET=${IMAGENET:-$DATA_DIR/imagenet}
PYTHON="$VENV/bin/python"

step() { echo; echo "##### $* #####"; }
cd "$REPO"

step "1/5 uv"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

step "2/5 venvs: $VENV (training), $DATA_VENV (unpacking)"
# uv.lock pins torch 2.6.0+cu124, which has kernels for the 3090 (sm_86), 4090 and L40S (sm_89),
# and Delta's A40 (sm_86) and A100 (sm_80). It needs an NVIDIA driver >= 550.
UV_PROJECT_ENVIRONMENT="$VENV" uv sync --frozen
# pyarrow is kept out of the training lockfile; the jobs that unpack the shards use this venv.
if ! "$DATA_VENV/bin/python" -c 'import pyarrow' 2>/dev/null; then
    uv venv --python 3.12 "$DATA_VENV"
    uv pip install --python "$DATA_VENV/bin/python" pyarrow
fi
# Empty when no GPU is visible, e.g. on a Slurm login node.
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)
if [ -n "$CAP" ]; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) (compute capability $CAP)"
fi

step "3/5 credentials (.env)"
# The keys live in .env (git-ignored, never committed). .env.example is only the committed template.
KEYS=(WANDB_API_TOKEN PAIEMAIL PAITOKEN)
if [ ! -f .env ]; then
    cp .env.example .env
    chmod 600 .env
    echo "No .env yet: created $REPO/.env from .env.example"
fi
set -a; source .env; set +a
MISSING=()
for var in "${KEYS[@]}"; do
    if [ -z "${!var:-}" ]; then MISSING+=("$var"); fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    cat >&2 <<EOF

Missing keys in $REPO/.env: ${MISSING[*]}
Open that file (e.g. nano $REPO/.env) and fill in each line as KEY=value, with no quotes or
spaces around '=':
  WANDB_API_TOKEN=   your W&B API key, from https://wandb.ai/authorize
  PAIEMAIL=          PerforatedAI license email
  PAITOKEN=          PerforatedAI license token
Or copy a filled-in .env from another machine: scp .env <user>@<this host>:$REPO/.env
Then re-run this script; finished steps are skipped.
Edit .env, not .env.example: .env.example is committed to git, so keys in it could be pushed.
EOF
    exit 1
fi
echo ".env has ${KEYS[*]}"
if grep -qE '^(WANDB_API_TOKEN|PAIEMAIL|PAITOKEN)=.+' .env.example; then
    echo "WARNING: .env.example contains keys. It is committed to git; clear the values there." >&2
fi

step "4/5 dataset"
if [ -n "${PARQUET_DIR:-}" ]; then
    SHARDS="$PARQUET_DIR"
    echo "using the shards already in $SHARDS"
elif [ "${DOWNLOAD_DATA:-1}" = 0 ]; then
    SHARDS=
    echo "DOWNLOAD_DATA=0: skipping"
else
    SHARDS="$DATA_DIR/imagenet_1k_resized_256/data"
    # train + val only (~22 GB); the unlabeled test-* shards are not used. Resumes if interrupted,
    # and finishes at once when everything is already there.
    INCLUDE=()
    for pattern in ${DATA_INCLUDE:-data/train-* data/val-*}; do
        INCLUDE+=(--include "$pattern")
    done
    # The dataset is public, so download without any Hugging Face login or mirror settings this
    # machine may have: a stale or restricted token (HF_TOKEN, ~/.cache/huggingface/token) or an
    # HF_ENDPOINT mirror makes the Hub answer "Dataset not found".
    env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HF_ENDPOINT HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
        uvx --from huggingface_hub hf download evanarlian/imagenet_1k_resized_256 \
        --repo-type dataset --local-dir "$DATA_DIR/imagenet_1k_resized_256" "${INCLUDE[@]}"
fi
if [ -n "$SHARDS" ]; then
    n_train=$(find "$SHARDS" -maxdepth 1 -name 'train-*.parquet' | wc -l)
    n_val=$(find "$SHARDS" -maxdepth 1 -name 'val-*.parquet' | wc -l)
    echo "$SHARDS: $n_train train and $n_val val shards ($(du -sh "$SHARDS" | cut -f1))"
    if [ "$n_train" = 0 ] || [ "$n_val" = 0 ]; then
        echo "no train-*/val-*.parquet shards in $SHARDS" >&2
        exit 1
    fi
fi
if [ "${UNPACK:-0}" = 1 ]; then
    if [ -f "$IMAGENET/.unpacked" ]; then
        echo "already unpacked in $IMAGENET"
    elif [ -z "$SHARDS" ]; then
        echo "UNPACK=1 needs the shards: drop DOWNLOAD_DATA=0 or set PARQUET_DIR" >&2
        exit 1
    else
        # $IMAGENET/{train,val}/<wnid>/*.JPEG (~25 GB more), the layout main.py reads.
        echo "unpacking to $IMAGENET"
        "$DATA_VENV/bin/python" scripts/parquet_to_imagefolder.py "$SHARDS" "$IMAGENET" --workers "$(nproc)"
        touch "$IMAGENET/.unpacked"   # marks a complete unpack, so re-runs skip it
    fi
fi

step "5/5 smoke test"
if [ "${SKIP_SMOKE:-0}" = 1 ]; then
    echo "SKIP_SMOKE=1: skipping"
elif [ -z "$CAP" ]; then
    echo "No GPU visible (expected on a Slurm login node), so the smoke test cannot run here."
    echo "Optional: run it in a GPU job, e.g."
    echo "  srun --gpus=1 --cpus-per-task=8 --time=1:00:00 -p <partition> -A <account> --pty ./scripts/setup_machine.sh"
else
    # PAI runs refuse to start when more than one GPU is visible (even with --gpu 0), so show the
    # smoke test exactly one: the first of CUDA_VISIBLE_DEVICES if set (e.g. by Slurm), otherwise
    # the GPU with the most free memory. PCI_BUS_ID makes CUDA number GPUs as nvidia-smi does.
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
    else
        CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    fi
    export CUDA_VISIBLE_DEVICES
    echo "smoke test on GPU $CUDA_VISIBLE_DEVICES ($(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.free --format=csv,noheader) free)"
    DATA_ARGS=()
    if [ -f "$IMAGENET/.unpacked" ]; then DATA_ARGS=(--data "$IMAGENET"); fi
    "$PYTHON" scripts/smoke_test.py ${DATA_ARGS[@]+"${DATA_ARGS[@]}"}

    # Short end-to-end PAI run on fake data (dendrites on layer 11): thresholds of 1 make PAI
    # switch to a dendrite phase, add a dendrite and restructure within 30 steps. Nothing is
    # logged to W&B.
    SMOKE_LOGS=$(mktemp -d)
    SMOKE_LOG="$SMOKE_LOGS/pai_smoke.log"
    echo; echo "30-step PAI training run on fake data (about 5 min; full log: $SMOKE_LOG)"
    if ! "$PYTHON" -u main.py --fake-data --gpu 0 --train-fraction 0.2 \
        --hidden-dim 192 --num-layers 12 --num-heads 3 --patch-size 16 --mlp-head \
        --batch-size 64 --accum-freq 1 --tf32 --lr 5e-4 --workers 2 \
        --max-steps 30 --log-steps 5 --print-freq 5 \
        --pai-layers 11 --pai-target ln_mlp --max-dendrites 1 \
        --pai-n-epochs-to-switch 1 --pai-p-epochs-to-switch 1 --pai-initial-correlation-batches 2 \
        --logs "$SMOKE_LOGS" --name pai_smoke > "$SMOKE_LOG" 2>&1; then
        tail -n 30 "$SMOKE_LOG"
        echo "SMOKE TEST FAILED: the PAI training run crashed (see $SMOKE_LOG)" >&2
        exit 1
    fi
    # Training progress and PAI phase changes only; the log also has every validation batch.
    grep -E '^\[ *[0-9]+/|PAI restructured|Switching back to N|Importing best Model' "$SMOKE_LOG" || true
    if ! grep -q "PAI restructured the model" "$SMOKE_LOG"; then
        echo "SMOKE TEST FAILED: the PAI run never restructured the model (see $SMOKE_LOG)" >&2
        exit 1
    fi
    echo "PAI training run ok: switched to a dendrite phase and back"
fi

echo
echo "Setup complete."
echo "  training python:  $PYTHON"
echo "  unpacking python: $DATA_VENV/bin/python"
if [ -n "$SHARDS" ]; then echo "  parquet shards:   $SHARDS"; fi
if [ -f "$IMAGENET/.unpacked" ]; then echo "  unpacked data:    $IMAGENET"; fi
