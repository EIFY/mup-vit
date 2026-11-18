# PerforatedAI transformer example

PerforatedAI's own transformer example — a 2-layer transformer language model on WikiText-2 with
every attention and feed-forward `Linear` perforated — used here as a reference point for our ViT
experiments: does Perforated Backpropagation learn dendrite correlations on PAI's own transformer?

- `train.py`, `model.py`, `data_preparation.py`, `test_setup.py`, `requirements.txt` are copied
  unchanged from [PerforatedAI/PerforatedAI](https://github.com/PerforatedAI/PerforatedAI) at tag
  `e4.0.0`, `examples/base_examples/transformer/` (Apache-2.0, see `LICENSE`). Their README is
  kept as `UPSTREAM_README.md`.
- `run_example.py` is ours: it applies PAI settings the example has no option for (mode, control,
  non-interactive configuration) and then runs `train.py` unchanged.

## Running it with the project environment

No separate venv is needed: the project's environment has torch, PerforatedAI, wandb, numpy and
matplotlib, and `uv run --with` adds the example's two extra packages temporarily without
touching `.venv` or the lockfile. Set the PAI license first, and run each experiment from its own
directory — the example writes `dendritic_128d_2l/` (PAI graphs and CSVs),
`best_dendritic_model.pt` and `wandb/` into the current directory under fixed names.

```bash
export PAIEMAIL=... PAITOKEN=...          # or: source ~/.pai_env
REPO=/path/to/ViT-perforated
RUN="uv run --project $REPO --with datasets --with tqdm python $REPO/pai_examples/transformer/run_example.py"

# Perforated Backpropagation (cascade correlation): the default with perforatedbp installed
mkdir -p runs/pb && (cd runs/pb && $RUN --model_type dendritic --dropout 0.25)

# Gradient descent: dendrites trained by backprop together with the network
mkdir -p runs/gd && (cd runs/gd && PAI_MODE=gd RUN_SUFFIX=_gd $RUN --model_type dendritic --dropout 0.25)

# Control: same PAI schedule and LR restarts, no dendrites (cap the restarts, e.g. at GD's count)
mkdir -p runs/control && (cd runs/control && \
    PAI_MODE=gd PAI_CONTROL=1 PAI_MAX_DENDRITES=8 RUN_SUFFIX=_control \
    $RUN --model_type dendritic --dropout 0.25)
```

Add `--no_wandb` to skip W&B; otherwise runs log to the `dendritic-transformer-comparison`
project of your default W&B entity. WikiText-2 is downloaded through Hugging Face `datasets` on
the first run. The dendritic and control runs continue until PAI reports training complete, not
until `--epochs`; the example's vanilla model (`--model_type vanilla`) stops at `--epochs`, so it is
not a like-for-like control.

## What to look at

PAI's graph `dendritic_128d_2l/dendritic_128d_2l.png` and its CSVs; in particular
`dendritic_128d_2lBest PBScores.csv` (correlation scores; PAI considers > 0.001 as learning) and,
in Perforated Backpropagation mode, the log line
`Nodes that improved their correlation score this epoch: N` after every dendrite epoch. GD mode
has no correlation phase, so it prints neither.

Results on one laptop GPU (RTX A5500), `--dropout 0.25`, Nov 2026:

| Run | Dendrite sets | Parameters | Epochs | Best val perplexity |
|---|---|---|---|---|
| Perforated Backpropagation | 2 (PB scores all 0) | 3.77M | 122 | 90.33 |
| Gradient descent | 8 | 6.29M | 254 | 86.10 |
| Control (`PAI_MAX_DENDRITES=8`; PAI stopped after 1 restart) | 0 | 2.97M | 112 | 90.92 |
