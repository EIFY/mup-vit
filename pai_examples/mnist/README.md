# PerforatedAI MNIST example

PerforatedAI's MNIST example — PyTorch's standard MNIST CNN (2 conv + 2 linear layers, Adadelta)
with dendrites on every layer — used as a reference: unlike the transformer example, its committed
result (`pai_graph.png`) shows Perforated Backpropagation learning real correlations (best PB
scores ~0.01 for the convolutions and ~0.08 for the linear layers, one dendrite set, test accuracy
99.16% → 99.33%).

- `mnist.py`, `mnist_perforatedai.py`, `requirements.txt`, `PAI/PAI_config.json`,
  `best_arch_scores.csv` and `pai_graph.png` are copied unchanged from
  [PerforatedAI/PerforatedAI](https://github.com/PerforatedAI/PerforatedAI) at tag `e4.0.0`,
  `examples/base_examples/mnist/` (Apache-2.0, see `LICENSE`); their README is
  `UPSTREAM_README.md`. `PAI/PAI_config.json` is the PAI configuration saved with the example,
  which PAI reads from the run directory.
- `run_example.py` is ours: it copies that config into the run directory (as if run from the
  example's folder), applies the PAI settings the example has no option for, and runs
  `mnist_perforatedai.py` unchanged.

## Running it with the project environment

The project environment already has everything the example needs (torch, torchvision,
PerforatedAI). Set the PAI license and run each experiment from its own directory: the example
writes `PAI/` (graphs, CSVs, saved config) and `data/` (MNIST, downloaded on first use) there.

```bash
export PAIEMAIL=... PAITOKEN=...          # or: source ~/.pai_env
REPO=/path/to/ViT-perforated
RUN="uv run --project $REPO python $REPO/pai_examples/mnist/run_example.py"

# Perforated Backpropagation (cascade correlation): the default with perforatedbp installed
mkdir -p runs/pb && (cd runs/pb && $RUN)

# Gradient descent: dendrites trained by backprop together with the network
mkdir -p runs/gd && (cd runs/gd && PAI_MODE=gd $RUN)

# Control: same PAI schedule and LR restarts, no dendrites (cap the restarts, e.g. at GD's count)
mkdir -p runs/control && (cd runs/control && PAI_MODE=gd PAI_CONTROL=1 PAI_MAX_DENDRITES=1 $RUN)
```

Each run continues until PAI reports training complete (`--epochs` defaults to 10000). The
example does not log to W&B.

## What to look at

`PAI/PAI.png` (PAI's graph), `PAI/PAI_best_arch_scores.csv` (best test accuracy per architecture),
`PAI/PAIBest PBScores.csv` (correlation scores; PAI considers > 0.001 as learning), and in
Perforated Backpropagation mode the log line
`Nodes that improved their correlation score this epoch: N` after every dendrite epoch.
