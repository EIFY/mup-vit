# PerforatedAI ResNet knowledge-distillation example

PerforatedAI's ResNet18 example on Food-101, used as a third reference next to the transformer
and MNIST examples. We reproduce the example README's first command — ResNet18 trained on 25% of
Food-101 *without* knowledge distillation — whose committed result (`pai_graph.png`,
`best_arch_scores.csv`) is 75.2% → 77.4% top-1 with one dendrite set. (The KD variants need a
ResNet50 teacher pre-trained first.)

- `train_perforated_resnet_KD.py`, `resnet.py`, `resnet_double.py`, `presets.py`, `sampler.py`,
  `transforms.py`, `utils.py`, `PAI.json`, `best_arch_scores.csv` and `pai_graph.png` are copied
  unchanged from [PerforatedAI/PerforatedAI](https://github.com/PerforatedAI/PerforatedAI) at tag
  `e4.0.0`, `examples/base_examples/resnet_knowledge_distillation/` (Apache-2.0, see `LICENSE`);
  their README is `UPSTREAM_README.md`.
- `run_example.py` (ours) adds a no-dendrite control with LR restarts (`PAI_CONTROL=1`), a
  `max_dendrites` override and W&B run-name suffixes, and gives the run directory a copy of
  `PAI.json` with only its two confirmation flags set, so PAI does not ask interactively.
- `run_comparison.py` (ours) runs the three comparison runs back to back on one GPU.

The example selects the dendrite mode itself: `--dendrite-mode 2` Perforated Backpropagation
(default), `1` gradient descent, `0` no dendrites (PAI stops at the first switch, so it is a
plain baseline, not a restart-matched control).

## Running the comparison

The project environment has everything the example needs. Set the PAI license (environment or
`~/.pai_env`), then:

```bash
uv run --project /path/to/ViT-perforated python /path/to/ViT-perforated/pai_examples/resnet_kd/run_comparison.py --use-wandb
```

It picks an idle GPU (or `--gpu N`), downloads Food-101 once to `--data-path`
(default `~/data/food101`, ~5 GB), and runs `pb`, `gd` and `control` in turn under `--runs-dir`
(default `~/runs/pai_resnet_kd`), each logging to its `run.log`. The control's restarts are capped
at the number of dendrite sets the gd run added. Finished runs are marked with `run.done` and
skipped if the script is restarted. With `--use-wandb`, runs log to the example's W&B project
("ImageNet-100 PerforatedAI") with `_pb`, `_gd`, `_control` name suffixes; the key comes from
`WANDB_API_KEY` or `WANDB_API_TOKEN` in the repo's `.env`.

Each run trains until PAI reports completion (the example uses `n_epochs_to_switch` 40 and up to
5 dendrite sets), so a run can take several hours.

## A single run by hand

```bash
mkdir -p runs/pb && cd runs/pb
uv run --project $REPO python $REPO/pai_examples/resnet_kd/run_example.py \
  --dataset food101 --download-food101 --data-path ~/data/food101 \
  --wd 0.001 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 0.6 \
  --random-erase 0.2 --auto-augment ta_wide --dropout 0.2 --dendrite-mode 2
```

## What to look at

PAI's folder in the run directory (`food101_..._dmode<N>_<timestamp>/`): the graph `<name>.png`,
`<name>_best_arch_scores.csv` (accuracy per architecture) and `<name>Best PBScores.csv`
(correlation scores; PAI considers > 0.001 as learning), plus, in Perforated Backpropagation mode,
the log line `Nodes that improved their correlation score this epoch: N`.
