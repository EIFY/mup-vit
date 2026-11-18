"""Run the ResNet KD example three ways, back to back on one GPU:

    pb       Perforated Backpropagation dendrites   (--dendrite-mode 2)
    gd       gradient-descent dendrites             (--dendrite-mode 1)
    control  same PAI schedule and LR restarts, no dendrites; restarts capped at the number of
             dendrite sets the gd run added          (--dendrite-mode 1, PAI_CONTROL=1)

All three use the example README's first command (ResNet18 on Food-101, no knowledge distillation).

    python pai_examples/resnet_kd/run_comparison.py [--gpu N] [--use-wandb] [--runs-dir DIR]

Picks an idle GPU unless --gpu is given. Needs PAIEMAIL/PAITOKEN in the environment or in
~/.pai_env; with --use-wandb, a W&B key from the environment or the repo's .env. Each run has its
own directory under --runs-dir (log in run.log); Food-101 is downloaded once to --data-path.
Already finished runs (run.done present) are skipped, so the script can be restarted.
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXAMPLE = os.path.join(HERE, "run_example.py")

# The example README's first command (student without knowledge distillation).
EXAMPLE_ARGS = [
    "--dataset", "food101", "--download-food101",
    "--wd", "0.001", "--label-smoothing", "0.1",
    "--mixup-alpha", "0.2", "--cutmix-alpha", "0.6",
    "--random-erase", "0.2", "--auto-augment", "ta_wide", "--dropout", "0.2",
]


def idle_gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    gpus = [tuple(int(x) for x in line.split(",")) for line in out.stdout.strip().splitlines()]
    idle = [g for g in gpus if g[1] < 1000 and g[2] < 10]
    if not idle:
        sys.exit(f"no idle GPU (index, MiB used, % util): {gpus}; pass --gpu")
    return idle[0][0]


def load_env_file(path, names):
    """Read NAME=value / export NAME=value lines (quotes stripped) for the given names."""
    values = {}
    if os.path.exists(path):
        for line in open(path):
            m = re.match(r"\s*(?:export\s+)?(\w+)=(.*)", line)
            if m and m.group(1) in names:
                values[m.group(1)] = m.group(2).strip().strip("'\"")
    return values


def gd_dendrite_sets(gd_dir):
    """Dendrite sets the gd run added: rows of PAI's best_arch_scores minus the base model."""
    for path in glob.glob(os.path.join(gd_dir, "*", "*_best_arch_scores.csv")):
        folder = os.path.basename(os.path.dirname(path))
        if os.path.basename(path) == f"{folder}_best_arch_scores.csv":
            with open(path) as f:
                return max(sum(1 for _ in f) - 2, 1)
    sys.exit(f"no PAI best_arch_scores.csv found in {gd_dir}; did the gd run finish?")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=int, default=None, help="GPU index (default: an idle one)")
    parser.add_argument("--runs-dir", default=os.path.expanduser("~/runs/pai_resnet_kd"))
    parser.add_argument("--data-path", default=os.path.expanduser("~/data/food101"))
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--python", default=os.path.join(REPO, ".venv", "bin", "python"))
    args, extra = parser.parse_known_args()

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu if args.gpu is not None else idle_gpu())
    env.update({k: v for k, v in load_env_file(os.path.expanduser("~/.pai_env"),
                                                {"PAIEMAIL", "PAITOKEN"}).items() if k not in env})
    if not (env.get("PAIEMAIL") and env.get("PAITOKEN")):
        sys.exit("set PAIEMAIL and PAITOKEN (environment or ~/.pai_env)")
    if args.use_wandb and "WANDB_API_KEY" not in env:
        token = load_env_file(os.path.join(REPO, ".env"), {"WANDB_API_TOKEN"}).get("WANDB_API_TOKEN")
        if token:
            env["WANDB_API_KEY"] = token
    os.makedirs(args.data_path, exist_ok=True)

    common = EXAMPLE_ARGS + ["--data-path", args.data_path] + (["--use-wandb"] if args.use_wandb else []) + extra
    runs = [
        ("pb", ["--dendrite-mode", "2"], {"RUN_SUFFIX": "_pb"}),
        ("gd", ["--dendrite-mode", "1"], {"RUN_SUFFIX": "_gd"}),
        ("control", ["--dendrite-mode", "1"], {"RUN_SUFFIX": "_control", "PAI_CONTROL": "1"}),
    ]
    print(f"GPU {env['CUDA_VISIBLE_DEVICES']}, runs in {args.runs_dir}", flush=True)
    for name, mode_args, run_env in runs:
        run_dir = os.path.join(args.runs_dir, name)
        os.makedirs(run_dir, exist_ok=True)
        if os.path.exists(os.path.join(run_dir, "run.done")):
            print(f"[{name}] already done, skipping", flush=True)
            continue
        if name == "control":
            run_env["PAI_MAX_DENDRITES"] = str(gd_dendrite_sets(os.path.join(args.runs_dir, "gd")))
        start = time.time()
        print(f"[{name}] starting {time.strftime('%F %T')} {run_env}", flush=True)
        with open(os.path.join(run_dir, "run.log"), "w") as log:
            code = subprocess.call([args.python, "-u", EXAMPLE] + common + mode_args,
                                   cwd=run_dir, env={**env, **run_env}, stdout=log, stderr=subprocess.STDOUT)
        print(f"[{name}] exit {code} after {(time.time() - start) / 3600:.2f} h", flush=True)
        if code != 0:
            sys.exit(f"[{name}] failed; see {run_dir}/run.log")
        open(os.path.join(run_dir, "run.done"), "w").close()


if __name__ == "__main__":
    main()
