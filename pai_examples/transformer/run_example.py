"""Run PerforatedAI's transformer example (train.py, unchanged) with PAI settings applied first.

The example has no option for these settings, so they are set here before train.py runs:

    PAI_MODE=pb|gd      pb (default): Perforated Backpropagation / cascade correlation, which PAI
                        enables automatically when perforatedbp is installed. gd: dendrites are
                        trained by gradient descent together with the network (open-source mode).
    PAI_CONTROL=1       control: perforate_model(doing_pai=False), so PAI only restarts the
                        learning rate at each would-be switch and never adds dendrites.
    PAI_MAX_DENDRITES=N cap on dendrite sets (or, for a control, on LR restarts). PAI may stop sooner.
    RUN_SUFFIX=_x       appended to the W&B run name so runs can be told apart.

PAI's configuration is confirmed non-interactively (PAI 4.0 otherwise asks in a terminal before
perforate_model), so this also works under nohup or Slurm. Outputs are written to the current
directory; run each experiment from its own directory. Arguments are passed on to train.py.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # train.py imports its sibling modules

from perforatedai import globals_perforatedai as GPA  # noqa: E402
from perforatedai import utils_perforatedai as UPA  # noqa: E402
import wandb  # noqa: E402

GPA.pc.set_configuration_confirmed(True)
mode = os.environ.get("PAI_MODE", "pb")
if mode == "gd":
    GPA.pc.set_perforated_backpropagation(False)
    GPA.pc.set_no_extra_n_modes(True)
elif mode != "pb":
    sys.exit("PAI_MODE must be pb or gd")
if os.environ.get("PAI_MAX_DENDRITES"):
    GPA.pc.set_max_dendrites(int(os.environ["PAI_MAX_DENDRITES"]))
control = os.environ.get("PAI_CONTROL") == "1"
if control:
    _perforate = UPA.perforate_model
    UPA.perforate_model = lambda model, *a, **k: _perforate(model, *a, **{**k, "doing_pai": False})
if os.environ.get("RUN_SUFFIX"):
    _init = wandb.init
    wandb.init = lambda *a, **k: _init(*a, **{**k, "name": k.get("name", "run") + os.environ["RUN_SUFFIX"]})

print(f"[run_example] mode={mode} control={control} "
      f"perforated_backpropagation={GPA.pc.get_perforated_backpropagation()} "
      f"max_dendrites={GPA.pc.get_max_dendrites()}", flush=True)
sys.argv = [os.path.join(HERE, "train.py")] + sys.argv[1:]
runpy.run_path(os.path.join(HERE, "train.py"), run_name="__main__")
