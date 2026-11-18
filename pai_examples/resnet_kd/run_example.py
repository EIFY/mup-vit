"""Run PerforatedAI's ResNet KD example (train_perforated_resnet_KD.py, unchanged) with extra settings.

The example selects the dendrite mode itself with --dendrite-mode (2 = Perforated Backpropagation,
1 = gradient descent, 0 = no dendrites at all). This wrapper adds what it has no option for:

    PAI_CONTROL=1       control: perforate_model(doing_pai=False), so PAI only restarts the
                        learning rate at each would-be switch and never adds dendrites (unlike
                        --dendrite-mode 0, which stops at the first switch). Use with --dendrite-mode 1.
    PAI_MAX_DENDRITES=N overrides the example's max_dendrites (5); for a control, the cap on restarts.
    RUN_SUFFIX=_x       appended to the W&B run name (with --use-wandb), so runs can be told apart.

The script reads its PAI configuration from PAI.json in the current directory. Like running it
from the example's folder, the run directory gets a copy of the example's PAI.json (unless one
exists), with only the two confirmation flags set so PAI does not ask interactively. Outputs go to
the current directory; run each experiment from its own directory. Arguments are passed on.
"""
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "train_perforated_resnet_KD.py")
sys.path.insert(0, HERE)  # the script imports its sibling modules (presets, utils, ...)

if not os.path.exists("PAI.json"):
    with open(os.path.join(HERE, "PAI.json")) as f:
        config = json.load(f)
    config["configuration_confirmed"] = True
    config["unwrapped_modules_confirmed"] = True
    with open("PAI.json", "w") as f:
        json.dump(config, f, indent=2)

from perforatedai import globals_perforatedai as GPA  # noqa: E402
from perforatedai import utils_perforatedai as UPA  # noqa: E402
import wandb  # noqa: E402

GPA.pc.set_configuration_confirmed(True)
GPA.pc.set_unwrapped_modules_confirmed(True)
if os.environ.get("PAI_MAX_DENDRITES"):
    cap = int(os.environ["PAI_MAX_DENDRITES"])
    _set_max = GPA.pc.set_max_dendrites
    GPA.pc.set_max_dendrites = lambda n: _set_max(cap)  # the example sets 5 after this point
    _set_max(cap)
control = os.environ.get("PAI_CONTROL") == "1"
if control:
    _perforate = UPA.perforate_model
    UPA.perforate_model = lambda model, *a, **k: _perforate(model, *a, **{**k, "doing_pai": False})
if os.environ.get("RUN_SUFFIX"):
    _init = wandb.init
    wandb.init = lambda *a, **k: _init(*a, **{**k, "name": k.get("name", "run") + os.environ["RUN_SUFFIX"]})

print(f"[run_example] control={control} max_dendrites_override={os.environ.get('PAI_MAX_DENDRITES')} "
      f"args={' '.join(sys.argv[1:])}", flush=True)
sys.argv = [SCRIPT] + sys.argv[1:]
runpy.run_path(SCRIPT, run_name="__main__")
