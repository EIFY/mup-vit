"""Run PerforatedAI's MNIST example (mnist_perforatedai.py, unchanged) with PAI settings applied first.

    PAI_MODE=pb|gd      pb (default): Perforated Backpropagation / cascade correlation, which PAI
                        enables automatically when perforatedbp is installed. gd: dendrites are
                        trained by gradient descent together with the network (open-source mode).
    PAI_CONTROL=1       control: perforate_model(doing_pai=False), so PAI only restarts the
                        learning rate at each would-be switch and never adds dendrites.
    PAI_MAX_DENDRITES=N cap on dendrite sets (or, for a control, on LR restarts). PAI may stop sooner.
    PAI_DENDRITE_LOSS=covariance
                        pb only: train candidates with perforatedbp's CovarianceDendriteLoss instead
                        of 4.0.0's default ImprovedCovarianceDendriteLoss, which scores 0.

Like running the example from its own folder, the run directory gets the example's saved PAI
config (PAI/PAI_config.json, copied unless one exists). PAI's configuration is confirmed
non-interactively, so this also works under nohup, tsp or Slurm. Outputs (PAI/ graphs and CSVs,
data/ with MNIST) go to the current directory; run each experiment from its own directory.
Arguments are passed on to mnist_perforatedai.py.
"""
import os
import runpy
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "mnist_perforatedai.py")

if not os.path.exists(os.path.join("PAI", "PAI_config.json")):
    os.makedirs("PAI", exist_ok=True)
    shutil.copy(os.path.join(HERE, "PAI", "PAI_config.json"), os.path.join("PAI", "PAI_config.json"))

from perforatedai import globals_perforatedai as GPA  # noqa: E402
from perforatedai import utils_perforatedai as UPA  # noqa: E402

GPA.pc.set_configuration_confirmed(True)
GPA.pc.set_unwrapped_modules_confirmed(True)
mode = os.environ.get("PAI_MODE", "pb")
if mode == "gd":
    GPA.pc.set_perforated_backpropagation(False)
    GPA.pc.set_no_extra_n_modes(True)
elif mode != "pb":
    sys.exit("PAI_MODE must be pb or gd")
dendrite_loss = os.environ.get("PAI_DENDRITE_LOSS", "default")
if dendrite_loss == "covariance":
    if mode != "pb":
        sys.exit("PAI_DENDRITE_LOSS applies only to PAI_MODE=pb")
    import perforatedbp.tracker_pbp as TPB
    import perforatedbp.dendrite_losses.dendrite_loss_pbp as DLP
    TPB.setup_new_global_dendrite_mode(
        DLP.CovarianceDendriteLoss, TPB.best_cascor_score_improved_this_epoch,
        dendrite_learn_mode=True)
elif dendrite_loss != "default":
    sys.exit("PAI_DENDRITE_LOSS must be default or covariance")
if os.environ.get("PAI_MAX_DENDRITES"):
    GPA.pc.set_max_dendrites(int(os.environ["PAI_MAX_DENDRITES"]))
control = os.environ.get("PAI_CONTROL") == "1"
if control:
    _perforate = UPA.perforate_model
    UPA.perforate_model = lambda model, *a, **k: _perforate(model, *a, **{**k, "doing_pai": False})

print(f"[run_example] mode={mode} control={control} dendrite_loss={dendrite_loss} "
      f"perforated_backpropagation={GPA.pc.get_perforated_backpropagation()} "
      f"max_dendrites={GPA.pc.get_max_dendrites()}", flush=True)
sys.argv = [SCRIPT] + sys.argv[1:]
runpy.run_path(SCRIPT, run_name="__main__")
