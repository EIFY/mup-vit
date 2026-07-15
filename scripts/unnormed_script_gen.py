import collections, math, os, pathlib, statistics, sys, torch

IMAGENET_TRAIN_SIZE = 1281167
BS = 4096

def read_best(p):
    best_path = os.path.join(p, 'checkpoints/model_best.pth.tar')
    ckpt = torch.load(best_path, weights_only=True)
    return ckpt['best_acc1']

N_REPEATS = 3
TOLERANCE = 0.002

branch = 'unnormed'

preface = f"""#!/bin/bash

MUPVIT_MAIN=~/Downloads/mup-vit/main.py
PYTHON=torchrun
N_WORKERS=100
N_THREADS=124
BS={BS}

git -C /home/ubuntu/Downloads/mup-vit checkout {branch}
"""

BEST_CKPT = 'checkpoints/model_best.pth.tar'

prefix = "NUMEXPR_MAX_THREADS=$N_THREADS $PYTHON $MUPVIT_MAIN /data/ImageNet/ "

def run_name(opt, d, repeat=0):
    l = [opt]
    for k, v in d.items():
        if v is not None:
            l.append(k)
            if v != '':
                if type(v) is float:
                    v = f"{v:.3g}"
                else:
                    v = str(v)
                l.append(v)
    l.append(str(repeat))
    return '-'.join(l)

def flags(d):
    l = []
    for k, v in d.items():
        if v is not None:
            l.append('--' + k.replace('_', '-'))
            if v != '':
                l.append(str(v))
    return ' '.join(l)


fixed = dict(workers="$N_WORKERS", multiprocessing_distributed='', batch_size="$BS", mlp_head='', scaled='', torchvision_inception_crop='', grad_clip_norm=100000000., report_to='wandb', print_freq=25)


def test_params(curr, fixed=fixed, opt='scion-t212', prefix=prefix, path='logs/', repeat=0):
    name = run_name(opt, curr, repeat=repeat)
    step = round(IMAGENET_TRAIN_SIZE * curr['ep'] / BS)
    path_name = os.path.join(path, name)
    last_ckpt = os.path.join(path_name, f'checkpoints/model_step_{step}.pth.tar')
    command = prefix + flags(curr | fixed | dict(name=name))
    accuracy = None
    if not os.path.exists(path_name):
        pass
    elif os.path.exists(last_ckpt):
        command = '# ' + command  # Done
        accuracy = read_best(path_name)
    else:
        curr_ckpt = os.path.join(path_name, 'checkpoints/checkpoint.pth.tar')
        command = prefix + flags(curr | fixed | dict(name=name) | {'resume': curr_ckpt})
    return command, accuracy


def read_repeats(curr, fixed=fixed, opt='scion-t212', prefix=prefix, path='logs/', repeats=N_REPEATS):
    commands, acc = [], []
    for repeat in range(repeats):
        command, accuracy = test_params(curr=curr, repeat=repeat)
        commands.append(command)
        acc.append(accuracy)
    return commands, [a for a in acc if a is not None]


def test_training_budgets(default, eps, f):
    done = True
    commands = []
    curr = dict(default)
    for curr['ep'] in eps:
        cmds, acc = read_repeats(curr=curr, repeats=N_REPEATS)
        commands.extend(cmds)
        done = done and len(acc) == N_REPEATS
    for command in commands:
        print(command, file=f)
    return done


# Due to the naming convention AutoTuner can't distinguish beyond 3 significant digits.
# Should be sufficient given the grid granularity.
def almost_eq(x, y):
    return f"{x:.3g}" == f"{y:.3g}"


class AutoTuner:

    def __init__(self, initial_values, curr, f):
        self.initial_values = initial_values
        self.curr = dict(curr)
        self.f = f

    def next_value(self):
        return None, False

    def prev_value(self):
        return None, False

    def test_value(self, val):
        to_test = self.curr | val
        commands, acc = read_repeats(curr=to_test, repeats=N_REPEATS)
        return val, commands[:1], acc  # Read all the accuracies but only return the command we need for sure, i.e. the first

    def run(self):
        best_val, commands, final_acc = self.optimize()
        for command in commands:
            print(command, file=self.f)
        return self.curr | best_val, final_acc

    def optimize(self):

        done = True
        commands = []
        self.values = collections.deque()
        accs = collections.deque()
        final_acc = []
        best_val = {}

        commands.append('')
        commands.append(f"# {self.initial_values=}")
        commands.append('')

        for val in self.initial_values:
            val, cmds, acc = self.test_value(val)
            done = done and bool(acc)
            self.values.append(val)
            commands.extend(cmds)
            accs.append(acc)

        if done:
            while True:
                nxt, nxt_ok = self.next_value()
                if not nxt_ok:
                    break
                nxt, nxt_commands, acc = self.test_value(nxt)
                if acc:
                    self.values.append(nxt)
                    accs.append(acc)
                else:
                    break
            while True:
                prev, prev_ok = self.prev_value()
                if not prev_ok:
                    break
                prev, prev_commands, acc = self.test_value(prev)
                if acc:
                    self.values.appendleft(prev)
                    accs.appendleft(acc)
                else:
                    break

        print(self.values, accs)

        if done and len(self.values) >= 2:
            last2 = (accs[-2], accs[-1])
            pen, ult = map(statistics.fmean, last2)
            if abs(pen - ult) < TOLERANCE and min(len(acc) for acc in last2) < N_REPEATS:
                commands.append('')
                commands.append(f"# abs({pen} - {ult}) < {TOLERANCE}, run N={N_REPEATS}:")
                commands.append('')
                done = False
                for val, acc in zip((self.values[-2], self.values[-1]), last2):
                    to_test = self.curr | val
                    for repeat in range(len(acc), N_REPEATS):
                        command, _ = test_params(curr=to_test, repeat=repeat)
                        commands.append(command)

        if done and (len(self.values) < 2 or pen < ult) and nxt_ok:
            commands.append('')
            if len(self.values) >= 2:
                commands.append(f"# {pen} < {ult}:")
                commands.append('')
            done = False
            commands.extend(nxt_commands)

        if done and len(self.values) >= 2:
            first2 = (accs[0], accs[1])
            first, second = map(statistics.fmean, first2)
            if abs(first - second) < TOLERANCE and min(len(acc) for acc in first2) < N_REPEATS:
                commands.append('')
                commands.append(f"# abs({first} - {second}) < {TOLERANCE}, run N={N_REPEATS}:")
                commands.append('')
                done = False
                for val, acc in zip((self.values[0], self.values[1]), first2):
                    to_test = self.curr | val
                    for repeat in range(len(acc), N_REPEATS):
                        command, _ = test_params(curr=to_test, repeat=repeat)
                        commands.append(command)

        if done and (len(self.values) < 2 or first > second) and prev_ok:
            commands.append('')
            if len(self.values) >= 2:
                commands.append(f"# {first} > {second}:")
                commands.append('')
            done = False
            commands.extend(prev_commands)

        if done:
            avg, index = max((statistics.fmean(acc), i) for i, acc in enumerate(accs))
            best_val, final_acc = self.values[index], accs[index]
            commands.append('')
            commands.append(f"# {best_val=}, {avg=}")
            commands.append(f"# {self.curr=}")

        return best_val, commands, final_acc


class LRAutoTuner(AutoTuner):

    def __init__(self, key, initial_lr, factor, curr, f):
        self.key = key
        self.factor = factor
        super().__init__(initial_values=[{self.key: initial_lr}], curr=curr, f=f)

    def next_value(self):
        nxt_lr = dict(self.values[-1])
        nxt_lr[self.key] *= self.factor
        return nxt_lr, True

    def prev_value(self):
        prev_lr = dict(self.values[0])
        prev_lr[self.key] /= self.factor
        return prev_lr, True


def lr_factor(momentum, nesterov):
    factor = math.sqrt((2 - momentum) / momentum)
    if nesterov:
        factor *= (1 + 4*momentum - 6*momentum**2 + 2*momentum**3) ** -0.5
    return factor


class JointCsqLRTuner(AutoTuner):
    """Jointly tune c_sq and lr based on rel. LR"""
    def __init__(self, factor, curr, f):
        assert curr['corrected'] == '', 'Must be a corrected experiment'
        self.factor = factor
        super().__init__(initial_values=[{'c_sq': curr['c_sq'], 'lr': curr['lr']}], curr=curr, f=f)

    def next_value(self):
        curr = self.values[-1]
        next_val = {'c_sq': curr['c_sq'] * self.factor, 'lr': curr['lr'] * math.sqrt(self.factor)}
        return next_val, True

    def prev_value(self):
        curr = self.values[0]
        prev_val = {'c_sq': curr['c_sq'] / self.factor, 'lr': curr['lr'] / math.sqrt(self.factor)}
        return prev_val, True


# None is tombstone value, '' (empty string) is for store_true flags
# Best hyperparameters w/ cosine LR schedule, taken from corrected_c_sq_lr.sh
default = {'corrected': '', 'ep': 90, 'momentum': 0.1, 'lr': 0.011584472366059664, 'sign_lr': 0.1, 'c_sq': 0.8396893026590251, 'wd': None, 'sign_wd': 0.00282842712474619, 'nesterov': '', 'cos_power': None, 'power': None}

file_prefix = 'unnormed_corrected_'

with open(file_prefix + "lr.sh", "w") as f:

    print(preface, file=f)
    print("# LR tuning:", file=f)

    key = 'lr'
    initial_lr = default[key]
    tuner = LRAutoTuner(key, initial_lr, 2 ** 0.5, default, f)
    default, final_acc = tuner.run()

if not final_acc:
    sys.exit()

with open(file_prefix + "wd.sh", "w") as f:

    print(preface, file=f)
    print("# Corrected WD tuning:", file=f)
    key = 'c_sq'
    initial_wd = default[key]
    tuner = LRAutoTuner(key, initial_wd, 2 ** 0.5, default, f)
    default, final_acc = tuner.run()

if not final_acc:
    sys.exit()

with open(file_prefix + "sign_lr.sh", "w") as f:

    print(preface, file=f)
    print("# Sign LR tuning:", file=f)

    key = 'sign_lr'
    initial_lr = default[key]
    tuner = LRAutoTuner(key, initial_lr, 2 ** 0.5, default, f)
    default, final_acc = tuner.run()

if not final_acc:
    sys.exit()

with open(file_prefix + "sign_wd.sh", "w") as f:

    print(preface, file=f)
    print("# Sign WD tuning:", file=f)

    key = 'sign_wd'
    initial_wd = default[key]
    tuner = LRAutoTuner(key, initial_wd, 2 ** 0.5, default, f)
    default, final_acc = tuner.run()

if not final_acc:
    sys.exit()

with open(file_prefix + "c_sq_lr.sh", "w") as f:

    print(preface, file=f)
    print("# Joint c_sq and lr tuning:", file=f)
    tuner = JointCsqLRTuner(factor=2 ** 0.5, curr=default, f=f)
    default, final_acc = tuner.run()

if not final_acc:
    sys.exit()

corrected_default = dict(default)

# Prepare uncorrected default
c_sq = default['c_sq']
mo, nesterov = default['momentum'], default.get('nesterov') == ''
initial_wd = lr_factor(mo, nesterov) ** 2 * default['lr'] / c_sq / 2
# Initial WD guess: half of the initial WD of the best corrected counterpart,
# so the average throughout the training is about the same
default['wd'], default['c_sq'] = initial_wd / 2, None

default['corrected'] = None
file_prefix = 'unnormed_'

with open(file_prefix + "wd.sh", "w") as f:

    print(preface, file=f)
    print("# Uncorrected WD tuning:", file=f)
    key = 'wd'
    initial_wd = default[key]
    tuner = LRAutoTuner(key, initial_wd, 2 ** 0.5, default, f)
    default, final_acc = tuner.run()

if not final_acc:
    sys.exit()

pathlib.Path('done').touch()
print('Done!')

# Files opened:
# ['unnormed_corrected_lr.sh', 'unnormed_corrected_wd.sh', 'unnormed_corrected_sign_lr.sh', 'unnormed_corrected_sign_wd.sh', 'unnormed_corrected_c_sq_lr.sh', 'unnormed_wd.sh', 'done']
