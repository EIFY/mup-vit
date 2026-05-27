import collections, decimal, math, os, pathlib, statistics, sys, torch

IMAGENET_TRAIN_SIZE = 1281167
BS = 4096

def read_best(p):
    best_path = os.path.join(p, 'checkpoints/model_best.pth.tar')
    ckpt = torch.load(best_path, weights_only=True)
    return ckpt['best_acc1']


# def make_up(curr, repeat, acc, done=True, path='logs/'):
#     """For testing sandbox only!!!"""
#     name = run_name('scion', curr, repeat=repeat)
#     ckpt_path = os.path.join(path, name, 'checkpoints')
#     pathlib.Path(ckpt_path).mkdir(parents=True, exist_ok=True)
#     best_ckpt = os.path.join(ckpt_path, 'model_best.pth.tar')
#     ckpt = {'best_acc1': acc}
#     torch.save(ckpt, best_ckpt)
#     if done:
#         step = round(IMAGENET_TRAIN_SIZE * curr['ep'] / BS)
#         last_ckpt = os.path.join(ckpt_path, f'model_step_{step}.pth.tar')
#         torch.save(ckpt, last_ckpt)


N_REPEATS = 3
TOLERANCE = 0.002

branch = 'unbiased'

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

def train_command(curr, fixed, opt='scion', prefix=prefix, path = 'logs/', repeat=0):
    name = run_name('scion', curr, repeat=repeat)
    step = round(IMAGENET_TRAIN_SIZE * curr['ep'] / BS)
    path_name = os.path.join(path, name)
    last_ckpt = os.path.join(path_name, f'checkpoints/model_step_{step}.pth.tar')
    command = prefix + flags(curr | fixed | dict(name=name))
    if not os.path.exists(path_name):
        return command
    if os.path.exists(last_ckpt):
        return '# ' + command  # Done
    else:
        curr_ckpt = os.path.join(path_name, 'checkpoints/checkpoint.pth.tar')
        return prefix + flags(curr | fixed | dict(name=name) | {'resume': curr_ckpt})

def read_repeats(curr, fixed, opt='scion', prefix=prefix, path = 'logs/'):
    step = round(IMAGENET_TRAIN_SIZE * curr['ep'] / BS)
    acc, commands = [], []
    for repeat in range(N_REPEATS):
        name = run_name('scion', curr, repeat=repeat)
        path_name = os.path.join(path, name)
        last_ckpt = os.path.join(path_name, f'checkpoints/model_step_{step}.pth.tar')
        if not os.path.exists(path_name):
            break
        if os.path.exists(last_ckpt):
            acc.append(read_best(path_name))
        else:
            curr_ckpt = os.path.join(path_name, 'checkpoints/checkpoint.pth.tar')
            commands.append(prefix + flags(curr | fixed | dict(name=name) | {'resume': curr_ckpt}))
    return acc, commands


def test_training_budgets(default, eps, f):
    done = True
    curr = dict(default)
    for curr['ep'] in eps:
        for repeat in range(N_REPEATS):
            command = train_command(curr, fixed, opt='scion', prefix=prefix, repeat=repeat)
            done = done and command[0] == '#'
            print(command, file=f)
    return done


class AutoTuner:

    def __init__(self, initial_values, curr, f):
        self.initial_values = initial_values
        self.curr = dict(curr)
        self.f = f

    def next_value(self):
        return None, False

    def prev_value(self):
        return None, False

    def run(self):

        done = True

        print(file=self.f)
        print(f"# {self.initial_values=}", file=self.f)
        print(file=self.f)

        for val in self.initial_values:
            self.curr |= val
            command = train_command(self.curr, fixed, opt='scion', prefix=prefix)
            done = done and command[0] == '#'
            print(command, file=self.f)

        self.values = collections.deque()
        accs = collections.deque()

        if done:
            for val in self.initial_values:
                self.curr |= val
                acc, commands = read_repeats(self.curr, fixed, opt='scion', prefix=prefix)
                for command in commands:
                    done = False
                    print(command, file=self.f)
                if acc:
                    self.values.append(val)
                    accs.append(acc)
            done = done and len(self.values) == len(self.initial_values)

        if done:
            while True:
                nxt, ok = self.next_value()
                if not ok:
                    break
                self.curr |= nxt
                acc, commands = read_repeats(self.curr, fixed, opt='scion', prefix=prefix)
                for command in commands:
                    done = False
                    print(command, file=self.f)
                if acc:
                    self.values.append(nxt)
                    accs.append(acc)
                else:
                    break
            while True:
                prev, ok = self.prev_value()
                if not ok:
                    break
                self.curr |= prev
                acc, commands = read_repeats(self.curr, fixed, opt='scion', prefix=prefix)
                for command in commands:
                    done = False
                    print(command, file=self.f)
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
                print(file=self.f)
                print(f"# abs({pen} - {ult}) < {TOLERANCE}, run N={N_REPEATS}:", file=self.f)
                print(file=self.f)
                done = False
                for val, acc in zip((self.values[-2], self.values[-1]), last2):
                    self.curr |= val
                    for repeat in range(len(acc), N_REPEATS):
                        command = train_command(self.curr, fixed, opt='scion', prefix=prefix, repeat=repeat)
                        print(command, file=self.f)

        if done and (len(self.values) < 2 or pen < ult):
            nxt, ok = self.next_value()
            if ok:
                print(file=self.f)
                if len(self.values) >= 2:
                    print(f"# {pen} < {ult}:", file=self.f)
                    print(file=self.f)
                done = False
                self.curr |= nxt
                command = train_command(self.curr, fixed, opt='scion', prefix=prefix)
                print(command, file=self.f)

        if done and len(self.values) >= 2:
            first2 = (accs[0], accs[1])
            first, second = map(statistics.fmean, first2)
            if abs(first - second) < TOLERANCE and min(len(acc) for acc in first2) < N_REPEATS:
                print(file=self.f)
                print(f"# abs({first} - {second}) < {TOLERANCE}, run N={N_REPEATS}:", file=self.f)
                print(file=self.f)
                done = False
                for val, acc in zip((self.values[0], self.values[1]), first2):
                    self.curr |= val
                    for repeat in range(len(acc), N_REPEATS):
                        command = train_command(self.curr, fixed, opt='scion', prefix=prefix, repeat=repeat)
                        print(command, file=self.f)

        if done and (len(self.values) < 2 or first > second):
            prev, ok = self.prev_value()
            if ok:
                print(file=self.f)
                if len(self.values) >= 2:
                    print(f"# {first} > {second}:", file=self.f)
                    print(file=self.f)
                done = False
                self.curr |= prev
                command = train_command(self.curr, fixed, opt='scion', prefix=prefix)
                print(command, file=self.f)

        if done:
            avg, index = max((statistics.fmean(acc), i) for i, acc in enumerate(accs))
            best_val = self.values[index]
            # avg, best_val = max((statistics.fmean(acc), val) for acc, val in zip(accs, self.values))
            self.curr |= best_val
            print(file=self.f)
            print(f"# {best_val=}, {avg=}", file=self.f)
            print(f"# {self.curr=}", file=self.f)

        return done and self.curr


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


def next_mo(mo):
    if str(mo)[-1] in '15':
        mo *= 2
    else:
        mo *= 5
        mo /= 2
    return mo


def prev_mo(mo):
    if str(mo)[-1] in '12':
        mo /= 2
    else:
        mo /= 5
        mo *= 2
    return mo


class MomentumAutoTuner(AutoTuner):

    def __init__(self, curr, f):
        self.nesterov = curr.get('nesterov') == ''
        self.lr_eff = curr['lr'] * lr_factor(curr['momentum'], nesterov=self.nesterov)
        init_val = {
            'momentum': decimal.Decimal(str(curr['momentum'])),  # Floating-point precision workaround
            'lr': curr['lr'],
        }
        super().__init__(initial_values=[init_val], curr=curr, f=f)

    def next_value(self):
        mo = self.values[-1]['momentum']
        if mo == 1.0:
            return None, False
        mo = next_mo(mo)
        lr = self.lr_eff / lr_factor(momentum=float(mo), nesterov=self.nesterov)
        return dict(momentum=mo, lr=lr), True

    def prev_value(self):
        mo = self.values[0]['momentum']
        mo = prev_mo(mo)
        lr = self.lr_eff / lr_factor(momentum=float(mo), nesterov=self.nesterov)
        return dict(momentum=mo, lr=lr), True


# None is tombstone value, '' (empty string) is for store_true flags
default = dict(corrected='', ep=90, momentum=0.1, lr=0.01, sign_lr=0.2, c_sq=1.1875, wd=None, sign_wd=0.004, am_gm_reg=None, nesterov=None, timescale_inv=None)

fixed = dict(workers="$N_WORKERS", multiprocessing_distributed='', batch_size="$BS", mlp_head='', torchvision_inception_crop='', grad_clip_norm=100000000., report_to='wandb', print_freq=25)

# old_open = open
# files_opened = []

# def open(file, mode):
#     files_opened.append(file)
#     return old_open(file, mode)

for default['corrected'] in ('', None):

    file_prefix = 'corrected_' if default['corrected'] == '' else ''

    with open(file_prefix + "lr.sh", "w") as f:

        print(preface, file=f)
        print("# LR tuning:", file=f)

        key = 'lr'
        initial_lr = default[key]
        tuner = LRAutoTuner(key, initial_lr, 2 ** 0.25, default, f)
        default = tuner.run()

    if not default:
        sys.exit()

    with open(file_prefix + "wd.sh", "w") as f:

        print(preface, file=f)
        print("# Corrected WD tuning:", file=f)

        if default['corrected'] == '':
            key = 'c_sq'
        else:
            key = 'wd'
        initial_wd = default[key]
        tuner = LRAutoTuner(key, initial_wd, 2 ** 0.25, default, f)
        default = tuner.run()

    if not default:
        sys.exit()

    with open(file_prefix + "nesterov.sh", "w") as f:

        print(preface, file=f)
        print("# Nesterov or not:", file=f)

        initial_vals = [{k: default.get(k) for k in ['lr', 'nesterov']}]
        nesterov = default.get('nesterov') == ''
        lr_eff = default['lr'] * lr_factor(default['momentum'], nesterov=nesterov)
        new_val = dict(lr=lr_eff / lr_factor(default['momentum'], nesterov=not nesterov), nesterov=None if nesterov else '')
        initial_vals.append(new_val)

        tuner = AutoTuner(initial_values=initial_vals, curr=default, f=f)
        default = tuner.run()

    if not default:
        sys.exit()

    with open(file_prefix + "momentum.sh", "w") as f:

        print(preface, file=f)
        print("# Momentum tuning:", file=f)

        tuner = MomentumAutoTuner(default, f)
        default = tuner.run()

    if not default:        
        sys.exit()

    default['momentum'] = float(default['momentum'])  # Avoid pitfall of inter-op between Decimal & float

    with open(file_prefix + "sign_lr.sh", "w") as f:

        print(preface, file=f)
        print("# Sign LR tuning:", file=f)

        key = 'sign_lr'
        initial_lr = default[key]
        tuner = LRAutoTuner(key, initial_lr, 2 ** 0.25, default, f)
        default = tuner.run()

    if not default:
        sys.exit()

    with open(file_prefix + "sign_wd.sh", "w") as f:

        print(preface, file=f)
        print("# Sign WD tuning:", file=f)

        key = 'sign_wd'
        initial_wd = default[key]
        tuner = LRAutoTuner(key, initial_wd, 2 ** 0.25, default, f)
        default = tuner.run()

    if not default:
        sys.exit()

    if default.get('corrected') == '':

        with open(file_prefix + "lr_eff_transfer.sh", "w") as f:

            print(preface, file=f)
            print("# Effective LR transfer:", file=f)

            curr = dict(default)
            done = True
            mo = curr['momentum']
            lr_eff = curr['lr'] * lr_factor(mo, nesterov=curr.get('nesterov') == '')
            mo = decimal.Decimal(str(mo))  
            mos = collections.deque([mo])
            while len(mos) < 3 and mos[-1] < 1:
                mos.append(next_mo(mos[-1]))
            while len(mos) < 6:
                mos.appendleft(prev_mo(mos[0]))
            factors = [2 ** -0.5, 2**-0.25, 1., 2**0.25, 2 ** 0.5]

            for curr['nesterov'] in ('', None):
                for curr['momentum'] in mos:
                    for factor in factors:
                        base_lr = lr_eff / lr_factor(float(curr['momentum']), nesterov=curr.get('nesterov') == '')
                        curr['lr'] = factor * base_lr
                        command = train_command(curr, fixed, opt='scion', prefix=prefix)
                        done = done and command[0] == '#'
                        print(command, file=f)
            if not done:
                sys.exit()

            accuracies = {}
            for curr['nesterov'] in ('', None):
                for curr['momentum'] in mos:
                    for factor in factors:
                        curr['lr'] = lr_eff * factor / lr_factor(float(curr['momentum']), nesterov=curr.get('nesterov') == '')
                        accuracies[curr['nesterov'], curr['momentum'], curr['lr']], _ = read_repeats(curr, fixed, opt='scion', prefix=prefix, path = 'logs/')
            key = max(accuracies, key=lambda k: statistics.fmean(accuracies[k]))
            avg = statistics.fmean(accuracies[key])
            default['nesterov'], default['momentum'], default['lr'] = key
            default['momentum'] = float(default['momentum'])  # Avoid pitfall of inter-op between Decimal & float
            print(file=f)
            print(f"# {(default['nesterov'], default['momentum'], default['lr'])=}, {avg=}", file=f)
            print(f"# {default=}", file=f)

    with open(file_prefix + "training_budgets.sh", "w") as f:

        print(preface, file=f)
        print("# Corrected with various training budgets:", file=f)

        done = test_training_budgets(default=default, eps=[30, 60, 90, 150, 300], f=f)

    if not done:
        sys.exit()

    if default.get('corrected') == '':

        corrected_default = dict(default)

        with open(file_prefix + "log_time_momentum.sh", "w") as f:

            print(preface, file=f)
            print("# Log-time momentum tuning:", file=f)

            lr_eff = default['lr'] * lr_factor(default['momentum'], nesterov=default.get('nesterov') == '')
            ratio = 2 / default['momentum']  # Start with end momentum half of the optimal (constant) momentum 
            step = round(IMAGENET_TRAIN_SIZE * default['ep'] / BS)
            log_time_val = dict(momentum=1.0, lr=lr_eff)
            timescale_inv = (ratio - 1) / step

            tuner = LRAutoTuner('timescale_inv', timescale_inv, 2 ** 0.25, default | log_time_val, f)
            log_time_default = tuner.run()

        if not log_time_default:
            sys.exit()

        with open(file_prefix + "baseline_comparison.sh", "w") as f:
            diff = ['momentum', 'lr', 'timescale_inv']
            baseline = {k: corrected_default[k] for k in diff}
            log_time = {k: log_time_default[k] for k in diff}

            print(preface, file=f)
            print("# Log-time vs. baseline:", file=f)

            tuner = AutoTuner(initial_values=[baseline, log_time], curr=corrected_default, f=f)
            better = tuner.run()

        if not better:
            sys.exit()

        with open(file_prefix + "log_time_training_budgets.sh", "w") as f:

            print(preface, file=f)
            print("# Log-time momentum with various training budgets:", file=f)
            if better['timescale_inv'] is None:
                print("# Skipped. Log-time momentum is not better.", file=f)
            else:
                done = test_training_budgets(default=better, eps=[30, 60, 90, 150, 300], f=f)

        if not done:
            sys.exit()

        # Prepare uncorrected default
        default = dict(corrected_default)
        c_sq = default['c_sq']
        default['c_sq'] = None
        mo, nesterov = default['momentum'], default.get('nesterov') == ''
        initial_wd = lr_factor(mo, nesterov) ** 2 * default['lr'] / c_sq / 2
        # Initial WD guess: half of the initial WD of the best corrected counterpart,
        # so the average throughout the training is about the same
        default['wd'], default['c_sq'] = initial_wd / 2, None

with open("misc.sh", "w") as f:
    branch = 'am-gm'

    preface = f"""#!/bin/bash

MUPVIT_MAIN=~/Downloads/mup-vit/main.py
PYTHON=torchrun
N_WORKERS=100
N_THREADS=124
BS={BS}

git -C /home/ubuntu/Downloads/mup-vit checkout {branch}
"""

    print(preface, file=f)
    print("# AM-GM regularization exp.:", file=f)

    tuner = AutoTuner(initial_values=[default, corrected_default, log_time_default], curr={}, f=f)
    best = tuner.run()
    best['sign_wd'] = 0.0
    if best['corrected'] == '':
        best['c_sq'] = 'inf'
    else:
        best['wd'] = 0.0

    tuner = LRAutoTuner('am_gm_reg', 1.0, 2 ** 0.25, best, f)
    best = tuner.run()
    if not best:
        sys.exit()

pathlib.Path('done').touch()
print('Done!')

# print(files_opened)
# ['corrected_lr.sh', 'corrected_wd.sh', 'corrected_nesterov.sh', 'corrected_momentum.sh', 'corrected_sign_lr.sh', 'corrected_sign_wd.sh', 'corrected_lr_eff_transfer.sh', 'corrected_training_budgets.sh', 'corrected_log_time_momentum.sh', 'corrected_baseline_comparison.sh', 'corrected_log_time_training_budgets.sh', 'lr.sh', 'wd.sh', 'nesterov.sh', 'momentum.sh', 'sign_lr.sh', 'sign_wd.sh', 'training_budgets.sh', 'misc.sh', 'done']
