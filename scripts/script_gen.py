import collections, decimal, math, os, pathlib, statistics, sys, torch

IMAGENET_TRAIN_SIZE = 1281167
BS = 4096

def read_best(p):
    best_path = os.path.join(p, 'checkpoints/model_best.pth.tar')
    ckpt = torch.load(best_path, weights_only=True)
    return ckpt['best_acc1']


# def make_up(curr, repeat, acc, done=True, path='logs/'):
#     """For testing sandbox only!!!"""
#     name = run_name('scion-t212', curr, repeat=repeat)
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

branch = 'power'

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


fixed = dict(workers="$N_WORKERS", multiprocessing_distributed='', batch_size="$BS", mlp_head='', torchvision_inception_crop='', grad_clip_norm=100000000., report_to='wandb', print_freq=25)


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


def test_training_budgets_with_mosch(default, eps, f):
    # Interpolate / extrapolate momentum schedule instead of shrinking / stretching
    done = True
    commands = []
    curr = dict(default)
    original_steps = round(IMAGENET_TRAIN_SIZE * curr['ep'] / BS)
    original_ratio = curr.get('end_mo_ratio', 1.0)
    max_ratio = 1 / curr['momentum']
    for curr['ep'] in eps:
        new_steps = round(IMAGENET_TRAIN_SIZE * curr['ep'] / BS)
        new_ratio = min(original_ratio ** (new_steps / original_steps), max_ratio)
        curr['end_mo_ratio'] = None if almost_eq(new_ratio, 1.0) else new_ratio
        cmds, acc = read_repeats(curr=curr, repeats=N_REPEATS)
        commands.extend(cmds)
        done = done and len(acc) == N_REPEATS
    for command in commands:
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


class PowerAutoTuner(AutoTuner):

    def __init__(self, key, initial_val, diff, curr, f):
        self.key = key
        self.diff = diff
        super().__init__(initial_values=[{self.key: initial_val}], curr=curr, f=f)

    def next_value(self):
        nxt_val = dict(self.values[-1])
        nxt_val[self.key] += self.diff
        return nxt_val, True

    def prev_value(self):
        prev_val = dict(self.values[0])
        prev_val[self.key] -= self.diff
        if almost_eq(prev_val[self.key], 0.0):
            return None, False
        return prev_val, True


class CosPowerAutoTuner(AutoTuner):

    def __init__(self, key, initial_val, diff, curr, f):
        self.key = key
        self.diff = diff
        if almost_eq(initial_val, 1.0):
            initial_val = None
        super().__init__(initial_values=[{self.key: initial_val}], curr=curr, f=f)

    def next_value(self):
        nxt_val = self.values[-1][self.key]
        if nxt_val is None:
            nxt_val = 1.0
        nxt_val += self.diff
        if almost_eq(nxt_val, 1.0):
            nxt_val = None
        return {self.key: nxt_val}, True

    def prev_value(self):
        prev_val = self.values[0][self.key]
        if prev_val is None:
            prev_val = 1.0
        prev_val -= self.diff
        if almost_eq(prev_val, 0.0):
            return None, False
        if almost_eq(prev_val, 1.0):
            prev_val = None
        return {self.key: prev_val}, True


class LRPowerAutoTuner(AutoTuner):
    """Nested AutoTuner for LR & schedule power"""
    def __init__(self, factor, initial_value, comp, p_tuner, key, diff, curr, f):
        self.factor = factor
        self.comp = comp
        self.p_tuner = p_tuner
        self.key = key
        self.diff = diff
        super().__init__(initial_values=[initial_value], curr=curr, f=f)

    def test_value(self, val):
        commands = [f"# Inner {self.key} optimization:"]
        tuner = self.p_tuner(key=self.key, initial_val=val[self.key], diff=self.diff, curr=self.curr | val, f=self.f)
        best_p, cmds, acc = tuner.optimize()
        val |= best_p
        commands.extend(cmds)
        return val, commands, acc  # All commands tuner ordered are necessary.

    def next_value(self):
        nxt_lr = dict(self.values[-1])
        nxt_lr['lr'] *= self.factor
        if nxt_lr[self.key] is None:
            nxt_lr[self.key] = 1.0
        nxt_lr[self.key] += self.comp
        return nxt_lr, True

    def prev_value(self):
        prev_lr = dict(self.values[0])
        prev_lr['lr'] /= self.factor
        if prev_lr[self.key] is None:
            prev_lr[self.key] = 1.0
        prev_lr[self.key] -= self.comp
        prev_lr[self.key] = max(prev_lr[self.key], self.diff)
        return prev_lr, True


class LimitedAutoTuner(LRAutoTuner):

    def __init__(self, limit, key, initial_lr, factor, curr, f):
        self.limit = limit
        super().__init__(key=key, initial_lr=initial_lr, factor=factor, curr=curr, f=f)

    def next_value(self):
        if len(self.values) >= self.limit:
            return None, False
        return super().next_value()

    def prev_value(self):
        if len(self.values) >= self.limit:
            return None, False
        return super().prev_value()


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


class EndMoRatioAutoTuner(AutoTuner):

    def __init__(self, factor, curr, f):
        self.key = 'end_mo_ratio'
        self.factor = factor
        self.max_ratio = 1 / float(curr['momentum'])  # Doesn't make sense to have momentum > 1, right?
        super().__init__(initial_values=[{self.key: curr.get(self.key)}], curr=curr, f=f)

    def next_value(self):
        curr_ratio = self.values[-1].get(self.key) or 1.0  # end_mo_ratio = 0 never makes sense!
        if curr_ratio == self.max_ratio:
            return None, False
        next_ratio = min(curr_ratio * self.factor, self.max_ratio)
        if almost_eq(next_ratio, 1.0):
            next_ratio = None
        return {self.key: next_ratio}, True

    def prev_value(self):
        curr_ratio = self.values[0].get(self.key) or 1.0
        prev_ratio = curr_ratio / self.factor
        if almost_eq(prev_ratio, 1.0):
            prev_ratio = None
        return {self.key: prev_ratio}, True


def copy_end_mo(curr, new_mo, key):
    ratio = curr.get(key) or 1.0
    end_mo = ratio * float(curr['momentum'])
    ratio = end_mo / float(new_mo)
    return None if almost_eq(ratio, 1.0) else ratio


class MoschAutoTuner(MomentumAutoTuner):
    """Nested AutoTuner for momentum schedule"""
    def __init__(self, factor, curr, f):
        self.key = 'end_mo_ratio'
        self.factor = factor
        super().__init__(curr, f)
        # self.initial_values[0][self.key] = self.curr.get(self.key)

    def test_value(self, val):
        commands = [f"# Inner {self.key} optimization:"]
        ratio_tuner = EndMoRatioAutoTuner(self.factor, self.curr | val, self.f)
        best_ratio, cmds, acc = ratio_tuner.optimize()
        val |= best_ratio
        commands.extend(cmds)
        return val, commands, acc  # All commands ratio_tuner ordered are necessary.

    def next_value(self):
        nxt, ok = super().next_value()
        if ok:
            nxt[self.key] = copy_end_mo(self.values[-1], nxt['momentum'], self.key)
        return nxt, ok

    def prev_value(self):
        prev, ok = super().prev_value()
        if ok:
            prev[self.key] = copy_end_mo(self.values[0], prev['momentum'], self.key)
        return prev, ok


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


class JointWDLRTuner(AutoTuner):
    """Jointly tune wd and lr based on rel. LR"""
    def __init__(self, factor, curr, f):
        assert curr['corrected'] is None, 'Must be an uncorrected experiment'
        self.factor = factor
        super().__init__(initial_values=[{'wd': curr['wd'], 'lr': curr['lr']}], curr=curr, f=f)

    def next_value(self):
        curr = self.values[-1]
        next_val = {'wd': curr['wd'] * self.factor, 'lr': curr['lr'] / math.sqrt(self.factor)}
        return next_val, True

    def prev_value(self):
        curr = self.values[0]
        prev_val = {'wd': curr['wd'] / self.factor, 'lr': curr['lr'] * math.sqrt(self.factor)}
        return prev_val, True


# None is tombstone value, '' (empty string) is for store_true flags
default = dict(corrected='', ep=90, momentum=0.1, lr=0.01, sign_lr=0.2, c_sq=1.1875, wd=None, sign_wd=0.004, nesterov=None, cos_power=None, power=None)

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
        tuner = LRAutoTuner(key, initial_lr, 2 ** 0.5, default, f)
        default, final_acc = tuner.run()

    if not final_acc:
        sys.exit()

    with open(file_prefix + "wd.sh", "w") as f:

        print(preface, file=f)
        if default['corrected'] == '':
            print("# Corrected WD tuning:", file=f)
            key = 'c_sq'
        else:
            print("# Uncorrected WD tuning:", file=f)
            key = 'wd'
        initial_wd = default[key]
        tuner = LRAutoTuner(key, initial_wd, 2 ** 0.5, default, f)
        default, final_acc = tuner.run()

    if not final_acc:
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
        default, final_acc = tuner.run()

    if not final_acc:
        sys.exit()

    with open(file_prefix + "momentum.sh", "w") as f:

        print(preface, file=f)
        print("# Momentum tuning:", file=f)

        tuner = MomentumAutoTuner(default, f)
        default, final_acc = tuner.run()

    if not final_acc:
        sys.exit()

    default['momentum'] = float(default['momentum'])  # Avoid pitfall of inter-op between Decimal & float

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

    if default.get('corrected') == '':

        with open(file_prefix + "lr_eff_transfer.sh", "w") as f:

            print(preface, file=f)
            print("# Effective LR transfer:", file=f)

            curr = dict(default)
            mo = curr['momentum']
            lr_eff = curr['lr'] * lr_factor(mo, nesterov=curr.get('nesterov') == '')
            mo = decimal.Decimal(str(mo))  
            mos = collections.deque([mo])
            while len(mos) < 3 and mos[-1] < 1:
                mos.append(next_mo(mos[-1]))
            while len(mos) < 6:
                mos.appendleft(prev_mo(mos[0]))

            factors = [0.5, 2**-0.5, 1., 2**0.5, 2.0]
            accuracies = {}
            for curr['nesterov'] in ('', None):
                for curr['momentum'] in mos:
                    for factor in factors:
                        base_lr = lr_eff / lr_factor(float(curr['momentum']), nesterov=curr.get('nesterov') == '')
                        curr['lr'] = factor * base_lr
                        cmds, acc = read_repeats(curr=curr, repeats=N_REPEATS)
                        accuracies[curr['nesterov'], curr['momentum'], curr['lr']] = acc
                        print(cmds[0], file=f)  # We only need one datapoint

            if not all(accuracies.values()):
                sys.exit()

        with open(file_prefix + "mo_baseline_comparison.sh", "w") as f:

            print(preface, file=f)
            print("# Double-check after the momentum sweep:", file=f)

            key = max(accuracies, key=lambda k: statistics.fmean(accuracies[k]))
            final_acc = accuracies[key]
            avg = statistics.fmean(final_acc)
            nesterov, momentum, lr = key
            momentum = float(momentum)  # Avoid pitfall of inter-op between Decimal & float
            if default['nesterov'] == nesterov and default['momentum'] == momentum:
                print(file=f)
                print(f"# {(default['nesterov'], default['momentum'], default['lr'])=}, {avg=}", file=f)
                print(f"# {default=}", file=f)
            else:
                alt = {'nesterov': nesterov, 'momentum': momentum, 'lr': lr}
                tuner = AutoTuner(initial_values=[{}, alt], curr=default, f=f)
                default, final_acc = tuner.run()

        if not final_acc:
            sys.exit()

        with open(file_prefix + "bias.sh", "w") as f:

            print(preface, file=f)
            print("# Double-check model performance with bias:", file=f)
            tuner = LimitedAutoTuner(limit=5, key='bias_c_sq', initial_lr=default['c_sq'] / 2, factor=2.0, curr=default, f=f)
            bias_default, final_acc = tuner.run()

        if not final_acc:
            sys.exit()

        with open(file_prefix + "c_sq_lr.sh", "w") as f:

            print(preface, file=f)
            print("# Joint c_sq and lr tuning:", file=f)
            tuner = JointCsqLRTuner(factor=2 ** 0.5, curr=default, f=f)
            default, final_acc = tuner.run()

        if not final_acc:
            sys.exit()

    with open(file_prefix + "power.sh", "w") as f:

        print(preface, file=f)
        print("# Polynomial decay power tuning:", file=f)

        key = 'power'
        initial_val = 1.0
        initial_value = {key: initial_val, 'lr': default['lr']}
        tuner = LRPowerAutoTuner(
            factor=2**0.5, initial_value=initial_value, comp=0.5, p_tuner=PowerAutoTuner, key=key, diff=0.1, curr=default, f=f)
        power_default, final_acc = tuner.run()

    if not final_acc:
        sys.exit()

    with open(file_prefix + "cos_power.sh", "w") as f:

        print(preface, file=f)
        print("# Cosine decay power tuning:", file=f)

        key = 'cos_power'
        initial_val = 1.0
        initial_value = {key: initial_val, 'lr': default['lr']}
        tuner = LRPowerAutoTuner(
            factor=2**0.5, initial_value=initial_value, comp=0.5, p_tuner=CosPowerAutoTuner, key=key, diff=0.1, curr=default, f=f)
        default, final_acc = tuner.run()

    if not final_acc:
        sys.exit()

    with open(file_prefix + "cosine_power_comparison.sh", "w") as f:

        print(preface, file=f)
        print("# Cosine vs. polynomial decay:", file=f)

        initial_values = [{'cos_power': default['cos_power'], 'lr': default['lr']}]
        initial_values.append({'power': power_default['power'], 'lr': power_default['lr']})
        default['cos_power'] = None

        tuner = AutoTuner(initial_values=initial_values, curr=default, f=f)
        default, final_acc = tuner.run()

    if not final_acc:
        sys.exit()

    with open(file_prefix + "training_budgets.sh", "w") as f:

        print(preface, file=f)
        print("# Corrected with various training budgets:", file=f)

        done = test_training_budgets(default=default, eps=[30, 60, 90, 150, 300], f=f)

    if not done:
        sys.exit()

    if default.get('corrected') == '':

        corrected_default = dict(default)

        # Prepare uncorrected default
        c_sq = default['c_sq']
        mo, nesterov = default['momentum'], default.get('nesterov') == ''
        initial_wd = lr_factor(mo, nesterov) ** 2 * default['lr'] / c_sq / 2
        # Initial WD guess: half of the initial WD of the best corrected counterpart,
        # so the average throughout the training is about the same
        default['wd'], default['c_sq'] = initial_wd / 2, None
        # Reset fancy schedule
        default['cos_power'] = default['power'] = None

pathlib.Path('done').touch()
print('Done!')

# print(files_opened)
# ['corrected_lr.sh', 'corrected_wd.sh', 'corrected_nesterov.sh', 'corrected_momentum.sh', 'corrected_sign_lr.sh', 'corrected_sign_wd.sh', 'corrected_lr_eff_transfer.sh', 'corrected_mo_baseline_comparison.sh', 'corrected_bias.sh', 'corrected_c_sq_lr.sh', 'corrected_power.sh', 'corrected_cos_power.sh', 'corrected_cosine_power_comparison.sh', 'corrected_training_budgets.sh', 'lr.sh', 'wd.sh', 'nesterov.sh', 'momentum.sh', 'sign_lr.sh', 'sign_wd.sh', 'power.sh', 'cos_power.sh', 'cosine_power_comparison.sh', 'training_budgets.sh', 'done']
