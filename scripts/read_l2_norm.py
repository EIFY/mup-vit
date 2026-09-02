import collections, decimal, math, os, pathlib, statistics, sys, torch, pickle, collections, statistics
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

IMAGENET_TRAIN_SIZE = 1281167
BS = 4096


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


def read_last(curr, opt='scion-t212', path='logs/', repeat=0):
    name = run_name(opt, curr, repeat=repeat)
    step = round(IMAGENET_TRAIN_SIZE * curr['ep'] / BS)
    path_name = os.path.join(path, name)
    last_ckpt = os.path.join(path_name, f'checkpoints/model_step_{step}.pth.tar')
    ckpt = torch.load(last_ckpt, weights_only=True)
    return ckpt


def l2_norms(ckpt):
    hidden = 0.
    for n, p in ckpt['state_dict'].items():
        if n == 'module.pos_embedding': # With sincos2d pos embedding this is constant
            continue
        elif n == 'module.heads.head.weight':
            output = torch.linalg.vector_norm(p).item()
        else:
            hidden += torch.sum(p ** 2).item()
    return dict(hidden=math.sqrt(hidden), output=output)


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
    return mo.normalize()


def prev_mo(mo):
    if str(mo)[-1] in '12':
        mo /= 2
    else:
        mo /= 5
        mo *= 2
    return mo.normalize()


# None is tombstone value, '' (empty string) is for store_true flags
default = {'corrected': '', 'ep': 90, 'momentum': 0.1, 'lr': 0.011584472366059664, 'sign_lr': 0.09999999999999999, 'c_sq': 0.8396893026590251, 'wd': None, 'sign_wd': 0.00282842712474619, 'nesterov': '', 'cos_power': None, 'power': None}
filename = 'vit_l2_norms.pkl'

mo = default['momentum']
lr_eff = default['lr'] * lr_factor(mo, nesterov=default.get('nesterov') == '')
mo = decimal.Decimal(str(mo))  
mos = collections.deque([mo])
while len(mos) < 3 and mos[-1] < 1:
    mos.append(next_mo(mos[-1]))
while len(mos) < 6:
    mos.appendleft(prev_mo(mos[0]))

def read_l2_norm(mos, default):
    curr = dict(default)
    res = {}
    factors = [0.5, 2**-0.5, 1., 2**0.5, 2.0]
    for curr['nesterov'] in ('', None):
        for curr['momentum'] in mos:
            for factor in factors:
                base_lr = lr_eff / lr_factor(float(curr['momentum']), nesterov=curr.get('nesterov') == '')
                curr['lr'] = factor * base_lr
                res[run_name('scion-t212', curr)] = l2_norms(read_last(curr))

    with open(filename, 'wb') as file:
        pickle.dump(res, file)
    return res

if os.path.exists(filename):
    with open(filename, 'rb') as file:
        res = pickle.load(file)
else:
    res = read_l2_norm(mos, default)


regular = {k: collections.defaultdict(list) for k in ['hidden', 'output']}
nesterov = {k: collections.defaultdict(list) for k in ['hidden', 'output']}

curr = dict(default)
factors = [0.5, 2**-0.5, 1., 2**0.5, 2.0]
for curr['nesterov'] in ('', None):
    for curr['momentum'] in mos:
        for factor in factors:
            base_lr = lr_eff / lr_factor(float(curr['momentum']), nesterov=curr.get('nesterov') == '')
            curr['lr'] = factor * base_lr
            key = run_name('scion-t212', curr)
            d = regular if curr['nesterov'] is None else nesterov
            for k, v in res[key].items():
                d[k][float(curr['momentum'])].append(v)

fig = plt.figure()
ax = plt.gca()

for label, d, c in [('regular', regular, 'tab:blue'), ('Nesterov', nesterov, 'tab:orange')]:
    for k, v in d.items():
        x = v.keys()
        avg = [statistics.fmean(l) for l in v.values()]
        std = [statistics.stdev(l) for l in v.values()]
        ls = '--' if k == 'output' else '-'
        ax.errorbar(x, avg, yerr=std, linestyle=ls, color=c, label=label)

handles, labels = ax.get_legend_handles_labels()
handles = [h[0] for h in handles]
leg1 = ax.legend(handles[::2], labels[::2], bbox_to_anchor=(0.23, 1.0))
ax.add_artist(leg1)
leg2 = ax.legend(handles[:2], ['hidden', 'output'], bbox_to_anchor=(0.43, 1.0))
for line in leg2.legend_handles:
    line.set_color('black')

ax.set_xscale('log')

ax.set(xlabel='Momentum $\\alpha$')
ax.set(ylabel='$L_2$ norm')

plt.tight_layout()
plt.savefig('l2_norm.png')
