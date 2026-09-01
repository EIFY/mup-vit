import collections, decimal, math, os, pathlib, statistics, sys, torch, pickle

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

curr = dict(default)
mo = curr['momentum']
lr_eff = curr['lr'] * lr_factor(mo, nesterov=curr.get('nesterov') == '')
mo = decimal.Decimal(str(mo))  
mos = collections.deque([mo])
while len(mos) < 3 and mos[-1] < 1:
    mos.append(next_mo(mos[-1]))
while len(mos) < 6:
    mos.appendleft(prev_mo(mos[0]))

res = {}

factors = [0.5, 2**-0.5, 1., 2**0.5, 2.0]
for curr['nesterov'] in ('', None):
    for curr['momentum'] in mos:
        for factor in factors:
            base_lr = lr_eff / lr_factor(float(curr['momentum']), nesterov=curr.get('nesterov') == '')
            curr['lr'] = factor * base_lr
            res[run_name('scion-t212', curr)] = l2_norms(read_last(curr))

print(res)

with open('vit_l2_norms.pkl', 'wb') as file:
    pickle.dump(res, file)
