import itertools
import math

import torch
import torch.distributed as dist


#######################################################
# Scion
#######################################################
eps = 1e-8


class Norm(object):
    def lmo(self, g):
        raise NotImplementedError

    def init(self, w):
        raise NotImplementedError

    def momentum_buffer_shape(self, w):
        return w.shape

    prev_param_shape = prev_grad_shape = momentum_buffer_shape


class ColNorm(Norm):
    """
    Column-wise normalization.

    Args:
        normalized (bool, optional): If True, normalizes by the input dimension. Use True only for non-input layers.
        transpose (bool, optional): If True, transposes input before normalization. Use True for embedding layers
                which store weights as (vocab_size, embedding_dim).
    """
    def __init__(self, normalized=False, transpose=False):
        self.normalized = normalized
        self.transpose = transpose

    @torch.compile
    def lmo(self, g):
        if self.transpose:
            g = g.transpose(0, 1) 
        rms_values = 1/math.sqrt(g.size(0))*torch.sqrt(torch.sum(g ** 2, dim=0, keepdim=True))
        if self.normalized:
            rms_values *= g.size(1)
        g = g / (rms_values + eps)
        if self.transpose:
            g = g.transpose(0, 1) 
        return g

    @torch.compile
    def local_decay(self, w, v, wd, repeat=1):
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        for _ in range(repeat):
            col_norm = w.norm(dim=0)
            index = torch.argmax(col_norm)
            w.data[:,index].mul_(1-wd)
        norm = (1 - wd) * col_norm[index] / math.sqrt(w.size(0))
        if self.normalized:
            norm *= w.size(1)
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        return norm, v

    def init(self, w, init_dtype=torch.float64):
        dtype = w.data.dtype
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        torch.nn.init.normal_(w.data)
        w.data /= w.norm(dim=0, keepdim=True)
        w.data *= math.sqrt(w.size(0))
        if self.normalized:
            w.data /= w.size(1)
        w.data = w.data.to(dtype=dtype)
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        return torch.tensor(1.).to(w), w.new_empty((0,))

    def norm_shape(self, w):
        return ()

    smoothness_shape = norm_shape

    def singular_shape(self, w):
        return (0,)


class RowNorm(Norm):
    """
    Row-wise normalization.

    Args:
        normalized (bool, optional): If True, normalizes by the input dimension. Use False only for the input layer.
        transpose (bool, optional): If True, transposes input before normalization. Use True for embedding layers
                which store weights as (vocab_size, embedding_dim).
    """
    def __init__(self, normalized=True, transpose=False):
        self.normalized = normalized
        self.transpose = transpose

    @torch.compile
    def lmo(self, g):
        if self.transpose:
            g = g.transpose(0, 1) 
        rms_values = torch.sqrt(torch.sum(g ** 2, dim=-1, keepdim=True))
        if self.normalized:
            rms_values *= math.sqrt(g.size(-1))
        g = g / (rms_values + eps)
        if self.transpose:
            g = g.transpose(0, 1) 
        return g

    @torch.compile
    def local_decay(self, w, v, wd, repeat=1):
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        for _ in range(repeat):
            row_norm = w.norm(dim=-1)
            index = torch.argmax(row_norm)
            w.data[index,:].mul_(1-wd)
        norm = (1 - wd) * row_norm[index]
        if self.normalized:
            norm *= math.sqrt(w.size(-1))
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        return norm, v

    def init(self, w, init_dtype=torch.float64):
        dtype = w.data.dtype
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        torch.nn.init.normal_(w.data)
        w.data /= w.norm(dim=-1, keepdim=True)
        if self.normalized:
            w.data /= math.sqrt(w.size(-1))
        w.data = w.data.to(dtype=dtype)
        if self.transpose:
            w.data = w.data.transpose(0, 1)       
        return torch.tensor(1.).to(w), w.new_empty((0,))

    def norm_shape(self, w):
        return ()

    smoothness_shape = norm_shape

    def singular_shape(self, w):
        return (0.)


class BiasRMS(Norm):

    @torch.compile
    def lmo(self, g):
        rms_values = torch.sqrt(torch.mean(g ** 2))
        g = g / (rms_values + eps)
        return g

    def dual_norm(self, g):
        return math.sqrt(g.size(0)) * torch.linalg.vector_norm(g)

    def norm(self, w, v, repeat=1):
        return torch.sqrt(torch.mean(w ** 2)), v

    @torch.compile
    def local_decay(self, w, v, wd, repeat=1):
        # Same as regular weight decay
        w.data.mul_(1-wd)
        rms_values = torch.sqrt(torch.mean(w ** 2))
        return rms_values, v

    def init(self, w, init_dtype=torch.float64):
        torch.nn.init.zeros_(w)
        return torch.tensor(0.).to(w), w.new_empty((0,))

    def norm_shape(self, w):
        return ()

    smoothness_shape = norm_shape

    def singular_shape(self, w):
        return (0.)


class SpectralConv(Norm):
    def __init__(self, steps=5):
        self.steps = steps

    @torch.compile
    def lmo(self, g):
        g = PolarExpress(g.permute(2, 3, 0, 1), steps=self.steps).permute(2, 3, 0, 1)
        d_out, d_in, k, _ = g.shape
        g *= (d_out / d_in)**0.5 / (k ** 2)
        return g

    def dual_norm(self, g):
        return torch.sum(self.lmo(g) * g)

    @torch.compile
    def norm(self, w, v, repeat=1):
        d_out, d_in, _, k = w.shape
        w = w.permute(2, 3, 0, 1)
        for _ in range(repeat):
            u = w @ v
            u /= torch.linalg.vector_norm(u, dim=-2, keepdim=True)
            v = w.mT @ u
            s = torch.linalg.vector_norm(v, dim=-2, keepdim=True)
            v /= s
        return k**2 * (d_in / d_out)**0.5 * torch.max(s), v

    @torch.compile
    def local_decay(self, w, v, wd, repeat=1):
        d_out, d_in, _, k = w.shape
        w = w.permute(2, 3, 0, 1)
        for _ in range(repeat):
            u = w @ v
            u /= torch.linalg.vector_norm(u, dim=-2, keepdim=True)
            v = w.mT @ u
            s = torch.linalg.vector_norm(v, dim=-2, keepdim=True)
            flat_index = torch.argmax(s)
            row = flat_index // k
            col = flat_index % k
            # It may be justified & more efficient to only power-iterate
            # w[row,col,...] in the next iteration.
            w.data[row,col,...].add_(u[row,col,...] @ v[row,col,...].mT, alpha=-wd)
            v /= s
        s[row,col].mul_(1 - wd)
        return k**2 * (d_in / d_out)**0.5 * torch.max(s), v

    def init(self, w, init_dtype=torch.float64):
        w_fp = w.data.to(init_dtype)
        k = w.data.size(2)
        for kx in range(k):
            for ky in range(k):
                torch.nn.init.orthogonal_(w_fp[:,:,kx,ky])
        
        d_out, d_in, k, _ = w_fp.shape
        w_fp.mul_((d_out / d_in)**0.5 / (k ** 2))
        w.data = w_fp.to(dtype=w.data.dtype)
        v = torch.normal(0, 1, (k, k, d_in, 1))
        v /= torch.linalg.vector_norm(v, dim=-2, keepdim=True)
        s = torch.tensor(1.)
        return s.to(w), v.to(w)

    def norm_shape(self, w):
        return ()

    smoothness_shape = norm_shape

    def singular_shape(self, w):
        d_out, d_in, k, _ = w.shape
        return (k, k, d_in, 1)


class SpectralPatchifier(Norm):
    """For patchifier like ViT's with kernel_size == stride,
       which is just Linear() with a different input shape.
    """
    def __init__(self, steps=5):
        self.steps = steps

    def lmo(self, g):
        original_shape = g.shape
        g = PolarExpress(g.reshape(len(g), -1), steps=self.steps)
        d_out, d_in = g.shape
        g *= (d_out / d_in)**0.5
        return g.view(original_shape)

    def dual_norm(self, g):
        return torch.sum(self.lmo(g) * g)

    @torch.compile
    def norm(self, w, v, repeat=1):
        # Same as self.local_decay(w, norm, wd=0, repeat=repeat)
        # But it doesn't seem that torch can exploit wd=0.
        w = w.reshape(len(w), -1)
        for _ in range(repeat):
            u = w @ v
            u /= torch.linalg.vector_norm(u)
            v = w.mT @ u
            s = torch.linalg.vector_norm(v)
            v /= s
        d_out, d_in = w.size(-2), w.size(-1)
        return (d_in / d_out)**0.5 * s, v

    @torch.compile
    def local_decay(self, w, v, wd, repeat=1):
        w = w.reshape(len(w), -1)
        for _ in range(repeat):
            u = w @ v
            u /= torch.linalg.vector_norm(u)
            v = w.mT @ u
            w.data.add_(torch.outer(u, v), alpha=-wd)
            s = torch.linalg.vector_norm(v)
            v /= s
        d_out, d_in = w.size(-2), w.size(-1)
        return (1 - wd) * (d_in / d_out)**0.5 * s, v

    def init(self, w, init_dtype=torch.float64):
        w_fp = w.data.to(init_dtype)
        torch.nn.init.orthogonal_(w_fp)
        d_out, *rest = w_fp.shape
        d_in = math.prod(rest)
        w_fp.mul_((d_out / d_in)**0.5)
        w.data = w_fp.to(dtype=w.data.dtype)
        v = torch.normal(0, 1, (d_in,))
        v /= torch.linalg.vector_norm(v)
        s = torch.tensor(1.)
        return s.to(w), v.to(w)

    def norm_shape(self, w):
        return ()

    smoothness_shape = norm_shape

    def singular_shape(self, w):
        d_out, *rest = w.shape
        d_in = math.prod(rest)
        return (d_in,)


class Spectral(Norm):
    def __init__(self, max=False, normalized=True, steps=5):
        self.max = max
        self.steps = steps
        self.normalized = normalized

    def scale(self, d_out, d_in):
        if self.normalized:
            scale = (d_out / d_in)**0.5
        else:
            scale = d_out**0.5
        if self.max:
            scale = max(1,scale)
        return scale

    @torch.compile
    def lmo(self, g):
        g = PolarExpress(g, steps=self.steps)
        g *= self.scale(*g.shape[-2:])
        return g

    def dual_norm(self, g):
        return torch.sum(self.lmo(g) * g, dim=(-2, -1), keepdim=True)

    @torch.compile
    def norm(self, w, v, repeat=1):
        for _ in range(repeat):
            u = w @ v
            u /= torch.linalg.vector_norm(u, dim=-2, keepdim=True)
            v = w.mT @ u
            s = torch.linalg.vector_norm(v, dim=-2, keepdim=True)
            v /= s
        scale = self.scale(*w.shape[-2:])
        return s / scale, v

    @torch.compile
    def local_decay(self, w, v, wd, repeat=1):
        for _ in range(repeat):
            u = w @ v
            u /= torch.linalg.vector_norm(u, dim=-2, keepdim=True)
            v = w.mT @ u
            w.data.add_(-wd * u @ v.mT)
            s = torch.linalg.vector_norm(v, dim=-2, keepdim=True)
            v /= s
        scale = self.scale(*w.shape[-2:])
        return (1 - wd) * s / scale, v

    def init(self, w, init_dtype=torch.float64):
        w_fp = w.data.to(init_dtype)
        l = [range(s) for s in w_fp.shape[:-2]]
        l.append([...])
        for index in itertools.product(*l):
            torch.nn.init.orthogonal_(w_fp[index])
        scale = self.scale(*w.shape[-2:])
        w_fp.mul_(scale)
        w.data = w_fp.to(dtype=w.data.dtype)
        v_shape = w_fp.shape[:-2] + (w_fp.shape[-1], 1)
        v = torch.normal(0, 1, v_shape)
        v /= torch.linalg.vector_norm(v, dim=-2, keepdim=True)
        s = torch.ones(w_fp.shape[:-2] + (1, 1))
        return s.to(w), v.to(w)

    def norm_shape(self, w):
        return w.shape[:-2] + (1, 1)

    smoothness_shape = norm_shape

    def singular_shape(self, w):
        return w.shape[:-2] + (w.shape[-1], 1)


class Sign(Norm):
    def __init__(self, zero_init=False, normalized=True):
        self.zero_init = zero_init
        self.normalized = normalized

    def lmo(self, g):
        lmo = torch.sign(g)
        if self.normalized:
            d_out, d_in = g.shape
            lmo /= d_in
        return lmo

    def dual_norm(self, g):
        norm = torch.sum(g.abs())
        if self.normalized:
            d_out, d_in = g.shape
            norm /= d_in
        return norm

    def norm(self, w, v, repeat=1):
        norm = torch.max(w.abs())
        if self.normalized:
            d_out, d_in = w.shape
            norm *= d_in
        return norm, v

    @torch.compile
    def local_decay(self, w, _, wd, repeat=1):
        d_out, d_in = w.shape
        for _ in range(repeat):
            flat_index = torch.argmax(w.abs())
            row = flat_index // d_in
            col = flat_index % d_in
            w.data[row,col].mul_(1-wd)
        norm = torch.max(w.abs())
        if self.normalized:
            norm *= d_in
        return norm, w.new_empty((0,))

    def init(self, w, init_dtype=torch.float64):
        if self.zero_init:
            torch.nn.init.zeros_(w)
        else:
            # Generate -1/fan_in or 1/fan_in uniformly at random
            w.data = (torch.randint(0, 2, w.shape).to(w) * 2 - 1)
            if self.normalized:
                d_out, d_in = w.shape
                w.data /= d_in
        return torch.tensor(not self.zero_init).to(w), w.new_empty((0,))

    def norm_shape(self, w):
        return ()

    smoothness_shape = norm_shape

    def singular_shape(self, w):
        return (0,)


norm_dict = {
    'ColNorm': ColNorm,
    'RowNorm': RowNorm,
    'BiasRMS': BiasRMS,
    'SpectralConv': SpectralConv,
    'SpectralPatchifier':SpectralPatchifier,
    'Spectral': Spectral,
    'Sign': Sign,
}


class Scion(torch.optim.Optimizer):
    """Scion optimizer implementation.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): Learning rate (default: 1e-3)
        momentum (float, optional): One minus the traditional momentum factor. For example,
            a traditional momentum of 0.9 would be specified as momentum=0.1 here (default: 1.0)
        weight_decay (float, optional): Weight decay coefficient to be muliplied by the LR. WD * LR
            corresponds to the "learning rate" of the original constrained Scion.
        norm (str, optional): Choice of norm for gradient projection ('Auto', 'SpectralConv',
            'ColNorm', 'RowNorm', 'BiasRMS', 'Spectral', or 'Sign') (default: 'Auto')
        norm_kwargs (dict, optional): Additional arguments for the norm projection (default: None)

    Example:
        >>> radius = 50.0
        >>> optim_groups = [{
        ...     'params': model.transformer.h.parameters(),
        ...     'norm': 'Spectral',
        ...     'norm_kwargs': {},
        ...     'lr': radius,
        ... }, {
        ...     'params': model.lm_head.parameters(),
        ...     'norm': 'Sign',
        ...     'norm_kwargs': {},
        ...     'lr': radius*60.0,
        ... }]
        >>> optimizer = Scion(optim_groups, lr=2**-12, momentum=0.1)
    """
    state_keys = ('norm', 'singular', 'momentum_buffer')

    def __init__(self, params, defaults, rank=0, world_size=1):
        if defaults.get('lr', 1e-3) < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if defaults.get('momentum', 1.0) < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if defaults.get('norm_kwargs') is None:
            defaults['norm_kwargs'] = {}
        self.rank = rank
        self.world_size = world_size
        super().__init__(params, defaults)
        self.register_state_dict_pre_hook(self.sync_state)
        self.register_state_dict_post_hook(self.remove_unused_keys)
        self.register_load_state_dict_post_hook(self.remove_unused_keys)

    def assigned_parameters(self):
        """Simple round-robin"""
        index = 0
        for group in self.param_groups:
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            for p in group['params']:
                if self.rank == index % self.world_size:
                    yield group, norm_backend, p
                index += 1

    def not_assigned_parameters(self):
        index = 0
        for group in self.param_groups:
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            for p in group['params']:
                if self.rank != index % self.world_size:
                    yield group, norm_backend, p
                index += 1

    def remove_unused_keys(self, *_):
        if self.world_size == 1:
            return
        for group, norm_backend, p in self.not_assigned_parameters():
            state = self.state[p]
            for key in self.state_keys:
                state.pop(key, None)

    def sync_state(self, *_):
        for key in self.state_keys:
            self.sync_state_for(key)

    def sync_state_for(self, key):
        if self.world_size == 1:
            return
        index = 0
        buffer = []
        assigned_tensor = None
        for group in self.param_groups:
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            shape_f = getattr(norm_backend, key + '_shape')
            for p in group['params']:
                state = self.state[p]
                if key not in state:
                    state[key] = p.new_empty(shape_f(w))
                tensor = state[key]
                buffer.append(tensor)
                if self.rank == index % self.world_size:
                    assigned_tensor = tensor
                if len(buffer) == self.world_size:
                    dist.all_gather(buffer, assigned_tensor)
                    buffer.clear()
                    assigned_tensor = None
                index += 1
        if buffer:
            padding = self.world_size - len(buffer) % self.world_size
            buffer.extend(None for _ in range(padding))
            if assigned_tensor is None:
                buffer[self.rank] = assigned_tensor = p.new_empty((0,))
            dist.all_gather(buffer, assigned_tensor)

    def sync_params(self):
        if self.world_size == 1:
            return
        index = 0
        buffer = []
        assigned_p = None
        for group in self.param_groups:
            for p in group['params']:
                buffer.append(p)
                if self.rank == index % self.world_size:
                    assigned_p = p
                if len(buffer) == self.world_size:
                    dist.all_gather(buffer, assigned_p)
                    buffer.clear()
                    assigned_p = None
                index += 1
        if buffer:
            padding = self.world_size - len(buffer) % self.world_size
            buffer.extend(None for _ in range(padding))
            if assigned_p is None:
                buffer[self.rank] = assigned_p = p.new_empty((0,))
            dist.all_gather(buffer, assigned_p)

    @torch.no_grad()
    def step(self):
        for group, norm_backend, p in self.assigned_parameters():
            lr = group['lr']
            momentum = group['momentum']
            wd = lr * group['weight_decay']
            g = p.grad
            if g is None:
                continue
            state = self.state[p]

            if momentum != 1:
                buf = state['momentum_buffer']
                buf.mul_(1-momentum).add_(g, alpha=momentum)
                g = buf

            update = norm_backend.lmo(g)
            if group['local_decay']:
                state['norm'], state['singular'] = norm_backend.local_decay(p, state['singular'], wd, repeat=group['repeat'])
            else:
                p.data.mul_(1-wd)
                state['norm'], state['singular'] = norm_backend.local_decay(p, state['singular'], 0., repeat=1)
            p.data.add_(update, alpha=-lr)
        self.sync_params()

    def report_norms(self):
        self.sync_state_for('norm')
        spectral = []
        bias = []
        sign = []
        for group in self.param_groups:
            for p in group['params']:
                norm = self.state[p]['norm']
                if group['norm'].startswith('Spectral'):
                    spectral.extend(norm.flatten().tolist())
                elif group['norm'] == 'BiasRMS':
                    bias.append(norm.item())
                else:
                    sign.append(norm.item())
        return math.prod(spectral) ** (1 / len(spectral)), sum(bias) / len(bias), sum(sign) / len(sign)

    def init(self):
        init_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        for group, norm_backend, p in self.assigned_parameters():
            init_func = norm_backend.init
            state = self.state[p]
            state['norm'], state['singular'] = init_func(p, init_dtype=init_dtype)
            if group['momentum'] != 1:
                state['momentum_buffer'] = torch.zeros_like(p)


class Talon(Scion):
    """Talon optimizer implementation.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): Learning rate (default: 1e-3)
        momentum (float, optional): One minus the traditional momentum factor. For example,
            a traditional momentum of 0.9 would be specified as momentum=0.1 here (default: 1.0)
        weight_decay (float, optional): Weight decay coefficient to be muliplied by the LR. WD * LR
            corresponds to the "learning rate" of the original constrained Scion.
        norm (str, optional): Choice of norm for gradient projection ('Auto', 'SpectralConv',
            'ColNorm', 'RowNorm', 'BiasRMS', 'Spectral', or 'Sign') (default: 'Auto')
        norm_kwargs (dict, optional): Additional arguments for the norm projection (default: None)

    Example:
        >>> radius = 50.0
        >>> optim_groups = [{
        ...     'params': model.transformer.h.parameters(),
        ...     'norm': 'Spectral',
        ...     'norm_kwargs': {},
        ...     'lr': radius,
        ... }, {
        ...     'params': model.lm_head.parameters(),
        ...     'norm': 'Sign',
        ...     'norm_kwargs': {},
        ...     'lr': radius*60.0,
        ... }]
        >>> optimizer = Talon(optim_groups, lr=2**-12, momentum=0.1)
    """
    state_keys = ('norm', 'singular', 'momentum_buffer', 'diff_singular', 'smoothness', 'prev_param', 'prev_grad')

    @torch.no_grad()
    def step(self):
        for group, norm_backend, p in self.assigned_parameters():
            lr = group['lr']
            momentum = group['momentum']
            beta = group['beta']
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])

            g = p.grad
            if g is None:
                continue
            state = self.state[p]

            if momentum != 1:
                buf = state['momentum_buffer']
                buf.mul_(1-momentum).add_(g, alpha=momentum)
                g = buf

            if 'prev_param' in state:
                norm_param_diff, state['diff_singular'] = norm_backend.norm(p.data - state['prev_param'], state['diff_singular'], repeat=5)
                norm_grad_diff = norm_backend.dual_norm(p.grad - state['prev_grad'])
                nonzero = torch.minimum(norm_grad_diff, norm_param_diff) > eps
                state['smoothness'][nonzero] = beta * state['smoothness'][nonzero] + (1-beta) * (norm_grad_diff / norm_param_diff)[nonzero]

            update = norm_backend.lmo(g)

            adaptive_lr = lr / state['smoothness']
            wd = adaptive_lr * group['weight_decay']
            if group['corrected']:
                wd *= adaptive_lr

            state['prev_param'] = p.data.clone()
            state['prev_grad'] = p.grad

            if group['local_decay']:
                state['norm'], state['singular'] = norm_backend.local_decay(p, state['singular'], wd, repeat=group['repeat'])
            else:
                p.data.mul_(1-wd)
                state['norm'], state['singular'] = norm_backend.local_decay(p, state['singular'], 0., repeat=1)
            p.data.add_(-adaptive_lr * update)
        self.sync_params()

    def init(self):
        super().init()
        for group, norm_backend, p in self.assigned_parameters():
            state = self.state[p]
            state['diff_singular'] = state['singular'].clone()
            state['smoothness'] = torch.ones_like(state['norm']) / group['lr']
        for group in self.param_groups:
            initial_lr = group['lr'] * group['lr_multiplier']
            if group['corrected']:
                group['weight_decay'] /= initial_lr
            group['lr'] = group.pop('lr_multiplier')


# Polar Express (https://arxiv.org/abs/2505.16932) w/ eps to prevent divide-by-zero
coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]

# safety factor for numerical stability (but exclude last polynomial)
coeffs_list = [(a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in coeffs_list[:-1]] + [coeffs_list[-1]]

def PolarExpress(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16() # for speed
    if G.size(-2) > G.size(-1): X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim = True) * 1.01 + eps)
    hs = coeffs_list[:steps] + list(itertools.repeat(coeffs_list[-1], steps - len(coeffs_list)))
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX ˆ3 + cX ˆ5
    if G.size(-2) > G.size(-1): X = X.mT
    return X

if not torch.backends.mps.is_available():
    PolarExpress = torch.compile(PolarExpress)


def zeroth_power_via_svd(G):
    U, S, V = G.svd()
    return U @ V.T
