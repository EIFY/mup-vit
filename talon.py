import itertools
import math
import torch


#######################################################
# Scion
#######################################################
eps = 1e-8


class Norm(object):
    def lmo(self, g):
        raise NotImplementedError

    def init(self, w):
        raise NotImplementedError


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

    def init(self, w):
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
        return w


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

    def init(self, w):
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
        return w


class BiasRMS(Norm):
    def lmo(self, g):
        rms_values = torch.sqrt(torch.mean(g ** 2, dim=0, keepdim=True))
        g = g / (rms_values + eps)
        return g

    def init(self, g):
        return torch.nn.init.zeros_(g)


class SpectralConv(Norm):
    def __init__(self, steps=5):
        self.steps = steps

    def lmo(self, g):
        g = PolarExpress(g.reshape(len(g), -1), steps=self.steps).view(g.shape)
        out_channels, in_channels, k, _ = g.shape
        g *= (out_channels / in_channels)**0.5 / (k ** 2)
        return g
    
    def init(self, w):
        w_fp = w.data.double()
        k = w.data.size(2)
        for kx in range(k):
            for ky in range(k):
                torch.nn.init.orthogonal_(w_fp[:,:,kx,ky])
        
        out_channels, in_channels, k, _ = w_fp.shape
        w_fp.mul_((out_channels / in_channels)**0.5 / (k ** 2))
        w.data = w_fp.to(dtype=w.data.dtype)
        return w


class SpectralPatchifier(Norm):
    """For patchifier like ViT's with kernel_size == stride,
       which is just Linear() with a different input shape.
    """
    def __init__(self, steps=5):
        self.steps = steps

    def lmo(self, g):
        g = PolarExpress(g.reshape(len(g), -1), steps=self.steps).view(g.shape)
        out_channels, *rest = g.shape
        g *= (out_channels / math.prod(rest))**0.5
        return g
    
    def init(self, w):
        w_fp = w.data.double()
        torch.nn.init.orthogonal_(w_fp)
        out_channels, *rest = w_fp.shape
        w_fp.mul_((out_channels / math.prod(rest))**0.5)
        w.data = w_fp.to(dtype=w.data.dtype)
        return w


class Spectral(Norm):
    def __init__(self, max=False, normalized=True, steps=5):
        self.max = max
        self.steps = steps
        self.normalized = normalized

    def lmo(self, g):
        g = PolarExpress(g, steps=self.steps)
        d_out, d_in = g.size(-2), g.size(-1)
        
        if self.normalized:
            scale = (d_out / d_in)**0.5
        else:
            scale = d_out**0.5
        if self.max:
            scale = max(1,scale)
        g *= scale

        return g

    def init(self, w):
        w_fp = w.data.double()
        l = [range(s) for s in w_fp.shape[:-2]]
        l.append([...])
        for index in itertools.product(*l):
            torch.nn.init.orthogonal_(w_fp[index])
        d_out, d_in = w.size(-2), w.size(-1)
        
        if self.normalized:
            scale = (d_out / d_in)**0.5
        else:
            scale = d_out**0.5
        if self.max:
            scale = max(1,scale)
        w_fp.mul_(scale)
    
        w.data = w_fp.to(dtype=w.data.dtype)
        return w


class Sign(Norm):
    def __init__(self, zero_init=False, normalized=True):
        self.zero_init = zero_init
        self.normalized = normalized

    def lmo(self, g):
        d_out, d_in = g.shape
        if self.normalized:
            return (1/d_in)*torch.sign(g)    
        else:
            return torch.sign(g)

    def init(self, w):
        if self.zero_init:
            torch.nn.init.zeros_(w)
        else:
            # Generate -1/fan_in or 1/fan_in uniformly at random
            d_out, d_in = w.shape
            w.data = (torch.randint(0, 2, w.shape, dtype=w.dtype, device=w.device) * 2 - 1)
            if self.normalized:
                w.data *= (1/d_in)
        return w


class Auto(Norm):
    def lmo(self, g):
        if g.ndim >= 2:
            return Spectral().lmo(g)
        else:
            return BiasRMS().lmo(g)

    def init(self, w):
        if w.ndim >= 2:
            return Spectral().init(w)
        else:
            return BiasRMS().init(w)


norm_dict = {
    'ColNorm': ColNorm,
    'RowNorm': RowNorm,
    'BiasRMS': BiasRMS,
    'SpectralConv': SpectralConv,
    'SpectralPatchifier':SpectralPatchifier,
    'Spectral': Spectral,
    'Sign': Sign,
    'Auto': Auto,
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
    def __init__(self, params, lr=1e-3, momentum=1.0, weight_decay=0.01, norm: str='Auto', norm_kwargs: dict=None):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if norm_kwargs is None:
            norm_kwargs = {}
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, norm=norm, norm_kwargs=norm_kwargs)
        super().__init__(params, defaults)
        for group in self.param_groups:
            group['max_lr'] = group['lr']

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            wd = lr * group['weight_decay']
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if momentum != 1:
                    if 'momentum_buffer' not in state.keys():
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(1-momentum).add_(g, alpha=momentum)
                    g = buf

                update = norm_backend.lmo(g)
                p.data.mul_(1-wd)
                p.data.add_(update, alpha=-lr)

    def init(self):
        for group in self.param_groups:
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            init_func = norm_backend.init
            for p in group['params']:
                init_func(p)


class ScionLight(torch.optim.Optimizer):
    """Memory-efficient variant of the Scion optimizer.
    
    This implementation saves memory by storing only the averaged gradient instead of 
    both the gradient and its average. Note that gradients should not be zeroed since
    p.grad is used directly to store the gradient average.
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): Learning rate (default: 1e-3)
        momentum (float, optional): One minus the traditional momentum factor. For example,
            a traditional momentum of 0.9 would be specified as momentum=0.1 here (default: 1.0)
        norm (str, optional): Choice of norm for gradient projection ('Auto', 'SpectralConv', 
            'ColNorm', 'RowNorm', 'BiasRMS', 'Spectral', or 'Sign') (default: 'Auto')
        norm_kwargs (dict, optional): Additional arguments for the norm projection (default: None)
        scale (float, optional): Scale factor for updates (default: 1.0)
        unconstrained (bool, optional): Whether to use unconstrained updates (default: False)
    
    Example:
        >>> radius = 50.0
        >>> optim_groups = [{
        ...     'params': model.transformer.h.parameters(),
        ...     'norm': 'Spectral',
        ...     'norm_kwargs': {},
        ...     'scale': radius,
        ... }, {
        ...     'params': model.lm_head.parameters(),
        ...     'norm': 'Sign',
        ...     'norm_kwargs': {},
        ...     'scale': radius*60.0,
        ... }]
        >>> optimizer = ScionLight(optim_groups, lr=2**-12, momentum=0.1)
    """
    def __init__(self, params, lr=1e-3, momentum=1.0, norm: str='Auto', norm_kwargs: dict=None, scale=1.0, unconstrained=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if norm_kwargs is None:
            norm_kwargs = {}
        defaults = dict(lr=lr, momentum=momentum, scale=scale, unconstrained=unconstrained, norm=norm, norm_kwargs=norm_kwargs)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            scale = group['scale']
            unconstrained = group['unconstrained']
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            for p in group['params']:
                G = p.grad
                if G is None:
                    continue

                update = scale * norm_backend.lmo(G)
                if not unconstrained:
                    p.data.mul_(1-lr)
                p.data.add_(update, alpha=-lr)
                
                if momentum != 1:
                    G.mul_(1-momentum)

    def init(self):
        for group in self.param_groups:
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            init_func = norm_backend.init
            scale = group['scale']
            for p in group['params']:
                init_func(p)
                p.data *= scale


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

@torch.compile
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


def zeroth_power_via_svd(G):
    U, S, V = G.svd()
    return U @ V.T
