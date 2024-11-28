import itertools
import math

from collections import OrderedDict
from functools import partial
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import MLPBlock


# Taken from https://github.com/lucidrains/vit-pytorch, likely ported from https://github.com/google-research/big_vision/
def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# Adapted from https://github.com/antofuller/CROMA/blob/main/use_croma.py
# inspired by: https://github.com/ofirpress/attention_with_linear_biases
def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                           :n - closest_power_of_2]


def get_2dalibi(size):
    points = list(itertools.product(range(size), range(size)))
    idxs = []
    for p1 in points:
        for p2 in points:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            idxs.append(dist)
    all_bias = torch.Tensor(idxs).view(len(points), len(points))
    return all_bias


def generate_alibi_bias(num_heads, sizes):
    seq_length = sum(s ** 2 for s in sizes)
    mask = torch.block_diag(*[get_2dalibi(s) for s in sizes]).unsqueeze(0)
    slopes = torch.Tensor(get_slopes(num_heads)).view(num_heads, 1, 1)
    bias = -slopes * mask
    return bias


def generate_fractal_mask(sizes):
    summary_size = sizes[1]
    mask = torch.block_diag(*[torch.ones(s ** 2, s ** 2) for s in sizes])
    offset = 0
    for i, s in enumerate(sizes[:-1]):
        factor = summary_size
        next_offset = s ** 2
        next_s = s * summary_size
        for _ in range(len(sizes) - 1 - i):
            for r_i in range(s):
                for c_i in range(s):
                    index = offset + r_i * s + c_i
                    for r_j in range(r_i * factor, (r_i + 1) * factor):
                        start = offset + next_offset + r_j * next_s + c_i * factor
                        mask[index, start:start + factor] = mask[start:start + factor, index] = 1
            factor *= summary_size
            next_offset += next_s ** 2
            next_s *= summary_size
        offset += s ** 2
    return mask.bool()


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

        # Fix init discrepancy between nn.MultiheadAttention and that of big_vision
        bound = math.sqrt(3 / hidden_dim)
        nn.init.uniform_(self.self_attention.in_proj_weight, -bound, bound)
        nn.init.uniform_(self.self_attention.out_proj.weight, -bound, bound)

    def forward(self, input: torch.Tensor, attn_mask: Optional[torch.Tensor]):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False, attn_mask=attn_mask)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        layers = [
            EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            ) for i in range(num_layers)
        ]
        self.layers = nn.ModuleList(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor, attn_mask: Optional[torch.Tensor]):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.dropout(input)
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.ln(x)


def jax_lecun_normal(layer, fan_in):
    """(re-)initializes layer weight in the same way as jax.nn.initializers.lecun_normal and bias to zero"""

    # constant is stddev of standard normal truncated to (-2, 2)
    std = math.sqrt(1 / fan_in) / .87962566103423978
    nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class SimpleVisionTransformer(nn.Module):
    """Vision Transformer modified per https://arxiv.org/abs/2205.01580."""

    def _learned_embeddings(self, num):
        return nn.Parameter(torch.normal(mean=0., std=math.sqrt(1 / self.hidden_dim), size=(1, num, self.hidden_dim)))

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        posemb: str = "sincos2d",
        representation_size: Optional[int] = None,
        pool_type: str = "gap",
        summary_size: Optional[int] = None,
        register: int = 0,
        fractal_mask: bool = False,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.pool_type = pool_type
        self.norm_layer = norm_layer
        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        h = w = image_size // patch_size
        self.seq_length = h * w
        if posemb == "sincos2d":
            self.register_buffer("pos_embedding", posemb_sincos_2d(h=h, w=w, dim=hidden_dim))
        elif posemb == "learn":
            self.pos_embedding = self._learned_embeddings(self.seq_length)
        else:
            self.pos_embedding = None

        sizes = []
        if summary_size:
            s = h
            while s > 1:
                s, mod = divmod(s, summary_size)
                # assert not mod, "Number of patches along height/width must be powers of summary size for now."
                sizes.append(s)
            sizes.reverse()
            self.register = sum(s ** 2 for s in sizes)
            if posemb == "sincos2d":
                self.register_buffer("reg", torch.cat([posemb_sincos_2d(h=s, w=s, dim=hidden_dim) for s in sizes], dim=0))
            elif posemb == "learn":
                self.reg = self._learned_embeddings(self.register)
            else:
                self.register_buffer("reg", torch.zeros(1, self.register, hidden_dim))
        else:
            self.register = register + (pool_type == 'tok')  # [CLS] token is just another register
            if self.register == 1:
                self.register_buffer("reg", torch.zeros(1, 1, hidden_dim))
            elif self.register > 1:  # Random initialization needed to break the symmetry
                self.reg = self._learned_embeddings(self.register)
        sizes.append(h)

        mask = torch.zeros(self.register + self.seq_length, self.register + self.seq_length, dtype=torch.float32)
        if fractal_mask:
            assert summary_size
            bool_fractal_mask = generate_fractal_mask(sizes)
            mask.masked_fill_(~bool_fractal_mask, -math.inf)
        if posemb == 'alibi':
            alibi_bias = generate_alibi_bias(num_heads, sizes)
            mask = mask + alibi_bias
        if fractal_mask or posemb == 'alibi':
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        self.encoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        # Init the patchify stem
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1] // self.conv_proj.groups
        jax_lecun_normal(self.conv_proj, fan_in)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            jax_lecun_normal(self.heads.pre_logits, fan_in)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def _loss_fn(self, out: torch.Tensor, lam: float, target1: torch.Tensor, target2: torch.Tensor):
        logprob = F.log_softmax(out, dim=1)
        return lam * F.nll_loss(logprob, target1) + (1.0 - lam) * F.nll_loss(logprob, target2)

    def forward(self, x: torch.Tensor, lam: float, target1: torch.Tensor, target2: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]
        if self.pos_embedding is not None:
            x = x + self.pos_embedding
        if self.register:
            x = torch.cat([torch.tile(self.reg, (n, 1, 1)), x], dim=1)
        mask = self.mask
        if mask is not None and len(mask.shape) == 3:
            mask = torch.tile(mask, (n, 1, 1))
        x = self.encoder(x, mask)
        if self.pool_type == 'tok':
            x = x[:, 0]
        else:
            x = x[:, self.register:]
            x = x.mean(dim = 1)
        x = self.heads(x)
        loss = self._loss_fn(x, lam, target1, target2)
        return x, loss
