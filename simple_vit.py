import math

from collections import OrderedDict
from functools import partial
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import flex_attention
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


def generate_fractal_mask(sizes):
    summary_size = sizes[1]
    sizes.append(summary_size * sizes[-1])
    mask = torch.block_diag(*[torch.ones(s ** 2, s ** 2) for s in sizes])
    sizes.pop()
    offset = 0
    for i, s in enumerate(sizes):
        factor = summary_size
        next_offset = s ** 2
        next_s = s * summary_size
        for _ in range(len(sizes) - i):
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


class FlexAttention(nn.Module):
    """Minimum module to run flex_attention w/ block_mask."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.in_proj_weight = nn.parameter.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.parameter.Parameter(torch.empty(3 * embed_dim))
        self.out_proj_weight = nn.parameter.Parameter(torch.empty(embed_dim, embed_dim))
        self.out_proj_bias = nn.parameter.Parameter(torch.empty(embed_dim))

        # big_vision parameter initialization
        bound = math.sqrt(3 / embed_dim)
        nn.init.uniform_(self.in_proj_weight, -bound, bound)
        nn.init.uniform_(self.out_proj_weight, -bound, bound)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj_bias, 0.0)

    def forward(self, input, block_mask):
        b, s, d = input.shape
        qkv = F.linear(input, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.reshape((b, s, 3, self.num_heads, self.head_dim))
        # (b, s, 3, nh, hd) -> (3, b, nh, s, hd)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        out = flex_attention.flex_attention(q, k, v, block_mask=block_mask)
        # (b, nh, s, hd) -> (b, s, nh, hd)
        out = out.transpose(1, 2).reshape((b, s, d))
        out = F.linear(out, self.out_proj_weight, self.out_proj_bias)
        return out


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = FlexAttention(hidden_dim, num_heads)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor, block_mask: Optional[torch.Tensor]):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x = self.self_attention(x, block_mask=block_mask)
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
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.pool_type = pool_type
        self.norm_layer = norm_layer
        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        h = w = image_size // patch_size
        seq_length = h * w
        if posemb == "sincos2d":
            self.register_buffer("pos_embedding", posemb_sincos_2d(h=h, w=w, dim=hidden_dim))
        elif posemb == "learn":
            self.pos_embedding = self._learned_embeddings(seq_length)
        else:
            self.pos_embedding = None

        if summary_size:
            s = h
            self.sizes = []
            while s > 1:
                s, mod = divmod(s, summary_size)
                assert not mod, "Number of patches along height/width must be powers of summary size for now."
                self.sizes.append(s)
            self.sizes.reverse()
            self.register = sum(s ** 2 for s in self.sizes)
            if posemb == "sincos2d":
                self.register_buffer("reg", torch.cat([posemb_sincos_2d(h=s, w=s, dim=hidden_dim) for s in self.sizes], dim=0))
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

        self.fractal_mask = None
        self.encoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            norm_layer,
        )
        self.seq_length = seq_length

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

    def create_block_mask(self, device, block_size):
        mask = generate_fractal_mask(self.sizes)
        total_length = self.register + self.seq_length
        div, mod = divmod(total_length, block_size)
        # See https://github.com/pytorch/pytorch/issues/137801
        if mod:
            padding = (div + 1) * block_size - total_length
            mask = torch.block_diag(mask, torch.zeros(padding, padding, dtype=torch.bool))
        mask = mask.to(device)
        def mask_mod(b, h, q_idx, kv_idx):
            return mask[q_idx][kv_idx]
        self.fractal_mask = flex_attention.create_block_mask(
            mask_mod, B=None, H=None, Q_LEN=total_length, KV_LEN=total_length, device=device, BLOCK_SIZE=block_size)

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

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding
        if self.register:
            n = x.shape[0]
            x = torch.cat([torch.tile(self.reg, (n, 1, 1)), x], dim=1)
        x = self.encoder(x, self.fractal_mask)
        if self.pool_type == 'tok':
            x = x[:, 0]
        else:
            x = x[:, self.register:]
            x = x.mean(dim = 1)
        x = self.heads(x)

        return x
