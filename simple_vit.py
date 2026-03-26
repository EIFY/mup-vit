import math

from collections import OrderedDict
from functools import partial
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ScaledGELU(nn.Module):
    def forward(self, input: torch.Tensor):
        return math.sqrt(2) * F.gelu(input)


class MLPBlock(nn.Sequential):
    def __init__(self, in_dim: int, mlp_dim: int, dropout: float, bias: bool):
        layers = [
            nn.Linear(in_dim, mlp_dim, bias=bias),
            ScaledGELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, in_dim, bias=bias),
            nn.Dropout(dropout),
        ]
        super().__init__(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)


class SelfAttention(nn.Module):
    """Muon-friendly with merged QKV weights"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, bias = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Follow big_vision's init
        bound = math.sqrt(3 / hidden_dim)
        self.qkv_w = nn.Parameter(torch.empty(3, hidden_dim, hidden_dim).uniform_(-bound, bound))
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        nn.init.uniform_(self.out.weight, -bound, bound)

    def forward(self, x: torch.Tensor):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, -1).chunk(3, dim=-2)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=self.dropout).transpose(1, 2)
        y = y.contiguous().view(B, T, self.hidden_dim)
        y = self.out(y)
        return y


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
        bias : bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SelfAttention(hidden_dim, num_heads, dropout=attention_dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout, bias)


    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x = self.self_attention(x)
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
        bias : bool = True,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
                bias,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        return self.ln(self.layers(self.dropout(input)))


def jax_lecun_normal(layer, fan_in):
    """(re-)initializes layer weight in the same way as jax.nn.initializers.lecun_normal and bias to zero"""

    # constant is stddev of standard normal truncated to (-2, 2)
    std = math.sqrt(1 / fan_in) / .87962566103423978
    nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


eps = 1e-8


class PotentialHead(nn.Module):

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.weight = nn.parameter.Parameter(torch.empty((num_classes, hidden_dim)))

    def forward(self, x: torch.Tensor):
        x_sq = x.square().sum(-1, keepdim=True)
        w_sq = self.weight.square().sum(-1, keepdim=True)
        squared = x_sq - 2 * x @ self.weight.T + w_sq.T + eps
        return torch.rsqrt(squared)


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
        head: Optional[str] = None,
        pool_type: str = "gap",
        register: int = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.RMSNorm, eps=1e-6, elementwise_affine=False),
        bias: bool = True,
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
        self.pool_type = pool_type
        self.norm_layer = norm_layer
        self.register = register + (pool_type == 'tok')  # [CLS] token is just another register
        if self.register == 1:
            self.register_buffer("reg", torch.zeros(1, 1, hidden_dim))
        elif self.register > 1:  # Random initialization needed to break the symmetry
            self.reg = self._learned_embeddings(self.register)

        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )

        h = w = image_size // patch_size
        seq_length = h * w
        if posemb == "sincos2d":
            self.register_buffer("pos_embedding", posemb_sincos_2d(h=h, w=w, dim=hidden_dim))
        elif posemb == "learn":
            self.pos_embedding = self._learned_embeddings(seq_length)
        else:
            self.pos_embedding = None

        self.encoder = Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
            bias=bias,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if head is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes, bias=bias)
            self.heads = nn.Sequential(heads_layers)
        elif head == 'mlp':
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, hidden_dim, bias=bias)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes, bias=bias)
            self.heads = nn.Sequential(heads_layers)
        else:
            self.heads = PotentialHead(hidden_dim, num_classes)

        # Init the patchify stem
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1] // self.conv_proj.groups
        jax_lecun_normal(self.conv_proj, fan_in)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            jax_lecun_normal(self.heads.pre_logits, fan_in)

        if hasattr(self.heads, 'head') and isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            if self.heads.head.bias is not None:
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
        if self.pos_embedding is not None:
            x = x + self.pos_embedding
        if self.register:
            n = x.shape[0]
            x = torch.cat([torch.tile(self.reg, (n, 1, 1)), x], dim=1)
        x = self.encoder(x)
        if self.pool_type == 'tok':
            x = x[:, 0]
        else:
            x = x[:, self.register:]
            x = x.mean(dim = 1)
        x = self.heads(x)
        loss = self._loss_fn(x, lam, target1, target2)
        return x, loss
