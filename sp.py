"""'Standard parameterization', more specifically that of big_vision's ViT."""

import math

from collections import OrderedDict
from functools import partial
from typing import Callable, Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models.vision_transformer import MLPBlock
from torchvision.transforms import v2


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


Loss = partial(nn.CrossEntropyLoss, reduction='sum')
LayerNorm = partial(nn.LayerNorm, eps=1e-6)
RMSNorm = partial(nn.LayerNorm, eps=1e-6)
Identity = nn.Identity
add = torch.add


def tok(x: torch.Tensor, register: int, seq_length: int):
    return x[:, 0]


def gap(x: torch.Tensor, register: int, seq_length: int):
    x = x[:, register:]
    return x.mean(dim = 1)


def loss_reduction(loss, batch_size):
    return loss / batch_size


def fan_in_init(layer, fan_in):
    """(re-)initializes layer weight in the same way as jax.nn.initializers.lecun_normal and bias to zero"""

    # constant is stddev of standard normal truncated to (-2, 2)
    std = math.sqrt(1 / fan_in) / .87962566103423978
    nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class Patchifier(nn.Conv2d):

    def __init__(
        self,
        hidden_dim: int,
        patch_size: int,
    ):
        super().__init__(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)
        # Init the patchify stem
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1] // self.groups
        fan_in_init(self, fan_in)


def learned_embeddings(num, hidden_dim):
    return nn.Parameter(torch.normal(mean=0., std=math.sqrt(1 / hidden_dim), size=(1, num, hidden_dim)))


def embedding_scale(embeddings, batch_size):
    return embeddings


def classifier_head(hidden_dim, num_classes, representation_size):
    heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
    if representation_size is None:
        heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
    else:
        heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
        heads_layers["act"] = nn.Tanh()
        heads_layers["head"] = nn.Linear(representation_size, num_classes)

    heads = nn.Sequential(heads_layers)

    if hasattr(heads, "pre_logits") and isinstance(heads.pre_logits, nn.Linear):
        fan_in = heads.pre_logits.in_features
        fan_in_init(heads.pre_logits, fan_in)

    nn.init.zeros_(heads.head.weight)
    nn.init.zeros_(heads.head.bias)

    return heads


class MaskedAttention(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Fix init discrepancy between nn.MultiheadAttention and that of big_vision
        bound = math.sqrt(3 / hidden_dim)
        nn.init.uniform_(self.attention.in_proj_weight, -bound, bound)
        nn.init.uniform_(self.attention.out_proj.weight, -bound, bound)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]):
        x, _ = self.attention(x, x, x, need_weights=False, attn_mask=attn_mask)
        return x


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = LayerNorm,
    ):
        super().__init__()

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.attention = MaskedAttention(hidden_dim, num_heads, dropout=attention_dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: Tuple[torch.Tensor, Optional[torch.Tensor]]):
        input, attn_mask = input
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x = self.attention(x, attn_mask=attn_mask)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y, attn_mask


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
        norm_layer: Callable[..., torch.nn.Module] = LayerNorm,
        attn_mask: Optional[torch.Tensor] = None,
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
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)
        if attn_mask is not None:
            # Note the bitwise-not "~" below. Binary attn_mask for MHA forward pass follows the *opposite* convention
            # (https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward)
            # to that of F.scaled_dot_product_attention!
            # (https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
            self.register_buffer("attn_mask", ~attn_mask)
        else:
            self.attn_mask = None

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.dropout(input)
        x, _ = self.layers((x, self.attn_mask))
        return self.ln(x)


def weight_decay_param(n, p):
    return p.ndim >= 2 and n.endswith('weight')


def generate_parameter_groups(model, lr, weight_decay, decoupled_weight_decay):
    wd_params = [p for n, p in model.named_parameters() if weight_decay_param(n, p) and p.requires_grad]
    non_wd_params = [p for n, p in model.named_parameters() if not weight_decay_param(n, p) and p.requires_grad]
    if decoupled_weight_decay:
        weight_decay /= lr
    params = [
        {"params": wd_params, "lr": lr, "weight_decay": weight_decay},
        {"params": non_wd_params, "lr": lr, "weight_decay": 0.},
    ]
    return params


# Ignore input mean & std and just make input ranges from -1 to 1,
# equivalent to big_vision's value_range(-1, 1).
value_range = v2.Normalize(
    mean=[0.5] * 3,
    std=[0.5] * 3)
