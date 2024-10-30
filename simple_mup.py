"""The simplest block from (https://arxiv.org/abs/2311.01906), unit-scaled (https://arxiv.org/abs/2407.17465)"""

import math

from collections import OrderedDict
from functools import partial
from typing import Callable, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import unit_scaling as uu
import unit_scaling.functional as U
import unit_scaling.scale as S

import sp

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    unscaled = sp.posemb_sincos_2d(h, w, dim, temperature, dtype)
    return unscaled / unscaled.std()


Loss = partial(uu.CrossEntropyLoss, reduction='sum')
LayerNorm = partial(uu.LayerNorm, eps=1e-6)
RMSNorm = partial(uu.RMSNorm, eps=1e-6)
Identity = nn.Identity
add = U.add


def tok(x: torch.Tensor, register: int, seq_length: int):
    x = x[:, 0]
    return S.scale_bwd(x, register + seq_length)
    

def gap(x: torch.Tensor, register: int, seq_length: int):
    x = x[:, register:]
    x = x.sum(dim = 1)
    return S.scale_fwd(x, 1 / seq_length)


# We need to handle this ourselves because of gradient accumulation:
# input.shape[0] may not be the correct effective batch size.
def loss_reduction(loss, batch_size):
    return S.scale_fwd(loss, 1 / batch_size)


class Patchifier(uu.Conv2d):

    def __init__(
        self,
        hidden_dim: int,
        patch_size: int,
    ):
        super().__init__(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            weight_mup_type="input",
        )


def learned_embeddings(num, hidden_dim):
    weight = uu.Parameter(torch.normal(mean=0.0, std=1.0, size=(num, hidden_dim)), mup_type="weight")
    # Empirical 1/sqrt(hidden_dim) embedding scaling law from https://arxiv.org/abs/2407.17465
    # ViT doesn't need to look up embeddings but needs to tile/cat them, so almost-raw Parameter()
    # is preferred over uu.Embedding().
    return scale_bwd(weight, hidden_dim ** -0.5)


def classifier_head(hidden_dim, num_classes, representation_size):
    heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
    if representation_size is None:
        heads_layers["head"] = uu.LinearReadout(hidden_dim, num_classes)
    else:
        heads_layers["pre_logits"] = uu.Linear(hidden_dim, representation_size)
        heads_layers["act"] = uu.Tanh(constraint=None)
        heads_layers["head"] = uu.LinearReadout(representation_size, num_classes)

    heads = nn.Sequential(heads_layers)
    nn.init.zeros_(heads.head.weight)

    return heads


class MLP(nn.Module):
    """A **unit-scaled** implementation of an MLP layer using GELU.

    Args:
        hidden_size (int): the hidden dimension size of the input.
        intermediate_size (int): the MLP's intermediate size.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.ratio = intermediate_size / hidden_size
        self.linear_1 = uu.Linear(hidden_size, intermediate_size, constraint=None)
        self.linear_2 = uu.Linear(intermediate_size, hidden_size, constraint=None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        z = self.linear_1(input)
        z = U.gelu(z)
        return self.linear_2(z)


class ShapedAttention(nn.Module):
    """A **unit-scaled** implementation of shaped attention."""

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        dropout: float,
        attention_dropout: float,
        mult: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.dropout = uu.Dropout(dropout) if dropout else nn.Identity()
        self.attention_dropout = attention_dropout
        self.mult = mult
        self.linear_qk = uu.Linear(hidden_size, 2 * hidden_size, constraint=None)

    def forward(self, input: torch.Tensor, mask: torch.Tensor, centering_matrix: torch.Tensor) -> torch.Tensor:
        qk = self.linear_qk(input)
        q, k = einops.rearrange(qk, "b s (z h d) -> z b h s d", h=self.heads, z=2)
        v = einops.rearrange(input, "b s (h d) -> b h s d", h=self.heads)
        scale = self.mult * self.heads / self.hidden_size
        new_v = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.attention_dropout, scale=scale,
        )
        if centering_matrix is not None:
            centering = centering_matrix @ v
        else:
            centering = v.mean(dim=-2, keepdim=True)
        v = v + self.dropout(new_v - centering)
        v = einops.rearrange(v, "b h s d -> b s (h d)")
        return v


class SimplifiedBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        beta: float,
        attn = ShapedAttention,
        norm_layer = LayerNorm,
    ) -> None:
        super().__init__()
        self.head_size = hidden_dim // num_heads
        self.attn = attn(hidden_dim, num_heads, dropout, attention_dropout)
        self.mlp = MLP(hidden_dim, mlp_dim)
        self.beta = beta
        self.norm = norm_layer(hidden_dim)

    def forward(self, input: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]):
        input, mask, centering_matrix = input
        input = self.norm(input)
        residual, skip = U.residual_split(input, self.beta)
        skip = self.attn(skip, mask, centering_matrix)
        residual = self.mlp(residual)
        out = U.residual_add(residual, skip, self.beta)
        return out, mask, centering_matrix


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
        self.dropout = uu.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = SimplifiedBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                beta=num_layers ** -0.5,
                norm_layer=norm_layer,
            )
        self.layers = uu.DepthSequential(layers)
        self.ln = norm_layer(hidden_dim)
        if attn_mask is not None:
            self.register_buffer("attn_mask", attn_mask)
            mask_mean = attn_mask.float().sum(dim=-1, keepdim=True)
            self.register_buffer("centering_matrix", attn_mask / mask_mean)
        else:
            self.attn_mask = self.centering_matrix = None

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.dropout(input)
        x, _, _ = self.layers((x, self.attn_mask, self.centering_matrix))
        return self.ln(x)


def generate_parameter_groups(model, lr, weight_decay, decoupled_weight_decay):
    return uu.optim.scaled_parameters(
        model.parameters(),
        uu.optim.lr_scale_func_adam,
        lr=lr,
        weight_decay=weight_decay,
        independent_weight_decay=decoupled_weight_decay,
    )


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
value_range = v2.Normalize(mean=MEAN, std=STD)
