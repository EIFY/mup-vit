import math
from typing import Optional

import torch
import torch.nn as nn
import einops
import unit_scaling as uu
import unit_scaling.functional as U
import unit_scaling.scale as S
from unit_scaling.core.functional import transformer_residual_scaling_rule

from simple_vit import posemb_sincos_2d


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
        constraint: Optional[str] = "to_output_scale",
    ) -> None:
        super().__init__()
        self.ratio = intermediate_size / hidden_size
        self.linear_1 = uu.Linear(hidden_size, intermediate_size, constraint=constraint)
        self.linear_2 = uu.Linear(intermediate_size, hidden_size, constraint=constraint)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = S.scale_bwd(input, 1 / math.sqrt(self.ratio))
        z = self.linear_1(input)
        z = S.scale_bwd(z, 1 / 1.20)
        z = U.gelu(z)
        z = S.scale_fwd(z, 1 / 1.15)
        z = S.scale_bwd(z, math.sqrt(self.ratio))
        return self.linear_2(z)


class MHSA(nn.Module):
    """A **unit-scaled** implementation of a multi-head self-attention layer.

    Warning: using `constraint=None` here will likely give incorrect gradients.

    Args:
        hidden_size (int): the hidden dimension size of the input.
        heads (int): the number of attention heads.
        is_causal (bool): causal masking (for non-padded sequences).
        dropout_p (float, optional): the probability of the post-softmax dropout.
    """

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        is_causal: bool,
        dropout_p: float = 0.0,
        mult: float = 1.0,
        constraint: Optional[str] = "to_output_scale",
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.mult = mult
        self.linear_qkv = uu.Linear(hidden_size, 3 * hidden_size, constraint=constraint)
        self.linear_o = uu.Linear(hidden_size, hidden_size, constraint=constraint)
        self.constraint = constraint

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = S.scale_bwd(input, 3 / 5)
        q_k_v = self.linear_qkv(input)
        q, k, v = einops.rearrange(q_k_v, "b s (z h d) -> z b h s d", h=self.heads, z=3)
        qkv = U.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p, is_causal=self.is_causal, mult=self.mult
        )
        qkv = S.scale_bwd(qkv, 5 / 3)
        qkv = einops.rearrange(qkv, "b h s d -> b s (h d)")
        return self.linear_o(qkv)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mlp_dim: int,
        num_heads: int,
        attn_tau: float,
        mlp_tau: float,
        attn = MHSA,
        norm_layer = uu.LayerNorm,
    ) -> None:
        super().__init__()
        self.head_size = hidden_dim // num_heads
        self.attn_norm = norm_layer(hidden_dim)
        self.attn = attn(hidden_dim, num_heads, False)

        self.mlp_norm = norm_layer(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_dim)

        self.attn_tau = attn_tau
        self.mlp_tau = mlp_tau

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        residual, skip = U.residual_split(input, self.attn_tau)
        residual = self.attn_norm(residual)
        residual = self.attn(residual)
        input = U.residual_add(residual, skip, self.attn_tau)

        residual, skip = U.residual_split(input, self.mlp_tau)
        residual = self.mlp_norm(residual)
        residual = self.mlp(residual)
        out = U.residual_add(residual, skip, self.mlp_tau)
        return out


class MupVisionTransformer(nn.Module):
    """Vision Transformer modified per https://arxiv.org/abs/2205.01580."""

    def _learned_embeddings(self, num):
        return uu.Embedding(num, self.hidden_dim)

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
        pool_type: str = "gap",
        register: int = 0,
        attn = MHSA,
        norm_layer = uu.LayerNorm,
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.pool_type = pool_type
        self.register = register + (pool_type == 'tok')  # [CLS] token is just another register
        if self.register == 1:
            self.register_buffer("reg", torch.zeros(1, 1, hidden_dim))
        elif self.register > 1:  # Random initialization needed to break the symmetry
            self.reg = self._learned_embeddings(self.register)

        self.conv_proj = uu.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        h = w = image_size // patch_size
        seq_length = h * w
        if posemb == "sincos2d":
            posemb = posemb_sincos_2d(h=h, w=w, dim=hidden_dim)
            posemb = posemb / posemb.std()
            self.register_buffer("pos_embedding", posemb)
        elif posemb == "learn":
            self.pos_embedding = self._learned_embeddings(seq_length)
        else:
            self.pos_embedding = None

        l = [transformer_residual_scaling_rule()(i, num_layers * 2) for i in range(num_layers * 2)]
        it = iter(l)
        self.encoder = uu.DepthSequential(*[
            EncoderBlock(
                hidden_dim=self.hidden_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                attn_tau=next(it),
                mlp_tau=next(it),
                attn=attn,
            ) for i in range(num_layers)
        ])
        self.seq_length = seq_length
        self.norm = norm_layer(self.hidden_dim)
        self.heads = uu.LinearReadout(hidden_dim, num_classes)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape and permute the input tensor
        x = self._process_input(x)
        if self.pos_embedding is not None:
            x = U.add(x, self.pos_embedding)
        if self.register:
            n = x.shape[0]
            x = torch.cat([torch.tile(self.reg, (n, 1, 1)), x], dim=1)
        x = self.norm(self.encoder(x))
        if self.pool_type == 'tok':
            x = x[:, 0]
            x = S.scale_bwd(x, self.seq_length + self.register)
        elif self.pool_type == 'gap':
            x = x[:, self.register:]
            x = x.sum(dim = 1)
            x = S.scale_fwd(x, 1 / self.seq_length)
        x = self.heads(x)

        return x
