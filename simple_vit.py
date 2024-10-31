from typing import Callable, Optional

import torch
import torch.nn as nn

import sp


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


class SimpleVisionTransformer(nn.Module):
    """Vision Transformer modified per https://arxiv.org/abs/2205.01580."""

    def __init__(
        self,
        p,
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
        norm_layer: str = "LayerNorm",
        l2_attn: bool = False,
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.patchifier = p.Patchifier(hidden_dim=hidden_dim, patch_size=patch_size)
        self.add = p.add
        self.pool = getattr(p, pool_type)
        self.embedding_scale = p.embedding_scale

        h = w = image_size // patch_size
        self.seq_length = h * w
        if posemb == "sincos2d":
            self.register_buffer("pos_embedding", p.posemb_sincos_2d(h=h, w=w, dim=hidden_dim))
        elif posemb == "learn":
            self.pos_embedding = p.learned_embeddings(num=self.seq_length, hidden_dim=hidden_dim)
        else:
            self.pos_embedding = None

        if summary_size:
            s = h
            sizes = []
            while s > 1:
                s, mod = divmod(s, summary_size)
                assert not mod, "Number of patches along height/width must be powers of summary size for now."
                sizes.append(s)
            sizes.reverse()
            self.register = sum(s ** 2 for s in sizes)
            if posemb == "sincos2d":
                self.register_buffer("reg", torch.cat([p.posemb_sincos_2d(h=s, w=s, dim=hidden_dim) for s in sizes], dim=0))
            elif posemb == "learn":
                self.reg = p.learned_embeddings(num=self.register, hidden_dim=hidden_dim)
            else:
                self.register_buffer("reg", torch.zeros(1, self.register, hidden_dim))
        else:
            self.register = register + (pool_type == 'tok')  # [CLS] token is just another register
            if self.register == 1:
                self.register_buffer("reg", torch.zeros(1, 1, hidden_dim))
            elif self.register > 1:  # Random initialization needed to break the symmetry
                self.reg = p.learned_embeddings(num=self.register, hidden_dim=hidden_dim)

        if fractal_mask:
            assert summary_size
            fractal_mask = generate_fractal_mask(sizes)
        else:
            fractal_mask = None

        self.encoder = p.Encoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            getattr(p, norm_layer),
            fractal_mask,
            l2_attn=l2_attn,
        )

        self.heads = p.classifier_head(hidden_dim, num_classes, representation_size)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.patchifier(x)
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
        n = x.shape[0]
        if self.pos_embedding is not None:
            x = self.add(x, self.embedding_scale(self.pos_embedding, n))
        if self.register:
            reg = self.embedding_scale(self.reg, n)
            x = torch.cat([torch.tile(reg, (n, 1, 1)), x], dim=1)
        x = self.encoder(x)
        x = self.pool(x, self.register, self.seq_length)
        x = self.heads(x)
        return x
