from typing import Any, Optional, Union

from collections.abc import Iterable

from typing_extensions import TypeAlias

import torch

ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor], Iterable[dict[str, Any]], Iterable[tuple[str, torch.Tensor]]
]

from torch import Tensor


class Arc(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:

            lr = group["lr"]
            wd = group["weight_decay"] * lr
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1
                step = state["step"]

                exp_avg = state.setdefault("exp_avg", torch.zeros_like(
                    p, memory_format=torch.preserve_format
                ))
                exp_avg_sq = state.setdefault("exp_avg_sq", torch.zeros_like(
                    p, memory_format=torch.preserve_format
                ))

                # Decay the first and second moment running average coefficient
                exp_avg.lerp_(p.grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                update = -lr / bias_correction1 * exp_avg / ((exp_avg_sq / bias_correction2).sqrt() + eps)

                # AdamW will be just
                # p.data.mul_(1-wd).add_(update)

                p.data.mul_(1-wd)
                w_norm = torch.linalg.vector_norm(p.data)
                if w_norm < eps:
                    p.data.add_(update)
                else:
                    u_norm = torch.linalg.vector_norm(update)
                    inner = torch.sum(p.data * update)
                    target_norm = torch.abs(w_norm + inner / w_norm)
                    p.data.add_(update)
                    new_norm = torch.linalg.vector_norm(p.data)
                    p.data.mul_(target_norm / (new_norm + eps))
