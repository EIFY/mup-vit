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
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:

            lr = group["lr"]
            wd = group["weight_decay"] * lr
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                for key, b in zip(("exp_avg", "exp_avg_sq"), (beta1, beta2)):
                    if key not in state:
                        state[key] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state[key + '_zero'] = 1.0
                    state[key + '_zero'] *= b

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Decay the first and second moment running average coefficient
                exp_avg.lerp_(p.grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                bias_correction1 = 1 - state["exp_avg_zero"]
                bias_correction2 = 1 - state["exp_avg_sq_zero"]

                update = -lr / bias_correction1 * exp_avg / ((exp_avg_sq / bias_correction2).sqrt() + eps)

                p.data.mul_(1-wd)
                w_2 = torch.sum(p.data ** 2)
                u_2 = torch.sum(update ** 2)

                w_norm = w_2.sqrt()

                # Just AdamW within the epsilon-ball
                if w_norm < eps:
                    p.data.add_(update)
                else:
                    inner = torch.sum(p.data * update)
                    target_norm = torch.abs(w_norm + inner / w_norm)
                    p.data.add_(update)
                    new_norm = (w_2 + 2 * inner + u_2).sqrt()
                    p.data.mul_(target_norm / (new_norm + eps))
