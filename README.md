The `main` branch of this repository aims to reproduce [Better plain ViT baselines for ImageNet-1k](https://arxiv.org/abs/2205.01580) in pytorch,
in particular the 76.7% top-1 validation set accuracy of the Head: MLP → linear variant after 90 epochs. This variant is no inferior than the default,
and personally [I have better experience with simpler prediction head](https://iclr-blogposts.github.io/2024/blog/alibi-mlm/). The changes I have made
to the [big_vision](https://github.com/google-research/big_vision/tree/main) reference implementation in my attempts to make the results converge
reside in the [grad_accum_wandb](https://github.com/EIFY/big_vision/tree/grad_accum_wandb) branch. In the rest of this README I would like to highlight
some of the discrepancies I resolved and the remaining issues.
