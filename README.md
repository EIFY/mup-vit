The `main` branch of this repository aims to reproduce [Better plain ViT baselines for ImageNet-1k](https://arxiv.org/abs/2205.01580) in pytorch,
in particular the 76.7% top-1 validation set accuracy of the Head: MLP → linear variant after 90 epochs. This variant is no inferior than the default,
and personally [I have better experience with simpler prediction head](https://iclr-blogposts.github.io/2024/blog/alibi-mlm/). The changes I have made
to the [big_vision](https://github.com/google-research/big_vision/tree/main) reference implementation in my attempts to make the results converge
reside in the [grad_accum_wandb](https://github.com/EIFY/big_vision/tree/grad_accum_wandb) branch. In the rest of this README I would like to highlight
some of the discrepancies I resolved and the remaining issues.

# `mup-vit` `main` branch
## Training data and budget
In [Better plain ViT baselines for ImageNet-1k](https://arxiv.org/abs/2205.01580) only the first 99% of the training data is used for training while the
remaining 1% is used for minival "to encourage the community to stop selecting design choices on the validation (de-facto test) set". This however is
difficult to reproduce with `torchvision.datasets` since `datasets.ImageNet()` is ordered by the class label, unlike [tdfs](https://www.tensorflow.org/datasets/overview)
where the ordering is somewhat randomized:
