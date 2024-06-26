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
difficult to reproduce with `torchvision.datasets` since `datasets.ImageNet()` is ordered by class label, unlike [tfds](https://www.tensorflow.org/datasets/overview)
where the ordering is somewhat randomized:

```python
import tensorflow_datasets as tfds
ds = tfds.builder('imagenet2012').as_dataset(split='train[99%:]')
from collections import Counter
c = Counter(int(e['label']) for e in ds)
>>> len(c)
999
>>> max(c.values())
27
>>> min(c.values())
3
```

Naively trying to do the same with `torchvision.datasets` prevented the model from learning the last few classes and resulted in [near-random performance
on the minival](https://api.wandb.ai/links/eify/3ju0jben): the model only learned the class that happened to stride across the first 99% and the last 1%.
Instead of randomly selecting 99% of the training data or copying the tfds 99% slice, I just fell back to training on 100% of the training data. ImageNet-1k
has 1281167 training images, so 1024 batch size results in `1281167 // 1024 = 1251` steps if we drop the last odd lot. big_vision however doesn't train the
model epoch by epoch: Instead, it makes the dataset iterator infinite and trains for the equivalent number of steps. Furthermore, [it `round()` the number of
steps](https://github.com/google-research/big_vision/blob/01edb81a4716f93a48be43b3a4af14e29cdb3a7f/big_vision/utils.py#L1014) instead of dropping the last.
The 90-epoch equivalent therefore would be `round(1281167 / 1024 * 90) = 112603` steps and `mup-vit` `main` follows this practice.

## Warmup
big_vision [warms up from 0 learning rate](https://github.com/google-research/big_vision/blob/01edb81a4716f93a48be43b3a4af14e29cdb3a7f/big_vision/utils.py#L1082)
but [`torch.optim.lr_scheduler.LinearLR()` disallows starting from 0 learning rate](https://github.com/pytorch/pytorch/blob/e62073d7997c9e63896cb5289ffd0874a8cc1838/torch/optim/lr_scheduler.py#L736).
I implemented warming up from 0 learning rate with [`torch.optim.lr_scheduler.LambdaLR()`](https://github.com/EIFY/mup-vit/blob/425cd9ac039367dbd96e2015f8d387e5958af998/main.py#L348) instead.

## Weight decay
In big_vision `config.wd` is only scaled by the global LR scheduling, but for `torch.optim.AdamW()` "`weight_decay`" is [first multiplied by the LR](https://fabian-sp.github.io/posts/2024/02/decoupling/).
The correct equivalent value for `weight_decay` is therefore `0.1` to match `config.lr = 0.001` and `config.wd = 0.0001`.

## Model
The simplified ViT described in [Better plain ViT baselines for ImageNet-1k](https://arxiv.org/abs/2205.01580) is not readily available in pytorch. E.g.
[vit_pytorch](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/)'s [simple_vit](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py)
and [simple_flash_attn_vit](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_flash_attn_vit.py) are rather dated without
taking advantage of [`torch.nn.MultiheadAttention()`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html), so I rolled my own.
I have to fix some of the parameter initialization, however:

1. `torch.nn.MultiheadAttention()` comes with its own issues. When QKV are of the same dimension, their projection matrices are combined into
`self.in_proj_weight` whose initial values are [set with `xavier_uniform_()`](https://github.com/pytorch/pytorch/blob/e62073d7997c9e63896cb5289ffd0874a8cc1838/torch/nn/modules/activation.py#L1074).
Likely unintentionally, this means that the values are sampled from uniform distribution U(−a,a) where a = sqrt(3 / (2 * hidden_dim)) instead sqrt(3 / hidden_dim).
Furthermore, the output projection is [initialized as `NonDynamicallyQuantizableLinear()`](https://github.com/pytorch/pytorch/blob/e62073d7997c9e63896cb5289ffd0874a8cc1838/torch/nn/modules/activation.py#L1097)
[whose initial values are sampled from U(-sqrt(k), sqrt(k)), k = 1 / hidden_dim](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html).
Both are therefore re-initialized with U(−a,a) where a = sqrt(3 / hidden_dim) to conform with the [`jax.nn.initializers.xavier_uniform()`
used by the reference ViT from big_vision](https://github.com/google-research/big_vision/blob/ec86e4da0f4e9e02574acdead5bd27e282013ff1/big_vision/models/vit.py#L93).
2. pytorch's own `nn.init.trunc_normal_()` doesn't take the effect of truncation on stddev into account, so I used [the magic factor](https://github.com/google/jax/blob/1949691daabe815f4b098253609dc4912b3d61d8/jax/_src/nn/initializers.py#L334) from the JAX repo to re-initialize the patchifying `nn.Conv2d`.

After 1 and 2 all of the summary statistics of the model parameters match that of the reference implementation at initialization.

## Data preprocessing and augmentation
Torchvision [`v2.RandomResizedCrop()` defaults to cropping 8%-100%](https://pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomResizedCrop.html) of the area of the image whereas big_vision `decode_jpeg_and_inception_crop()` [defaults to 5%-100%](https://github.com/google-research/big_vision/blob/01edb81a4716f93a48be43b3a4af14e29cdb3a7f/big_vision/pp/ops_image.py#L199). Torchvision transforms of [v2.RandAugment() default to zero paddling](https://pytorch.org/vision/main/generated/torchvision.transforms.RandAugment.html) whereas big_vision `randaug()` [uses RGB values (128, 128, 128)](https://github.com/google-research/big_vision/blob/01edb81a4716f93a48be43b3a4af14e29cdb3a7f/big_vision/pp/autoaugment.py#L676) as the replacement value. In both cases I have specified the latter to conform to the reference implementation.
