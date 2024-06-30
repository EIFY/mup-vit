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
