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
Torchvision [`v2.RandomResizedCrop()` defaults to cropping 8%-100%](https://pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomResizedCrop.html) of the area of the image whereas big_vision `decode_jpeg_and_inception_crop()` [defaults to 5%-100%](https://github.com/google-research/big_vision/blob/01edb81a4716f93a48be43b3a4af14e29cdb3a7f/big_vision/pp/ops_image.py#L199). Torchvision transforms of [v2.RandAugment() default to zero padding](https://pytorch.org/vision/main/generated/torchvision.transforms.RandAugment.html) whereas big_vision `randaug()` [uses RGB values (128, 128, 128)](https://github.com/google-research/big_vision/blob/01edb81a4716f93a48be43b3a4af14e29cdb3a7f/big_vision/pp/autoaugment.py#L676) as the replacement value. In both cases I have specified the latter to conform to the reference implementation.
Model trained with all of the above for 90 epoches reached 76.91% top-1 validation set accuracy, but the loss curve and the gradient L2 norm clearly show that it deviates from the reference:

[<img width="1074" alt="Screenshot 2024-07-01 at 10 28 58 PM" src="https://github.com/EIFY/mup-vit/assets/2584418/80ce12d5-8cae-4729-8556-a146fd351e83">](https://api.wandb.ai/links/eify/5l4dv2p8)

It turned out that `RandAugment(num_ops=2, magnitude=10)` means very different things in torchvision vs. big_vision. I created the following 224 × 224 black & white calibration grid consists of 56 × 56 black & white squares:

![calibration_grid](https://github.com/EIFY/mup-vit/assets/2584418/d1da19c1-9c45-4e8a-b40e-c028ee2cf0af)

and [applied both versions of `RandAugment(2, 10)` 100000 times](notebooks/RandAugmentOnCalibrationGrid.ipynb) to gather the stats. All of the resulting pixels remain colorless
(i.e. for RGB values (r, g, b) r == g == b remains true) so we can sort them [from black to white into a spectrum](notebooks/GradientVisual.ipynb). For the following 2000 × 200 spectra, pixels are sorted top-down, left-right, and each pixel represents 224 * 224 * 100000 / (2000 * 200) = 112 * 112 pixels of the aggregated output, i.e. 1/4 of one output image. In case one batch of 12544 pixels happens to be of different values, I took the average. Here is the spectrum of torchvision `RandAugment(2, 10)`:

![torch_vision_randaugment_2_10](https://github.com/EIFY/mup-vit/assets/2584418/2b2cdd14-9ae6-49f9-8e9f-f5958a14c14e)

Here is the spectrum of torchvision `RandAugment(2, 10, fill=[128] * 3)`. We can see that it just shifts the zero-padding part of the black into the (128, 128, 128) gray:

![torch_vision_randaugment_2_10_mid_fill](https://github.com/EIFY/mup-vit/assets/2584418/1c3a0129-f48f-47c4-9f5e-00eb4ea2d57a)

And here is the spectrum of big_vision `randaug(2, 10)`:

![big_vision_randaugment_2_10](https://github.com/EIFY/mup-vit/assets/2584418/97e74d75-7002-4be1-888d-2aa8ae3d1e51)

Digging into the codebase, we can see that while torchvision's `v2.RandAugment()` sticks with the original [14-transform lineup](https://github.com/pytorch/vision/blob/bf01bab6125c5f1152e4f336b470399e52a8559d/torchvision/transforms/v2/_auto_augment.py#L375) of [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719), big_vision's own `randaug()` omits the `Identity` no-op and adds 3 new transforms `Invert`, `SolarizeAdd`, and `Cutout`, along with other subtler discrepancies (e.g. `Sharpness` is considered "signed" in torchvision so half of the time the transform blurs the image instead, while in big_vision it always sharpens the image). What I did then is to [subclass torchvision's `v2.RandAugment()`](notebooks/RandAugmentCalibration.ipynb), remove & add transforms accordingly, and use a variety of calibration grids to make sure that they are within ±1 of the RGB values given by the big_vision's counterpart. The sole exception is `Contrast`: more on that later. Even with that exception, the near-replication of big_vision's `randaug(2, 10)` results in near-identical spectrum:

![torch_vision_randaugment17_2_10](https://github.com/EIFY/mup-vit/assets/2584418/692bf214-87c0-442e-807c-98181f2efc62)

Training with the near-replication of big_vision `randaug(2, 10)` for 90 epoches reached 77.27% top-1 validation set accuracy and the gradient L2 norm looks the same, but the loss curve still differs:

[<img width="1074" alt="Screenshot 2024-07-02 at 1 43 47 PM" src="https://github.com/EIFY/mup-vit/assets/2584418/50fe571b-cba8-40f3-aef3-a63c3e7c65d2">](https://api.wandb.ai/links/eify/8d0wix47)

[<img width="1074" alt="Screenshot 2024-07-02 at 1 45 30 PM" src="https://github.com/EIFY/mup-vit/assets/2584418/489a3193-1c91-4045-ba02-d6e25420625c">](https://api.wandb.ai/links/eify/8d0wix47)

There is no more to be done on the pytorch side, however. Let's turn our attention to big_vision itself.

# `big_vision` [`grad_accum_wandb`](https://github.com/EIFY/big_vision/tree/grad_accum_wandb) branch
I first bolted on `wandb` logging and revived `utils.accumulate_gradient()` to run 1024 batch size on my GeForce RTX 3080 Laptop GPU. TensorBook is unable to handle `shuffle_buffer_size = 250_000` so I shrank it to `150_000`. Finally, I fell back to training on 100% of the training data to converge to what I had to do with pytorch. This resulted in 76.74% top-1 validation set accuracy `big-vision-repo-attempt` referenced above and consistent with the reported 76.7% top-1 validation set accuracy.

## `contrast()` transform
It turned out that one of big_vision's `randaug()` transforms, `contrast()`, [is broken](https://github.com/google-research/big_vision/issues/109). In short, what meant to calculate the average grayscale of the image
```python
  # Compute the grayscale histogram, then compute the mean pixel value,
  # and create a constant image size of that value.  Use that as the
  # blending degenerate target of the original image.
  hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
  mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
```
is instead calculating image_area / 256, so in our case of 224 × 224 image, mean is always 196. What it should do is the following:
```python
  # Compute the grayscale histogram, then compute the mean pixel value,
  # and create a constant image size of that value.  Use that as the
  # blending degenerate target of the original image.
  hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
  mean = tf.reduce_sum(
      tf.cast(hist, tf.float32) * tf.linspace(0., 255., 256)) / float(image_height * image_width)
```
We can visualize this bug by using the following calibration grid as the input:

![download (8)](https://github.com/EIFY/mup-vit/assets/2584418/d53438b8-0b9b-4dd4-8c60-7c2aa3aaa790)

and compare the output given by the broken `contrast()`:

![download (11)](https://github.com/EIFY/mup-vit/assets/2584418/23eccaa7-069f-44d5-9406-88132fbac5c3)

vs. the output after the fix:

![download (12)](https://github.com/EIFY/mup-vit/assets/2584418/9224b72b-7100-402b-80d8-8b032e731816)

Some CV people are aware of this bug ([1](https://x.com/giffmana/status/1798978504689938560), [2](https://x.com/wightmanr/status/1799170879697686897)) but AFAIK it wasn't documented anywhere in the public. As an aside, [`solarize()` transform has its own integer overflow bug](https://github.com/google-research/big_vision/issues/110) but just happens to have no effect when `magnitude=_MAX_LEVEL` here.

## Inconsistent anti-aliasing between training vs. validation
`decode_jpeg_and_inception_crop()` used by the training data pipeline defaults to [bilinear interpolation without anti-aliasing](https://github.com/google-research/big_vision/blob/01edb81a4716f93a48be43b3a4af14e29cdb3a7f/big_vision/pp/ops_image.py#L201) for resizing, but `resize_small()` used by the validation data pipeline defaults to [area interpolation](https://github.com/google-research/big_vision/blob/01edb81a4716f93a48be43b3a4af14e29cdb3a7f/big_vision/pp/ops_image.py#L108) that ["always anti-aliases"](https://www.tensorflow.org/api_docs/python/tf/image/resize). Furthermore, torchvision doesn't support resizing with area interpolation. For consistency, I changed both to bilinear interpolation with anti-aliasing.

## JPEG decoding
`tf.io.decode_jpeg()` by default [lets the system decide the JPEG decompression algorithm](https://www.tensorflow.org/api_docs/python/tf/io/decode_jpeg). Specifying `dct_method="INTEGER_ACCURATE"` makes it [behave like the PIL/cv2/PyTorch counterpart](https://github.com/google-research/big_vision/blob/01edb81a4716f93a48be43b3a4af14e29cdb3a7f/big_vision/pp/ops_image.py#L39) (see also the last few cells of [RandAugmentCalibration.ipynb](notebooks/RandAugmentCalibration.ipynb)). This option is exposed as `decode(precise=True)` in big_vision but is left unexposed for `decode_jpeg_and_inception_crop()`, so I [added the `precise` argument](https://github.com/EIFY/big_vision/commit/a31822116a9377d6f6dbfbd78372964ed48d8b9a) to the latter.

Changing all of the above seems to have [no apparent effect on the model](https://api.wandb.ai/links/eify/njwkx3dv), however (76.87% top-1 validation set accuracy).

## Adam 1st order accumulator precision
`optax.scale_by_adam()` supports the unusual option of using a different dtype for the 1st order accumulator, [`mu_dtype`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_adam) and the reference implementation uses [`bfloat16`](https://github.com/google-research/big_vision/blob/main/big_vision/configs/vit_s16_i1k.py#L80) instead of `float32` like the rest of the model. Changing it back to `float32`, however, still has [no apparent effect](https://api.wandb.ai/links/eify/dr9b8q4w) (76.77% top-1 validation set accuracy).

## Shuffle buffer size
Finally, back to `shuffle_buffer_size`. Unlike `torch.utils.data.DataLoader(shuffle=True)` which always fully shuffles by indices, [`tf.data.Dataset.shuffle(buffer_size)`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle) needs to load `buffer_size`'s worth of training examples into the main memory and fully shuffles iff `buffer_size=dataset.cardinality()`. To test whether incomplete shuffle so far has hurt performance, I launched a 8x A100-SXM4-40GB instance on [Lambda](https://lambdalabs.com/) and trained a big_vision model on it with all of the above and `config.input.shuffle_buffer_size = 1281167`, size of the ImageNet-1k training set. It still has [no apparent effect](https://api.wandb.ai/links/eify/huigfbka) (76.85% top-1 validation set accuracy).

As a by-product, this also proves that big_vision gradient accumulation and multi-GPU training are fully equivalent.

# Conclusions
I have run out of candidate causes of discrepancies to investigate. Can it be just due to randomness? While the discrepancy of the top-1 validation set accuracy is small (77.27% vs. 76.7%-76.87%), the loss curves suggest that it's real. Furthermore, I have trained a model by [grafting the pytorch model/optimizer/scheduler on the big_vision data pipelines](https://github.com/EIFY/big_vision/tree/grafted) and it clocked at [76.38% top-1 validation set accuracy](https://wandb.ai/eify/mup-vit/reports/torch-on-big-vision-input3--Vmlldzo4NTMzMTU4). If we believe that it behaves the same as the big_vision models on the big_vision data pipelines, the discrepancy can be up to +0.9%. I am therefore opening up the results here for new ideas and further investigation.

*Postscript*: Metrics of the models aside, in terms of training walltime, modern (2.2+) PyTorch with `compile()` and JAX are [nearly identical on the same GPU](https://api.wandb.ai/links/eify/rprx0dqy). The tiny difference may well be fully-explained by the overhead of transposing from channels-last to channels-first and converting from `tf.Tensor` to `torch.Tensor`. As for hardware comparison, here are the walltimes reported:

| Hardware | Walltime |
| --- | --- |
| TPUv3-8 node | 6h30 (99%) |
| 8x A100-SXM4-40GB | 5h41m |
| RTX 3080 Laptop | 5d19h32m |

8x A100-SXM4-40GB is comparable but faster than a TPUv3-8 node. RTX 3080 Laptop is unsurprisingly out of the league: 1 day on it is about the same as 1 hour on the other two.
