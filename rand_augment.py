from torchvision.transforms import v2

import math
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type, Union

import PIL.Image
import torch

from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec
from torchvision import transforms as _transforms, tv_tensors
from torchvision.transforms import _functional_tensor as _FT
from torchvision.transforms.v2 import AutoAugmentPolicy, functional as F, InterpolationMode, Transform
from torchvision.transforms.v2.functional._geometry import _check_interpolation
from torchvision.transforms.v2.functional._meta import get_size
from torchvision.transforms.v2.functional._utils import _FillType, _FillTypeJIT

from torchvision.transforms.v2._utils import _get_fill, _setup_fill_arg, check_type, is_pure_tensor

ImageOrVideo = Union[torch.Tensor, PIL.Image.Image, tv_tensors.Image, tv_tensors.Video]


# Implemented with references to big_vision and https://github.com/pytorch/vision/pull/6609

def _solarize_add(
    image: ImageOrVideo, addition: int = 0, threshold: int = 128
) -> ImageOrVideo:
    bound = _FT._max_value(image.dtype) if isinstance(image, torch.Tensor) else 255
    added_image = image.to(torch.int64) + addition
    added_image = added_image.clip(0, bound).to(torch.uint8)
    return torch.where(image < threshold, added_image, image)


def _cutout(
    image: ImageOrVideo,
    pad_size: int,
    replace: int = 0,
) -> ImageOrVideo:
    _, img_h, img_w = F.get_dimensions(image)

    # Sample the center location in the image where the zero mask will be applied.
    cutout_center_height = int(torch.randint(img_h, ()))
    cutout_center_width = int(torch.randint(img_w, ()))

    lower_pad = max(0, cutout_center_height - pad_size)
    upper_pad = max(0, img_h - cutout_center_height - pad_size)
    left_pad = max(0, cutout_center_width - pad_size)
    right_pad = max(0, img_w - cutout_center_width - pad_size)

    cutout_shape = [img_h - (lower_pad + upper_pad), img_w - (left_pad + right_pad)]
    return F.erase(image, lower_pad, left_pad, cutout_shape[0], cutout_shape[1], torch.tensor(replace).unsqueeze(1).unsqueeze(1))


class RandAugment17(v2.RandAugment):
    def _apply_image_or_video_transform(
        self,
        image: ImageOrVideo,
        transform_id: str,
        magnitude: float,
        interpolation: Union[InterpolationMode, int],
        fill: Dict[Union[Type, str], _FillTypeJIT],
    ) -> ImageOrVideo:
        # Note: this cast is wrong and is only here to make mypy happy (it disagrees with torchscript)
        image = cast(torch.Tensor, image)
        fill_ = _get_fill(fill, type(image))

        if transform_id == "Identity":
            return image
        elif transform_id == "ShearX":
            # magnitude should be arctan(magnitude)
            # official autoaug: (1, level, 0, 0, 1, 0)
            # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
            # compared to
            # torchvision:      (1, tan(level), 0, 0, 1, 0)
            # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
            return F.affine(
                image,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[math.degrees(math.atan(magnitude)), 0.0],
                interpolation=interpolation,
                fill=fill_,
                center=[0, 0],
            )
        elif transform_id == "ShearY":
            # magnitude should be arctan(magnitude)
            # See above
            return F.affine(
                image,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[0.0, math.degrees(math.atan(magnitude))],
                interpolation=interpolation,
                fill=fill_,
                center=[0, 0],
            )
        elif transform_id == "TranslateX":
            return F.affine(
                image,
                angle=0.0,
                translate=[int(magnitude), 0],
                scale=1.0,
                interpolation=interpolation,
                shear=[0.0, 0.0],
                fill=fill_,
            )
        elif transform_id == "TranslateY":
            return F.affine(
                image,
                angle=0.0,
                translate=[0, int(magnitude)],
                scale=1.0,
                interpolation=interpolation,
                shear=[0.0, 0.0],
                fill=fill_,
            )
        elif transform_id == "Rotate":
            return F.rotate(image, angle=magnitude, interpolation=interpolation, fill=fill_)
        elif transform_id == "Brightness":
            return F.adjust_brightness(image, brightness_factor=1.0 + magnitude)
        elif transform_id == "Color":
            return F.adjust_saturation(image, saturation_factor=1.0 + magnitude)
        elif transform_id == "Contrast":
            return F.adjust_contrast(image, contrast_factor=1.0 + magnitude)
        elif transform_id == "Sharpness":
            return F.adjust_sharpness(image, sharpness_factor=1.0 + magnitude)
        elif transform_id == "Posterize":
            return F.posterize(image, bits=int(magnitude))
        elif transform_id == "Solarize":
            bound = _FT._max_value(image.dtype) if isinstance(image, torch.Tensor) else 255.0
            return F.solarize(image, threshold=bound * magnitude)
        elif transform_id == "AutoContrast":
            return F.autocontrast(image)
        elif transform_id == "Equalize":
            return F.equalize(image)
        elif transform_id == "Invert":
            return F.invert(image)
        elif transform_id == "SolarizeAdd":
            return _solarize_add(image, addition=int(magnitude))
        elif transform_id == "Cutout":
            return _cutout(image, pad_size=int(magnitude), replace=fill_)
        else:
            raise ValueError(f"No transform available for {transform_id}")
