from torchvision.transforms import v2

import math
import warnings
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple, Type, Union

import PIL.Image
import torch

from torchvision import tv_tensors
from torchvision.transforms import _functional_tensor as _FT
from torchvision.transforms.v2 import functional as F, InterpolationMode, Transform
from torchvision.transforms.v2.functional._meta import get_size
from torchvision.transforms.v2.functional._utils import _FillTypeJIT
from torchvision.transforms.v2._utils import _get_fill, _setup_size, query_size

ImageOrVideo = Union[torch.Tensor, PIL.Image.Image, tv_tensors.Image, tv_tensors.Video]


class TwoHotMixUp:
    """This implementation of MixUp returns both targets as class indices instead of
    class probabilities and reshape & mix-up (prefetch_factor * batch_size) samples
    into (prefetch_factor) batches at once. Note that this does mean that (prefetch_factor)
    batches share the same lam(bda) value.
    """

    def __init__(self, alpha: float, prefetch_factor: int, batch_size: int):
        self.prefetch_factor = prefetch_factor
        self.batch_size = batch_size
        self._dist = None
        if alpha:
            self._dist = torch.distributions.Beta(alpha, alpha)

    def __call__(self, images, labels):
        _, *sample_shape = images.shape
        images = images.reshape(self.prefetch_factor, self.batch_size, *sample_shape)
        labels = labels.reshape(self.prefetch_factor, self.batch_size)
        if self._dist:
            lam = self._dist.sample()
            images = images.roll(1, dims=1).mul_(1.0 - lam).add_(images, alpha=lam)
            return images, lam, labels, labels.roll(1, dims=1)
        else:
            return images, 1, labels, labels


class BetaCrop(Transform):
    def __init__(
        self,
        size: Union[int, Sequence[int]],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
        max_attempts: int = 100,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.antialias = antialias
        self.max_attempts = max_attempts
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([beta]))

        self._log_ratio = torch.log(torch.tensor(self.ratio))

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        original_height, original_width = query_size(flat_inputs)
        original_area = original_height * original_width
        in_ratio = float(original_width) / float(original_height)
        min_area = self.scale[0] * original_area
        max_area = self.scale[1] * original_area

        # If the input aspect ratio is out of the range,
        # we can never take the full image.
        if in_ratio > self.ratio[1]:
            max_area = min(max_area, float(original_height ** 2 * self.ratio[1]))
        elif in_ratio < self.ratio[0]:
            max_area = min(max_area, float(original_width ** 2 / self.ratio[0]))

        log_ratio = self._log_ratio
        for _ in range(self.max_attempts):
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(
                    log_ratio[0],  # type: ignore[arg-type]
                    log_ratio[1],  # type: ignore[arg-type]
                )
            ).item()

            area = min_area + (max_area - min_area) * float(self._dist.sample(()))
            height = round((area / aspect_ratio) ** 0.5)
            width = round((area * aspect_ratio) ** 0.5)

            # If the constraints can be satisfied: break out of the loop.
            if 0 < width <= original_width and 0 < height <= original_height and min_area <= area <= max_area:
                i = torch.randint(0, original_height - height + 1, size=(1,)).item()
                j = torch.randint(0, original_width - width + 1, size=(1,)).item()
                break
        else:
            # Fallback to central crop
            if in_ratio < min(self.ratio):
                width = original_width
                height = int(round(width / min(self.ratio)))
            elif in_ratio > max(self.ratio):
                height = original_height
                width = int(round(height * max(self.ratio)))
            else:  # whole image
                width = original_width
                height = original_height
            i = (original_height - height) // 2
            j = (original_width - width) // 2

        return dict(top=i, left=j, height=height, width=width)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(
            F.resized_crop, inpt, **params, size=self.size, interpolation=self.interpolation, antialias=self.antialias
        )


class TFInceptionCrop(Transform):
    """TensorFlow-style Inception crop, i.e. tf.slice() with the bbox returned by
    tf.image.sample_distorted_bounding_box(). Note that get_params() is not supported. 
    """

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
        max_attempts: int = 100,
    ) -> None:
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.antialias = antialias
        self.max_attempts = max_attempts

        self._ratio = torch.tensor(self.ratio)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        """Almost line-by-line translation of the core logic of
        tensorflow/core/kernels/image/sample_distorted_bounding_box_op.cc
        """
        original_height, original_width = query_size(flat_inputs)
        original_area = original_height * original_width
        min_area = self.scale[0] * original_area
        max_area = self.scale[1] * original_area

        ratio = self._ratio
        for _ in range(self.max_attempts):
            aspect_ratio = torch.empty(1).uniform_(
                ratio[0],  # type: ignore[arg-type]
                ratio[1],  # type: ignore[arg-type]
            ).item()

            min_height = round(math.sqrt(min_area / aspect_ratio))
            max_height = round(math.sqrt(max_area / aspect_ratio))

            # TODO(b/140767341): Rewrite the generation logic to be more tolerant
            # of floating point behavior.
            if round(max_height * aspect_ratio) > original_width:
                # We must find the smallest max_height satisfying
                # round(max_height * aspect_ratio) <= original_width:
                EPSILON = 0.0000001
                max_height = int((original_width + 0.5 - EPSILON) / aspect_ratio)
                # If due to some precision issues, we still cannot guarantee
                # round(max_height * aspect_ratio) <= original_width, subtract 1 from
                # max height.
                if round(max_height * aspect_ratio) > original_width:
                    max_height -= 1

            max_height = min(max_height, original_height)
            min_height = min(min_height, max_height)

            # We need to generate a random number in the closed range
            # [min_height, max_height].
            height = torch.randint(min_height, max_height + 1, size=(1,)).item()
            width = round(height * aspect_ratio)

            # Let us not fail if rounding error causes the area to be
            # outside the constraints.
            # Try first with a slightly bigger rectangle first.
            area = width * height
            if area < min_area:
                height += 1
                width = round(height * aspect_ratio)
                area = width * height

            # Let us not fail if rounding error causes the area to be
            # outside the constraints.
            # Try first with a slightly smaller rectangle first.
            if area > max_area:
                height -= 1
                width = round(height * aspect_ratio)
                area = width * height

            # Now, we explored all options to rectify small rounding errors.
            # If the constraints can be satisfied: break out of the loop.
            if 0 < width <= original_width and 0 < height <= original_height and min_area <= area <= max_area:
                i = torch.randint(0, original_height - height + 1, size=(1,)).item()
                j = torch.randint(0, original_width - width + 1, size=(1,)).item()
                break
        else:
            # Fallback to the entire image
            width = original_width
            height = original_height
            i = j = 0

        return dict(top=i, left=j, height=height, width=width)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(
            F.resized_crop, inpt, **params, size=self.size, interpolation=self.interpolation, antialias=self.antialias
        )


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
