"""EEG-appropriate data augmentations.

All augmentations operate on tensors of shape ``(C, T)`` (channels x time) and are
composable. Each is registered in the ``AUGMENTATIONS`` registry so augmentation
pipelines can be specified entirely from config. A ``WeakStrongView`` wrapper
produces the (weak, strong) augmented pair required by FixMatch-style methods.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ssleeg.utils.registry import Registry

AUGMENTATIONS: "Registry" = Registry("augmentations")


class Augmentation:
    """Base class. Subclasses implement ``__call__(x) -> x`` on a ``(C, T)`` tensor."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose(Augmentation):
    """Apply a list of augmentations in sequence."""

    def __init__(self, transforms: Sequence[Augmentation]) -> None:
        self.transforms = list(transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"Compose([{inner}])"


class RandomApply(Augmentation):
    """Apply ``transform`` with probability ``p``."""

    def __init__(self, transform: Augmentation, p: float = 0.5) -> None:
        self.transform = transform
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x) if random.random() < self.p else x


# --------------------------------------------------------------------------- #
# Time-domain augmentations
# --------------------------------------------------------------------------- #
@AUGMENTATIONS.register("identity")
class Identity(Augmentation):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


@AUGMENTATIONS.register("gaussian_noise")
class GaussianNoise(Augmentation):
    """Add zero-mean Gaussian noise scaled by the per-sample std."""

    def __init__(self, sigma: float = 0.1, relative: bool = True) -> None:
        self.sigma = sigma
        self.relative = relative

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.std() if self.relative else 1.0
        return x + torch.randn_like(x) * self.sigma * scale


@AUGMENTATIONS.register("jitter")
class Jitter(Augmentation):
    """Per-element jitter (additive uniform noise)."""

    def __init__(self, amount: float = 0.05) -> None:
        self.amount = amount

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + (torch.rand_like(x) * 2 - 1) * self.amount * x.std()


@AUGMENTATIONS.register("scaling")
class Scaling(Augmentation):
    """Multiply the whole trial by a random scalar drawn from N(1, sigma)."""

    def __init__(self, sigma: float = 0.1) -> None:
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        factor = 1.0 + torch.randn(1, device=x.device) * self.sigma
        return x * factor


@AUGMENTATIONS.register("crop_resize")
class CropAndResize(Augmentation):
    """Randomly crop a contiguous window and resize back to the original length."""

    def __init__(self, min_ratio: float = 0.6) -> None:
        self.min_ratio = min_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        c, t = x.shape
        length = int(t * random.uniform(self.min_ratio, 1.0))
        start = random.randint(0, t - length)
        cropped = x[:, start : start + length].unsqueeze(0)
        resized = torch.nn.functional.interpolate(
            cropped, size=t, mode="linear", align_corners=False
        )
        return resized.squeeze(0)


@AUGMENTATIONS.register("time_mask")
class TimeMask(Augmentation):
    """Zero out one or more random time segments (cutout in time)."""

    def __init__(self, n_masks: int = 1, max_width: float = 0.1) -> None:
        self.n_masks = n_masks
        self.max_width = max_width

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        c, t = x.shape
        for _ in range(self.n_masks):
            width = random.randint(1, max(1, int(t * self.max_width)))
            start = random.randint(0, t - width)
            x[:, start : start + width] = 0.0
        return x


@AUGMENTATIONS.register("permutation")
class Permutation(Augmentation):
    """Split the signal into segments and randomly permute their order."""

    def __init__(self, n_segments: int = 5) -> None:
        self.n_segments = n_segments

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        c, t = x.shape
        n = min(self.n_segments, t)
        bounds = np.array_split(np.arange(t), n)
        order = list(range(n))
        random.shuffle(order)
        return torch.cat([x[:, bounds[i]] for i in order], dim=1)


@AUGMENTATIONS.register("time_shift")
class TimeShift(Augmentation):
    """Circularly shift the signal in time."""

    def __init__(self, max_shift: float = 0.1) -> None:
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        shift = random.randint(-int(t * self.max_shift), int(t * self.max_shift))
        return torch.roll(x, shifts=shift, dims=1)


# --------------------------------------------------------------------------- #
# Channel-domain augmentations
# --------------------------------------------------------------------------- #
@AUGMENTATIONS.register("channel_dropout")
class ChannelDropout(Augmentation):
    """Randomly zero entire channels."""

    def __init__(self, p: float = 0.1) -> None:
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mask = (torch.rand(x.shape[0], 1, device=x.device) > self.p).float()
        return x * mask


@AUGMENTATIONS.register("channel_mask")
class ChannelMask(Augmentation):
    """Mask a fixed number of randomly-chosen channels."""

    def __init__(self, n_channels: int = 1) -> None:
        self.n_channels = n_channels

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        c = x.shape[0]
        idx = random.sample(range(c), min(self.n_channels, c))
        x[idx] = 0.0
        return x


# --------------------------------------------------------------------------- #
# Frequency-domain augmentations
# --------------------------------------------------------------------------- #
@AUGMENTATIONS.register("freq_mask")
class FrequencyMask(Augmentation):
    """Mask a random band in the frequency domain via rFFT."""

    def __init__(self, max_width: float = 0.1) -> None:
        self.max_width = max_width

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.fft.rfft(x, dim=1)
        f = spec.shape[1]
        width = random.randint(1, max(1, int(f * self.max_width)))
        start = random.randint(0, f - width)
        spec[:, start : start + width] = 0.0
        return torch.fft.irfft(spec, n=x.shape[1], dim=1)


@AUGMENTATIONS.register("band_perturb")
class BandPerturbation(Augmentation):
    """Randomly scale the magnitude of a random frequency band."""

    def __init__(self, max_width: float = 0.15, scale_range: Tuple[float, float] = (0.7, 1.3)) -> None:
        self.max_width = max_width
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.fft.rfft(x, dim=1)
        f = spec.shape[1]
        width = random.randint(1, max(1, int(f * self.max_width)))
        start = random.randint(0, f - width)
        scale = random.uniform(*self.scale_range)
        spec[:, start : start + width] *= scale
        return torch.fft.irfft(spec, n=x.shape[1], dim=1)


def build_augmentation(spec: Optional[Sequence]) -> Augmentation:
    """Build a ``Compose`` from a config list.

    Each item is either a string name or a ``{name: ..., p: ..., **kwargs}`` dict.
    A top-level ``p`` wraps the augmentation in ``RandomApply``.
    """
    if not spec:
        return Identity()
    transforms: List[Augmentation] = []
    for item in spec:
        if isinstance(item, str):
            transforms.append(AUGMENTATIONS.build(item))
            continue
        item = dict(item)
        name = item.pop("name")
        p = item.pop("p", None)
        aug = AUGMENTATIONS.build(name, **item)
        transforms.append(RandomApply(aug, p) if p is not None else aug)
    return Compose(transforms)


class WeakStrongView:
    """Produce ``(weak, strong)`` augmented views of an input (FixMatch family)."""

    def __init__(self, weak: Augmentation, strong: Augmentation) -> None:
        self.weak = weak
        self.strong = strong

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.weak(x), self.strong(x)


# Convenience presets referenced by configs as ``weak_default`` / ``strong_default``.
def default_weak() -> Augmentation:
    return Compose([RandomApply(GaussianNoise(0.05), p=0.5), RandomApply(TimeShift(0.05), p=0.5)])


def default_strong() -> Augmentation:
    return Compose(
        [
            RandomApply(GaussianNoise(0.2), p=0.7),
            RandomApply(TimeMask(2, 0.15), p=0.7),
            RandomApply(ChannelDropout(0.2), p=0.5),
            RandomApply(FrequencyMask(0.15), p=0.5),
            RandomApply(Scaling(0.2), p=0.5),
        ]
    )


PRESETS: Dict[str, callable] = {"weak_default": default_weak, "strong_default": default_strong}


def build_view(weak_spec, strong_spec) -> WeakStrongView:
    """Build a WeakStrongView from config specs, honouring named presets."""
    weak = PRESETS["weak_default"]() if weak_spec == "weak_default" else build_augmentation(weak_spec)
    strong = (
        PRESETS["strong_default"]()
        if strong_spec == "strong_default"
        else build_augmentation(strong_spec)
    )
    return WeakStrongView(weak, strong)
