"""EEGNet: a compact convolutional network for EEG (Lawhern et al., 2018).

Input shape ``(B, C, T)`` is treated as a single-image ``(B, 1, C, T)`` and passed
through a temporal conv, a depthwise spatial conv (per-channel filters), and a
separable conv, followed by global pooling to a feature vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ssleeg.models.base import EEGBackbone
from ssleeg.utils.registry import MODELS


@MODELS.register("eegnet")
class EEGNet(EEGBackbone):
    def __init__(
        self,
        num_channels: int,
        num_timepoints: int,
        F1: int = 8,
        D: int = 2,
        F2: int | None = None,
        kernel_length: int = 64,
        pool1: int = 4,
        pool2: int = 8,
        dropout: float = 0.25,
        **kwargs,
    ):
        super().__init__()
        F2 = F2 or F1 * D

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        # Depthwise spatial convolution across EEG channels.
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(dropout),
        )
        # Separable convolution.
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = F2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = self.pool(x)
        return torch.flatten(x, 1)
