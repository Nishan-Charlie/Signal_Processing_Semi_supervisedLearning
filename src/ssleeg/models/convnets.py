"""ShallowConvNet, DeepConvNet (Schirrmeister et al., 2017) and a CNN-LSTM backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from ssleeg.models.base import EEGBackbone
from ssleeg.utils.registry import MODELS


class _Square(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=1e-6).pow(2)


class _Log(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


@MODELS.register("shallowconvnet")
class ShallowConvNet(EEGBackbone):
    """Shallow ConvNet: temporal conv -> spatial conv -> square -> avgpool -> log."""

    def __init__(
        self,
        num_channels: int,
        num_timepoints: int,
        n_filters: int = 40,
        temporal_kernel: int = 25,
        pool_size: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.temporal = nn.Conv2d(1, n_filters, (1, temporal_kernel), bias=False)
        self.spatial = nn.Conv2d(n_filters, n_filters, (num_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.square = _Square()
        # Adapt the (large, canonical) pooling window to short EEG segments so the
        # backbone never produces a non-positive temporal dimension.
        conv_out = num_timepoints - temporal_kernel + 1
        pool_size = max(1, min(pool_size, conv_out))
        pool_stride = max(1, min(pool_stride, pool_size))
        self.pool = nn.AvgPool2d((1, pool_size), stride=(1, pool_stride))
        self.log = _Log()
        self.drop = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = n_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.bn(self.spatial(x))
        x = self.square(x)
        x = self.pool(x)
        x = self.log(x)
        x = self.drop(x)
        return torch.flatten(self.gap(x), 1)


@MODELS.register("deepconvnet")
class DeepConvNet(EEGBackbone):
    """Deep ConvNet: 4 conv-pool blocks of increasing width."""

    def __init__(
        self,
        num_channels: int,
        num_timepoints: int,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        # Padded temporal convs + ceil-mode pooling keep the temporal dimension
        # positive for short EEG windows (canonical DeepConvNet assumes long inputs).
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), padding=(0, 5), bias=False),
            nn.Conv2d(25, 25, (num_channels, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3), ceil_mode=True),
            nn.Dropout(dropout),
        )

        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, (1, 10), padding=(0, 5), bias=False),
                nn.BatchNorm2d(cout),
                nn.ELU(),
                nn.MaxPool2d((1, 3), stride=(1, 3), ceil_mode=True),
                nn.Dropout(dropout),
            )

        self.block2 = conv_block(25, 50)
        self.block3 = conv_block(50, 100)
        self.block4 = conv_block(100, 200)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 200

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return torch.flatten(self.gap(x), 1)


@MODELS.register("cnn_lstm")
class CNNLSTM(EEGBackbone):
    """Temporal CNN feature extractor followed by an LSTM over time."""

    def __init__(
        self,
        num_channels: int,
        num_timepoints: int,
        cnn_channels: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, cnn_channels, 7, padding=3),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_channels, cnn_channels, 5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            cnn_channels,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.feature_dim = hidden_size * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)  # (B, F, T')
        x = x.permute(0, 2, 1)  # (B, T', F)
        out, _ = self.lstm(x)
        return out.mean(dim=1)  # temporal average pooling
