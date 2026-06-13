"""Transformer-based EEG backbones: a patch Transformer and an EEG Conformer.

* ``eeg_transformer`` -- tokenizes the signal into temporal patches with a small
  conv stem, adds positional embeddings and a CLS token, then applies a standard
  Transformer encoder.
* ``eeg_conformer`` -- a convolutional stem (temporal + spatial conv, ala
  ShallowConvNet) feeding a Transformer encoder (Song et al., 2022 "EEG Conformer").
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ssleeg.models.base import EEGBackbone
from ssleeg.utils.registry import MODELS


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


@MODELS.register("eeg_transformer")
class EEGTransformer(EEGBackbone):
    def __init__(
        self,
        num_channels: int,
        num_timepoints: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        patch_size: int = 16,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        # Conv stem maps (C, T) -> (d_model, T/patch_size) patch tokens.
        self.tokenizer = nn.Conv1d(num_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.feature_dim = d_model
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x).permute(0, 2, 1)  # (B, N, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.norm(x[:, 0])  # CLS token


@MODELS.register("eeg_conformer")
class EEGConformer(EEGBackbone):
    def __init__(
        self,
        num_channels: int,
        num_timepoints: int,
        n_filters: int = 40,
        temporal_kernel: int = 25,
        pool_size: int = 75,
        pool_stride: int = 15,
        d_model: int = 40,
        n_heads: int = 5,
        n_layers: int = 3,
        dim_feedforward: int = 160,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        # Clamp the pooling window to the (padded) conv output length for short inputs.
        conv_out = num_timepoints + 2 * (temporal_kernel // 2) - temporal_kernel + 1
        pool_size = max(1, min(pool_size, conv_out))
        pool_stride = max(1, min(pool_stride, pool_size))
        self.conv = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, temporal_kernel), padding=(0, temporal_kernel // 2)),
            nn.Conv2d(n_filters, n_filters, (num_channels, 1)),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
            nn.AvgPool2d((1, pool_size), stride=(1, pool_stride)),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(n_filters, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.feature_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B,1,C,T)
        x = self.conv(x)  # (B, F, 1, T')
        x = x.squeeze(2).permute(0, 2, 1)  # (B, T', F)
        x = self.proj(x)
        x = self.encoder(x)
        return self.norm(x.mean(dim=1))
