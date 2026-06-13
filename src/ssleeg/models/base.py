"""Backbone interface, classifier/projection heads, and the model builder.

Every backbone is an :class:`EEGBackbone` mapping ``(B, C, T)`` -> a flat feature
vector ``(B, feature_dim)``. The :class:`EEGClassifier` composes a backbone with a
linear classification head and (optionally) an MLP projection head used by
contrastive methods (SimCLR/BYOL/...). Separating *features* from *logits* is what
lets a single backbone be reused across every SSL method unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ssleeg.utils.config import Config
from ssleeg.utils.registry import MODELS


class EEGBackbone(nn.Module):
    """Base class for feature-extracting backbones.

    Subclasses must set ``self.feature_dim`` and implement ``forward`` returning
    ``(B, feature_dim)`` features.
    """

    feature_dim: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class ProjectionHead(nn.Module):
    """MLP projection head (SimCLR-style) for contrastive representation learning."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128, n_layers: int = 2):
        super().__init__()
        layers = []
        dim = in_dim
        for i in range(n_layers - 1):
            layers += [nn.Linear(dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
            dim = hidden_dim
        layers += [nn.Linear(dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EEGClassifier(nn.Module):
    """Backbone + linear classifier (+ optional projection head)."""

    def __init__(
        self,
        backbone: EEGBackbone,
        num_classes: int,
        dropout: float = 0.0,
        projection: Optional[Config] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(backbone.feature_dim, num_classes)
        self.projection_head: Optional[ProjectionHead] = None
        if projection is not None:
            self.projection_head = ProjectionHead(
                backbone.feature_dim,
                hidden_dim=projection.get("hidden_dim", 256),
                out_dim=projection.get("out_dim", 128),
                n_layers=projection.get("n_layers", 2),
            )

    @property
    def feature_dim(self) -> int:
        return self.backbone.feature_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.backbone(x)
        logits = self.classifier(self.dropout(feats))
        if return_features:
            return logits, feats
        return logits

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized projection-head embeddings (for contrastive losses)."""
        assert self.projection_head is not None, "Model was built without a projection head."
        z = self.projection_head(self.backbone(x))
        return nn.functional.normalize(z, dim=1)


def build_model(
    cfg: Config, num_channels: int, num_timepoints: int, num_classes: int
) -> EEGClassifier:
    """Instantiate an :class:`EEGClassifier` from the ``model`` config block."""
    backbone = MODELS.build(
        cfg.name,
        num_channels=num_channels,
        num_timepoints=num_timepoints,
        **cfg.get("args", {}),
    )
    return EEGClassifier(
        backbone,
        num_classes=num_classes,
        dropout=cfg.get("dropout", 0.0),
        projection=cfg.get("projection", None),
    )
