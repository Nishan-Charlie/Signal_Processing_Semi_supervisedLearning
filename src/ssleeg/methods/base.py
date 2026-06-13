"""Base class and shared utilities for SSL methods."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ssleeg.models.base import EEGClassifier
from ssleeg.utils.config import Config
from ssleeg.utils.registry import METHODS


def sigmoid_rampup(current: int, rampup_length: int) -> float:
    """Sigmoid ramp-up from 0 to 1 over ``rampup_length`` steps (Laine & Aila, 2017)."""
    if rampup_length <= 0:
        return 1.0
    current = max(0.0, min(current, rampup_length))
    phase = 1.0 - current / rampup_length
    return float(math.exp(-5.0 * phase * phase))


def consistency_weight(step: int, max_weight: float, rampup_length: int) -> float:
    """Ramped consistency / unsupervised loss weight."""
    return max_weight * sigmoid_rampup(step, rampup_length)


def interleave(x: torch.Tensor, size: int) -> torch.Tensor:
    """Interleave labeled and unlabeled batches so BatchNorm statistics are shared
    consistently (MixMatch/FixMatch trick)."""
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


class SSLMethod(nn.Module):
    """Base interface for all semi-supervised methods.

    Args:
        model: the :class:`EEGClassifier` (student).
        cfg: the ``method`` config block.
        num_classes: number of classes.
        device: torch device.
        total_steps: total number of optimization steps (for ramp-up schedules).
    """

    #: whether ``compute_loss`` requires an unlabeled batch
    requires_unlabeled: bool = True

    def __init__(
        self,
        model: EEGClassifier,
        cfg: Config,
        num_classes: int,
        device: torch.device,
        total_steps: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.num_classes = num_classes
        self.device = device
        self.total_steps = total_steps
        self.ce = nn.CrossEntropyLoss()

    # -- core API ------------------------------------------------------------
    def compute_loss(
        self,
        labeled: Dict[str, torch.Tensor],
        unlabeled: Optional[Dict[str, torch.Tensor]],
        step: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:  # pragma: no cover - interface
        """Return ``(total_loss, logs)`` for one optimization step."""
        raise NotImplementedError

    def on_step_end(self, step: int) -> None:
        """Hook called after each optimizer step (e.g. EMA teacher update)."""

    def eval_module(self) -> nn.Module:
        """Module to use at evaluation time (override for EMA-teacher methods)."""
        return self.model

    # -- helpers -------------------------------------------------------------
    def _sup_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = batch["x"].to(self.device), batch["y"].to(self.device)
        logits = self.model(x)
        return self.ce(logits, y), logits


def build_method(
    name: str,
    model: EEGClassifier,
    cfg: Config,
    num_classes: int,
    device: torch.device,
    total_steps: int,
) -> SSLMethod:
    return METHODS.build(name, model=model, cfg=cfg, num_classes=num_classes, device=device, total_steps=total_steps)
