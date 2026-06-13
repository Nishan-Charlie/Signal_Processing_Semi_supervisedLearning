"""Evaluator: runs inference and computes the full metric suite + raw predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from ssleeg.metrics.classification import compute_metrics, predict_logits


@dataclass
class EvalResult:
    metrics: Dict[str, float]
    probs: np.ndarray
    labels: np.ndarray
    logits: np.ndarray = field(repr=False, default=None)

    @property
    def preds(self) -> np.ndarray:
        return self.probs.argmax(axis=1)


def evaluate_model(model: nn.Module, loader, device: torch.device, num_classes: int) -> EvalResult:
    logits, probs, labels = predict_logits(model, loader, device)
    metrics = compute_metrics(probs, labels, num_classes)
    return EvalResult(metrics=metrics, probs=probs, labels=labels, logits=logits)


class Evaluator:
    """Thin stateful wrapper for repeated evaluation against fixed loaders."""

    def __init__(self, device: torch.device, num_classes: int) -> None:
        self.device = device
        self.num_classes = num_classes

    def __call__(self, model: nn.Module, loader) -> EvalResult:
        return evaluate_model(model, loader, self.device, self.num_classes)
