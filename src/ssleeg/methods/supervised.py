"""Supervised baseline -- trains only on the labeled subset (lower bound for SSL)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ssleeg.methods.base import SSLMethod
from ssleeg.utils.registry import METHODS


@METHODS.register("supervised")
class Supervised(SSLMethod):
    requires_unlabeled = False

    def compute_loss(self, labeled, unlabeled, step) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss, logits = self._sup_loss(labeled)
        acc = (logits.argmax(1) == labeled["y"].to(self.device)).float().mean()
        return loss, {"loss": loss.item(), "loss_sup": loss.item(), "train_acc": acc.item()}
