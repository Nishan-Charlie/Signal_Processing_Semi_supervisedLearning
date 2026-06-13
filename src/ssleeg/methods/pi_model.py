"""Pi-Model (Laine & Aila, 2017): consistency between two stochastic augmentations.

The unsupervised loss penalizes the squared difference between the softmax outputs
of two differently-augmented views of the same unlabeled input. The weight is
ramped up with a sigmoid schedule.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ssleeg.methods.base import SSLMethod, consistency_weight
from ssleeg.utils.registry import METHODS


@METHODS.register("pi_model")
class PiModel(SSLMethod):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_weight = self.cfg.get("consistency_weight", 1.0)
        self.rampup = self.cfg.get("rampup_steps", max(1, self.total_steps // 5))

    def compute_loss(self, labeled, unlabeled, step) -> Tuple[torch.Tensor, Dict[str, float]]:
        sup_loss, _ = self._sup_loss(labeled)

        # Two augmented views (weak/strong serve as two stochastic perturbations).
        v1 = unlabeled["weak"].to(self.device)
        v2 = unlabeled["strong"].to(self.device)
        p1 = F.softmax(self.model(v1), dim=1)
        p2 = F.softmax(self.model(v2), dim=1)
        cons_loss = F.mse_loss(p1, p2)

        w = consistency_weight(step, self.max_weight, self.rampup)
        loss = sup_loss + w * cons_loss
        return loss, {
            "loss": loss.item(),
            "loss_sup": sup_loss.item(),
            "loss_cons": cons_loss.item(),
            "cons_weight": w,
        }
