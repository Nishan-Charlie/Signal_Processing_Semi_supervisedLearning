"""Pseudo-Labeling / Self-Training with confidence thresholding (Lee, 2013).

Hard pseudo-labels are generated from the model's own (weakly-augmented)
predictions; only predictions above a confidence threshold contribute to the
unsupervised cross-entropy loss.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ssleeg.methods.base import SSLMethod, consistency_weight
from ssleeg.utils.registry import METHODS


@METHODS.register("pseudo_label")
class PseudoLabel(SSLMethod):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = self.cfg.get("threshold", 0.95)
        self.max_weight = self.cfg.get("consistency_weight", 1.0)
        self.rampup = self.cfg.get("rampup_steps", max(1, self.total_steps // 5))

    def compute_loss(self, labeled, unlabeled, step) -> Tuple[torch.Tensor, Dict[str, float]]:
        sup_loss, _ = self._sup_loss(labeled)

        x_u = unlabeled["weak"].to(self.device)
        logits_u = self.model(x_u)
        with torch.no_grad():
            probs = F.softmax(logits_u, dim=1)
            max_p, pseudo = probs.max(dim=1)
            mask = (max_p >= self.threshold).float()

        unsup = (F.cross_entropy(logits_u, pseudo, reduction="none") * mask).mean()
        w = consistency_weight(step, self.max_weight, self.rampup)
        loss = sup_loss + w * unsup
        return loss, {
            "loss": loss.item(),
            "loss_sup": sup_loss.item(),
            "loss_unsup": unsup.item(),
            "mask_rate": mask.mean().item(),
            "cons_weight": w,
        }
