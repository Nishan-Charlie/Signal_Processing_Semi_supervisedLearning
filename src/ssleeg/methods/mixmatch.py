"""MixMatch (Berthelot et al., 2019).

Guesses a sharpened label for each unlabeled example by averaging predictions over
K augmentations, then applies MixUp across the combined labeled+unlabeled batch.
The labeled portion is trained with cross-entropy and the unlabeled portion with a
Brier/L2 consistency loss.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ssleeg.methods.base import SSLMethod, consistency_weight
from ssleeg.utils.registry import METHODS


@METHODS.register("mixmatch")
class MixMatch(SSLMethod):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = self.cfg.get("mixup_alpha", 0.75)
        self.temperature = self.cfg.get("temperature", 0.5)
        self.max_weight = self.cfg.get("consistency_weight", 100.0)
        self.rampup = self.cfg.get("rampup_steps", max(1, self.total_steps // 5))

    def _sharpen(self, p: torch.Tensor) -> torch.Tensor:
        pt = p ** (1.0 / self.temperature)
        return pt / pt.sum(dim=1, keepdim=True)

    def compute_loss(self, labeled, unlabeled, step) -> Tuple[torch.Tensor, Dict[str, float]]:
        x_l = labeled["x"].to(self.device)
        y_l = F.one_hot(labeled["y"].to(self.device), self.num_classes).float()

        u1 = unlabeled["weak"].to(self.device)
        u2 = unlabeled["strong"].to(self.device)

        # Guess and sharpen the unlabeled label by averaging over the two views.
        with torch.no_grad():
            q = (F.softmax(self.model(u1), dim=1) + F.softmax(self.model(u2), dim=1)) / 2
            q = self._sharpen(q).detach()

        all_x = torch.cat([x_l, u1, u2], dim=0)
        all_y = torch.cat([y_l, q, q], dim=0)

        # MixUp with lambda biased towards the first element (MixMatch convention).
        lam = float(np.random.beta(self.alpha, self.alpha))
        lam = max(lam, 1 - lam)
        perm = torch.randperm(all_x.size(0), device=self.device)
        mixed_x = lam * all_x + (1 - lam) * all_x[perm]
        mixed_y = lam * all_y + (1 - lam) * all_y[perm]

        logits = self.model(mixed_x)
        n_l = x_l.size(0)
        log_p = F.log_softmax(logits, dim=1)

        sup_loss = -(mixed_y[:n_l] * log_p[:n_l]).sum(dim=1).mean()
        unsup_loss = F.mse_loss(F.softmax(logits[n_l:], dim=1), mixed_y[n_l:])

        w = consistency_weight(step, self.max_weight, self.rampup)
        loss = sup_loss + w * unsup_loss
        return loss, {
            "loss": loss.item(),
            "loss_sup": sup_loss.item(),
            "loss_unsup": unsup_loss.item(),
            "cons_weight": w,
        }
