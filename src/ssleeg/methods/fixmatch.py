"""FixMatch (Sohn et al., 2020).

Pseudo-labels are derived from confident predictions on *weakly* augmented
unlabeled inputs; the model is then trained to match those hard labels on the
*strongly* augmented version of the same inputs. A fixed confidence threshold
gates which unlabeled examples contribute.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ssleeg.methods.base import SSLMethod
from ssleeg.utils.registry import METHODS


@METHODS.register("fixmatch")
class FixMatch(SSLMethod):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = self.cfg.get("threshold", 0.95)
        self.lambda_u = self.cfg.get("lambda_u", 1.0)
        self.temperature = self.cfg.get("temperature", 1.0)

    def compute_loss(self, labeled, unlabeled, step) -> Tuple[torch.Tensor, Dict[str, float]]:
        sup_loss, _ = self._sup_loss(labeled)

        weak = unlabeled["weak"].to(self.device)
        strong = unlabeled["strong"].to(self.device)

        with torch.no_grad():
            probs = F.softmax(self.model(weak) / self.temperature, dim=1)
            max_p, pseudo = probs.max(dim=1)
            mask = (max_p >= self.threshold).float()

        logits_s = self.model(strong)
        unsup = (F.cross_entropy(logits_s, pseudo, reduction="none") * mask).mean()
        loss = sup_loss + self.lambda_u * unsup
        return loss, {
            "loss": loss.item(),
            "loss_sup": sup_loss.item(),
            "loss_unsup": unsup.item(),
            "mask_rate": mask.mean().item(),
        }
