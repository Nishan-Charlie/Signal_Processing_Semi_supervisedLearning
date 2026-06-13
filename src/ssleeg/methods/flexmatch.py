"""FlexMatch (Zhang et al., 2021): curriculum pseudo-labeling on top of FixMatch.

Instead of a single global threshold, FlexMatch maintains a per-class threshold
that adapts to the model's *learning status* for each class: classes the model is
less confident about get lower thresholds, so their (rarer) confident samples are
not discarded early. Learning status is estimated from the running count of
confident, threshold-passing unlabeled samples per class.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ssleeg.methods.base import SSLMethod
from ssleeg.utils.registry import METHODS


@METHODS.register("flexmatch")
class FlexMatch(SSLMethod):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = self.cfg.get("threshold", 0.95)
        self.lambda_u = self.cfg.get("lambda_u", 1.0)
        self.temperature = self.cfg.get("temperature", 1.0)
        self.thresh_warmup = self.cfg.get("thresh_warmup", True)
        # Running per-class count of samples currently assigned (selected) to each class.
        self.register_buffer("class_counts", torch.zeros(self.num_classes, device=self.device))

    def _flex_thresholds(self) -> torch.Tensor:
        counts = self.class_counts
        if self.thresh_warmup:
            # Normalize by max(max_count, #unselected) so early training warms up smoothly.
            denom = torch.clamp(counts.max(), min=1.0)
        else:
            denom = torch.clamp(counts.max(), min=1.0)
        beta = counts / denom  # learning effect per class in [0, 1]
        # Convex warmup mapping (the "non-linear" mapping from the paper).
        beta = beta / (2.0 - beta)
        return self.threshold * beta

    def compute_loss(self, labeled, unlabeled, step) -> Tuple[torch.Tensor, Dict[str, float]]:
        sup_loss, _ = self._sup_loss(labeled)

        weak = unlabeled["weak"].to(self.device)
        strong = unlabeled["strong"].to(self.device)

        with torch.no_grad():
            probs = F.softmax(self.model(weak) / self.temperature, dim=1)
            max_p, pseudo = probs.max(dim=1)
            class_thresh = self._flex_thresholds()
            mask = (max_p >= class_thresh[pseudo]).float()
            # Update per-class selection counts (decayed) for next step's thresholds.
            for c in range(self.num_classes):
                self.class_counts[c] = 0.9 * self.class_counts[c] + 0.1 * (
                    (pseudo == c).float() * mask
                ).sum()

        logits_s = self.model(strong)
        unsup = (F.cross_entropy(logits_s, pseudo, reduction="none") * mask).mean()
        loss = sup_loss + self.lambda_u * unsup
        return loss, {
            "loss": loss.item(),
            "loss_sup": sup_loss.item(),
            "loss_unsup": unsup.item(),
            "mask_rate": mask.mean().item(),
            "min_class_thresh": self._flex_thresholds().min().item(),
        }
