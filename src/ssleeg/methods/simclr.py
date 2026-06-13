"""SimCLR-style contrastive regularization adapted for EEG SSL.

This implements the NT-Xent (normalized temperature-scaled cross-entropy) loss on
two augmented views of unlabeled trials, trained *jointly* with the supervised loss
on the labeled subset. It requires the model to be built with a projection head
(``model.projection: {...}`` in the config). For a pure pretrain->finetune
protocol, set ``lambda_sup: 0`` for a warmup phase, or run a supervised stage after.

The same NT-Xent machinery is the basis for adapting other representation-learning
methods (BYOL, Barlow Twins, VICReg, SimSiam) -- see the README for guidance.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ssleeg.methods.base import SSLMethod
from ssleeg.utils.registry import METHODS


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent contrastive loss for a batch of paired embeddings (already L2-normed)."""
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim = z @ z.t() / temperature  # (2N, 2N)
    # Mask self-similarities.
    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))
    # Positive pairs: i <-> i+n.
    targets = torch.arange(2 * n, device=z.device)
    targets = (targets + n) % (2 * n)
    return F.cross_entropy(sim, targets)


@METHODS.register("simclr")
class SimCLR(SSLMethod):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.model.projection_head is not None, (
            "SimCLR requires a projection head: add `projection: {out_dim: 128}` "
            "to the model config block."
        )
        self.temperature = self.cfg.get("temperature", 0.5)
        self.lambda_contrast = self.cfg.get("lambda_contrast", 1.0)
        self.lambda_sup = self.cfg.get("lambda_sup", 1.0)

    def compute_loss(self, labeled, unlabeled, step) -> Tuple[torch.Tensor, Dict[str, float]]:
        sup_loss, _ = self._sup_loss(labeled)

        v1 = unlabeled["weak"].to(self.device)
        v2 = unlabeled["strong"].to(self.device)
        z1 = self.model.project(v1)
        z2 = self.model.project(v2)
        contrast = nt_xent(z1, z2, self.temperature)

        loss = self.lambda_sup * sup_loss + self.lambda_contrast * contrast
        return loss, {
            "loss": loss.item(),
            "loss_sup": sup_loss.item(),
            "loss_contrast": contrast.item(),
        }
