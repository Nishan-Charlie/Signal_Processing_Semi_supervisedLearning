"""Virtual Adversarial Training (Miyato et al., 2018).

Computes the adversarial perturbation r_adv that most changes the model's output
distribution (via one power-iteration step) and penalizes the KL divergence
between predictions on the clean and adversarially-perturbed inputs. Also adds an
optional conditional-entropy minimization term.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ssleeg.methods.base import SSLMethod, consistency_weight
from ssleeg.utils.registry import METHODS


def _l2_normalize(d: torch.Tensor) -> torch.Tensor:
    d = d / (d.flatten(1).norm(dim=1).view(-1, *([1] * (d.dim() - 1))) + 1e-8)
    return d


@METHODS.register("vat")
class VAT(SSLMethod):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xi = self.cfg.get("xi", 1e-6)
        self.eps = self.cfg.get("epsilon", 1.0)
        self.n_power = self.cfg.get("n_power_iterations", 1)
        self.max_weight = self.cfg.get("consistency_weight", 1.0)
        self.ent_weight = self.cfg.get("entropy_weight", 0.0)
        self.rampup = self.cfg.get("rampup_steps", max(1, self.total_steps // 5))

    def _vat_loss(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pred = F.softmax(self.model(x), dim=1)

        d = _l2_normalize(torch.randn_like(x))
        for _ in range(self.n_power):
            d.requires_grad_(True)
            pred_hat = self.model(x + self.xi * d)
            adv_dist = F.kl_div(F.log_softmax(pred_hat, dim=1), pred, reduction="batchmean")
            (grad,) = torch.autograd.grad(adv_dist, d)
            d = _l2_normalize(grad.detach())

        r_adv = d * self.eps
        pred_hat = self.model(x + r_adv)
        return F.kl_div(F.log_softmax(pred_hat, dim=1), pred, reduction="batchmean")

    def compute_loss(self, labeled, unlabeled, step) -> Tuple[torch.Tensor, Dict[str, float]]:
        sup_loss, _ = self._sup_loss(labeled)

        x_u = unlabeled["weak"].to(self.device)
        vat_loss = self._vat_loss(x_u)

        ent_loss = torch.tensor(0.0, device=self.device)
        if self.ent_weight > 0:
            probs = F.softmax(self.model(x_u), dim=1)
            ent_loss = -(probs * torch.log(probs + 1e-8)).sum(1).mean()

        w = consistency_weight(step, self.max_weight, self.rampup)
        loss = sup_loss + w * vat_loss + self.ent_weight * ent_loss
        return loss, {
            "loss": loss.item(),
            "loss_sup": sup_loss.item(),
            "loss_vat": vat_loss.item(),
            "loss_ent": float(ent_loss),
            "cons_weight": w,
        }
