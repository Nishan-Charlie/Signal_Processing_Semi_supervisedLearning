"""Interpolation Consistency Training (Verma et al., 2019).

Encourages the model's prediction at an interpolation of two unlabeled points to
match the interpolation of the (EMA teacher's) predictions:
    f(mix(u_i, u_j)) ~= mix(f_teacher(u_i), f_teacher(u_j)).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ssleeg.methods.base import SSLMethod, consistency_weight
from ssleeg.utils.ema import ModelEMA
from ssleeg.utils.registry import METHODS


@METHODS.register("ict")
class ICT(SSLMethod):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = self.cfg.get("mixup_alpha", 1.0)
        self.max_weight = self.cfg.get("consistency_weight", 1.0)
        self.rampup = self.cfg.get("rampup_steps", max(1, self.total_steps // 5))
        self.ema = ModelEMA(self.model, decay=self.cfg.get("ema_decay", 0.999))
        self.ema.ema_model.to(self.device)

    def compute_loss(self, labeled, unlabeled, step) -> Tuple[torch.Tensor, Dict[str, float]]:
        sup_loss, _ = self._sup_loss(labeled)

        x_u = unlabeled["weak"].to(self.device)
        lam = float(np.random.beta(self.alpha, self.alpha))
        perm = torch.randperm(x_u.size(0), device=self.device)
        mixed = lam * x_u + (1 - lam) * x_u[perm]

        with torch.no_grad():
            t = F.softmax(self.ema.ema_model(x_u), dim=1)
            t_mixed = lam * t + (1 - lam) * t[perm]
        pred = F.softmax(self.model(mixed), dim=1)
        cons_loss = F.mse_loss(pred, t_mixed)

        w = consistency_weight(step, self.max_weight, self.rampup)
        loss = sup_loss + w * cons_loss
        return loss, {
            "loss": loss.item(),
            "loss_sup": sup_loss.item(),
            "loss_cons": cons_loss.item(),
            "cons_weight": w,
        }

    def on_step_end(self, step: int) -> None:
        self.ema.update(self.model)

    def eval_module(self) -> nn.Module:
        return self.ema.ema_model
