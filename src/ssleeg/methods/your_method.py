"""TEMPLATE: your own proposed semi-supervised EEG emotion recognition method.

This is the single file you edit to plug *your* method into the benchmark so it is
compared against every baseline under identical data splits, backbones, seeds,
augmentations, and evaluation -- i.e. a fair comparison by construction.

How to use:
1. Implement ``compute_loss`` below (and optionally ``on_step_end`` / ``eval_module``).
2. Reference it from a config with ``method.name: your_method`` (see
   ``configs/method/your_method.yaml``).
3. Run exactly the same commands as for the baselines.

You have access to:
* ``self.model``        -- the EEGClassifier (``model(x)`` -> logits;
                           ``model(x, return_features=True)`` -> (logits, feats);
                           ``model.project(x)`` -> contrastive embedding if a
                           projection head is configured).
* ``self.num_classes``, ``self.device``, ``self.total_steps``, ``self.cfg``.
* helpers ``self._sup_loss(batch)``, ``consistency_weight(step, w, rampup)``.

The example below is a strong, simple starting point: FixMatch-style confident
pseudo-labeling on strong views + a Mean-Teacher EMA consistency term. Replace the
body with your contribution.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssleeg.methods.base import SSLMethod, consistency_weight
from ssleeg.utils.ema import ModelEMA
from ssleeg.utils.registry import METHODS


@METHODS.register("your_method")
class YourMethod(SSLMethod):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = self.cfg.get("threshold", 0.95)
        self.lambda_u = self.cfg.get("lambda_u", 1.0)
        self.lambda_cons = self.cfg.get("lambda_cons", 1.0)
        self.rampup = self.cfg.get("rampup_steps", max(1, self.total_steps // 5))
        self.ema = ModelEMA(self.model, decay=self.cfg.get("ema_decay", 0.999))
        self.ema.ema_model.to(self.device)

    def compute_loss(self, labeled, unlabeled, step) -> Tuple[torch.Tensor, Dict[str, float]]:
        # --- supervised term -------------------------------------------------
        sup_loss, _ = self._sup_loss(labeled)

        weak = unlabeled["weak"].to(self.device)
        strong = unlabeled["strong"].to(self.device)

        # --- confident pseudo-labeling on strong views (FixMatch-like) -------
        with torch.no_grad():
            t_probs = F.softmax(self.ema.ema_model(weak), dim=1)
            max_p, pseudo = t_probs.max(dim=1)
            mask = (max_p >= self.threshold).float()
        logits_s = self.model(strong)
        pl_loss = (F.cross_entropy(logits_s, pseudo, reduction="none") * mask).mean()

        # --- teacher-student consistency (Mean-Teacher-like) ----------------
        with torch.no_grad():
            t_soft = F.softmax(self.ema.ema_model(weak), dim=1)
        cons_loss = F.mse_loss(F.softmax(logits_s, dim=1), t_soft)

        w = consistency_weight(step, self.lambda_cons, self.rampup)
        loss = sup_loss + self.lambda_u * pl_loss + w * cons_loss
        return loss, {
            "loss": loss.item(),
            "loss_sup": sup_loss.item(),
            "loss_pl": pl_loss.item(),
            "loss_cons": cons_loss.item(),
            "mask_rate": mask.mean().item(),
        }

    def on_step_end(self, step: int) -> None:
        self.ema.update(self.model)

    def eval_module(self) -> nn.Module:
        return self.ema.ema_model
