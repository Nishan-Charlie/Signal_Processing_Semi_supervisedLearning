"""Mean Teacher (Tarvainen & Valpola, 2017).

A teacher network -- an EMA of the student's weights -- produces targets for a
consistency loss against the student's predictions on perturbed inputs. Evaluation
uses the (more stable) teacher.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssleeg.methods.base import SSLMethod, consistency_weight
from ssleeg.utils.ema import ModelEMA
from ssleeg.utils.registry import METHODS


@METHODS.register("mean_teacher")
class MeanTeacher(SSLMethod):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema = ModelEMA(
            self.model,
            decay=self.cfg.get("ema_decay", 0.999),
            warmup_steps=self.cfg.get("ema_warmup", 0),
        )
        self.ema.ema_model.to(self.device)
        self.max_weight = self.cfg.get("consistency_weight", 1.0)
        self.rampup = self.cfg.get("rampup_steps", max(1, self.total_steps // 5))

    def compute_loss(self, labeled, unlabeled, step) -> Tuple[torch.Tensor, Dict[str, float]]:
        sup_loss, _ = self._sup_loss(labeled)

        student_in = unlabeled["strong"].to(self.device)
        teacher_in = unlabeled["weak"].to(self.device)
        student_logits = self.model(student_in)
        with torch.no_grad():
            teacher_logits = self.ema.ema_model(teacher_in)
        # MSE between probability distributions (the canonical MT consistency loss).
        cons_loss = F.mse_loss(
            F.softmax(student_logits, dim=1), F.softmax(teacher_logits, dim=1)
        )

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

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        sd["_ema"] = self.ema.state_dict()
        return sd

    def load_state_dict(self, state_dict, *args, **kwargs):
        ema = state_dict.pop("_ema", None)
        if ema is not None:
            self.ema.load_state_dict(ema)
        return super().load_state_dict(state_dict, *args, **kwargs)
