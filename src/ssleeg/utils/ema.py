"""Exponential Moving Average of model weights (used by Mean Teacher, FixMatch, BYOL...)."""

from __future__ import annotations

import copy
from typing import Iterator

import torch
import torch.nn as nn


class ModelEMA:
    """Maintains an exponentially-moving-averaged copy of a model's parameters.

    The EMA ("teacher") model is updated after every optimizer step as::

        theta_ema = decay * theta_ema + (1 - decay) * theta_student

    Buffers (e.g. BatchNorm running stats) are copied directly. Optionally applies
    a warmup schedule on the decay so early, noisy weights are tracked faster.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup_steps: int = 0) -> None:
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates = 0
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def _current_decay(self) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        # Ramp the decay from 0 up to the target over the warmup period.
        return min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        d = self._current_decay()
        ema_params = dict(self.ema_model.named_parameters())
        for name, param in model.named_parameters():
            if param.dtype.is_floating_point:
                ema_params[name].mul_(d).add_(param.detach(), alpha=1.0 - d)
        # Copy buffers (non-trainable statistics) directly.
        ema_buffers = dict(self.ema_model.named_buffers())
        for name, buf in model.named_buffers():
            if name in ema_buffers:
                ema_buffers[name].copy_(buf)

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.ema_model.parameters()

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.ema_model.load_state_dict(state_dict)
