"""Optimizer and learning-rate scheduler builders, configured from the ``optim`` block."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim.lr_scheduler import LambdaLR

from ssleeg.utils.config import Config


def build_optimizer(params: Iterable, cfg: Config) -> torch.optim.Optimizer:
    name = cfg.get("optimizer", "adamw").lower()
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 5e-4))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=cfg.get("betas", (0.9, 0.999)))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, weight_decay=wd, momentum=cfg.get("momentum", 0.9), nesterov=True
        )
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, total_steps: int) -> LambdaLR:
    """Build a per-step LR scheduler with linear warmup."""
    name = cfg.get("scheduler", "cosine").lower()
    warmup = int(cfg.get("warmup_steps", 0))

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, max(0.0, progress))
        if name == "cosine":
            # FixMatch-style cosine decay (to ~0).
            return max(0.0, math.cos(7.0 / 16.0 * math.pi * progress))
        if name == "linear":
            return 1.0 - progress
        if name == "constant":
            return 1.0
        raise ValueError(f"Unknown scheduler: {name}")

    return LambdaLR(optimizer, lr_lambda)
