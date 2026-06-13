"""Checkpoint management: save/load full training state with best-model tracking."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from ssleeg.utils.logging import get_logger


class CheckpointManager:
    """Save and restore full training state, tracking the best model by a metric.

    A checkpoint bundles model weights, optimizer/scheduler/scaler state, the EMA
    model (if any), the current epoch/step and the best metric so that training can
    be resumed bit-for-bit.
    """

    def __init__(self, ckpt_dir: str, mode: str = "max", keep_last: int = 1) -> None:
        assert mode in {"max", "min"}
        self.ckpt_dir = ckpt_dir
        self.mode = mode
        self.keep_last = keep_last
        self.best_metric: Optional[float] = None
        os.makedirs(ckpt_dir, exist_ok=True)
        self._logger = get_logger()

    def _is_better(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        return metric > self.best_metric if self.mode == "max" else metric < self.best_metric

    def save(self, state: Dict[str, Any], epoch: int, metric: Optional[float] = None) -> str:
        """Save ``last.ckpt`` and, if ``metric`` improved, ``best.ckpt``."""
        state = {**state, "epoch": epoch, "best_metric": self.best_metric}
        last_path = os.path.join(self.ckpt_dir, "last.ckpt")
        torch.save(state, last_path)

        if metric is not None and self._is_better(metric):
            self.best_metric = metric
            state["best_metric"] = metric
            best_path = os.path.join(self.ckpt_dir, "best.ckpt")
            torch.save(state, best_path)
            self._logger.info("New best checkpoint (metric=%.4f) -> %s", metric, best_path)
        return last_path

    @staticmethod
    def load(path: str, map_location: str = "cpu") -> Dict[str, Any]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, map_location=map_location, weights_only=False)

    def best_path(self) -> str:
        return os.path.join(self.ckpt_dir, "best.ckpt")

    def last_path(self) -> str:
        return os.path.join(self.ckpt_dir, "last.ckpt")
