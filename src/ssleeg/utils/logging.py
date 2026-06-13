"""Logging and experiment tracking.

Provides a console+file Python logger and an ``ExperimentLogger`` that fans out
scalar metrics to TensorBoard, an optional Weights & Biases run, and a JSONL file
for offline parsing -- all behind a single ``log_scalars`` call.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, Optional

_LOGGER_NAME = "ssleeg"


def setup_logger(log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the root ``ssleeg`` logger with a console and optional file handler."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


class ExperimentLogger:
    """Unified scalar logger: TensorBoard + optional W&B + JSONL.

    Args:
        log_dir: Directory for TensorBoard event files and the JSONL log.
        use_tensorboard: Enable TensorBoard logging.
        use_wandb: Enable Weights & Biases (requires ``wandb`` installed).
        wandb_project / wandb_run_name / config: W&B initialisation arguments.
    """

    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self._jsonl_path = os.path.join(log_dir, "metrics.jsonl")
        self._logger = get_logger()

        self.tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=log_dir)
            except Exception as exc:  # pragma: no cover - optional dep
                self._logger.warning("TensorBoard unavailable (%s); disabling.", exc)

        self.wandb = None
        if use_wandb:
            try:
                import wandb

                wandb.init(
                    project=wandb_project or "ssleeg",
                    name=wandb_run_name,
                    dir=log_dir,
                    config=config or {},
                )
                self.wandb = wandb
            except Exception as exc:  # pragma: no cover - optional dep
                self._logger.warning("W&B unavailable (%s); disabling.", exc)

    def log_scalars(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        tag_metrics = {f"{prefix}{k}" if prefix else k: float(v) for k, v in metrics.items()}
        if self.tb is not None:
            for tag, value in tag_metrics.items():
                self.tb.add_scalar(tag, value, step)
        if self.wandb is not None:
            self.wandb.log(tag_metrics, step=step)
        with open(self._jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": step, **tag_metrics}) + "\n")

    def log_figure(self, tag: str, figure, step: int = 0) -> None:
        if self.tb is not None:
            self.tb.add_figure(tag, figure, step)
        if self.wandb is not None:
            self.wandb.log({tag: self.wandb.Image(figure)}, step=step)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()
        if self.wandb is not None:
            self.wandb.finish()
