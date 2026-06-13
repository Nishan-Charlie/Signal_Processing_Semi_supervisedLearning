"""The Trainer: a method-agnostic SSL training loop.

Features: AMP mixed precision, gradient clipping, per-step LR scheduling with
warmup, EMA-teacher support (via the method), TensorBoard/W&B/JSONL logging,
best-checkpointing on a validation metric, early stopping, and resume.

The loop draws one labeled and one unlabeled mini-batch per step (the unlabeled
batch is ``mu`` times larger, FixMatch convention), cycling the shorter loader.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

import torch

from ssleeg.data.datamodule import EEGDataModule
from ssleeg.engine.evaluator import Evaluator, EvalResult
from ssleeg.engine.optim import build_optimizer, build_scheduler
from ssleeg.methods.base import build_method
from ssleeg.models.base import build_model
from ssleeg.utils.checkpoint import CheckpointManager
from ssleeg.utils.config import Config
from ssleeg.utils.logging import ExperimentLogger, get_logger


class Trainer:
    def __init__(self, cfg: Config, output_dir: str, seed: int, device: Optional[str] = None):
        self.cfg = cfg
        self.seed = seed
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = get_logger()

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.logger.info("Using device: %s", self.device)

        # Data.
        self.dm = EEGDataModule(cfg, seed=seed)

        # Training schedule.
        t = cfg.train
        self.epochs = int(t.epochs)
        self.bs = int(t.batch_size)
        self.mu = int(t.get("mu", 1))  # unlabeled/labeled batch-size ratio
        self.grad_clip = float(t.get("grad_clip", 0.0))
        self.use_amp = bool(t.get("amp", True)) and self.device.type == "cuda"
        self.eval_metric = t.get("eval_metric", "balanced_accuracy")
        self.patience = int(t.get("early_stopping_patience", 0))

        self.labeled_loader = self.dm.labeled_loader(self.bs)
        self.unlabeled_loader = self.dm.unlabeled_loader(self.bs * self.mu)
        self.val_loader = self.dm.val_loader(t.get("eval_batch_size", 256))
        self.test_loader = self.dm.test_loader(t.get("eval_batch_size", 256))

        self.steps_per_epoch = int(t.get("steps_per_epoch", 0)) or self._infer_steps_per_epoch()
        self.total_steps = self.steps_per_epoch * self.epochs

        # Model + method.
        self.model = build_model(
            cfg.model, self.dm.num_channels, self.dm.num_timepoints, self.dm.num_classes
        ).to(self.device)
        self.method = build_method(
            cfg.method.name, self.model, cfg.method, self.dm.num_classes, self.device, self.total_steps
        ).to(self.device)

        # Optimization.
        self.optimizer = build_optimizer(self.model.parameters(), cfg.optim)
        self.scheduler = build_scheduler(self.optimizer, cfg.optim, self.total_steps)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Bookkeeping.
        self.evaluator = Evaluator(self.device, self.dm.num_classes)
        self.ckpt = CheckpointManager(os.path.join(output_dir, "checkpoints"), mode="max")
        self.exp_logger = ExperimentLogger(
            log_dir=output_dir,
            use_tensorboard=t.get("tensorboard", True),
            use_wandb=t.get("wandb", False),
            wandb_project=t.get("wandb_project", "ssleeg"),
            wandb_run_name=os.path.basename(output_dir),
            config=cfg.to_dict(),
        )
        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = -float("inf")
        self.epochs_no_improve = 0
        self.history: list = []

    def _infer_steps_per_epoch(self) -> int:
        n_lab = len(self.labeled_loader)
        n_unlab = len(self.unlabeled_loader) if self.unlabeled_loader is not None else 0
        return max(n_lab, n_unlab, 1)

    # -- training ------------------------------------------------------------
    def _cycle(self, loader):
        if len(loader) == 0:
            raise RuntimeError(
                "A data loader yielded zero batches (likely batch_size > pool size "
                "with drop_last). Reduce train.batch_size or increase data.label_ratio."
            )
        while True:
            for batch in loader:
                yield batch

    def train(self) -> Dict[str, float]:
        self.logger.info(
            "Training '%s' for %d epochs (%d steps/epoch, %d total).",
            self.cfg.method.name, self.epochs, self.steps_per_epoch, self.total_steps,
        )
        lab_iter = self._cycle(self.labeled_loader)
        unlab_iter = (
            self._cycle(self.unlabeled_loader) if self.unlabeled_loader is not None else None
        )
        if self.method.requires_unlabeled and unlab_iter is None:
            self.logger.warning(
                "Method '%s' expects unlabeled data but the unlabeled pool is empty.",
                self.cfg.method.name,
            )

        for epoch in range(self.start_epoch, self.epochs):
            self._train_epoch(epoch, lab_iter, unlab_iter)
            val_result = self.evaluator(self.method.eval_module(), self.val_loader)
            self._log_validation(epoch, val_result)

            metric = val_result.metrics[self.eval_metric]
            improved = metric > self.best_metric
            if improved:
                self.best_metric = metric
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            self._save_checkpoint(epoch, metric)

            if self.patience and self.epochs_no_improve >= self.patience:
                self.logger.info("Early stopping at epoch %d (no improvement).", epoch)
                break

        return self.test()

    def _train_epoch(self, epoch: int, lab_iter, unlab_iter) -> None:
        self.method.train()
        t0 = time.time()
        running: Dict[str, float] = {}
        for _ in range(self.steps_per_epoch):
            labeled = next(lab_iter)
            unlabeled = next(unlab_iter) if unlab_iter is not None else None

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, logs = self.method.compute_loss(labeled, unlabeled, self.global_step)

            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.method.on_step_end(self.global_step)

            logs["lr"] = self.scheduler.get_last_lr()[0]
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v
            if self.global_step % self.cfg.train.get("log_every", 50) == 0:
                self.exp_logger.log_scalars(logs, self.global_step, prefix="train/")
            self.global_step += 1

        avg = {k: v / self.steps_per_epoch for k, v in running.items()}
        self.logger.info(
            "Epoch %d | %s | %.1fs",
            epoch,
            " | ".join(f"{k}={v:.4f}" for k, v in avg.items() if k in {"loss", "loss_sup", "mask_rate"}),
            time.time() - t0,
        )

    def _log_validation(self, epoch: int, result: EvalResult) -> None:
        self.exp_logger.log_scalars(result.metrics, self.global_step, prefix="val/")
        self.history.append({"epoch": epoch, "step": self.global_step, **{f"val_{k}": v for k, v in result.metrics.items()}})
        self.logger.info(
            "Epoch %d | val %s=%.4f acc=%.4f f1=%.4f",
            epoch, self.eval_metric, result.metrics[self.eval_metric],
            result.metrics["accuracy"], result.metrics["f1"],
        )

    # -- checkpointing -------------------------------------------------------
    def _save_checkpoint(self, epoch: int, metric: float) -> None:
        state = {
            "model": self.model.state_dict(),
            "method": self.method.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "epochs_no_improve": self.epochs_no_improve,
            "config": self.cfg.to_dict(),
            "seed": self.seed,
        }
        self.ckpt.save(state, epoch=epoch, metric=metric)

    def resume(self, path: str) -> None:
        state = CheckpointManager.load(path, map_location=str(self.device))
        self.model.load_state_dict(state["model"])
        self.method.load_state_dict(state["method"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.global_step = state["global_step"]
        self.best_metric = state.get("best_metric", -float("inf"))
        self.epochs_no_improve = state.get("epochs_no_improve", 0)
        self.start_epoch = state["epoch"] + 1
        self.ckpt.best_metric = self.best_metric
        self.logger.info("Resumed from %s at epoch %d.", path, self.start_epoch)

    # -- final test ----------------------------------------------------------
    def test(self) -> Dict[str, float]:
        """Load the best checkpoint and evaluate on the held-out test set."""
        best_path = self.ckpt.best_path()
        if os.path.isfile(best_path):
            state = CheckpointManager.load(best_path, map_location=str(self.device))
            self.model.load_state_dict(state["model"])
            self.method.load_state_dict(state["method"])
            self.logger.info("Loaded best checkpoint (val %s=%.4f).", self.eval_metric, state.get("best_metric", float("nan")))

        result = self.evaluator(self.method.eval_module(), self.test_loader)
        self.exp_logger.log_scalars(result.metrics, self.global_step, prefix="test/")
        self.logger.info(
            "TEST | acc=%.4f bacc=%.4f f1=%.4f auc=%.4f",
            result.metrics["accuracy"], result.metrics["balanced_accuracy"],
            result.metrics["f1"], result.metrics["roc_auc"],
        )
        # Persist predictions + metrics for downstream visualization/stats.
        import numpy as np

        np.savez(
            os.path.join(self.output_dir, "test_predictions.npz"),
            probs=result.probs, labels=result.labels, logits=result.logits,
        )
        self._dump_results(result.metrics)
        self.exp_logger.close()
        return result.metrics

    def _dump_results(self, metrics: Dict[str, float]) -> None:
        import json

        payload = {
            "method": self.cfg.method.name,
            "model": self.cfg.model.name,
            "dataset": self.cfg.data.name,
            "label_ratio": self.cfg.data.label_ratio,
            "protocol": self.cfg.data.get("protocol", "random"),
            "seed": self.seed,
            "best_val_metric": self.best_metric,
            "test_metrics": metrics,
            "history": self.history,
        }
        with open(os.path.join(self.output_dir, "results.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
