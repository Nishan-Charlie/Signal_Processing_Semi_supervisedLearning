"""Classification metrics computed from logits/labels."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@torch.no_grad()
def predict_logits(
    model: nn.Module, loader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference, returning ``(logits, probs, labels)`` as numpy arrays."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        x = batch["x"].to(device)
        logits = model(x)
        all_logits.append(logits.detach().cpu())
        all_labels.append(batch["y"])
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    return logits, probs, labels


def compute_metrics(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Compute the full suite of classification metrics."""
    preds = probs.argmax(axis=1)
    avg = "binary" if num_classes == 2 else "macro"

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average=avg, zero_division=0)),
        "recall": float(recall_score(labels, preds, average=avg, zero_division=0)),
        "f1": float(f1_score(labels, preds, average=avg, zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(labels, preds)),
    }

    # ROC-AUC where well-defined (needs >=2 classes present in labels).
    try:
        if num_classes == 2:
            metrics["roc_auc"] = float(roc_auc_score(labels, probs[:, 1]))
        else:
            metrics["roc_auc"] = float(
                roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            )
    except (ValueError, IndexError):
        metrics["roc_auc"] = float("nan")

    return metrics
