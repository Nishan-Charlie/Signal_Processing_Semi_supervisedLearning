"""Matplotlib/seaborn plotting helpers that save publication-quality figures."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np


def set_publication_style() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.frameon": False,
            "pdf.fonttype": 42,  # editable text in vector outputs
            "ps.fonttype": 42,
        }
    )


def _save(fig, path: Optional[str]):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path)
    return fig


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    set_publication_style()
    cm = confusion_matrix(labels, preds)
    if normalize:
        cm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
        xticklabels=class_names or "auto", yticklabels=class_names or "auto",
        cbar=True, ax=ax, square=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return _save(fig, save_path)


def plot_roc_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    title: str = "ROC Curves",
    save_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    set_publication_style()
    num_classes = probs.shape[1]
    fig, ax = plt.subplots(figsize=(5, 4))

    if num_classes == 2:
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
    else:
        y_bin = label_binarize(labels, classes=list(range(num_classes)))
        for c in range(num_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
            name = class_names[c] if class_names else f"class {c}"
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, save_path)


def plot_learning_curves(
    history: List[Dict],
    metrics: Sequence[str] = ("val_accuracy", "val_f1"),
    title: str = "Learning Curves",
    save_path: Optional[str] = None,
):
    """Plot validation metric trajectories from a Trainer ``history`` list."""
    import matplotlib.pyplot as plt

    set_publication_style()
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in metrics:
        if history and m in history[0]:
            ax.plot(epochs, [h[m] for h in history], marker="o", ms=3, label=m)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_title(title)
    ax.legend()
    return _save(fig, save_path)


def plot_label_efficiency(
    results: Dict[str, Dict[float, Sequence[float]]],
    metric_name: str = "accuracy",
    title: str = "Label Efficiency",
    save_path: Optional[str] = None,
):
    """Plot metric vs labeled-ratio for each method (mean +/- std band).

    ``results`` maps method -> {label_ratio -> [scores over seeds]}.
    """
    import matplotlib.pyplot as plt

    set_publication_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method, by_ratio in results.items():
        ratios = sorted(by_ratio)
        means = [np.mean(by_ratio[r]) for r in ratios]
        stds = [np.std(by_ratio[r]) for r in ratios]
        ax.plot([r * 100 for r in ratios], means, marker="o", label=method)
        ax.fill_between(
            [r * 100 for r in ratios],
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.15,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Labeled data (%)")
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.legend(fontsize=9, ncol=2)
    return _save(fig, save_path)


def plot_critical_difference(
    avg_ranks: Dict[str, float], cd: float, title: str = "Critical Difference", save_path=None
):
    """A simple critical-difference (Nemenyi) diagram of average ranks."""
    import matplotlib.pyplot as plt

    set_publication_style()
    methods = sorted(avg_ranks, key=lambda m: avg_ranks[m])
    ranks = [avg_ranks[m] for m in methods]
    fig, ax = plt.subplots(figsize=(7, 1.6 + 0.3 * len(methods)))
    y = np.arange(len(methods))[::-1]
    ax.scatter(ranks, y, zorder=3)
    for yi, m, r in zip(y, methods, ranks):
        ax.text(r, yi + 0.15, f"{m} ({r:.2f})", ha="center", fontsize=9)
    best = min(ranks)
    ax.axvspan(best, best + cd, alpha=0.1, color="green", label=f"CD = {cd:.2f}")
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend()
    return _save(fig, save_path)
