"""``ssleeg-visualize`` -- generate publication figures from runs / benchmarks.

Per-run figures (confusion matrix, ROC, learning curves, optional t-SNE/UMAP):
    ssleeg-visualize --run outputs/synthetic/fixmatch/eegnet/lr0.1_seed0 --embeddings

Benchmark figures (label-efficiency curves, critical-difference diagram):
    ssleeg-visualize --benchmark outputs/bench --metric accuracy
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

import ssleeg  # noqa: F401
from ssleeg.metrics.statistics import average_ranks, critical_difference
from ssleeg.reporting.tables import collect_results
from ssleeg.utils.config import load_config
from ssleeg.viz.embeddings import extract_features, plot_embeddings
from ssleeg.viz.plots import (
    plot_confusion_matrix,
    plot_critical_difference,
    plot_label_efficiency,
    plot_learning_curves,
    plot_roc_curves,
)


def _visualize_run(run_dir: str, do_embeddings: bool, device: str | None) -> None:
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    preds_path = os.path.join(run_dir, "test_predictions.npz")
    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    class_names = None

    if os.path.isfile(preds_path):
        data = np.load(preds_path)
        probs, labels = data["probs"], data["labels"]
        preds = probs.argmax(axis=1)
        plot_confusion_matrix(labels, preds, class_names, save_path=os.path.join(fig_dir, "confusion_matrix.png"))
        plot_roc_curves(labels, probs, class_names, save_path=os.path.join(fig_dir, "roc_curves.png"))
        print(f"Saved confusion matrix + ROC to {fig_dir}")

    results_path = os.path.join(run_dir, "results.json")
    if os.path.isfile(results_path):
        with open(results_path, "r", encoding="utf-8") as fh:
            history = json.load(fh).get("history", [])
        if history:
            plot_learning_curves(history, save_path=os.path.join(fig_dir, "learning_curves.png"))
            print(f"Saved learning curves to {fig_dir}")

    if do_embeddings:
        _embed_run(run_dir, cfg, fig_dir, device)


def _embed_run(run_dir, cfg, fig_dir, device) -> None:
    from ssleeg.data.datamodule import EEGDataModule
    from ssleeg.methods.base import build_method
    from ssleeg.models.base import build_model
    from ssleeg.utils.checkpoint import CheckpointManager

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed = int(cfg.get("seed", 0))
    dm = EEGDataModule(cfg, seed=seed)
    model = build_model(cfg.model, dm.num_channels, dm.num_timepoints, dm.num_classes).to(dev)
    method = build_method(cfg.method.name, model, cfg.method, dm.num_classes, dev, total_steps=1).to(dev)
    best = os.path.join(run_dir, "checkpoints", "best.ckpt")
    if os.path.isfile(best):
        state = CheckpointManager.load(best, map_location=str(dev))
        model.load_state_dict(state["model"])
        method.load_state_dict(state["method"])
    feats, labels = extract_features(method.eval_module(), dm.test_loader(256), dev)
    for m in ("tsne", "pca"):
        plot_embeddings(feats, labels, method=m, seed=seed, save_path=os.path.join(fig_dir, f"embed_{m}.png"))
    print(f"Saved embeddings to {fig_dir}")


def _visualize_benchmark(root: str, metric: str) -> None:
    df = collect_results(root)
    fig_dir = os.path.join(root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        results = defaultdict(lambda: defaultdict(list))
        for _, r in sub.iterrows():
            results[r["method"]][r["label_ratio"]].append(r[metric])
        plot_label_efficiency(
            {m: dict(v) for m, v in results.items()},
            metric_name=metric,
            title=f"Label efficiency on {dataset}",
            save_path=os.path.join(fig_dir, f"label_efficiency_{dataset}_{metric}.png"),
        )

    # Critical-difference diagram across methods, treating (dataset,ratio) as "datasets".
    pivot = df.groupby(["method", "dataset", "label_ratio"])[metric].mean().reset_index()
    methods = sorted(pivot["method"].unique())
    conditions = sorted(set(map(tuple, pivot[["dataset", "label_ratio"]].values.tolist())))
    matrix = {m: [] for m in methods}
    for (ds, lr) in conditions:
        for m in methods:
            cell = pivot[(pivot["method"] == m) & (pivot["dataset"] == ds) & (pivot["label_ratio"] == lr)]
            matrix[m].append(float(cell[metric].mean()) if not cell.empty else np.nan)
    if len(methods) >= 2 and len(conditions) >= 2:
        ranks = average_ranks(matrix)
        cd = critical_difference(len(methods), len(conditions))
        plot_critical_difference(ranks, cd, save_path=os.path.join(fig_dir, f"critical_difference_{metric}.png"))
    print(f"Saved benchmark figures to {fig_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate figures from runs/benchmarks.")
    p.add_argument("--run", type=str, default=None, help="A single run directory.")
    p.add_argument("--benchmark", type=str, default=None, help="A benchmark output root.")
    p.add_argument("--embeddings", action="store_true", help="Also compute t-SNE/PCA embeddings.")
    p.add_argument("--metric", type=str, default="accuracy")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    if args.run:
        _visualize_run(args.run, args.embeddings, args.device)
    if args.benchmark:
        _visualize_benchmark(args.benchmark, args.metric)
    if not args.run and not args.benchmark:
        raise SystemExit("Provide --run <dir> and/or --benchmark <root>.")


if __name__ == "__main__":
    main()
