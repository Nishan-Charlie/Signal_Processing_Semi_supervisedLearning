"""Feature-embedding visualization via t-SNE / UMAP / PCA."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def extract_features(model: nn.Module, loader, device) -> tuple:
    """Collect backbone features and labels over a loader."""
    model.eval()
    feats, labels = [], []
    for batch in loader:
        x = batch["x"].to(device)
        if hasattr(model, "forward_features"):
            f = model.forward_features(x)
        else:
            _, f = model(x, return_features=True)
        feats.append(f.cpu().numpy())
        labels.append(batch["y"].numpy())
    return np.concatenate(feats), np.concatenate(labels)


def _reduce(features: np.ndarray, method: str, seed: int) -> np.ndarray:
    method = method.lower()
    if method == "tsne":
        from sklearn.manifold import TSNE

        perp = min(30, max(5, len(features) // 4))
        return TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed).fit_transform(features)
    if method == "umap":
        try:
            import umap
        except ImportError as exc:  # pragma: no cover
            raise ImportError("UMAP embedding requires `pip install umap-learn`.") from exc
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(features)
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(features)
    raise ValueError(f"Unknown embedding method: {method}")


def plot_embeddings(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = "tsne",
    class_names: Optional[Sequence[str]] = None,
    seed: int = 0,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    from ssleeg.viz.plots import set_publication_style, _save

    set_publication_style()
    emb = _reduce(features, method, seed)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    palette = sns.color_palette("tab10", n_colors=len(np.unique(labels)))
    for i, c in enumerate(np.unique(labels)):
        m = labels == c
        name = class_names[c] if class_names else f"class {c}"
        ax.scatter(emb[m, 0], emb[m, 1], s=12, alpha=0.7, color=palette[i], label=name)
    ax.set_title(title or f"{method.upper()} of learned features")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=9, markerscale=1.5)
    return _save(fig, save_path)
