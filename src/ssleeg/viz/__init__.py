"""Publication-quality visualization."""

from ssleeg.viz.plots import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_learning_curves,
    plot_label_efficiency,
    plot_critical_difference,
    set_publication_style,
)
from ssleeg.viz.embeddings import plot_embeddings

__all__ = [
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_learning_curves",
    "plot_label_efficiency",
    "plot_critical_difference",
    "plot_embeddings",
    "set_publication_style",
]
