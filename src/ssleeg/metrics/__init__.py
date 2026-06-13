"""Metrics and statistical testing."""

from ssleeg.metrics.classification import compute_metrics, predict_logits
from ssleeg.metrics.statistics import (
    mean_std,
    confidence_interval,
    wilcoxon_test,
    paired_ttest,
    friedman_test,
    nemenyi_posthoc,
)

__all__ = [
    "compute_metrics",
    "predict_logits",
    "mean_std",
    "confidence_interval",
    "wilcoxon_test",
    "paired_ttest",
    "friedman_test",
    "nemenyi_posthoc",
]
