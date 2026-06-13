"""Statistical analysis for comparing methods across multiple seeds/datasets.

Provides descriptive statistics (mean +/- std, confidence intervals) and the
significance tests standard in SSL/ML benchmarking:

* paired Wilcoxon signed-rank and paired t-test for pairwise method comparison,
* Friedman test for comparing many methods across many datasets,
* Nemenyi post-hoc with critical-difference ranking after a significant Friedman.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def confidence_interval(values: Sequence[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Two-sided Student-t confidence interval for the mean."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < 2:
        return (float(arr.mean()), float(arr.mean())) if n else (float("nan"), float("nan"))
    mean = arr.mean()
    sem = stats.sem(arr)
    h = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return float(mean - h), float(mean + h)


def wilcoxon_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test (non-parametric)."""
    try:
        stat, p = stats.wilcoxon(a, b)
    except ValueError:  # e.g. all differences zero
        return {"statistic": float("nan"), "p_value": 1.0}
    return {"statistic": float(stat), "p_value": float(p)}


def paired_ttest(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    stat, p = stats.ttest_rel(a, b)
    return {"statistic": float(stat), "p_value": float(p)}


def friedman_test(results: Dict[str, Sequence[float]]) -> Dict[str, float]:
    """Friedman test across methods. ``results`` maps method -> per-dataset scores."""
    arrays = [np.asarray(v, dtype=float) for v in results.values()]
    stat, p = stats.friedmanchisquare(*arrays)
    return {"statistic": float(stat), "p_value": float(p)}


def average_ranks(results: Dict[str, Sequence[float]]) -> Dict[str, float]:
    """Average rank of each method across datasets (rank 1 = best/highest)."""
    methods = list(results.keys())
    matrix = np.array([results[m] for m in methods], dtype=float)  # (methods, datasets)
    # Higher score -> better -> lower rank number.
    ranks = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        order = stats.rankdata(-matrix[:, j])
        ranks[:, j] = order
    return {m: float(ranks[i].mean()) for i, m in enumerate(methods)}


def nemenyi_posthoc(results: Dict[str, Sequence[float]]) -> "np.ndarray":
    """Nemenyi post-hoc p-value matrix (requires scikit-posthocs)."""
    try:
        import scikit_posthocs as sp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Nemenyi post-hoc requires `pip install scikit-posthocs`.") from exc
    methods = list(results.keys())
    matrix = np.array([results[m] for m in methods], dtype=float).T  # (datasets, methods)
    return sp.posthoc_nemenyi_friedman(matrix)


def critical_difference(n_methods: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference for CD diagrams."""
    # q_alpha values for the two-tailed Nemenyi test at alpha=0.05 (Studentized range / sqrt(2)).
    q05 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
        9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
    }
    q = q05.get(n_methods, 3.391)
    return float(q * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_datasets)))
