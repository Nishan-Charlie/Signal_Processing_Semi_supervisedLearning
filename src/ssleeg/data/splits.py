"""Reproducible dataset splitting for SSL label-efficiency experiments.

Two orthogonal axes are supported:

1. **Evaluation protocol** -- how train/val/test are carved out:
   * ``random``    : stratified random split over all trials.
   * ``subject``   : cross-subject (leave-subjects-out) -- subjects in test never
                     appear in train. The standard generalization protocol for EEG.
   * ``session``   : cross-session -- train on some sessions, test on others.

2. **Label efficiency** -- within the *training* pool, a stratified fraction is kept
   labeled and the rest are treated as unlabeled. Crucially the unlabeled pool is
   drawn only from the training trials, so there is **no leakage** from val/test.

All routines take an explicit ``seed`` and return integer index arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ssleeg.data.base import EEGArrayDataset


@dataclass
class DataSplit:
    """Index arrays defining one experimental split."""

    labeled: np.ndarray
    unlabeled: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def summary(self) -> Dict[str, int]:
        return {
            "labeled": len(self.labeled),
            "unlabeled": len(self.unlabeled),
            "val": len(self.val),
            "test": len(self.test),
        }


def _stratified_indices(y: np.ndarray, idx: np.ndarray, frac: float, rng) -> Tuple[np.ndarray, np.ndarray]:
    """Split ``idx`` into (selected, remaining) keeping class proportions."""
    selected = []
    for cls in np.unique(y[idx]):
        cls_idx = idx[y[idx] == cls]
        rng.shuffle(cls_idx)
        n_sel = max(1, int(round(len(cls_idx) * frac)))
        selected.append(cls_idx[:n_sel])
    selected = np.concatenate(selected) if selected else np.array([], dtype=np.int64)
    remaining = np.setdiff1d(idx, selected, assume_unique=False)
    return selected, remaining


def subject_split(
    ds: EEGArrayDataset, test_frac: float, val_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-subjects-out split: returns (train_idx, val_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    subjects = np.unique(ds.subjects)
    rng.shuffle(subjects)
    n_test = max(1, int(round(len(subjects) * test_frac)))
    n_val = max(1, int(round(len(subjects) * val_frac)))
    test_subj = set(subjects[:n_test])
    val_subj = set(subjects[n_test : n_test + n_val])
    train_idx = np.where(~np.isin(ds.subjects, list(test_subj | val_subj)))[0]
    val_idx = np.where(np.isin(ds.subjects, list(val_subj)))[0]
    test_idx = np.where(np.isin(ds.subjects, list(test_subj)))[0]
    return train_idx, val_idx, test_idx


def session_split(
    ds: EEGArrayDataset, test_sessions: list, val_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-session split: ``test_sessions`` held out; val carved from train."""
    rng = np.random.default_rng(seed)
    test_idx = np.where(np.isin(ds.sessions, test_sessions))[0]
    train_pool = np.where(~np.isin(ds.sessions, test_sessions))[0]
    val_idx, train_idx = _stratified_indices(ds.y, train_pool, val_frac, rng)
    return train_idx, val_idx, test_idx


def random_split(
    ds: EEGArrayDataset, test_frac: float, val_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified random split: returns (train_idx, val_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(ds))
    test_idx, rest = _stratified_indices(ds.y, all_idx, test_frac, rng)
    val_idx, train_idx = _stratified_indices(ds.y, rest, val_frac / (1 - test_frac), rng)
    return train_idx, val_idx, test_idx


def make_label_efficiency_split(
    ds: EEGArrayDataset,
    label_ratio: float,
    protocol: str = "random",
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    seed: int = 0,
    test_sessions: Optional[list] = None,
) -> DataSplit:
    """Build a full SSL split for a given labeled ratio and evaluation protocol.

    Args:
        ds: the dataset container.
        label_ratio: fraction of the *training pool* that remains labeled
            (e.g. 0.01, 0.05, 0.1, ...). The rest become the unlabeled pool.
        protocol: ``"random"`` | ``"subject"`` | ``"session"``.
        test_frac / val_frac: sizes for random/subject protocols.
        seed: RNG seed for reproducibility.
        test_sessions: required for the ``"session"`` protocol.

    Returns:
        A :class:`DataSplit` with labeled / unlabeled / val / test index arrays.
    """
    if protocol == "subject":
        train_idx, val_idx, test_idx = subject_split(ds, test_frac, val_frac, seed)
    elif protocol == "session":
        if not test_sessions:
            raise ValueError("protocol='session' requires test_sessions=[...]")
        train_idx, val_idx, test_idx = session_split(ds, test_sessions, val_frac, seed)
    elif protocol == "random":
        train_idx, val_idx, test_idx = random_split(ds, test_frac, val_frac, seed)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    # Within the training pool only, split labeled vs unlabeled (stratified).
    rng = np.random.default_rng(seed + 1)
    labeled_idx, unlabeled_idx = _stratified_indices(ds.y, train_idx, label_ratio, rng)
    return DataSplit(labeled=labeled_idx, unlabeled=unlabeled_idx, val=val_idx, test=test_idx)
