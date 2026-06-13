"""Base dataset containers and PyTorch ``Dataset`` wrappers for EEG trials.

The framework separates two concerns:

* ``EEGArrayDataset`` -- a lightweight in-memory container of raw arrays
  (``X`` of shape ``(N, C, T)``, integer labels ``y``, and per-trial ``subjects``
  / ``sessions`` metadata) returned by every dataset loader. It owns
  normalization and indexing/subsetting.
* ``EEGTensorDataset`` / ``SSLDataset`` -- thin ``torch.utils.data.Dataset``
  views that apply augmentations and yield dict batches consumed by the trainers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class EEGArrayDataset:
    """In-memory container for an EEG emotion dataset.

    Attributes:
        X: float array of shape ``(N, C, T)`` -- trials x channels x time.
        y: int array of shape ``(N,)`` -- class labels in ``[0, num_classes)``.
        subjects: int array of shape ``(N,)`` -- subject id per trial.
        sessions: int array of shape ``(N,)`` -- session id per trial.
        num_classes: number of emotion classes.
        name: dataset identifier.
        class_names: optional human-readable class labels.
    """

    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    num_classes: int
    name: str = "eeg"
    class_names: Optional[list] = None
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.int64)
        self.subjects = np.asarray(self.subjects, dtype=np.int64)
        self.sessions = np.asarray(self.sessions, dtype=np.int64)
        n = len(self.X)
        assert self.y.shape[0] == n, "X and y length mismatch"
        assert self.subjects.shape[0] == n and self.sessions.shape[0] == n

    # -- shape helpers -------------------------------------------------------
    def __len__(self) -> int:
        return len(self.X)

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def num_timepoints(self) -> int:
        return self.X.shape[2]

    def subset(self, indices: np.ndarray) -> "EEGArrayDataset":
        """Return a new container restricted to ``indices`` (no copy of dtype/meta)."""
        idx = np.asarray(indices, dtype=np.int64)
        return EEGArrayDataset(
            X=self.X[idx],
            y=self.y[idx],
            subjects=self.subjects[idx],
            sessions=self.sessions[idx],
            num_classes=self.num_classes,
            name=self.name,
            class_names=self.class_names,
            meta=dict(self.meta),
        )

    # -- normalization -------------------------------------------------------
    def normalize(
        self, mode: str = "channel", stats: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """Z-score normalize ``X`` in place.

        Args:
            mode: one of ``"channel"`` (per-channel over train), ``"trial"``
                (per-trial), or ``"none"``.
            stats: precomputed ``{"mean", "std"}`` (e.g. fit on train, applied to
                val/test to avoid leakage). If None they are computed here.

        Returns:
            The statistics used, so they can be reused on other splits.
        """
        if mode == "none":
            return {}
        if mode == "trial":
            mean = self.X.mean(axis=2, keepdims=True)
            std = self.X.std(axis=2, keepdims=True) + 1e-6
            self.X = (self.X - mean) / std
            return {}
        if mode == "channel":
            if stats is None:
                mean = self.X.mean(axis=(0, 2), keepdims=True)
                std = self.X.std(axis=(0, 2), keepdims=True) + 1e-6
            else:
                mean, std = stats["mean"], stats["std"]
            self.X = (self.X - mean) / std
            return {"mean": mean, "std": std}
        raise ValueError(f"Unknown normalization mode: {mode}")


class EEGTensorDataset(Dataset):
    """A labeled ``Dataset`` view that applies an optional augmentation.

    Yields ``{"x": (C, T), "y": int, "index": int}``.
    """

    def __init__(self, container: EEGArrayDataset, transform=None, return_index: bool = False):
        self.X = torch.from_numpy(container.X)
        self.y = torch.from_numpy(container.y)
        self.transform = transform
        self.return_index = return_index

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        x = self.X[i]
        if self.transform is not None:
            x = self.transform(x)
        out = {"x": x, "y": self.y[i]}
        if self.return_index:
            out["index"] = torch.tensor(i, dtype=torch.long)
        return out


class SSLDataset(Dataset):
    """An unlabeled ``Dataset`` view producing weak/strong augmented views.

    Yields ``{"weak": (C, T), "strong": (C, T), "index": int, "y": int}``.
    The label ``y`` is included only for *evaluation/oracle analysis*; SSL methods
    must not use it during training.
    """

    def __init__(self, container: EEGArrayDataset, view, return_index: bool = True):
        self.X = torch.from_numpy(container.X)
        self.y = torch.from_numpy(container.y)
        self.view = view
        self.return_index = return_index

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        weak, strong = self.view(self.X[i])
        out = {"weak": weak, "strong": strong, "y": self.y[i]}
        if self.return_index:
            out["index"] = torch.tensor(i, dtype=torch.long)
        return out
