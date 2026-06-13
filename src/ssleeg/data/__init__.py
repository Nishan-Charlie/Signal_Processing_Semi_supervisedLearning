"""Data subpackage: datasets, EEG augmentations, and label-efficiency splitting.

Importing this module registers all dataset classes in ``DATASETS``.
"""

from ssleeg.data.base import EEGArrayDataset, EEGTensorDataset, SSLDataset
from ssleeg.data import synthetic, deap, seed  # noqa: F401  (registers datasets)
from ssleeg.data.augment import (
    Augmentation,
    Compose,
    build_augmentation,
    WeakStrongView,
    AUGMENTATIONS,
)
from ssleeg.data.splits import (
    make_label_efficiency_split,
    subject_split,
    session_split,
)
from ssleeg.data.datamodule import EEGDataModule

__all__ = [
    "EEGArrayDataset",
    "EEGTensorDataset",
    "SSLDataset",
    "Augmentation",
    "Compose",
    "build_augmentation",
    "WeakStrongView",
    "AUGMENTATIONS",
    "make_label_efficiency_split",
    "subject_split",
    "session_split",
    "EEGDataModule",
]
