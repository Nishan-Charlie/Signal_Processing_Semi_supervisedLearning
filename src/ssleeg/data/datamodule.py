"""DataModule: turns a config into ready-to-use labeled/unlabeled/val/test loaders.

Responsibilities:
* load the raw dataset via the ``DATASETS`` registry,
* build the label-efficiency split (with no leakage),
* fit normalization statistics on the *labeled+unlabeled training* trials only and
  apply them to every split,
* construct augmentation pipelines and ``DataLoader`` objects with reproducible
  shuffling.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import numpy as np
from torch.utils.data import DataLoader

from ssleeg.data.augment import build_augmentation, build_view
from ssleeg.data.base import EEGTensorDataset, SSLDataset
from ssleeg.data.splits import make_label_efficiency_split
from ssleeg.utils.config import Config
from ssleeg.utils.logging import get_logger
from ssleeg.utils.registry import DATASETS
from ssleeg.utils.seed import make_generator, worker_init_fn


class EEGDataModule:
    def __init__(self, cfg: Config, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self.logger = get_logger()
        self._build()

    def _build(self) -> None:
        d = self.cfg.data
        self.dataset = DATASETS.build(d.name, **d.get("loader", {}))
        self.num_classes = self.dataset.num_classes
        self.num_channels = self.dataset.num_channels
        self.num_timepoints = self.dataset.num_timepoints

        self.split = make_label_efficiency_split(
            self.dataset,
            label_ratio=float(d.label_ratio),
            protocol=d.get("protocol", "random"),
            test_frac=d.get("test_frac", 0.2),
            val_frac=d.get("val_frac", 0.1),
            seed=self.seed,
            test_sessions=d.get("test_sessions", None),
        )
        self.logger.info("Split sizes: %s", self.split.summary())

        # Fit normalization on the union of train trials (labeled+unlabeled), no
        # information from val/test is used.
        norm_mode = d.get("normalize", "channel")
        train_idx = np.concatenate([self.split.labeled, self.split.unlabeled])
        stats = self.dataset.subset(train_idx).normalize(norm_mode)
        # `stats` is computed on the train subset copy; recompute+apply on full set
        # by normalizing each split container with the same stats.
        self._norm_mode = norm_mode
        self._norm_stats = stats or self._fit_stats(train_idx, norm_mode)

        self.labeled_ds = self._make_container(self.split.labeled)
        self.unlabeled_ds = self._make_container(self.split.unlabeled)
        self.val_ds = self._make_container(self.split.val)
        self.test_ds = self._make_container(self.split.test)

        # Augmentations.
        aug_cfg = self.cfg.get("augment", Config({}))
        self.labeled_aug = build_augmentation(aug_cfg.get("labeled", None))
        self.eval_aug = build_augmentation(aug_cfg.get("eval", None))  # usually identity
        self.view = build_view(
            aug_cfg.get("weak", "weak_default"), aug_cfg.get("strong", "strong_default")
        )

    def _fit_stats(self, train_idx, mode):
        if mode != "channel":
            return {}
        X = self.dataset.X[train_idx]
        return {
            "mean": X.mean(axis=(0, 2), keepdims=True),
            "std": X.std(axis=(0, 2), keepdims=True) + 1e-6,
        }

    def _make_container(self, idx):
        sub = self.dataset.subset(idx)
        sub.normalize(self._norm_mode, stats=self._norm_stats or None)
        return sub

    # -- loaders -------------------------------------------------------------
    def _loader(self, dataset, shuffle: bool, batch_size: int, drop_last: bool = False) -> DataLoader:
        l = self.cfg.get("loader", Config({}))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=l.get("num_workers", 0),
            pin_memory=l.get("pin_memory", True),
            drop_last=drop_last,
            persistent_workers=bool(l.get("num_workers", 0)) and l.get("persistent_workers", False),
            worker_init_fn=partial(worker_init_fn, base_seed=self.seed),
            generator=make_generator(self.seed),
        )

    def labeled_loader(self, batch_size: int, drop_last: bool = True) -> DataLoader:
        ds = EEGTensorDataset(self.labeled_ds, transform=self.labeled_aug)
        # Never drop the only (partial) batch when the labeled pool is tiny -- doing
        # so would yield an empty loader (common at low label ratios).
        drop_last = drop_last and len(ds) > batch_size
        return self._loader(ds, shuffle=True, batch_size=batch_size, drop_last=drop_last)

    def unlabeled_loader(self, batch_size: int, drop_last: bool = True) -> Optional[DataLoader]:
        if len(self.unlabeled_ds) == 0:
            return None
        ds = SSLDataset(self.unlabeled_ds, view=self.view)
        drop_last = drop_last and len(ds) > batch_size
        return self._loader(ds, shuffle=True, batch_size=batch_size, drop_last=drop_last)

    def val_loader(self, batch_size: int) -> DataLoader:
        ds = EEGTensorDataset(self.val_ds, transform=self.eval_aug)
        return self._loader(ds, shuffle=False, batch_size=batch_size)

    def test_loader(self, batch_size: int) -> DataLoader:
        ds = EEGTensorDataset(self.test_ds, transform=self.eval_aug)
        return self._loader(ds, shuffle=False, batch_size=batch_size)

    @property
    def num_unlabeled(self) -> int:
        return len(self.unlabeled_ds)
