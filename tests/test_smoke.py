"""Fast smoke tests that exercise the core pipeline on tiny synthetic data (CPU)."""

import numpy as np
import torch

import ssleeg  # noqa: F401  (populate registries)
from ssleeg.data.base import EEGArrayDataset
from ssleeg.data.splits import make_label_efficiency_split
from ssleeg.models.base import build_model
from ssleeg.utils.config import Config
from ssleeg.utils.registry import DATASETS, METHODS, MODELS


def _tiny_dataset() -> EEGArrayDataset:
    return DATASETS.build(
        "synthetic",
        num_classes=3,
        num_subjects=3,
        num_sessions=1,
        trials_per_class_per_session=10,
        num_channels=8,
        num_timepoints=64,
        seed=0,
    )


def test_registries_populated():
    for key in ["supervised", "fixmatch", "mean_teacher", "your_method"]:
        assert key in METHODS
    for key in ["eegnet", "shallowconvnet", "eeg_conformer"]:
        assert key in MODELS
    assert "synthetic" in DATASETS


def test_label_efficiency_split_no_leakage():
    ds = _tiny_dataset()
    split = make_label_efficiency_split(ds, label_ratio=0.2, protocol="random", seed=0)
    all_idx = np.concatenate([split.labeled, split.unlabeled, split.val, split.test])
    # No index appears in more than one split.
    assert len(all_idx) == len(np.unique(all_idx))
    # Labeled+unlabeled are disjoint from val/test.
    train = set(split.labeled.tolist()) | set(split.unlabeled.tolist())
    assert not (train & set(split.val.tolist()))
    assert not (train & set(split.test.tolist()))


def test_subject_split_is_cross_subject():
    ds = _tiny_dataset()
    split = make_label_efficiency_split(ds, label_ratio=0.5, protocol="subject", seed=0)
    train_subj = set(ds.subjects[np.concatenate([split.labeled, split.unlabeled])].tolist())
    test_subj = set(ds.subjects[split.test].tolist())
    assert not (train_subj & test_subj)


def test_backbones_forward_shapes():
    x = torch.randn(4, 8, 64)
    for name in ["eegnet", "shallowconvnet", "deepconvnet", "cnn_lstm", "eeg_transformer", "eeg_conformer"]:
        model = build_model(Config({"name": name, "args": {}}), 8, 64, 3)
        logits = model(x)
        assert logits.shape == (4, 3)


def test_methods_compute_loss_runs():
    x = torch.randn(8, 8, 64)
    labeled = {"x": x, "y": torch.randint(0, 3, (8,))}
    unlabeled = {"weak": x, "strong": x, "y": torch.randint(0, 3, (8,)), "index": torch.arange(8)}
    device = torch.device("cpu")
    for name in ["supervised", "pi_model", "mean_teacher", "pseudo_label", "fixmatch", "flexmatch", "mixmatch", "ict", "your_method"]:
        model = build_model(Config({"name": "eegnet", "args": {"kernel_length": 16}}), 8, 64, 3)
        method = METHODS.build(name, model=model, cfg=Config({"name": name}), num_classes=3, device=device, total_steps=10)
        loss, logs = method.compute_loss(labeled, unlabeled, step=1)
        assert torch.isfinite(loss), f"{name} produced non-finite loss"
        assert "loss" in logs


def test_simclr_with_projection_head():
    x = torch.randn(8, 8, 64)
    labeled = {"x": x, "y": torch.randint(0, 3, (8,))}
    unlabeled = {"weak": x, "strong": x, "y": torch.randint(0, 3, (8,)), "index": torch.arange(8)}
    model = build_model(
        Config({"name": "eegnet", "args": {"kernel_length": 16}, "projection": {"out_dim": 32}}), 8, 64, 3
    )
    method = METHODS.build("simclr", model=model, cfg=Config({"name": "simclr"}), num_classes=3, device=torch.device("cpu"), total_steps=10)
    loss, logs = method.compute_loss(labeled, unlabeled, step=1)
    assert torch.isfinite(loss)
