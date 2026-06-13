"""A synthetic EEG emotion dataset for fast, dependency-free end-to-end testing.

This loader fabricates class-conditional oscillatory signals (distinct frequency
bands + spatial channel patterns per emotion class, plus subject-specific shifts)
so the *entire* training/evaluation/benchmarking pipeline can be exercised without
downloading any real data. It is the default in the smoke-test config.

It is registered as ``synthetic`` and mirrors the API of the real loaders.
"""

from __future__ import annotations

import numpy as np

from ssleeg.data.base import EEGArrayDataset
from ssleeg.utils.registry import DATASETS


@DATASETS.register("synthetic")
def load_synthetic(
    root: str = "",
    num_classes: int = 4,
    num_subjects: int = 8,
    num_sessions: int = 2,
    trials_per_class_per_session: int = 40,
    num_channels: int = 32,
    num_timepoints: int = 256,
    fs: float = 128.0,
    noise: float = 0.6,
    seed: int = 0,
    **kwargs,
) -> EEGArrayDataset:
    """Generate a class-separable synthetic EEG dataset.

    Each class is associated with a base frequency and a random spatial topography;
    each subject adds a small frequency/topography offset to emulate inter-subject
    variability (the core challenge SSL aims to mitigate).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(num_timepoints) / fs

    class_freqs = np.linspace(6.0, 30.0, num_classes)  # theta..beta-ish bands
    class_topo = rng.normal(size=(num_classes, num_channels))
    subj_freq_shift = rng.normal(scale=1.0, size=num_subjects)
    subj_topo_shift = rng.normal(scale=0.3, size=(num_subjects, num_channels))

    X, y, subjects, sessions = [], [], [], []
    for subj in range(num_subjects):
        for sess in range(num_sessions):
            for cls in range(num_classes):
                for _ in range(trials_per_class_per_session):
                    freq = class_freqs[cls] + subj_freq_shift[subj] + rng.normal(scale=0.3)
                    topo = class_topo[cls] + subj_topo_shift[subj]
                    phase = rng.uniform(0, 2 * np.pi)
                    base = np.sin(2 * np.pi * freq * t + phase)  # (T,)
                    # Outer product: per-channel amplitude modulation of the oscillation.
                    trial = np.outer(topo, base)  # (C, T)
                    trial += rng.normal(scale=noise, size=trial.shape)
                    X.append(trial.astype(np.float32))
                    y.append(cls)
                    subjects.append(subj)
                    sessions.append(sess)

    return EEGArrayDataset(
        X=np.stack(X),
        y=np.array(y),
        subjects=np.array(subjects),
        sessions=np.array(sessions),
        num_classes=num_classes,
        name="synthetic",
        class_names=[f"emotion_{i}" for i in range(num_classes)],
        meta={"fs": fs, "synthetic": True},
    )
