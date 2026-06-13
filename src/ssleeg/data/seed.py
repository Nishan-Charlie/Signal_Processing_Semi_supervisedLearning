"""SEED-family dataset loaders (SEED, SEED-IV, SEED-V) and registry placeholders.

SEED (raw "Preprocessed_EEG" release) layout::

    <root>/
        Preprocessed_EEG/
            1_20131027.mat  1_20131030.mat  ...   (subject_session.mat)
            label.mat                              (per-trial emotion labels)

Each subject/session ``.mat`` holds 15 trial variables (``*_eeg1`` ... ``*_eeg15``)
of shape ``(62 channels, time)`` at 200 Hz. ``label.mat`` provides the 15 labels
(``-1/0/1`` -> negative/neutral/positive, remapped to ``0/1/2``).

Other SEED variants and additional datasets (DREAMER, AMIGOS, MPED, FACED) are
registered as informative placeholders that point at this template -- implement a
loader returning an :class:`EEGArrayDataset` and register it the same way.
"""

from __future__ import annotations

import glob
import os
from typing import List

import numpy as np

from ssleeg.data.base import EEGArrayDataset
from ssleeg.utils.registry import DATASETS

SEED_FS = 200


def _segment(sig: np.ndarray, win: int, step: int) -> List[np.ndarray]:
    return [sig[:, s : s + win] for s in range(0, sig.shape[1] - win + 1, step)]


@DATASETS.register("seed")
def load_seed(
    root: str,
    window_sec: float = 4.0,
    overlap: float = 0.5,
    subjects: List[int] | None = None,
    **kwargs,
) -> EEGArrayDataset:
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Loading SEED requires scipy (`pip install scipy`).") from exc

    eeg_dir = os.path.join(root, "Preprocessed_EEG")
    eeg_dir = eeg_dir if os.path.isdir(eeg_dir) else root
    label_path = os.path.join(eeg_dir, "label.mat")
    if not os.path.isfile(label_path):
        raise FileNotFoundError(
            f"SEED label.mat not found under '{eeg_dir}'. Point `data.root` at the "
            f"SEED release containing Preprocessed_EEG/."
        )
    labels = loadmat(label_path)["label"].flatten()  # 15 trial labels in {-1,0,1}
    labels = (labels + 1).astype(int)  # -> {0,1,2}

    win = int(window_sec * SEED_FS)
    step = max(1, int(win * (1 - overlap)))

    files = sorted(glob.glob(os.path.join(eeg_dir, "*.mat")))
    files = [f for f in files if "label" not in os.path.basename(f).lower()]

    X, y, subj_arr, sess_arr = [], [], [], []
    subj_to_id: dict = {}
    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        subj_key = base.split("_")[0]
        if subjects is not None and int(subj_key) not in subjects:
            continue
        sid = subj_to_id.setdefault(subj_key, len(subj_to_id))
        mat = loadmat(path)
        trial_keys = sorted(
            [k for k in mat if not k.startswith("__")],
            key=lambda s: int("".join(c for c in s if c.isdigit()) or 0),
        )
        for trial_idx, key in enumerate(trial_keys):
            sig = np.asarray(mat[key], dtype=np.float32)  # (62, time)
            for seg in _segment(sig, win, step):
                X.append(seg)
                y.append(int(labels[trial_idx % len(labels)]))
                subj_arr.append(sid)
                sess_arr.append(0)

    if not X:
        raise RuntimeError(f"No SEED trials loaded from '{eeg_dir}'.")

    return EEGArrayDataset(
        X=np.stack(X),
        y=np.array(y),
        subjects=np.array(subj_arr),
        sessions=np.array(sess_arr),
        num_classes=3,
        name="seed",
        class_names=["negative", "neutral", "positive"],
        meta={"fs": SEED_FS},
    )


def _make_placeholder(dataset_name: str, hint: str):
    def _loader(root: str = "", **kwargs):
        raise NotImplementedError(
            f"The '{dataset_name}' loader is a placeholder. {hint}\n"
            f"To add it, implement a function returning an EEGArrayDataset and "
            f"register it with @DATASETS.register('{dataset_name}') -- see "
            f"ssleeg/data/seed.py and deap.py for working templates."
        )

    return _loader


# Register informative placeholders so these names appear in `--list datasets`.
for _name, _hint in {
    "seed_iv": "SEED-IV has 4 classes and 3 sessions per subject.",
    "seed_v": "SEED-V has 5 classes (movie-elicited).",
    "dreamer": "DREAMER is distributed as a single DREAMER.mat (23 subjects, ECG+EEG).",
    "amigos": "AMIGOS provides preprocessed per-subject .mat files.",
    "mped": "MPED has 7 emotion categories.",
    "faced": "FACED is a large (123-subject) affective EEG dataset.",
}.items():
    DATASETS.register(_name)(_make_placeholder(_name, _hint))
