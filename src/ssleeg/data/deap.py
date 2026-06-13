"""DEAP dataset loader.

Expected layout (DEAP "preprocessed python" release)::

    <root>/
        s01.dat  s02.dat  ...  s32.dat

Each ``.dat`` is a pickled dict with ``data`` of shape ``(40, 40, 8064)``
(40 trials x 40 channels x 8064 samples @ 128 Hz; first 32 channels are EEG and
the first 3 s are a pre-trial baseline) and ``labels`` of shape ``(40, 4)``
(valence, arousal, dominance, liking on a 1-9 scale).

Continuous valence/arousal are binarized at a configurable threshold, and the
60 s trials are segmented into fixed-length windows to form classification trials.
"""

from __future__ import annotations

import os
import pickle
from typing import List

import numpy as np

from ssleeg.data.base import EEGArrayDataset
from ssleeg.utils.registry import DATASETS

DEAP_FS = 128
DEAP_EEG_CHANNELS = 32
DEAP_BASELINE_SEC = 3


def _label_for(labels_row: np.ndarray, target: str, threshold: float) -> int:
    """Map a (valence, arousal, dominance, liking) row to a class index."""
    valence, arousal = labels_row[0], labels_row[1]
    if target == "valence":
        return int(valence >= threshold)
    if target == "arousal":
        return int(arousal >= threshold)
    if target == "valence_arousal":  # 4-class quadrant
        return int(valence >= threshold) * 2 + int(arousal >= threshold)
    raise ValueError(f"Unknown DEAP target: {target}")


@DATASETS.register("deap")
def load_deap(
    root: str,
    target: str = "valence",
    threshold: float = 5.0,
    window_sec: float = 4.0,
    overlap: float = 0.5,
    drop_baseline: bool = True,
    subjects: List[int] | None = None,
    **kwargs,
) -> EEGArrayDataset:
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"DEAP root '{root}' not found. Download the 'preprocessed python' "
            f"release and point `data.root` at the folder containing s01.dat ... s32.dat."
        )

    num_classes = 4 if target == "valence_arousal" else 2
    win = int(window_sec * DEAP_FS)
    step = max(1, int(win * (1 - overlap)))
    baseline = DEAP_BASELINE_SEC * DEAP_FS if drop_baseline else 0

    subject_ids = subjects or list(range(1, 33))
    X, y, subj_arr, sess_arr = [], [], [], []

    for sid in subject_ids:
        path = os.path.join(root, f"s{sid:02d}.dat")
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        data = d["data"][:, :DEAP_EEG_CHANNELS, baseline:]  # (40, 32, T)
        labels = d["labels"]
        for trial in range(data.shape[0]):
            sig = data[trial]
            cls = _label_for(labels[trial], target, threshold)
            for start in range(0, sig.shape[1] - win + 1, step):
                X.append(sig[:, start : start + win].astype(np.float32))
                y.append(cls)
                subj_arr.append(sid - 1)
                sess_arr.append(0)  # DEAP has a single session

    if not X:
        raise RuntimeError(f"No DEAP trials loaded from '{root}'. Check the file layout.")

    class_names = (
        ["LV", "HV"]
        if target == "valence"
        else ["LA", "HA"]
        if target == "arousal"
        else ["LVLA", "LVHA", "HVLA", "HVHA"]
    )
    return EEGArrayDataset(
        X=np.stack(X),
        y=np.array(y),
        subjects=np.array(subj_arr),
        sessions=np.array(sess_arr),
        num_classes=num_classes,
        name=f"deap_{target}",
        class_names=class_names,
        meta={"fs": DEAP_FS, "target": target},
    )
