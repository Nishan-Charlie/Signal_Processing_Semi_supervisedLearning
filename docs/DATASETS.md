# Datasets

Raw EEG emotion datasets are **not bundled** — obtain each from its official source
(most require accepting a license/EULA) and point the loader `root` at it.

| Name | Registry key | Classes | Status | Expected layout |
| --- | --- | --- | --- | --- |
| Synthetic | `synthetic` | configurable | ✅ built-in | none (generated) |
| DEAP | `deap` | 2 / 4 | ✅ loader | `data_preprocessed_python/s01.dat … s32.dat` |
| SEED | `seed` | 3 | ✅ loader | `Preprocessed_EEG/*.mat` + `label.mat` |
| SEED-IV | `seed_iv` | 4 | 🟡 placeholder | — |
| SEED-V | `seed_v` | 5 | 🟡 placeholder | — |
| DREAMER | `dreamer` | 2 (V/A) | 🟡 placeholder | `DREAMER.mat` |
| AMIGOS | `amigos` | 2 (V/A) | 🟡 placeholder | per-subject `.mat` |
| MPED | `mped` | 7 | 🟡 placeholder | — |
| FACED | `faced` | varies | 🟡 placeholder | — |

🟡 placeholders raise an informative `NotImplementedError` pointing at the loader
template — implement one by following `deap.py`/`seed.py` and the
[component guide](ADDING_COMPONENTS.md).

## DEAP

Download the *preprocessed Python* release. Set:

```yaml
data:
  name: deap
  loader:
    root: /path/to/DEAP/data_preprocessed_python
    target: valence       # valence | arousal | valence_arousal (4-class quadrants)
    threshold: 5.0        # binarization threshold on the 1–9 rating
    window_sec: 4.0
    overlap: 0.5
```

32 EEG channels @ 128 Hz; the 3 s pre-trial baseline is dropped by default.

## SEED

Download the SEED release containing `Preprocessed_EEG/`. Set:

```yaml
data:
  name: seed
  loader:
    root: /path/to/SEED
    window_sec: 4.0
    overlap: 0.5
```

62 channels @ 200 Hz, 3 emotion classes (negative/neutral/positive).

## Evaluation protocols

Set `data.protocol`:

* `subject` — leave-subjects-out (cross-subject generalization; the recommended
  protocol for reporting). Test subjects never appear in train/val.
* `session` — train/test on different sessions (set `data.test_sessions: [..]`).
* `random` — stratified random split (optimistic; useful for debugging).

Within the training pool, `data.label_ratio` controls the labeled fraction; the
remainder becomes the unlabeled pool. Normalization statistics are fit on the
training pool only, so there is no leakage into val/test.
