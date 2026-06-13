# Extending the framework

Everything is wired through three registries (`DATASETS`, `MODELS`, `METHODS`). To
add a component you write one class/function, decorate it, and reference it by name
from a config — no other file changes.

## Add your own SSL method (the main use case)

1. Open [`src/ssleeg/methods/your_method.py`](../src/ssleeg/methods/your_method.py).
2. Implement `compute_loss(self, labeled, unlabeled, step) -> (loss, logs)`.
   * `labeled`  = `{"x": (B,C,T), "y": (B,)}` (already weakly augmented).
   * `unlabeled` = `{"weak": (B',C,T), "strong": (B',C,T), "index", "y"}`.
     **Do not use `unlabeled["y"]` for training** — it exists only for analysis.
   * `self.model(x)` → logits; `self.model(x, return_features=True)` → `(logits, feats)`;
     `self.model.project(x)` → contrastive embedding (needs a projection head).
   * Helpers: `self._sup_loss(batch)`, `consistency_weight(step, w, rampup)`.
3. (Optional) override `on_step_end(step)` for an EMA teacher, and `eval_module()`
   to evaluate the teacher instead of the student.
4. Add hyperparameters to `configs/method/your_method.yaml` and include
   `your_method` in a benchmark `methods:` list.

Registering a *new* method file instead:

```python
from ssleeg.methods.base import SSLMethod
from ssleeg.utils.registry import METHODS

@METHODS.register("my_cool_method")
class MyCoolMethod(SSLMethod):
    def compute_loss(self, labeled, unlabeled, step):
        ...
        return loss, {"loss": loss.item()}
```

Then import it in `src/ssleeg/methods/__init__.py` so the decorator runs.

## Add a backbone

```python
import torch.nn as nn
from ssleeg.models.base import EEGBackbone
from ssleeg.utils.registry import MODELS

@MODELS.register("my_backbone")
class MyBackbone(EEGBackbone):
    def __init__(self, num_channels, num_timepoints, **kwargs):
        super().__init__()
        ...
        self.feature_dim = 128          # REQUIRED: output feature size
    def forward(self, x):               # x: (B, C, T)
        return feats                    # (B, feature_dim)
```

Import it in `src/ssleeg/models/__init__.py`. Use it via `model.name=my_backbone`.

## Add a dataset

Return an `EEGArrayDataset` (X `(N,C,T)`, y, subjects, sessions, num_classes):

```python
from ssleeg.data.base import EEGArrayDataset
from ssleeg.utils.registry import DATASETS

@DATASETS.register("my_dataset")
def load_my_dataset(root, **kwargs):
    ...
    return EEGArrayDataset(X=X, y=y, subjects=subj, sessions=sess, num_classes=K)
```

Import it in `src/ssleeg/data/__init__.py`. Loader kwargs come from
`data.loader.*` in the config. Splitting, normalization, augmentation, label
efficiency, and loaders are then handled automatically by the `EEGDataModule`.

## Add an augmentation

```python
from ssleeg.data.augment import Augmentation, AUGMENTATIONS

@AUGMENTATIONS.register("my_aug")
class MyAug(Augmentation):
    def __call__(self, x):   # x: (C, T) tensor
        return x_aug
```

Reference it from config: `augment.strong: [{name: my_aug, p: 0.5, ...kwargs}]`.

## Adapting other contrastive methods (BYOL / Barlow Twins / VICReg / SimSiam)

`simclr.py` shows the pattern: use the two unlabeled views, call
`self.model.project(view)`, and define the appropriate loss. BYOL/SimSiam need a
predictor MLP + EMA target (reuse `ModelEMA`); Barlow Twins/VICReg replace NT-Xent
with their covariance-based losses on the projected embeddings.
