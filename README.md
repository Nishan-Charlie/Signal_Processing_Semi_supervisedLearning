# ssleeg — Semi-Supervised Learning Benchmark for EEG Emotion Recognition

The framework ships with a **synthetic EEG dataset** so the _entire_ pipeline
(train → evaluate → benchmark → tables → figures → statistics) runs end-to-end in
under a minute, with **no data downloads required**.

---

## ✨ Features

| Area                 | What's included                                                                                                                                     |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SSL methods**      | Supervised baseline, Π-Model, Mean Teacher, VAT, ICT, Pseudo-Label, MixMatch, FixMatch, FlexMatch, SimCLR (contrastive), + a `your_method` template |
| **Backbones**        | EEGNet, ShallowConvNet, DeepConvNet, CNN-LSTM, EEG-Transformer, EEG-Conformer                                                                       |
| **Datasets**         | Synthetic (built-in), DEAP, SEED (+ informative placeholders for SEED-IV/V, DREAMER, AMIGOS, MPED, FACED)                                           |
| **Protocols**        | Cross-subject, cross-session, random — all with stratified label-efficiency splits and**no leakage**                                                |
| **Label efficiency** | Arbitrary labeled ratios (1%, 2%, 5%, 10%, 20%, 30%, 50%, …)                                                                                        |
| **Augmentations**    | Time/frequency/channel domain, MixUp, weak/strong views for FixMatch-family                                                                         |
| **Training**         | AMP mixed precision, grad clipping, LR warmup+cosine, EMA teachers, early stopping, checkpoint/resume                                               |
| **Tracking**         | TensorBoard + optional W&B + JSONL                                                                                                                  |
| **Metrics**          | Accuracy, Balanced Acc, Precision/Recall/F1, Cohen's κ, ROC-AUC                                                                                     |
| **Statistics**       | Mean±std, CIs, paired t-test, Wilcoxon, Friedman + Nemenyi, critical-difference diagrams                                                            |
| **Reporting**        | Auto-generated benchmark tables in Markdown / CSV / LaTeX                                                                                           |
| **Figures**          | Confusion matrices, ROC, learning curves, label-efficiency curves, t-SNE/UMAP/PCA embeddings                                                        |

---

## 📦 Installation

```bash
# (recommended) create an environment with PyTorch matching your CUDA version first
pip install -e .            # core install
pip install -e ".[viz,eeg,wandb]"   # optional: UMAP, MNE loaders, W&B
```

This exposes the console scripts `ssleeg-train`, `ssleeg-eval`, `ssleeg-benchmark`,
`ssleeg-visualize`, `ssleeg-stats`. (You can equally run `python -m ssleeg.cli.<x>`
with `PYTHONPATH=src` without installing.)

---

## 📓 Notebook

Prefer an interactive walkthrough? Open
[`notebooks/ssleeg_quickstart.ipynb`](notebooks/ssleeg_quickstart.ipynb) — it runs
the full pipeline (data → train → evaluate → figures → benchmark → tables → stats →
plug in your method) on synthetic data, CPU-friendly, top-to-bottom in a couple of
minutes.

## 🚀 Quickstart (no data needed)

```bash
# 1. Smoke test: train FixMatch on synthetic EEG (~1 min, CPU-friendly)
ssleeg-train -c configs/experiment/smoke.yaml -o outputs

# 2. Run a small benchmark grid (methods × ratios × seeds) and emit tables
ssleeg-benchmark -c configs/benchmark/synthetic_quick.yaml -o outputs/bench

# 3. Generate figures for one run + the benchmark
ssleeg-visualize --run outputs/synthetic/fixmatch/eegnet/lr0.1_seed0 --embeddings
ssleeg-visualize --benchmark outputs/bench --metric accuracy

# 4. Statistical significance analysis (vs your method)
ssleeg-stats -i outputs/bench --metric accuracy --reference your_method
```

Inspect what's available at any time:

```bash
ssleeg-train --list methods
ssleeg-train --list models
ssleeg-train --list datasets
```

---

## 🗂️ Project structure

```
configs/                  YAML configs (base + method + experiment + benchmark)
src/ssleeg/
  utils/      registry, seeding, config, logging, checkpoint, EMA
  data/       datasets (synthetic/DEAP/SEED), augmentations, splits, datamodule
  models/     EEGNet, ConvNets, Transformer/Conformer + classifier wrapper
  methods/    all SSL methods + base interface + your_method template
  engine/     trainer, evaluator, optimizer/scheduler builders
  metrics/    classification metrics + statistical tests
  viz/        plots + embedding visualization
  reporting/  benchmark table generation (md/csv/latex)
  cli/        train / evaluate / benchmark / visualize / stats entry points
tests/                    pytest smoke tests
```

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for an example command for every
experiment and [docs/ADDING_COMPONENTS.md](docs/ADDING_COMPONENTS.md) for how to add
datasets, backbones, and **your own method**.

---

## 🔧 Configuration system

Configs are YAML with `_base_` inheritance and dot-path CLI overrides:

```bash
ssleeg-train -c configs/experiment/deap_fixmatch.yaml \
    --set data.label_ratio=0.05 method.threshold=0.9 optim.lr=5e-4 \
    --seeds 0 1 2
```

Every field in [configs/base.yaml](configs/base.yaml) is documented inline.

---

## 🧪 Plugging in _your_ method (fair comparison)

1. Edit [`src/ssleeg/methods/your_method.py`](src/ssleeg/methods/your_method.py) —
   implement `compute_loss(labeled, unlabeled, step)` (a strong FixMatch+MeanTeacher
   starting point is provided).
2. Add it to a benchmark config's `methods:` list (already included in
   `synthetic_quick.yaml`).
3. Run the **same** benchmark/stats commands as the baselines. Because the data
   splits, backbone, seeds, and evaluation are shared, the comparison is fair by
   construction.

---

## 📥 Using real datasets

Download the dataset and point the loader `root` at it (see each loader's docstring
and [docs/DATASETS.md](docs/DATASETS.md)). For DEAP:

```bash
ssleeg-train -c configs/experiment/deap_fixmatch.yaml \
    --set data.loader.root=/path/to/DEAP/data_preprocessed_python
```

Raw datasets must be obtained from their official sources (license/EULA required);
they are **not** bundled.

---

## ♻️ Reproducibility

- Global seeding of Python/NumPy/PyTorch + deterministic cuDNN (`deterministic: true`).
- Seeded DataLoader workers and generators.
- The exact resolved config is saved to each run directory (`config.yaml`).
- Multi-seed runs report mean ± std and confidence intervals.

---

## 📊 Output layout

```
outputs/<dataset>/<method>/<model>/lr<ratio>_seed<k>/
  config.yaml            resolved config
  train.log              full log
  metrics.jsonl          per-step/epoch scalars
  results.json           final test metrics + val history
  test_predictions.npz   probs/labels/logits for figures & stats
  checkpoints/{best,last}.ckpt
  figures/               generated plots
events.out.tfevents...   TensorBoard
```

## License

MIT (framework code). Datasets retain their original licenses.
