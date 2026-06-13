# Experiment Guide — example command for every workflow

All commands assume the package is installed (`pip install -e .`) or that you prefix
with `PYTHONPATH=src python -m ssleeg.cli.<tool>` instead of the console script.

## 0. Inspect available components

```bash
ssleeg-train --list methods
ssleeg-train --list models
ssleeg-train --list datasets
```

## 1. Single training run

```bash
# Synthetic smoke test
ssleeg-train -c configs/experiment/smoke.yaml -o outputs

# DEAP, FixMatch, EEGNet, 10% labels, 3 seeds
ssleeg-train -c configs/experiment/deap_fixmatch.yaml --seeds 0 1 2

# SEED, Mean Teacher, EEG-Conformer
ssleeg-train -c configs/experiment/seed_meanteacher.yaml
```

## 2. Override anything from the CLI

```bash
# Change method, labeled ratio, backbone, LR, threshold in one line
ssleeg-train -c configs/experiment/deap_fixmatch.yaml \
  --set method.name=flexmatch data.label_ratio=0.05 model.name=shallowconvnet \
        optim.lr=5e-4 method.threshold=0.9
```

## 3. Resume training

```bash
ssleeg-train -c configs/experiment/deap_fixmatch.yaml \
  --resume outputs/deap_valence/fixmatch/eegnet/lr0.1_seed0/checkpoints/last.ckpt
```

## 4. Evaluate a checkpoint

```bash
ssleeg-eval -c outputs/synthetic/fixmatch/eegnet/lr0.1_seed0/config.yaml \
  --ckpt outputs/synthetic/fixmatch/eegnet/lr0.1_seed0/checkpoints/best.ckpt
```

## 5. Full benchmark grid + tables

```bash
ssleeg-benchmark -c configs/benchmark/synthetic_quick.yaml -o outputs/bench
ssleeg-benchmark -c configs/benchmark/full_benchmark.yaml -o outputs/full --skip-existing
```

Produces `outputs/bench/tables/{dataset}_{metric}.{csv,tex}` and `benchmark.md`.

## 6. Figures

```bash
# Per-run: confusion matrix, ROC, learning curves, t-SNE/PCA embeddings
ssleeg-visualize --run outputs/synthetic/fixmatch/eegnet/lr0.1_seed0 --embeddings

# Benchmark-level: label-efficiency curves + critical-difference diagram
ssleeg-visualize --benchmark outputs/bench --metric balanced_accuracy
```

## 7. Statistical analysis

```bash
ssleeg-stats -i outputs/bench --metric accuracy --reference your_method
```

Reports per-condition mean±std + 95% CI, paired t-test / Wilcoxon vs the reference
method, and a Friedman + Nemenyi omnibus across methods.

## 8. Ablations

Ablations are just config sweeps. Examples:

```bash
# Confidence threshold ablation (FixMatch)
for thr in 0.7 0.8 0.9 0.95; do
  ssleeg-train -c configs/experiment/deap_fixmatch.yaml --set method.threshold=$thr \
    -o outputs/ablation_threshold
done

# Backbone ablation
for m in eegnet shallowconvnet deepconvnet eeg_conformer; do
  ssleeg-train -c configs/experiment/deap_fixmatch.yaml --set model.name=$m \
    -o outputs/ablation_backbone
done

# Augmentation ablation: swap the strong view
ssleeg-train -c configs/experiment/deap_fixmatch.yaml \
  --set augment.strong='[{name: time_mask, n_masks: 2}, {name: channel_dropout, p: 0.3}]'
```

Then aggregate with `ssleeg-benchmark`/`ssleeg-stats` pointed at the ablation root.
