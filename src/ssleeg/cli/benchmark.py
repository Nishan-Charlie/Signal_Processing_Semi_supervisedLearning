"""``ssleeg-benchmark`` -- run a full grid of (dataset x method x backbone x ratio x seed)
and emit publication-ready benchmark tables.

Driven by a benchmark config (see ``configs/benchmark/synthetic_quick.yaml``)::

    base: configs/experiment/base.yaml
    datasets: [synthetic]
    methods:  [supervised, pseudo_label, fixmatch, mean_teacher]
    models:   [eegnet]
    label_ratios: [0.01, 0.05, 0.1, 0.2]
    seeds: [0, 1, 2]
    method_configs:            # optional per-method hyperparameter files
      fixmatch: configs/method/fixmatch.yaml

Example:
    ssleeg-benchmark -c configs/benchmark/synthetic_quick.yaml -o outputs/bench
"""

from __future__ import annotations

import argparse
import itertools
import os
import traceback

from ssleeg.cli.common import make_output_dir
from ssleeg.engine.trainer import Trainer
from ssleeg.reporting.tables import build_benchmark_table, collect_results, to_csv, to_latex, to_markdown
from ssleeg.utils.config import Config, load_config, merge_overrides
from ssleeg.utils.logging import get_logger, setup_logger
from ssleeg.utils.seed import seed_everything

import ssleeg  # noqa: F401  (registry population)


def _merge_method_cfg(cfg: Config, method: str, method_configs: dict) -> Config:
    """Overlay a per-method hyperparameter file (if provided) onto the base config."""
    path = method_configs.get(method) if method_configs else None
    merged = cfg.to_dict()
    merged["method"]["name"] = method
    if path:
        method_block = load_config(path).to_dict()
        merged["method"].update(method_block.get("method", method_block))
    return Config(merged)


def main() -> None:
    p = argparse.ArgumentParser(description="Run an SSL benchmark grid and build tables.")
    p.add_argument("--config", "-c", type=str, required=True, help="Benchmark config YAML.")
    p.add_argument("--output", "-o", type=str, default="outputs/benchmark")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--metrics", nargs="*", default=["accuracy", "balanced_accuracy", "f1"])
    p.add_argument("--skip-existing", action="store_true", help="Skip runs with results.json present.")
    args = p.parse_args()

    setup_logger(args.output)
    logger = get_logger()
    bench = load_config(args.config)
    base_cfg = load_config(bench.base)

    datasets = bench.get("datasets", [base_cfg.data.name])
    methods = bench.get("methods", [base_cfg.method.name])
    models = bench.get("models", [base_cfg.model.name])
    ratios = bench.get("label_ratios", [base_cfg.data.label_ratio])
    seeds = bench.get("seeds", [0])
    method_configs = bench.get("method_configs", {})

    grid = list(itertools.product(datasets, methods, models, ratios, seeds))
    logger.info("Benchmark grid: %d runs", len(grid))

    for i, (dataset, method, model, ratio, seed) in enumerate(grid, 1):
        cfg = _merge_method_cfg(base_cfg, method, method_configs)
        cfg = merge_overrides(
            cfg, [f"data.name={dataset}", f"model.name={model}", f"data.label_ratio={ratio}", f"seed={seed}"]
        )
        out_dir = make_output_dir(args.output, cfg, seed)
        if args.skip_existing and os.path.isfile(os.path.join(out_dir, "results.json")):
            logger.info("[%d/%d] skip existing %s", i, len(grid), out_dir)
            continue
        logger.info("[%d/%d] %s | %s | %s | ratio=%s | seed=%s", i, len(grid), dataset, method, model, ratio, seed)
        try:
            seed_everything(seed, deterministic=cfg.get("deterministic", True))
            Trainer(cfg, output_dir=out_dir, seed=seed, device=args.device).train()
        except Exception:  # keep the grid going if one run fails
            logger.error("Run failed:\n%s", traceback.format_exc())

    _emit_tables(args.output, datasets, args.metrics, logger)


def _emit_tables(root: str, datasets, metrics, logger) -> None:
    try:
        df = collect_results(root)
    except FileNotFoundError:
        logger.warning("No results collected; skipping table generation.")
        return
    df.to_csv(os.path.join(root, "all_results_long.csv"), index=False)
    tables_dir = os.path.join(root, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    md_parts = ["# SSL EEG Benchmark Results\n"]
    for dataset in datasets:
        for metric in metrics:
            sub = df[df["dataset"] == dataset]
            if sub.empty:
                continue
            table = build_benchmark_table(sub, metric=metric, dataset=dataset)
            stem = f"{dataset}_{metric}"
            to_csv(table, os.path.join(tables_dir, f"{stem}.csv"))
            with open(os.path.join(tables_dir, f"{stem}.tex"), "w", encoding="utf-8") as fh:
                fh.write(to_latex(table, caption=f"{metric} on {dataset} (mean ± std %)", label=f"tab:{stem}"))
            md_parts.append(to_markdown(table, title=f"{dataset} - {metric} (%)"))
            md_parts.append("")
    with open(os.path.join(tables_dir, "benchmark.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_parts))
    logger.info("Tables written to %s", tables_dir)
    print("\n".join(md_parts))


if __name__ == "__main__":
    main()
