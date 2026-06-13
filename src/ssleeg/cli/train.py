"""``ssleeg-train`` -- train a single SSL method over one or more seeds.

Examples:
    ssleeg-train -c configs/experiment/smoke.yaml
    ssleeg-train -c configs/experiment/deap_fixmatch.yaml --set data.label_ratio=0.05 seed=1
    ssleeg-train -c configs/experiment/smoke.yaml --seeds 0 1 2
    ssleeg-train --list methods
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np

from ssleeg.cli.common import (
    add_common_args,
    load_run_config,
    make_output_dir,
    maybe_list_and_exit,
    seeds_from_config,
)
from ssleeg.engine.trainer import Trainer
from ssleeg.utils.logging import setup_logger
from ssleeg.utils.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train an SSL method on EEG emotion data.")
    add_common_args(p)
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="Override seed list.")
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if maybe_list_and_exit(args):
        return

    cfg = load_run_config(args)
    seeds = seeds_from_config(cfg, args.seeds)

    all_metrics: List[Dict[str, float]] = []
    for seed in seeds:
        seed_everything(seed, deterministic=cfg.get("deterministic", True))
        out_dir = make_output_dir(args.output, cfg, seed)
        setup_logger(out_dir)

        trainer = Trainer(cfg, output_dir=out_dir, seed=seed, device=args.device)
        if args.resume:
            trainer.resume(args.resume)
        metrics = trainer.train()
        all_metrics.append(metrics)

    if len(all_metrics) > 1:
        print("\n=== Aggregate over seeds ===")
        for key in ("accuracy", "balanced_accuracy", "f1", "roc_auc"):
            vals = np.array([m[key] for m in all_metrics])
            print(f"  {key:18s}: {vals.mean() * 100:.2f} ± {vals.std() * 100:.2f}")


if __name__ == "__main__":
    main()
