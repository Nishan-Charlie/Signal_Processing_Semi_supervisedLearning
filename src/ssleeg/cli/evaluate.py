"""``ssleeg-eval`` -- evaluate a trained checkpoint on the test split.

Example:
    ssleeg-eval -c outputs/synthetic/fixmatch/eegnet/lr0.1_seed0/config.yaml \
                --ckpt outputs/synthetic/fixmatch/eegnet/lr0.1_seed0/checkpoints/best.ckpt
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from ssleeg.cli.common import add_common_args, load_run_config, maybe_list_and_exit
from ssleeg.data.datamodule import EEGDataModule
from ssleeg.engine.evaluator import evaluate_model
from ssleeg.methods.base import build_method
from ssleeg.models.base import build_model
from ssleeg.utils.checkpoint import CheckpointManager
from ssleeg.utils.logging import setup_logger
from ssleeg.utils.seed import seed_everything


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained SSL checkpoint.")
    add_common_args(p)
    p.add_argument("--ckpt", type=str, required=False, help="Checkpoint path (defaults to best.ckpt).")
    args = p.parse_args()
    if maybe_list_and_exit(args):
        return

    cfg = load_run_config(args)
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    setup_logger()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dm = EEGDataModule(cfg, seed=seed)
    model = build_model(cfg.model, dm.num_channels, dm.num_timepoints, dm.num_classes).to(device)
    method = build_method(cfg.method.name, model, cfg.method, dm.num_classes, device, total_steps=1).to(device)

    ckpt_path = args.ckpt
    if ckpt_path is None:
        raise SystemExit("--ckpt is required.")
    state = CheckpointManager.load(ckpt_path, map_location=str(device))
    model.load_state_dict(state["model"])
    method.load_state_dict(state["method"])

    result = evaluate_model(method.eval_module(), dm.test_loader(256), device, dm.num_classes)
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
