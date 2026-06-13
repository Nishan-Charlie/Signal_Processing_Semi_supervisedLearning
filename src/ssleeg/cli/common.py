"""Shared CLI helpers: argument parsing, config loading, run-dir naming, listing."""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import ssleeg  # noqa: F401  (triggers registry population)
from ssleeg.utils.config import Config, load_config, merge_overrides, save_config
from ssleeg.utils.registry import DATASETS, METHODS, MODELS


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", "-c", type=str, help="Path to a YAML config.")
    parser.add_argument(
        "--set", dest="overrides", nargs="*", default=[],
        help="Override config fields, e.g. --set method.name=fixmatch optim.lr=1e-3 seed=7",
    )
    parser.add_argument("--output", "-o", type=str, default="outputs", help="Output root directory.")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu | cuda:0 ...")
    parser.add_argument(
        "--list", choices=["datasets", "models", "methods"], default=None,
        help="List available registered components and exit.",
    )


def maybe_list_and_exit(args: argparse.Namespace) -> bool:
    if args.list is None:
        return False
    registry = {"datasets": DATASETS, "models": MODELS, "methods": METHODS}[args.list]
    print(f"Available {args.list}:")
    for name in registry.keys():
        print(f"  - {name}")
    return True


def load_run_config(args: argparse.Namespace) -> Config:
    if not args.config:
        raise SystemExit("A --config is required (or use --list to inspect components).")
    cfg = load_config(args.config)
    if args.overrides:
        cfg = merge_overrides(cfg, args.overrides)
    return cfg


def run_name(cfg: Config, seed: int) -> str:
    return (
        f"{cfg.data.name}/{cfg.method.name}/{cfg.model.name}"
        f"/lr{cfg.data.label_ratio}_seed{seed}"
    )


def make_output_dir(root: str, cfg: Config, seed: int) -> str:
    out = os.path.join(root, run_name(cfg, seed))
    os.makedirs(out, exist_ok=True)
    save_config(cfg, os.path.join(out, "config.yaml"))
    return out


def seeds_from_config(cfg: Config, override: List[int] | None = None) -> List[int]:
    if override:
        return override
    seeds = cfg.get("seeds", None)
    if seeds:
        return list(seeds)
    return [int(cfg.get("seed", 0))]
