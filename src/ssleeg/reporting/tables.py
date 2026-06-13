"""Aggregate ``results.json`` files into benchmark tables (Markdown / CSV / LaTeX)."""

from __future__ import annotations

import glob
import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd


def collect_results(root: str) -> pd.DataFrame:
    """Recursively gather every ``results.json`` under ``root`` into a long DataFrame."""
    rows: List[dict] = []
    for path in glob.glob(os.path.join(root, "**", "results.json"), recursive=True):
        with open(path, "r", encoding="utf-8") as fh:
            r = json.load(fh)
        base = {
            "method": r.get("method"),
            "model": r.get("model"),
            "dataset": r.get("dataset"),
            "protocol": r.get("protocol"),
            "label_ratio": r.get("label_ratio"),
            "seed": r.get("seed"),
            "run_dir": os.path.dirname(path),
        }
        for metric, value in r.get("test_metrics", {}).items():
            base[metric] = value
        rows.append(base)
    if not rows:
        raise FileNotFoundError(f"No results.json found under '{root}'.")
    return pd.DataFrame(rows)


def _fmt(mean: float, std: float, pct: bool, decimals: int) -> str:
    if pct:
        return f"{mean * 100:.{decimals}f} ± {std * 100:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def build_benchmark_table(
    df: pd.DataFrame,
    metric: str = "accuracy",
    dataset: Optional[str] = None,
    method_order: Optional[List[str]] = None,
    as_percent: bool = True,
    decimals: int = 2,
) -> pd.DataFrame:
    """Pivot to a ``method x label_ratio`` table of ``mean +/- std`` over seeds."""
    if dataset is not None:
        df = df[df["dataset"] == dataset]
    grouped = (
        df.groupby(["method", "label_ratio"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .fillna({"std": 0.0})
    )
    grouped["cell"] = grouped.apply(
        lambda r: _fmt(r["mean"], r["std"], as_percent, decimals), axis=1
    )
    table = grouped.pivot(index="method", columns="label_ratio", values="cell")
    table.columns = [f"{int(c*100)}%" if c < 1 else "Full" for c in table.columns]
    if method_order:
        table = table.reindex([m for m in method_order if m in table.index])
    return table


def to_markdown(table: pd.DataFrame, title: Optional[str] = None) -> str:
    out = f"### {title}\n\n" if title else ""
    return out + table.to_markdown()


def to_csv(table: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    table.to_csv(path)


def to_latex(table: pd.DataFrame, caption: Optional[str] = None, label: Optional[str] = None) -> str:
    return table.to_latex(
        caption=caption,
        label=label,
        column_format="l" + "c" * table.shape[1],
        escape=False,
        na_rep="--",
    )
