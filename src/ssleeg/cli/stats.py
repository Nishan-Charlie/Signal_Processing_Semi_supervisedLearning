"""``ssleeg-stats`` -- statistical significance analysis over benchmark results.

Computes, for a chosen metric and condition (dataset + label ratio):
* mean +/- std and 95% confidence intervals per method,
* pairwise paired t-test and Wilcoxon signed-rank tests vs a reference method,
* a Friedman omnibus test + Nemenyi post-hoc across methods (using all conditions).

Example:
    ssleeg-stats -i outputs/bench --metric accuracy --reference your_method
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from ssleeg.metrics.statistics import (
    average_ranks,
    confidence_interval,
    friedman_test,
    mean_std,
    paired_ttest,
    wilcoxon_test,
)
from ssleeg.reporting.tables import collect_results


def main() -> None:
    p = argparse.ArgumentParser(description="Statistical analysis of benchmark results.")
    p.add_argument("--input", "-i", type=str, required=True, help="Benchmark output root.")
    p.add_argument("--metric", type=str, default="accuracy")
    p.add_argument("--reference", type=str, default=None, help="Reference method for pairwise tests.")
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    df = collect_results(args.input)
    metric = args.metric
    report: dict = {"metric": metric, "per_condition": {}, "omnibus": {}}

    print(f"\n=== Descriptive statistics ({metric}) ===")
    for (dataset, ratio), grp in df.groupby(["dataset", "label_ratio"]):
        cond = f"{dataset}@{int(ratio*100)}%"
        print(f"\n[{cond}]")
        per_method = {}
        for method, mgrp in grp.groupby("method"):
            vals = mgrp[metric].values
            mean, std = mean_std(vals)
            lo, hi = confidence_interval(vals)
            per_method[method] = list(map(float, vals))
            print(f"  {method:16s} {mean*100:6.2f} ± {std*100:4.2f}  (95% CI [{lo*100:.2f}, {hi*100:.2f}], n={len(vals)})")

        # Pairwise tests vs reference.
        ref = args.reference
        if ref and ref in per_method:
            print(f"  -- paired tests vs '{ref}' --")
            for method, vals in per_method.items():
                if method == ref or len(vals) != len(per_method[ref]):
                    continue
                tt = paired_ttest(per_method[ref], vals)
                wx = wilcoxon_test(per_method[ref], vals)
                sig = "*" if min(tt["p_value"], wx["p_value"]) < args.alpha else " "
                print(f"   {sig} {ref} vs {method:16s} t-test p={tt['p_value']:.4f}  wilcoxon p={wx['p_value']:.4f}")
        report["per_condition"][cond] = per_method

    # Friedman + ranks across methods over all (dataset, ratio) conditions.
    pivot = df.groupby(["method", "dataset", "label_ratio"])[metric].mean().reset_index()
    conditions = sorted(set(map(tuple, pivot[["dataset", "label_ratio"]].values.tolist())))
    methods = sorted(pivot["method"].unique())
    matrix = {m: [] for m in methods}
    for (ds, lr) in conditions:
        for m in methods:
            cell = pivot[(pivot["method"] == m) & (pivot["dataset"] == ds) & (pivot["label_ratio"] == lr)]
            matrix[m].append(float(cell[metric].mean()) if not cell.empty else np.nan)

    complete = {m: v for m, v in matrix.items() if not any(np.isnan(v))}
    if len(complete) >= 3 and len(conditions) >= 2:
        fr = friedman_test(complete)
        ranks = average_ranks(complete)
        print("\n=== Omnibus across methods (all conditions) ===")
        print(f"  Friedman chi2={fr['statistic']:.3f}, p={fr['p_value']:.4f}")
        print("  Average ranks (lower = better):")
        for m in sorted(ranks, key=lambda k: ranks[k]):
            print(f"    {m:16s} {ranks[m]:.3f}")
        report["omnibus"] = {"friedman": fr, "average_ranks": ranks}
        try:
            from ssleeg.metrics.statistics import nemenyi_posthoc

            nem = nemenyi_posthoc(complete)
            nem.to_csv(os.path.join(args.input, f"nemenyi_{metric}.csv"))
            print(f"  Nemenyi post-hoc matrix -> nemenyi_{metric}.csv")
        except ImportError:
            print("  (install scikit-posthocs for Nemenyi post-hoc)")

    out = os.path.join(args.input, f"stats_{metric}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved statistical report to {out}")


if __name__ == "__main__":
    main()
