"""Benchmark result aggregation and publication-ready table generation."""

from ssleeg.reporting.tables import (
    collect_results,
    build_benchmark_table,
    to_markdown,
    to_latex,
    to_csv,
)

__all__ = ["collect_results", "build_benchmark_table", "to_markdown", "to_latex", "to_csv"]
