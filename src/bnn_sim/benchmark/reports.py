"""Compact reports for dataframe-first benchmark rows."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from bnn_sim.benchmark.diagnostic_rows import validate_diagnostic_rows


def summarize_diagnostic_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a compact JSON-compatible summary of diagnostic rows."""
    validate_diagnostic_rows(rows)
    summary: dict[str, Any] = {
        "row_count": len(rows),
        "diagnostics": sorted({str(row["diagnostic"]) for row in rows}),
        "suites": sorted({suite for row in rows for suite in str(row["suite"]).split("|") if suite}),
        "models": sorted({str(row["model_id"]) for row in rows}),
        "teachers": sorted({str(row["teacher_id"]) for row in rows}),
        "morphologies": sorted({str(row["morphology_id"]) for row in rows}),
        "conditions": sorted({str(row["condition"]) for row in rows}),
        "trace_ids": _sample_trace_ids(rows),
        "numeric_metrics": _numeric_metric_summaries(rows),
        "label_metrics": _label_metric_summaries(rows),
    }
    return summary


def write_diagnostic_summary(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write a compact Markdown or JSON summary for diagnostic rows."""
    summary = summarize_diagnostic_rows(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".json":
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    if output.suffix == ".md":
        output.write_text(_summary_markdown(summary), encoding="utf-8")
        return
    raise ValueError("diagnostic summary output path must end in .json or .md")


def _numeric_metric_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row["value"]
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)) and np.isfinite(value):
            grouped[(str(row["diagnostic"]), str(row["metric"]), str(row["unit"]))].append(float(value))
    summaries = []
    for (diagnostic, metric, unit), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        summaries.append(
            {
                "diagnostic": diagnostic,
                "metric": metric,
                "unit": unit,
                "n_values": int(array.size),
                "mean": float(np.mean(array)),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
            }
        )
    return summaries


def _label_metric_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        value = row["value"]
        if value is None or isinstance(value, (int, float, bool)):
            continue
        grouped[(str(row["diagnostic"]), str(row["metric"]))][str(value)] += 1
    summaries = []
    for (diagnostic, metric), counter in sorted(grouped.items()):
        summaries.append(
            {
                "diagnostic": diagnostic,
                "metric": metric,
                "counts": dict(sorted(counter.items())),
            }
        )
    return summaries


def _sample_trace_ids(rows: list[dict[str, Any]], *, limit: int = 12) -> list[str]:
    seen = []
    for row in rows:
        trace_id = row.get("trace_id")
        if trace_id is None:
            continue
        trace_id = str(trace_id)
        if trace_id not in seen:
            seen.append(trace_id)
        if len(seen) >= limit:
            break
    return seen


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Diagnostic Summary",
        "",
        f"- rows: `{summary['row_count']}`",
        f"- diagnostics: `{', '.join(summary['diagnostics'])}`",
        f"- suites: `{', '.join(summary['suites'])}`",
        f"- models: `{', '.join(summary['models'])}`",
        f"- morphologies: `{', '.join(summary['morphologies'])}`",
        f"- conditions: `{', '.join(summary['conditions'])}`",
    ]
    if summary["trace_ids"]:
        lines.append(f"- sample trace IDs: `{', '.join(summary['trace_ids'])}`")
    lines.extend(["", "## Numeric Metrics", "", "| diagnostic | metric | unit | n | mean | min | max |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"])
    for item in summary["numeric_metrics"]:
        lines.append(
            "| {diagnostic} | {metric} | {unit} | {n_values} | {mean:.6g} | {min:.6g} | {max:.6g} |".format(**item)
        )
    if summary["label_metrics"]:
        lines.extend(["", "## Label Metrics", "", "| diagnostic | metric | counts |", "| --- | --- | --- |"])
        for item in summary["label_metrics"]:
            counts = ", ".join(f"{key}: {value}" for key, value in item["counts"].items())
            lines.append(f"| {item['diagnostic']} | {item['metric']} | {counts} |")
    return "\n".join(lines) + "\n"
