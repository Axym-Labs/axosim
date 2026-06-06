"""Runner glue for v1 dataframe-first benchmark diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from bnn_sim.benchmark.diagnostic_rows import (
    DiagnosticContext,
    evaluate_model_diagnostic_rows,
    teacher_qc_diagnostic_rows,
    write_diagnostic_rows,
)
from bnn_sim.benchmark.selectors import DatasetSelector


def load_selector_batch(
    selector: DatasetSelector,
    *,
    max_samples: int | None = None,
    cache_shards: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one selector into arrays for diagnostic evaluation."""
    dataset = selector.load(cache_shards=cache_shards)
    n = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    if n <= 0:
        raise ValueError("selector contains no samples")
    indices = list(range(n))
    inputs, targets = dataset.get_batch(indices)
    trace_ids = dataset.get_sample_ids(indices)
    return inputs, targets, trace_ids


def run_model_diagnostics_on_selector(
    model_fn: Callable[[np.ndarray], np.ndarray],
    selector: DatasetSelector,
    *,
    context: DiagnosticContext | None = None,
    max_samples: int | None = None,
    output: str | Path | None = None,
    cache_shards: int = 1,
) -> list[dict[str, Any]]:
    """Run model diagnostics on a selected v1 dataset category."""
    inputs, targets, trace_ids = load_selector_batch(selector, max_samples=max_samples, cache_shards=cache_shards)
    context = _context_for_selector(selector, context)
    rows = evaluate_model_diagnostic_rows(model_fn, inputs, targets, context=context, trace_ids=trace_ids)
    if output is not None:
        write_diagnostic_rows(rows, output)
    return rows


def run_teacher_qc_on_selector(
    selector: DatasetSelector,
    *,
    context: DiagnosticContext | None = None,
    raw_soma_mv: np.ndarray | None = None,
    max_samples: int | None = None,
    output: str | Path | None = None,
    cache_shards: int = 1,
) -> list[dict[str, Any]]:
    """Run teacher-QC diagnostics on a selected v1 dataset category."""
    inputs, targets, trace_ids = load_selector_batch(selector, max_samples=max_samples, cache_shards=cache_shards)
    context = _context_for_selector(selector, context)
    rows = teacher_qc_diagnostic_rows(inputs, targets, context=context, raw_soma_mv=raw_soma_mv, trace_ids=trace_ids)
    if output is not None:
        write_diagnostic_rows(rows, output)
    return rows


def _context_for_selector(selector: DatasetSelector, context: DiagnosticContext | None) -> DiagnosticContext:
    base = context or DiagnosticContext()
    data = {
        **base.__dict__,
        "split": selector.split or "",
        "condition": selector.condition,
        "intervention": selector.intervention,
    }
    data.pop("trace_id", None)
    return DiagnosticContext(**data)
