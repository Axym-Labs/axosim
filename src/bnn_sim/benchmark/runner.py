"""Runner glue for v1 dataframe-first benchmark diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from bnn_sim.benchmark.diagnostic_rows import (
    DiagnosticContext,
    evaluate_model_diagnostic_rows,
    paired_intervention_diagnostic_rows,
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
    indices = _leading_indices(n)
    inputs, targets = dataset.get_batch(indices)
    trace_ids = dataset.get_sample_ids(indices)
    return inputs, targets, trace_ids


def load_intervention_pair_npz(
    path: str | Path,
    *,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load a paired intervention `.npz` artifact into diagnostic arrays."""
    payload = np.load(Path(path), allow_pickle=True)
    required = ("baseline_inputs", "baseline_targets", "intervention_inputs", "intervention_targets")
    missing = [name for name in required if name not in payload.files]
    if missing:
        raise ValueError(f"intervention pair file missing arrays: {missing}")
    baseline_inputs = np.asarray(payload["baseline_inputs"])
    baseline_targets = np.asarray(payload["baseline_targets"])
    intervention_inputs = np.asarray(payload["intervention_inputs"])
    intervention_targets = np.asarray(payload["intervention_targets"])
    n = baseline_inputs.shape[0] if max_samples is None else min(int(max_samples), baseline_inputs.shape[0])
    if n <= 0:
        raise ValueError("intervention pair contains no samples")
    trace_ids = (
        [str(item) for item in payload["sample_ids"][:n]]
        if "sample_ids" in payload.files
        else [str(index) for index in range(n)]
    )
    return baseline_inputs[:n], baseline_targets[:n], intervention_inputs[:n], intervention_targets[:n], trace_ids


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


def run_paired_intervention_on_npz(
    path: str | Path,
    *,
    model_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    context: DiagnosticContext | None = None,
    max_samples: int | None = None,
    output: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run paired teacher/model intervention diagnostics on a pair file."""
    baseline_inputs, baseline_targets, intervention_inputs, intervention_targets, trace_ids = load_intervention_pair_npz(
        path,
        max_samples=max_samples,
    )
    rows = paired_intervention_diagnostic_rows(
        baseline_inputs,
        baseline_targets,
        intervention_inputs,
        intervention_targets,
        model_fn=model_fn,
        context=context,
        trace_ids=trace_ids,
    )
    if output is not None:
        write_diagnostic_rows(rows, output)
    return rows


def run_paired_intervention_on_selector(
    selector: DatasetSelector,
    *,
    model_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    context: DiagnosticContext | None = None,
    max_samples: int | None = None,
    output: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run paired intervention diagnostics from a v1 intervention selector."""
    if selector.intervention is None:
        raise ValueError("paired intervention diagnostics require an intervention selector")
    pair_paths = resolve_intervention_pair_paths(selector)
    context = _context_for_selector(selector, context)
    if len(pair_paths) == 1:
        return run_paired_intervention_on_npz(
            pair_paths[0],
            model_fn=model_fn,
            context=context,
            max_samples=max_samples,
            output=output,
        )
    baseline_inputs, baseline_targets, intervention_inputs, intervention_targets, trace_ids = load_intervention_pair_npzs(
        pair_paths,
        max_samples=max_samples,
    )
    rows = paired_intervention_diagnostic_rows(
        baseline_inputs,
        baseline_targets,
        intervention_inputs,
        intervention_targets,
        model_fn=model_fn,
        context=context,
        trace_ids=trace_ids,
    )
    if output is not None:
        write_diagnostic_rows(rows, output)
    return rows


def load_intervention_pair_npzs(
    paths: list[Path],
    *,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load and concatenate sharded intervention pair `.npz` artifacts."""
    arrays = []
    remaining = None if max_samples is None else int(max_samples)
    for path in paths:
        if remaining is not None and remaining <= 0:
            break
        loaded = load_intervention_pair_npz(path, max_samples=remaining)
        arrays.append(loaded)
        if remaining is not None:
            remaining -= loaded[0].shape[0]
    if not arrays:
        raise ValueError("intervention pair shards contain no samples")
    return (
        np.concatenate([item[0] for item in arrays], axis=0),
        np.concatenate([item[1] for item in arrays], axis=0),
        np.concatenate([item[2] for item in arrays], axis=0),
        np.concatenate([item[3] for item in arrays], axis=0),
        [trace_id for item in arrays for trace_id in item[4]],
    )


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
    dataset = selector.load(cache_shards=cache_shards)
    n = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    if n <= 0:
        raise ValueError("selector contains no samples")
    indices = _leading_indices(n)
    inputs, targets = dataset.get_batch(indices)
    trace_ids = dataset.get_sample_ids(indices)
    if raw_soma_mv is None:
        raw_soma_mv = dataset.get_optional_array_batch("soma_mv", indices)
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
    return DiagnosticContext(**data)


def _leading_indices(n: int) -> list[int]:
    return list(range(n))


def _resolve_pair_path(selector: DatasetSelector) -> Path:
    candidates = [
        selector.root / "intervention_pair.npz",
        selector.root / f"{selector.intervention}_pair.npz",
        selector.root / "event_dropout_pair.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "missing intervention pair file; expected one of "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def resolve_intervention_pair_paths(selector: DatasetSelector) -> list[Path]:
    """Resolve one or more intervention pair files from a v1 intervention selector."""
    candidates = [
        selector.root / "intervention_pair.npz",
        selector.root / f"{selector.intervention}_pair.npz",
        selector.root / "event_dropout_pair.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return [candidate]
    shards = sorted({*selector.root.glob("intervention_pair_*.npz"), *selector.root.glob("*_intervention_pair_*.npz")})
    if shards:
        return shards
    raise FileNotFoundError(
        "missing intervention pair file; expected one of "
        + ", ".join(str(candidate) for candidate in candidates)
        + f" or sharded files matching {selector.root / 'intervention_pair_*.npz'}"
    )
