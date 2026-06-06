"""Dataframe-first diagnostic rows for the v1 benchmark."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from bnn_sim.benchmark.dataset_schema import (
    DIAGNOSTIC_ROW_COLUMNS,
    default_diagnostic_specs,
)
from bnn_sim.benchmark.structured_intervention import compute_paired_event_intervention_metrics
from bnn_sim.benchmark.suite import BenchmarkSuite
from bnn_sim.official.constants import DEFAULT_Y_TRAIN_SOMA_BIAS


DEFAULT_MODEL_ID = "model"
DEFAULT_TEACHER_ID = "teacher"


@dataclass(frozen=True)
class DiagnosticContext:
    """Metadata attached to every v1 diagnostic row."""

    suites: tuple[str, ...] = ("paper-summary",)
    model_id: str = DEFAULT_MODEL_ID
    teacher_id: str = DEFAULT_TEACHER_ID
    morphology_id: str = "unknown"
    split: str = "val"
    condition: str = "val"
    intervention: str | None = None
    trace_id: str | None = None
    target_view: str = "neuronio_clipped_normalized"
    soma_scale: float = 0.1
    soma_bias_mv: float = DEFAULT_Y_TRAIN_SOMA_BIAS
    ignore_start: int = 500
    calibrated_spike_threshold: float | None = None


def evaluate_model_diagnostic_rows(
    model_fn: Callable[[np.ndarray], np.ndarray],
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    context: DiagnosticContext | None = None,
    trace_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a model and return tidy v1 diagnostic rows.

    This is a dataframe-compatible companion to `BenchmarkSuite.evaluate_model`.
    The nested report is still useful for backwards compatibility; this path is
    the package-facing substrate for suite composition, reports, and later plots.
    """
    context = context or DiagnosticContext()
    trace_ids = trace_ids or [str(i) for i in range(np.asarray(inputs).shape[0])]
    report = BenchmarkSuite().evaluate_model(
        model_fn,
        inputs,
        targets,
        soma_scale=context.soma_scale,
        soma_bias_mv=context.soma_bias_mv,
        ignore_start=context.ignore_start,
        run_perturbation=False,
        model_name=context.model_id,
        trace_ids=trace_ids,
        calibrated_spike_threshold=context.calibrated_spike_threshold,
    )
    rows: list[dict[str, Any]] = []
    rows.extend(_trace_core_rows(report["m1_trace_fidelity"], context))
    rows.extend(_state_dynamics_rows(report["state_conditioned_fidelity"], context))
    rows.extend(_spike_behavior_rows(report["m1_trace_fidelity"], context))
    rows.extend(_feature_fidelity_rows(report["m1b_feature_fidelity"], context))
    return _filter_rows_for_suites(rows, context.suites)


def teacher_qc_diagnostic_rows(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    context: DiagnosticContext | None = None,
    raw_soma_mv: np.ndarray | None = None,
    trace_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return teacher-QC rows for distribution and clipping checks."""
    context = context or DiagnosticContext(model_id="teacher")
    trace_ids = trace_ids or [str(i) for i in range(np.asarray(targets).shape[0])]
    rows = []
    rows.extend(_teacher_distribution_rows(inputs, targets, context, trace_ids))
    if raw_soma_mv is not None:
        rows.extend(_clipping_distortion_rows(raw_soma_mv, targets, context, trace_ids))
    return _filter_rows_for_suites(rows, context.suites)


def paired_intervention_diagnostic_rows(
    baseline_inputs: np.ndarray,
    baseline_targets: np.ndarray,
    intervention_inputs: np.ndarray,
    intervention_targets: np.ndarray,
    *,
    model_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    context: DiagnosticContext | None = None,
    trace_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return tidy rows for a paired intervention teacher/model comparison."""
    context = context or DiagnosticContext(condition="interventions/event_dropout", intervention="event_dropout")
    intervention_name = context.intervention or context.condition.rsplit("/", 1)[-1]
    report = compute_paired_event_intervention_metrics(
        model_fn,
        baseline_inputs,
        baseline_targets,
        intervention_inputs,
        intervention_targets,
        intervention_name=intervention_name,
        soma_scale=context.soma_scale,
        soma_bias_mv=context.soma_bias_mv,
        ignore_start=context.ignore_start,
        trace_ids=trace_ids,
        model_spike_threshold=(
            0.0 if context.calibrated_spike_threshold is None else context.calibrated_spike_threshold
        ),
    )
    rows: list[dict[str, Any]] = []
    teacher = report["teacher_response"]
    for metric in (
        "baseline_event_fraction",
        "intervention_event_fraction",
        "event_fraction_delta",
        "delta_soma_mean_mv",
        "delta_soma_abs_mean_mv",
        "delta_spike_rate_per_1000_steps",
    ):
        rows.append(_row("intervention_response", context, metric, teacher.get(metric), unit=_unit_for_metric(metric)))
    rows.append(_row("intervention_response", context, "response_detected", float(bool(teacher["response_detected"])), unit="bool"))
    rows.append(_row("intervention_response", context, "teacher_direction", teacher["teacher_direction"], unit="label"))

    model = report.get("model_response")
    if model is not None:
        for metric in (
            "soma_delta_rmse_mv",
            "soma_delta_gain_error",
            "spike_count_delta_mae_per_trace",
            "spike_count_delta_gain_error",
            "direction_agreement",
        ):
            rows.append(_row("intervention_response", context, metric, model.get(metric), unit=_unit_for_metric(metric)))
    return _filter_rows_for_suites(rows, context.suites)


def morphology_contrast_diagnostic_rows(
    source_rows: Iterable[dict[str, Any]],
    *,
    context: DiagnosticContext | None = None,
    min_morphologies: int = 2,
) -> list[dict[str, Any]]:
    """Derive rough cross-morphology contrast rows from existing diagnostics."""
    source_rows = list(source_rows)
    validate_diagnostic_rows(source_rows)
    if min_morphologies < 2:
        raise ValueError("min_morphologies must be at least 2")
    context = context or DiagnosticContext(suites=("morphology-contrast", "paper-summary"), morphology_id="all")
    grouped: dict[tuple[str, str, str, str | None, str, str, str | None], dict[str, list[float]]] = {}
    for row in source_rows:
        value = row["value"]
        morphology_id = row["morphology_id"]
        if (
            row["diagnostic"] == "morphology_contrast"
            or morphology_id in {None, "", "unknown"}
            or isinstance(value, bool)
            or value is None
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            continue
        key = (
            str(row["diagnostic"]),
            str(row["metric"]),
            str(row["unit"]),
            row["state"],
            str(row["model_id"]),
            str(row["condition"]),
            row["intervention"],
        )
        grouped.setdefault(key, {}).setdefault(str(morphology_id), []).append(float(value))

    rows: list[dict[str, Any]] = []
    for (diagnostic, metric, unit, state, model_id, condition, intervention), by_morphology in sorted(grouped.items()):
        morphology_values = {
            morphology: float(np.mean(values))
            for morphology, values in by_morphology.items()
            if values
        }
        if len(morphology_values) < min_morphologies:
            continue
        values = np.asarray(list(morphology_values.values()), dtype=float)
        row_context = _replace_context(
            context,
            model_id=model_id,
            morphology_id="all_v1",
            condition=condition,
            intervention=intervention,
        )
        prefix = f"{diagnostic}.{metric}"
        for suffix, value in (
            ("mean_across_morphologies", float(np.mean(values))),
            ("std_across_morphologies", float(np.std(values))),
            ("range_across_morphologies", float(np.max(values) - np.min(values))),
            ("min_across_morphologies", float(np.min(values))),
            ("max_across_morphologies", float(np.max(values))),
        ):
            row = _row(
                "morphology_contrast",
                row_context,
                f"{prefix}.{suffix}",
                value,
                state=state,
                unit=unit,
                n=len(morphology_values),
            )
            row["source_diagnostic"] = diagnostic
            row["source_metric"] = metric
            row["morphology_values"] = morphology_values
            rows.append(row)
    return _filter_rows_for_suites(rows, context.suites)


def validate_diagnostic_rows(rows: Iterable[dict[str, Any]]) -> None:
    """Validate required row columns for dataframe compatibility."""
    required = set(DIAGNOSTIC_ROW_COLUMNS)
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"diagnostic row {index} missing required columns: {sorted(missing)}")


def write_diagnostic_rows(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write diagnostic rows as `.json`, `.jsonl`, or `.csv`."""
    rows = list(rows)
    validate_diagnostic_rows(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".json":
        output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    if output.suffix == ".jsonl":
        output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        return
    if output.suffix == ".csv":
        fieldnames = list(DIAGNOSTIC_ROW_COLUMNS)
        extras = sorted({key for row in rows for key in row} - set(fieldnames))
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames + extras)
            writer.writeheader()
            writer.writerows(rows)
        return
    raise ValueError("diagnostic row output path must end in .json, .jsonl, or .csv")


def _trace_core_rows(report: dict[str, Any], context: DiagnosticContext) -> list[dict[str, Any]]:
    metrics = (
        "soma_rmse_mv",
        "soma_mae_mv",
        "state_weighted_trace_distance_mv",
        "dynamic_distance",
        "trace_shape_distance",
        "lag_tolerant_voltage_rmse_mv",
        "state_trace_shape_distance",
        "spike_train_distance",
        "spike_auc",
        "spike_brier",
        "spike_log_loss",
        "spike_count_mae_per_trace",
    )
    rows = [_row("trace_core", context, metric, report.get(metric), unit=_unit_for_metric(metric)) for metric in metrics]
    for item in report.get("per_trace_summary", {}).get("rows", []):
        trace_context = _replace_context_trace(context, item["trace_id"])
        for metric in ("soma_rmse_mv", "soma_mae_mv", "soma_max_abs_error_mv", "target_spike_count"):
            rows.append(_row("trace_core", trace_context, metric, item.get(metric), unit=_unit_for_metric(metric), n=1))
    return rows


def _state_dynamics_rows(report: dict[str, Any], context: DiagnosticContext) -> list[dict[str, Any]]:
    rows = []
    for state_row in report.get("states", []):
        state = state_row["state"]
        for metric in (
            "voltage_rmse_mv",
            "voltage_mae_mv",
            "dvdt_mae_mv_per_step",
            "spike_target_count",
            "spike_predicted_count",
            "spike_count_error",
            "fraction",
        ):
            rows.append(
                _row(
                    "state_dynamics",
                    context,
                    metric,
                    state_row.get(metric),
                    state=state,
                    unit=_unit_for_metric(metric),
                    n=state_row.get("count"),
                )
            )
    return rows


def _spike_behavior_rows(report: dict[str, Any], context: DiagnosticContext) -> list[dict[str, Any]]:
    rows = []
    for block_name in ("spike_threshold_zero", "spike_prevalence_matched", "spike_calibrated_threshold"):
        block = report.get(block_name)
        if not isinstance(block, dict):
            continue
        block_context = _replace_context_condition(context, f"{context.condition}:{block_name}")
        for metric in ("threshold", "predicted_count", "count_mae_per_trace"):
            rows.append(_row("spike_behavior", block_context, metric, block.get(metric), unit=_unit_for_metric(metric)))
    for metric in ("spike_auc", "spike_brier", "spike_log_loss"):
        rows.append(_row("spike_behavior", context, metric, report.get(metric), unit=_unit_for_metric(metric)))
    return rows


def _feature_fidelity_rows(report: dict[str, Any], context: DiagnosticContext) -> list[dict[str, Any]]:
    rows = []
    features = report.get("features", {})
    for feature_name, feature in features.items():
        if not isinstance(feature, dict):
            continue
        for metric in ("teacher_mean", "prediction_mean", "mae"):
            rows.append(
                _row(
                    "feature_fidelity",
                    context,
                    f"{feature_name}_{metric}",
                    feature.get(metric),
                    unit=_unit_for_metric(feature_name),
                )
            )
    return rows


def _teacher_distribution_rows(
    inputs: np.ndarray,
    targets: np.ndarray,
    context: DiagnosticContext,
    trace_ids: list[str],
) -> list[dict[str, Any]]:
    inp = _apply_time_mask(_as_trace_array(np.asarray(inputs)), context.ignore_start)
    tgt = _apply_time_mask(_as_trace_array(np.asarray(targets)), context.ignore_start)
    soma_mv = tgt[..., 1] / context.soma_scale + context.soma_bias_mv
    spikes = tgt[..., 0] > 0.5
    rows = [
        _row("teacher_distribution_qc", context, "soma_mean_mv", float(np.mean(soma_mv)), unit="mV", n=soma_mv.size),
        _row("teacher_distribution_qc", context, "soma_std_mv", float(np.std(soma_mv)), unit="mV", n=soma_mv.size),
        _row("teacher_distribution_qc", context, "spike_count", int(spikes.sum()), unit="count", n=spikes.size),
        _row(
            "teacher_distribution_qc",
            context,
            "spike_rate_per_1000_steps",
            float(1000.0 * spikes.sum() / max(1, spikes.size)),
            unit="spikes_per_1000_steps",
            n=spikes.size,
        ),
        _row(
            "teacher_distribution_qc",
            context,
            "event_fraction",
            float(np.count_nonzero(inp) / max(1, inp.size)),
            unit="fraction",
            n=inp.size,
        ),
    ]
    per_trace_spikes = spikes.sum(axis=1)
    for index, trace_id in enumerate(trace_ids):
        rows.append(
            _row(
                "teacher_distribution_qc",
                _replace_context_trace(context, trace_id),
                "trace_spike_count",
                int(per_trace_spikes[index]),
                unit="count",
                n=spikes.shape[1],
            )
        )
    return rows


def _clipping_distortion_rows(
    raw_soma_mv: np.ndarray,
    clipped_targets: np.ndarray,
    context: DiagnosticContext,
    trace_ids: list[str],
) -> list[dict[str, Any]]:
    raw = np.asarray(raw_soma_mv)
    if raw.ndim == 1:
        raw = raw[None, :]
    raw = raw[:, context.ignore_start:]
    tgt = _apply_time_mask(_as_trace_array(np.asarray(clipped_targets)), context.ignore_start)
    clipped_mv = tgt[..., 1] / context.soma_scale + context.soma_bias_mv
    if raw.shape != clipped_mv.shape:
        raise ValueError(f"raw_soma_mv shape must match target time shape after mask, got {raw.shape} and {clipped_mv.shape}")
    distortion = raw - clipped_mv
    clipped_mask = np.abs(distortion) > 1e-6
    spikes = tgt[..., 0] > 0.5
    near_spike = _near_spike_mask(spikes, radius=3)
    rows = [
        _row("clipping_distortion", context, "clipped_fraction", float(clipped_mask.mean()), unit="fraction", n=raw.size),
        _row("clipping_distortion", context, "distortion_rmse_mv", _rmse(distortion), unit="mV", n=raw.size),
        _row("clipping_distortion", context, "distortion_max_mv", float(np.max(np.abs(distortion))), unit="mV", n=raw.size),
        _row(
            "clipping_distortion",
            context,
            "near_spike_clipped_fraction",
            float(clipped_mask[near_spike].mean()) if near_spike.any() else None,
            unit="fraction",
            n=int(near_spike.sum()),
        ),
        _row(
            "clipping_distortion",
            context,
            "near_spike_distortion_rmse_mv",
            _rmse(distortion[near_spike]) if near_spike.any() else None,
            unit="mV",
            n=int(near_spike.sum()),
        ),
    ]
    for index, trace_id in enumerate(trace_ids):
        rows.append(
            _row(
                "clipping_distortion",
                _replace_context_trace(context, trace_id),
                "trace_distortion_max_mv",
                float(np.max(np.abs(distortion[index]))),
                unit="mV",
                n=raw.shape[1],
            )
        )
    return rows


def _filter_rows_for_suites(rows: list[dict[str, Any]], suites: tuple[str, ...]) -> list[dict[str, Any]]:
    specs = {spec.name: set(spec.suites) for spec in default_diagnostic_specs()}
    selected = set(suites)
    filtered = []
    for row in rows:
        diagnostic_suites = specs.get(row["diagnostic"], set())
        row_suites = tuple(suite for suite in suites if suite in diagnostic_suites)
        if row_suites:
            row = dict(row)
            row["suite"] = "|".join(row_suites)
            filtered.append(row)
    validate_diagnostic_rows(filtered)
    return filtered


def _row(
    diagnostic: str,
    context: DiagnosticContext,
    metric: str,
    value: Any,
    *,
    state: str | None = None,
    unit: str = "unitless",
    n: int | None = None,
) -> dict[str, Any]:
    return {
        "diagnostic": diagnostic,
        "suite": "",
        "model_id": context.model_id,
        "teacher_id": context.teacher_id,
        "morphology_id": context.morphology_id,
        "split": context.split,
        "condition": context.condition,
        "intervention": context.intervention,
        "trace_id": context.trace_id,
        "state": state,
        "metric": metric,
        "value": _json_scalar(value),
        "unit": unit,
        "n": n,
        "target_view": context.target_view,
        "ignore_start": context.ignore_start,
        "calibrated_spike_threshold": context.calibrated_spike_threshold,
    }


def _replace_context_trace(context: DiagnosticContext, trace_id: str) -> DiagnosticContext:
    return replace(context, trace_id=trace_id)


def _replace_context_condition(context: DiagnosticContext, condition: str) -> DiagnosticContext:
    return replace(context, condition=condition)


def _replace_context(context: DiagnosticContext, **updates: Any) -> DiagnosticContext:
    return replace(context, **updates)


def _as_trace_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim != 3:
        leading = int(np.prod(array.shape[:-2]))
        return array.reshape(leading, array.shape[-2], array.shape[-1])
    return array


def _apply_time_mask(array: np.ndarray, ignore_start: int) -> np.ndarray:
    return array[:, ignore_start:, :] if ignore_start > 0 else array


def _near_spike_mask(spikes: np.ndarray, *, radius: int) -> np.ndarray:
    out = np.zeros_like(spikes, dtype=bool)
    for trace in range(spikes.shape[0]):
        for idx in np.flatnonzero(spikes[trace]):
            start = max(0, idx - radius)
            stop = min(spikes.shape[1], idx + radius + 1)
            out[trace, start:stop] = True
    return out


def _rmse(values: np.ndarray) -> float:
    values = np.asarray(values)
    return float(np.sqrt(np.mean(values ** 2)))


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _unit_for_metric(metric: str) -> str:
    if metric.endswith("_mv") or "_mv_" in metric or metric in {"soma_mean_mv", "soma_std_mv"}:
        return "mV"
    if "rate_per_1000" in metric:
        return "spikes_per_1000_steps"
    if "fraction" in metric or metric in {"spike_auc", "spike_brier"}:
        return "fraction"
    if "count" in metric:
        return "count"
    if "threshold" in metric:
        return "logit"
    if "loss" in metric or "distance" in metric or "gain" in metric:
        return "unitless"
    return "unitless"
