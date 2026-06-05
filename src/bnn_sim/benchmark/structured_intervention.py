"""M4: structured event-intervention metrics.

This module evaluates paired teacher traces under a simulator-plausible input
intervention, such as event dropout or event-rate scaling. It deliberately does
not make global smoothness claims.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def make_event_dropout_inputs(
    inputs: np.ndarray,
    *,
    dropout_probability: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop a Bernoulli subset of non-zero input events.

    Returns the intervened inputs and a boolean mask marking removed events.
    """
    if not 0.0 <= dropout_probability <= 1.0:
        raise ValueError("dropout_probability must be in [0, 1]")
    rng = np.random.default_rng(seed)
    inputs = np.asarray(inputs)
    event_mask = inputs != 0
    drop_mask = event_mask & (rng.random(inputs.shape) < dropout_probability)
    intervened = inputs.copy()
    intervened[drop_mask] = 0
    return intervened, drop_mask


def compute_paired_event_intervention_metrics(
    model_fn: Callable[[np.ndarray], np.ndarray] | None,
    baseline_inputs: np.ndarray,
    baseline_targets: np.ndarray,
    intervention_inputs: np.ndarray,
    intervention_targets: np.ndarray,
    *,
    intervention_name: str = "event_intervention",
    soma_scale: float = 0.1,
    soma_bias_mv: float = -67.7,
    ignore_start: int = 0,
    trace_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Compare teacher and model deltas for one paired event intervention."""
    baseline_inputs = _as_trace_array(np.asarray(baseline_inputs))
    intervention_inputs = _as_trace_array(np.asarray(intervention_inputs))
    baseline_targets = _as_trace_array(np.asarray(baseline_targets))
    intervention_targets = _as_trace_array(np.asarray(intervention_targets))
    _check_pair_shapes("inputs", baseline_inputs, intervention_inputs)
    _check_pair_shapes("targets", baseline_targets, intervention_targets)
    if baseline_inputs.shape[:2] != baseline_targets.shape[:2]:
        raise ValueError(
            "input and target pairs must have matching batch/time dimensions, got "
            f"{baseline_inputs.shape[:2]} and {baseline_targets.shape[:2]}"
        )

    trace_ids = trace_ids or [str(i) for i in range(baseline_inputs.shape[0])]
    if len(trace_ids) != baseline_inputs.shape[0]:
        raise ValueError("trace_ids length must match number of traces")

    baseline_inputs_m = baseline_inputs[:, ignore_start:, :]
    intervention_inputs_m = intervention_inputs[:, ignore_start:, :]
    baseline_targets_m = baseline_targets[:, ignore_start:, :]
    intervention_targets_m = intervention_targets[:, ignore_start:, :]
    if baseline_targets_m.shape[1] == 0:
        raise ValueError("ignore_start leaves no timesteps to evaluate")

    teacher_response = _teacher_response_summary(
        baseline_inputs_m,
        baseline_targets_m,
        intervention_inputs_m,
        intervention_targets_m,
        soma_scale=soma_scale,
        soma_bias_mv=soma_bias_mv,
    )
    report: dict[str, Any] = {
        "status": "teacher_only" if model_fn is None else "model_compared",
        "kind": "paired_event_intervention",
        "intervention_name": intervention_name,
        "mask": {
            "ignore_start": ignore_start,
            "evaluated_timesteps_per_trace": int(baseline_targets_m.shape[1]),
        },
        "teacher_response": teacher_response,
        "model_response": None,
        "representative_trace_ids": _representative_trace_ids(
            trace_ids,
            baseline_targets_m,
            intervention_targets_m,
            soma_scale=soma_scale,
        ),
        "what_failure_means": (
            "If teacher response is weak or counter to the intended protocol, "
            "the intervention is not a clean robustness claim. If teacher "
            "response is interpretable but model delta error is high, the "
            "surrogate does not preserve the teacher's input-shift behavior."
        ),
    }

    if model_fn is not None:
        baseline_pred = _as_trace_array(np.asarray(model_fn(baseline_inputs)))
        intervention_pred = _as_trace_array(np.asarray(model_fn(intervention_inputs)))
        _check_prediction_shapes(baseline_pred, intervention_pred, baseline_targets)
        report["model_response"] = _model_delta_summary(
            baseline_pred[:, ignore_start:, :],
            intervention_pred[:, ignore_start:, :],
            baseline_targets_m,
            intervention_targets_m,
            soma_scale=soma_scale,
        )

    return report


def _as_trace_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim != 3:
        leading = int(np.prod(array.shape[:-2]))
        return array.reshape(leading, array.shape[-2], array.shape[-1])
    return array


def _check_pair_shapes(name: str, *arrays: np.ndarray) -> None:
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"paired intervention {name} arrays must have matching shapes, got {sorted(shapes)}")


def _check_prediction_shapes(
    baseline_pred: np.ndarray,
    intervention_pred: np.ndarray,
    targets: np.ndarray,
) -> None:
    if baseline_pred.shape != targets.shape or intervention_pred.shape != targets.shape:
        raise ValueError(
            "model predictions must match target shape, got "
            f"{baseline_pred.shape}, {intervention_pred.shape}, target {targets.shape}"
        )


def _event_fraction(inputs: np.ndarray) -> float:
    return float(np.count_nonzero(inputs) / max(1, inputs.size))


def _spike_rate_per_1000_steps(targets: np.ndarray) -> float:
    return float(1000.0 * np.mean(targets[..., 0] > 0.5))


def _teacher_response_summary(
    baseline_inputs: np.ndarray,
    baseline_targets: np.ndarray,
    intervention_inputs: np.ndarray,
    intervention_targets: np.ndarray,
    *,
    soma_scale: float,
    soma_bias_mv: float,
) -> dict[str, Any]:
    baseline_soma_mv = baseline_targets[..., 1] / soma_scale + soma_bias_mv
    intervention_soma_mv = intervention_targets[..., 1] / soma_scale + soma_bias_mv
    baseline_rate = _spike_rate_per_1000_steps(baseline_targets)
    intervention_rate = _spike_rate_per_1000_steps(intervention_targets)
    baseline_events = _event_fraction(baseline_inputs)
    intervention_events = _event_fraction(intervention_inputs)
    soma_delta = intervention_soma_mv - baseline_soma_mv
    spike_delta = intervention_rate - baseline_rate
    event_delta = intervention_events - baseline_events
    return {
        "baseline_event_fraction": baseline_events,
        "intervention_event_fraction": intervention_events,
        "event_fraction_delta": event_delta,
        "event_fraction_ratio": float(intervention_events / baseline_events) if baseline_events > 0 else None,
        "baseline_spike_rate_per_1000_steps": baseline_rate,
        "intervention_spike_rate_per_1000_steps": intervention_rate,
        "delta_spike_rate_per_1000_steps": spike_delta,
        "baseline_soma_mean_mv": float(np.mean(baseline_soma_mv)),
        "intervention_soma_mean_mv": float(np.mean(intervention_soma_mv)),
        "delta_soma_mean_mv": float(np.mean(soma_delta)),
        "delta_soma_abs_mean_mv": float(np.mean(np.abs(soma_delta))),
        "response_detected": bool(abs(spike_delta) >= 0.1 or np.mean(np.abs(soma_delta)) >= 0.1),
        "teacher_direction": _direction_label(event_delta, spike_delta, float(np.mean(soma_delta))),
    }


def _direction_label(event_delta: float, spike_delta: float, soma_delta: float) -> str:
    event = "higher_event_drive" if event_delta > 0 else "lower_event_drive" if event_delta < 0 else "same_event_drive"
    spike = "higher_spiking" if spike_delta > 0 else "lower_spiking" if spike_delta < 0 else "same_spiking"
    soma = "depolarized" if soma_delta > 0 else "hyperpolarized" if soma_delta < 0 else "same_soma_mean"
    return f"{event}/{spike}/{soma}"


def _model_delta_summary(
    baseline_pred: np.ndarray,
    intervention_pred: np.ndarray,
    baseline_targets: np.ndarray,
    intervention_targets: np.ndarray,
    *,
    soma_scale: float,
) -> dict[str, Any]:
    teacher_soma_delta = (intervention_targets[..., 1] - baseline_targets[..., 1]) / soma_scale
    model_soma_delta = (intervention_pred[..., 1] - baseline_pred[..., 1]) / soma_scale
    teacher_spike_delta = (intervention_targets[..., 0] > 0.5).sum(axis=1) - (baseline_targets[..., 0] > 0.5).sum(axis=1)
    model_spike_delta = (intervention_pred[..., 0] > 0.0).sum(axis=1) - (baseline_pred[..., 0] > 0.0).sum(axis=1)
    soma_delta_error = model_soma_delta - teacher_soma_delta
    return {
        "soma_delta_rmse_mv": float(np.sqrt(np.mean(soma_delta_error ** 2))),
        "soma_delta_mae_mv": float(np.mean(np.abs(soma_delta_error))),
        "soma_delta_correlation": _safe_corr(model_soma_delta.reshape(-1), teacher_soma_delta.reshape(-1)),
        "soma_mean_delta_error_mv": float(np.mean(model_soma_delta) - np.mean(teacher_soma_delta)),
        "spike_count_delta_mae_per_trace": float(np.mean(np.abs(model_spike_delta - teacher_spike_delta))),
        "spike_count_delta_teacher_mean": float(np.mean(teacher_spike_delta)),
        "spike_count_delta_model_mean": float(np.mean(model_spike_delta)),
        "direction_agreement": {
            "soma_mean": bool(np.sign(np.mean(model_soma_delta)) == np.sign(np.mean(teacher_soma_delta))),
            "spike_count": bool(np.sign(np.mean(model_spike_delta)) == np.sign(np.mean(teacher_spike_delta))),
        },
    }


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _representative_trace_ids(
    trace_ids: list[str],
    baseline_targets: np.ndarray,
    intervention_targets: np.ndarray,
    *,
    soma_scale: float,
    limit: int = 5,
) -> list[str]:
    soma_delta = np.mean(
        np.abs((intervention_targets[..., 1] - baseline_targets[..., 1]) / soma_scale),
        axis=1,
    )
    if soma_delta.size == 0:
        return []
    order = np.argsort(-soma_delta)
    return [trace_ids[int(i)] for i in order[:limit] if soma_delta[int(i)] > 0]
