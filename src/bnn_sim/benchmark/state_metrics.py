"""State-conditioned neuron-surrogate metrics.

The goal is to make trace fidelity less dominated by the most common quiet
timesteps.  The state atlas is intentionally simple and inspectable: masks are
derived from teacher voltage, teacher spikes, and signed input events when
available.
"""

from __future__ import annotations

from typing import Any

import numpy as np


STATE_DEFINITIONS = {
    "spiking": "teacher spike label is active",
    "refractory": "1-6 steps after a teacher spike, excluding the spike timestep",
    "near_threshold": "teacher soma is near the clipped AP threshold or shortly before a teacher spike",
    "excited_drive": "non-spiking state with net excitatory input drive, or rising voltage if inputs are unavailable",
    "inhibited_drive": "non-spiking state with net inhibitory input drive, or falling voltage if inputs are unavailable",
    "hyperpolarized_quiet": "low-voltage non-spiking state without recent high drive",
    "resting": "remaining non-spiking, non-refractory timesteps",
}


def compute_state_conditioned_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    inputs: np.ndarray | None = None,
    *,
    soma_scale: float = 0.1,
    soma_bias_mv: float = -67.7,
    ignore_start: int = 0,
    calibrated_spike_threshold: float | None = None,
    near_threshold_mv: float = -60.0,
    high_drive_quantile: float = 0.95,
) -> dict[str, Any]:
    """Compute state-conditioned voltage, dynamics, and spike metrics."""
    pred = _apply_mask(_as_trace_array(np.asarray(prediction)), ignore_start)
    tgt = _apply_mask(_as_trace_array(np.asarray(target)), ignore_start)
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction and target shapes must match, got {pred.shape} and {tgt.shape}")
    inp = _apply_mask(_as_trace_array(np.asarray(inputs)), ignore_start) if inputs is not None else None
    if pred.shape[1] == 0:
        raise ValueError("ignore_start leaves no timesteps to evaluate")

    pred_soma_mv = pred[..., 1] / soma_scale + soma_bias_mv
    tgt_soma_mv = tgt[..., 1] / soma_scale + soma_bias_mv
    residual_mv = pred_soma_mv - tgt_soma_mv
    pred_dvdt = _per_trace_diff(pred_soma_mv)
    tgt_dvdt = _per_trace_diff(tgt_soma_mv)
    dvdt_residual = pred_dvdt - tgt_dvdt

    spike_scores = pred[..., 0]
    spike_labels = tgt[..., 0] > 0.5
    threshold = calibrated_spike_threshold if calibrated_spike_threshold is not None else 0.0
    pred_spikes = spike_scores >= threshold

    states = build_neuron_state_masks(
        tgt_soma_mv,
        spike_labels,
        inp,
        near_threshold_mv=near_threshold_mv,
        high_drive_quantile=high_drive_quantile,
    )
    rows = [
        _state_row(
            name,
            mask,
            residual_mv,
            dvdt_residual,
            pred_spikes,
            spike_labels,
        )
        for name, mask in states["primary_masks"].items()
    ]
    supported = [row for row in rows if row["count"] > 0]
    state_weighted_trace_distance_mv = (
        float(np.mean([row["voltage_rmse_mv"] for row in supported if row["voltage_rmse_mv"] is not None]))
        if supported else None
    )
    state_weighted_voltage_mae_mv = (
        float(np.mean([row["voltage_mae_mv"] for row in supported if row["voltage_mae_mv"] is not None]))
        if supported else None
    )
    state_weighted_dvdt_mae_mv_per_step = (
        float(np.mean([row["dvdt_mae_mv_per_step"] for row in supported if row["dvdt_mae_mv_per_step"] is not None]))
        if supported else None
    )
    spike_timing = _spike_timing_summary(pred_spikes.reshape(-1), spike_labels.reshape(-1))
    target_counts = spike_labels.sum(axis=1)
    pred_counts = pred_spikes.sum(axis=1)
    count_mae = float(np.mean(np.abs(pred_counts - target_counts)))
    count_scale = max(1.0, float(np.mean(target_counts)))
    voltage_scale = _robust_scale(tgt_soma_mv)
    dvdt_scale = _robust_scale(tgt_dvdt)
    dynamic_distance = None
    if state_weighted_voltage_mae_mv is not None and state_weighted_dvdt_mae_mv_per_step is not None:
        components = {
            "state_weighted_voltage": state_weighted_voltage_mae_mv / voltage_scale,
            "state_weighted_dvdt": state_weighted_dvdt_mae_mv_per_step / dvdt_scale,
            "spike_timing": 1.0 - spike_timing["f1_at_2steps"],
            "spike_count": count_mae / count_scale,
        }
        dynamic_distance = float(np.mean(list(components.values())))
    else:
        components = None

    return {
        "kind": "state_conditioned_dynamic_fidelity",
        "state_definitions": STATE_DEFINITIONS,
        "state_parameters": {
            "near_threshold_mv": near_threshold_mv,
            "high_drive_quantile": high_drive_quantile,
            "spike_threshold": threshold,
            "spike_threshold_semantics": (
                "calibrated threshold learned on a separate split"
                if calibrated_spike_threshold is not None
                else "fixed zero-logit operating point"
            ),
            **states["parameters"],
        },
        "states": rows,
        "state_weighted_trace_distance_mv": state_weighted_trace_distance_mv,
        "state_weighted_voltage_mae_mv": state_weighted_voltage_mae_mv,
        "state_weighted_dvdt_mae_mv_per_step": state_weighted_dvdt_mae_mv_per_step,
        "dynamic_distance": dynamic_distance,
        "dynamic_distance_components": components,
        "spike_timing": spike_timing,
        "spike_count_mae_per_trace": count_mae,
        "what_failure_means": (
            "High state-weighted trace distance means the surrogate is wrong "
            "across behaviorally distinct neuron states, not merely on the "
            "dominant quiet-voltage distribution. High dynamic distance points "
            "to combined failures in voltage level, voltage derivative, spike "
            "timing, or spike count."
        ),
    }


def build_neuron_state_masks(
    soma_mv: np.ndarray,
    spikes: np.ndarray,
    inputs: np.ndarray | None = None,
    *,
    near_threshold_mv: float = -60.0,
    high_drive_quantile: float = 0.95,
) -> dict[str, Any]:
    """Build mutually exclusive primary state masks plus useful overlays."""
    soma_mv = np.asarray(soma_mv)
    spikes = np.asarray(spikes, dtype=bool)
    if soma_mv.shape != spikes.shape:
        raise ValueError(f"soma and spike masks must have matching shape, got {soma_mv.shape} and {spikes.shape}")
    if not 0.0 <= high_drive_quantile <= 1.0:
        raise ValueError("high_drive_quantile must be in [0, 1]")

    dvdt = _per_trace_diff(soma_mv)
    refractory = _dilate_after(spikes, after=6) & ~spikes
    pre_spike = _dilate_before(spikes, before=2) & ~spikes
    near_threshold = ((soma_mv >= near_threshold_mv) | pre_spike) & ~spikes & ~refractory
    event_count, net_drive, high_drive = _input_drive(inputs, soma_mv.shape, high_drive_quantile)
    available = ~(spikes | refractory | near_threshold)

    if inputs is not None:
        excited = available & high_drive & (net_drive > 0)
        inhibited = available & high_drive & (net_drive < 0)
    else:
        slope_scale = max(_robust_scale(dvdt), 1e-6)
        excited = available & (dvdt > 0.25 * slope_scale)
        inhibited = available & (dvdt < -0.25 * slope_scale)

    remaining = available & ~excited & ~inhibited
    low_voltage = soma_mv <= np.quantile(soma_mv, 0.25)
    hyperpolarized_quiet = remaining & low_voltage & ~high_drive
    resting = remaining & ~hyperpolarized_quiet
    masks = {
        "spiking": spikes,
        "refractory": refractory,
        "near_threshold": near_threshold,
        "excited_drive": excited,
        "inhibited_drive": inhibited,
        "hyperpolarized_quiet": hyperpolarized_quiet,
        "resting": resting,
    }
    assigned = np.zeros_like(spikes, dtype=bool)
    for mask in masks.values():
        assigned |= mask
    if not bool(np.all(assigned)):
        raise AssertionError("primary neuron states must cover every evaluated timestep")
    return {
        "primary_masks": masks,
        "overlays": {
            "high_input_drive": high_drive,
            "voltage_rising": dvdt > 0,
            "voltage_falling": dvdt < 0,
        },
        "parameters": {
            "high_drive_event_count_threshold": None if event_count is None else float(np.quantile(event_count, high_drive_quantile)),
            "uses_signed_input_drive": inputs is not None,
        },
    }


def _state_row(
    name: str,
    mask: np.ndarray,
    residual_mv: np.ndarray,
    dvdt_residual: np.ndarray,
    pred_spikes: np.ndarray,
    target_spikes: np.ndarray,
) -> dict[str, Any]:
    count = int(mask.sum())
    if count == 0:
        return {
            "state": name,
            "count": 0,
            "trace_support": 0,
            "fraction": 0.0,
            "voltage_rmse_mv": None,
            "voltage_mae_mv": None,
            "dvdt_mae_mv_per_step": None,
            "spike_target_count": 0,
            "spike_predicted_count": 0,
        }
    target_count = int(target_spikes[mask].sum())
    pred_count = int(pred_spikes[mask].sum())
    return {
        "state": name,
        "count": count,
        "trace_support": int(np.any(mask, axis=1).sum()),
        "fraction": float(count / mask.size),
        "voltage_rmse_mv": float(np.sqrt(np.mean(residual_mv[mask] ** 2))),
        "voltage_mae_mv": float(np.mean(np.abs(residual_mv[mask]))),
        "dvdt_mae_mv_per_step": float(np.mean(np.abs(dvdt_residual[mask]))),
        "spike_target_count": target_count,
        "spike_predicted_count": pred_count,
        "spike_count_error": pred_count - target_count,
    }


def _as_trace_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim != 3:
        leading = int(np.prod(array.shape[:-2]))
        return array.reshape(leading, array.shape[-2], array.shape[-1])
    return array


def _apply_mask(array: np.ndarray, ignore_start: int) -> np.ndarray:
    return array[:, ignore_start:, :] if ignore_start > 0 else array


def _per_trace_diff(values: np.ndarray) -> np.ndarray:
    return np.diff(values, axis=1, prepend=values[:, :1])


def _dilate_after(mask: np.ndarray, after: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for offset in range(1, min(after, mask.shape[1] - 1) + 1):
        out[:, offset:] |= mask[:, : mask.shape[1] - offset]
    return out


def _dilate_before(mask: np.ndarray, before: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for offset in range(1, min(before, mask.shape[1] - 1) + 1):
        out[:, : mask.shape[1] - offset] |= mask[:, offset:]
    return out


def _input_drive(
    inputs: np.ndarray | None,
    shape: tuple[int, int],
    high_drive_quantile: float,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    if inputs is None:
        return None, np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=bool)
    arr = _as_trace_array(np.asarray(inputs))
    if arr.shape[:2] != shape:
        raise ValueError(f"inputs must match batch/time shape {shape}, got {arr.shape[:2]}")
    event_count = np.count_nonzero(arr, axis=-1)
    threshold = float(np.quantile(event_count, high_drive_quantile))
    high_drive = event_count >= threshold
    if threshold <= 0:
        high_drive = event_count > 0
    excitatory = np.count_nonzero(arr > 0, axis=-1)
    inhibitory = np.count_nonzero(arr < 0, axis=-1)
    return event_count, excitatory - inhibitory, high_drive


def _robust_scale(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 1.0
    scale = float(np.quantile(arr, 0.95) - np.quantile(arr, 0.05))
    return max(scale, 1.0)


def _spike_timing_summary(pred_spikes: np.ndarray, target_spikes: np.ndarray) -> dict[str, Any]:
    target_idx = np.flatnonzero(target_spikes)
    pred_idx = np.flatnonzero(pred_spikes)
    if target_idx.size == 0 or pred_idx.size == 0:
        return {
            "precision_at_2steps": 0.0,
            "recall_at_2steps": 0.0,
            "f1_at_2steps": 0.0,
            "mean_timing_error_steps": None,
        }
    target_dists = np.min(np.abs(target_idx[:, None] - pred_idx[None, :]), axis=1)
    pred_dists = np.min(np.abs(pred_idx[:, None] - target_idx[None, :]), axis=1)
    recall = float(np.mean(target_dists <= 2))
    precision = float(np.mean(pred_dists <= 2))
    f1 = 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))
    return {
        "precision_at_2steps": precision,
        "recall_at_2steps": recall,
        "f1_at_2steps": f1,
        "mean_timing_error_steps": float(np.mean(target_dists)),
    }
