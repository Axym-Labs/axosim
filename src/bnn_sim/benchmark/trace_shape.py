"""Trace-shape metrics beyond pointwise RMSE.

Pointwise RMSE is useful but brittle for neuron traces: a small AP timing shift
can dominate the score, while a model can also match voltage levels and miss
derivatives, voltage distributions, or spike-triggered waveforms.  This module
keeps the calculation dependency-light and reportable.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_trace_shape_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    soma_scale: float = 0.1,
    soma_bias_mv: float = -67.7,
    ignore_start: int = 0,
    calibrated_spike_threshold: float | None = None,
    lag_tolerance_steps: int = 2,
    smoothing_windows: tuple[int, ...] = (5, 20),
    spike_pre_steps: int = 5,
    spike_post_steps: int = 15,
    quantile_count: int = 21,
) -> dict[str, Any]:
    """Compute multi-scale, timing-tolerant voltage trace-shape metrics."""
    pred = _apply_time_mask(_as_trace_array(np.asarray(prediction)), ignore_start)
    tgt = _apply_time_mask(_as_trace_array(np.asarray(target)), ignore_start)
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction and target shapes must match, got {pred.shape} and {tgt.shape}")
    if pred.shape[1] == 0:
        raise ValueError("ignore_start leaves no timesteps to evaluate")
    if lag_tolerance_steps < 0:
        raise ValueError("lag_tolerance_steps must be non-negative")

    pred_soma_mv = pred[..., 1] / soma_scale + soma_bias_mv
    tgt_soma_mv = tgt[..., 1] / soma_scale + soma_bias_mv
    residual_mv = pred_soma_mv - tgt_soma_mv
    pred_dvdt = _per_trace_diff(pred_soma_mv)
    tgt_dvdt = _per_trace_diff(tgt_soma_mv)
    dvdt_residual = pred_dvdt - tgt_dvdt
    threshold = 0.0 if calibrated_spike_threshold is None else calibrated_spike_threshold
    pred_spikes = pred[..., 0] > 0.0 if calibrated_spike_threshold is None else pred[..., 0] >= threshold
    target_spikes = tgt[..., 0] > 0.5

    lowpass_rows = []
    for window in smoothing_windows:
        if window <= 0:
            raise ValueError("smoothing windows must be positive")
        pred_smooth = _moving_average(pred_soma_mv, window)
        tgt_smooth = _moving_average(tgt_soma_mv, window)
        lowpass_rows.append({
            "window_steps": int(window),
            "rmse_mv": _rmse(pred_smooth - tgt_smooth),
            "mae_mv": float(np.mean(np.abs(pred_smooth - tgt_smooth))),
        })

    lag_tolerant = _lag_tolerant_voltage_distance(
        pred_soma_mv,
        tgt_soma_mv,
        tolerance=lag_tolerance_steps,
    )
    quantile_distance = _voltage_quantile_distance(
        pred_soma_mv,
        tgt_soma_mv,
        quantile_count=quantile_count,
    )
    spike_waveform = _spike_triggered_waveform_distance(
        pred_soma_mv,
        tgt_soma_mv,
        pred_spikes,
        target_spikes,
        tolerance=lag_tolerance_steps,
        pre=spike_pre_steps,
        post=spike_post_steps,
    )
    voltage_scale = _robust_scale(tgt_soma_mv)
    dvdt_scale = _robust_scale(tgt_dvdt)
    components = {
        "lag_tolerant_voltage": lag_tolerant["symmetric_rmse_mv"] / voltage_scale,
        "multiscale_lowpass": float(np.mean([row["rmse_mv"] for row in lowpass_rows])) / voltage_scale,
        "dvdt": float(np.mean(np.abs(dvdt_residual))) / dvdt_scale,
        "voltage_distribution": quantile_distance["mean_abs_quantile_error_mv"] / voltage_scale,
    }
    if spike_waveform["matched_spikes"] > 0 and spike_waveform["waveform_rmse_mv"] is not None:
        components["spike_triggered_waveform"] = spike_waveform["waveform_rmse_mv"] / voltage_scale
    if spike_waveform["match_fraction"] is not None:
        components["spike_waveform_coverage"] = 1.0 - spike_waveform["match_fraction"]
    trace_shape_distance = float(np.mean(list(components.values())))
    return {
        "kind": "multiscale_trace_shape_fidelity",
        "parameters": {
            "lag_tolerance_steps": int(lag_tolerance_steps),
            "smoothing_windows": [int(x) for x in smoothing_windows],
            "spike_pre_steps": int(spike_pre_steps),
            "spike_post_steps": int(spike_post_steps),
            "quantile_count": int(quantile_count),
            "spike_threshold": float(threshold),
            "spike_threshold_semantics": (
                "calibrated threshold learned on a separate split"
                if calibrated_spike_threshold is not None
                else "fixed zero-logit operating point"
            ),
        },
        "pointwise": {
            "rmse_mv": _rmse(residual_mv),
            "mae_mv": float(np.mean(np.abs(residual_mv))),
            "dvdt_rmse_mv_per_step": _rmse(dvdt_residual),
            "dvdt_mae_mv_per_step": float(np.mean(np.abs(dvdt_residual))),
        },
        "lag_tolerant_voltage": lag_tolerant,
        "multiscale_lowpass": lowpass_rows,
        "voltage_quantile_distance": quantile_distance,
        "spike_triggered_waveform": spike_waveform,
        "trace_shape_distance": trace_shape_distance,
        "trace_shape_distance_components": components,
        "what_failure_means": (
            "High trace-shape distance means the surrogate misses multi-scale "
            "voltage dynamics, even after allowing small event-timing shifts. "
            "The components separate slow voltage level, local timing tolerance, "
            "dV/dt, voltage-distribution, and spike-triggered waveform failures."
        ),
    }


def _as_trace_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim != 3:
        leading = int(np.prod(array.shape[:-2]))
        return array.reshape(leading, array.shape[-2], array.shape[-1])
    return array


def _apply_time_mask(array: np.ndarray, ignore_start: int) -> np.ndarray:
    return array[:, ignore_start:, :] if ignore_start > 0 else array


def _per_trace_diff(values: np.ndarray) -> np.ndarray:
    return np.diff(values, axis=1, prepend=values[:, :1])


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window == 1:
        return values.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, ((0, 0), (left, right)), mode="edge")
    cumsum = np.cumsum(padded, axis=1)
    cumsum = np.pad(cumsum, ((0, 0), (1, 0)), mode="constant")
    return (cumsum[:, window:] - cumsum[:, :-window]) / float(window)


def _lag_tolerant_voltage_distance(pred: np.ndarray, target: np.ndarray, *, tolerance: int) -> dict[str, Any]:
    pred_to_target = _directed_local_min_square(pred, target, tolerance)
    target_to_pred = _directed_local_min_square(target, pred, tolerance)
    return {
        "tolerance_steps": int(tolerance),
        "prediction_to_target_rmse_mv": float(np.sqrt(np.mean(pred_to_target))),
        "target_to_prediction_rmse_mv": float(np.sqrt(np.mean(target_to_pred))),
        "symmetric_rmse_mv": float(np.sqrt(np.mean(0.5 * (pred_to_target + target_to_pred)))),
        "semantics": "symmetric local-window voltage distance; small timing shifts are penalized less than in pointwise RMSE",
    }


def _directed_local_min_square(reference: np.ndarray, candidate: np.ndarray, tolerance: int) -> np.ndarray:
    options = []
    for shift in range(-tolerance, tolerance + 1):
        shifted = _shift_with_edge(candidate, shift)
        options.append((shifted - reference) ** 2)
    return np.min(np.stack(options, axis=0), axis=0)


def _shift_with_edge(values: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return values
    out = np.empty_like(values)
    if shift > 0:
        out[:, shift:] = values[:, :-shift]
        out[:, :shift] = values[:, :1]
    else:
        amount = -shift
        out[:, :-amount] = values[:, amount:]
        out[:, -amount:] = values[:, -1:]
    return out


def _voltage_quantile_distance(pred: np.ndarray, target: np.ndarray, *, quantile_count: int) -> dict[str, Any]:
    if quantile_count < 2:
        raise ValueError("quantile_count must be at least 2")
    q = np.linspace(0.0, 1.0, quantile_count)
    pred_q = np.quantile(pred.reshape(-1), q)
    target_q = np.quantile(target.reshape(-1), q)
    return {
        "quantile_count": int(quantile_count),
        "mean_abs_quantile_error_mv": float(np.mean(np.abs(pred_q - target_q))),
        "max_abs_quantile_error_mv": float(np.max(np.abs(pred_q - target_q))),
        "semantics": "1D voltage-distribution distance over evenly spaced quantiles",
    }


def _spike_triggered_waveform_distance(
    pred_soma: np.ndarray,
    target_soma: np.ndarray,
    pred_spikes: np.ndarray,
    target_spikes: np.ndarray,
    *,
    tolerance: int,
    pre: int,
    post: int,
) -> dict[str, Any]:
    matched_errors = []
    target_count = int(target_spikes.sum())
    pred_count = int(pred_spikes.sum())
    evaluable_target_count = 0
    for trace in range(target_spikes.shape[0]):
        pred_idx = np.flatnonzero(pred_spikes[trace])
        for target_index in np.flatnonzero(target_spikes[trace]):
            if target_index - pre < 0 or target_index + post >= target_spikes.shape[1]:
                continue
            evaluable_target_count += 1
            if pred_idx.size == 0:
                continue
            nearest = int(pred_idx[np.argmin(np.abs(pred_idx - target_index))])
            if abs(nearest - int(target_index)) > tolerance:
                continue
            if nearest - pre < 0 or nearest + post >= target_spikes.shape[1]:
                continue
            target_wave = target_soma[trace, target_index - pre: target_index + post + 1]
            pred_wave = pred_soma[trace, nearest - pre: nearest + post + 1]
            matched_errors.append(_rmse(pred_wave - target_wave))
    matched = len(matched_errors)
    return {
        "target_spikes": target_count,
        "evaluable_target_spikes": evaluable_target_count,
        "predicted_spikes": pred_count,
        "matched_spikes": matched,
        "match_fraction": float(matched / evaluable_target_count) if evaluable_target_count > 0 else None,
        "waveform_rmse_mv": float(np.mean(matched_errors)) if matched_errors else None,
        "window": {"pre_steps": int(pre), "post_steps": int(post)},
        "semantics": "AP-neighborhood waveform RMSE after nearest-spike alignment within the configured tolerance",
    }


def _robust_scale(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 1.0
    scale = float(np.quantile(arr, 0.95) - np.quantile(arr, 0.05))
    return max(scale, 1.0)


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))
