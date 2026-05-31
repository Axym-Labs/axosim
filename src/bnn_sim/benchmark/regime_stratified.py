"""M2: Regime-Stratified Error Metrics.

Decomposes prediction error by biophysical regime:
- Voltage regime: hyperpolarized, resting, depolarized, spike
- Slope regime: low dV/dt, medium dV/dt, high dV/dt, spike-rise
- Spike proximity: near-spike vs far-from-spike timesteps
- Post-event: error conditioned on recent input events
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _bin_errors(
    values: np.ndarray,
    residual: np.ndarray,
    bin_edges: np.ndarray,
    bin_labels: list[str],
) -> list[dict[str, Any]]:
    """Bin residual errors by value ranges."""
    rows = []
    for i in range(len(bin_edges) - 1):
        mask = (values >= bin_edges[i]) & (values < bin_edges[i + 1])
        if i == len(bin_edges) - 2:
            mask = (values >= bin_edges[i]) & (values <= bin_edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            rows.append({
                "regime_label": bin_labels[i],
                "bin_low": float(bin_edges[i]),
                "bin_high": float(bin_edges[i + 1]),
                "count": 0,
                "rmse": None,
                "mae": None,
                "max_error": None,
            })
            continue
        rows.append({
            "regime_label": bin_labels[i],
            "bin_low": float(bin_edges[i]),
            "bin_high": float(bin_edges[i + 1]),
            "count": count,
            "rmse": float(np.sqrt(np.mean(residual[mask] ** 2))),
            "mae": float(np.mean(np.abs(residual[mask]))),
            "max_error": float(np.max(np.abs(residual[mask]))),
        })
    return rows


def _as_trace_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim != 3:
        leading = int(np.prod(array.shape[:-2]))
        return array.reshape(leading, array.shape[-2], array.shape[-1])
    return array


def _dilate_per_trace(mask: np.ndarray, before: int, after: int) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    for row in range(mask.shape[0]):
        indices = np.flatnonzero(mask[row])
        for index in indices:
            start = max(0, index - before)
            stop = min(mask.shape[1], index + after + 1)
            result[row, start:stop] = True
    return result


def _post_event_per_trace(event_mask: np.ndarray, lookback: int) -> np.ndarray:
    result = np.zeros_like(event_mask, dtype=bool)
    for offset in range(lookback + 1):
        result[:, offset:] |= event_mask[:, : event_mask.shape[1] - offset]
    return result


def _row_for_mask(label: str, residual: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    count = int(mask.sum())
    if count == 0:
        return {
            "regime_label": label,
            "count": 0,
            "rmse": None,
            "mae": None,
            "max_error": None,
        }
    return {
        "regime_label": label,
        "count": count,
        "rmse": float(np.sqrt(np.mean(residual[mask] ** 2))),
        "mae": float(np.mean(np.abs(residual[mask]))),
        "max_error": float(np.max(np.abs(residual[mask]))),
    }


def compute_regime_stratified_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    inputs: np.ndarray | None = None,
    soma_scale: float = 0.1,
    ignore_start: int = 0,
) -> dict[str, Any]:
    """Compute M2: Regime-stratified error metrics.

    Args:
        prediction: Array of shape (..., 2) with [spike_score, soma_voltage] in normalized units.
        target: Array of shape (..., 2) with [spike_label, soma_voltage] in normalized units.
        inputs: Optional array of shape (...,) or (..., channels) with input events.
        soma_scale: Scale factor to convert normalized soma to mV (default 0.1).
        ignore_start: Number of initial timesteps to ignore.

    Returns:
        Dictionary with regime-stratified metrics.
    """
    pred = _as_trace_array(np.asarray(prediction))
    tgt = _as_trace_array(np.asarray(target))

    if pred.ndim < 2 or tgt.ndim < 2:
        raise ValueError(f"prediction and target must have at least 2 dims, got {pred.ndim} and {tgt.ndim}")

    if inputs is not None:
        inp = _as_trace_array(np.asarray(inputs))
    else:
        inp = None

    # Apply ignore_start
    if ignore_start > 0:
        pred = pred[:, ignore_start:, :]
        tgt = tgt[:, ignore_start:, :]
        if inp is not None:
            inp = inp[:, ignore_start:, :]

    if pred.shape[1] == 0:
        raise ValueError("ignore_start leaves no timesteps to evaluate")

    pred_soma = pred[..., 1]
    tgt_soma = tgt[..., 1]
    pred_spike = pred[..., 0]
    tgt_spike = tgt[..., 0]

    # Convert to mV
    pred_soma_mv = pred_soma / soma_scale
    tgt_soma_mv = tgt_soma / soma_scale
    residual_mv = pred_soma_mv - tgt_soma_mv
    tgt_soma_flat = tgt_soma_mv.reshape(-1)
    residual_flat = residual_mv.reshape(-1)
    pred_spike_flat = pred_spike.reshape(-1)
    tgt_spike_flat = tgt_spike.reshape(-1)

    # Global metrics
    global_rmse = float(np.sqrt(np.mean(residual_flat ** 2)))
    global_mae = float(np.mean(np.abs(residual_flat)))

    # M2a: Voltage regime stratification
    # Define bins based on target voltage distribution
    voltage_quantiles = np.quantile(tgt_soma_flat, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    voltage_labels = [
        "hyperpolarized",
        "resting_low",
        "resting_high",
        "depolarized",
        "spike_range",
    ]
    voltage_rows = _bin_errors(tgt_soma_flat, residual_flat, voltage_quantiles, voltage_labels)

    # M2b: Slope regime stratification
    # Compute dV/dt independently for each trace to avoid cross-trace leakage.
    dvdt = np.diff(tgt_soma_mv, axis=1, prepend=tgt_soma_mv[:, :1])
    abs_dvdt = np.abs(dvdt)
    abs_dvdt_flat = abs_dvdt.reshape(-1)
    slope_quantiles = np.quantile(abs_dvdt_flat, [0.0, 0.5, 0.8, 0.95, 1.0])
    slope_labels = [
        "low_dvdt",
        "medium_dvdt",
        "high_dvdt",
        "spike_rise",
    ]
    slope_rows = _bin_errors(abs_dvdt_flat, residual_flat, slope_quantiles, slope_labels)

    # M2c: Spike proximity
    # Define "near spike" as timesteps where target spike label is 1
    # and timesteps within ±2 steps of a spike
    spike_mask = tgt_spike > 0.5
    near_spike_mask = _dilate_per_trace(spike_mask, before=2, after=2)
    post_spike_mask = _dilate_per_trace(spike_mask, before=0, after=5) & ~spike_mask

    spike_proximity_rows = [
        _row_for_mask("near_spike", residual_mv, near_spike_mask),
        _row_for_mask("far_from_spike", residual_mv, ~near_spike_mask),
    ]

    # M2d: Post-event error (if inputs provided)
    post_event_rows = []
    post_event_mask = np.zeros_like(spike_mask, dtype=bool)
    if inp is not None:
        # Count events per timestep
        event_counts = np.count_nonzero(inp, axis=-1)

        # Within-window post-event mask (event at t-2, t-1, or t)
        post_event_mask = _post_event_per_trace(event_counts > 0, lookback=2)
        post_event_rows = [
            _row_for_mask("post_event", residual_mv, post_event_mask),
            _row_for_mask("no_recent_event", residual_mv, ~post_event_mask),
        ]

    # M2e: Spike timing precision
    # For each predicted spike, find nearest target spike; measure timing error
    spike_timing_errors = []
    spike_precision_recall = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Simple threshold-based spike detection on predictions
    pred_spike_binary = pred_spike_flat > 0.0
    tgt_spike_binary = tgt_spike_flat > 0.5

    if tgt_spike_binary.sum() > 0 and pred_spike_binary.sum() > 0:
        # Compute recall: for each target spike, nearest pred spike
        tgt_spike_idx = np.where(tgt_spike_binary)[0]
        pred_spike_idx = np.where(pred_spike_binary)[0]

        matched_dists = []
        for t_idx in tgt_spike_idx:
            dists = np.abs(pred_spike_idx - t_idx)
            if len(dists) > 0:
                matched_dists.append(np.min(dists))

        if matched_dists:
            spike_timing_errors = [float(d) for d in matched_dists]
            spike_precision_recall["recall"] = float(np.mean(np.array(matched_dists) <= 2))

        # Compute precision: for each pred spike, nearest target spike
        matched_dists_pred = []
        for p_idx in pred_spike_idx:
            dists = np.abs(tgt_spike_idx - p_idx)
            if len(dists) > 0:
                matched_dists_pred.append(np.min(dists))

        if matched_dists_pred:
            spike_precision_recall["precision"] = float(np.mean(np.array(matched_dists_pred) <= 2))

        if spike_precision_recall["precision"] + spike_precision_recall["recall"] > 0:
            spike_precision_recall["f1"] = float(
                2 * spike_precision_recall["precision"] * spike_precision_recall["recall"]
                / (spike_precision_recall["precision"] + spike_precision_recall["recall"])
            )

    far_subthreshold = (~near_spike_mask) & (~post_event_mask)
    mvp_regimes = [
        _row_for_mask("far_from_spike_subthreshold", residual_mv, far_subthreshold),
        _row_for_mask("near_spike", residual_mv, near_spike_mask),
        _row_for_mask("post_spike_refractory", residual_mv, post_spike_mask),
        _row_for_mask("post_input_event", residual_mv, post_event_mask),
    ]

    return {
        "global": {
            "rmse_mv": global_rmse,
            "mae_mv": global_mae,
            "timesteps": int(residual_flat.size),
        },
        "mvp_regimes": mvp_regimes,
        "voltage_regime": voltage_rows,
        "slope_regime": slope_rows,
        "spike_proximity": spike_proximity_rows,
        "post_event": post_event_rows,
        "spike_timing": {
            "mean_timing_error_steps": float(np.mean(spike_timing_errors)) if spike_timing_errors else None,
            "median_timing_error_steps": float(np.median(spike_timing_errors)) if spike_timing_errors else None,
            "precision_at_2steps": spike_precision_recall["precision"],
            "recall_at_2steps": spike_precision_recall["recall"],
            "f1_at_2steps": spike_precision_recall["f1"],
        },
    }
