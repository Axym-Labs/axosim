from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bnn_sim.checkpoint import load_checkpoint
from bnn_sim.data import ShardedNeuronIODataset
from bnn_sim.evaluate import _synchronize_device, _tensor_from_numpy
from bnn_sim.official.constants import DEFAULT_Y_TRAIN_SOMA_SCALE


def analyze_mamba_rnn_failure_modes(
    *,
    data_path: str | Path,
    rnn_checkpoint: str | Path,
    mamba_checkpoint: str | Path,
    output_dir: str | Path,
    batch_size: int = 16,
    device: str = "cuda",
    ignore_start: int = 500,
    max_samples: int | None = None,
    y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE,
) -> dict[str, Any]:
    dataset = ShardedNeuronIODataset(data_path, cache_shards=1)
    sample_count = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
    rnn, _ = load_checkpoint(rnn_checkpoint, map_location=device)
    mamba, _ = load_checkpoint(mamba_checkpoint, map_location=device)
    rnn = rnn.to(device).eval()
    mamba = mamba.to(device).eval()

    targets: list[np.ndarray] = []
    spikes: list[np.ndarray] = []
    rnn_preds: list[np.ndarray] = []
    mamba_preds: list[np.ndarray] = []
    _synchronize_device(device)
    with torch.inference_mode():
        for start in range(0, sample_count, batch_size):
            indices = range(start, min(start + batch_size, sample_count))
            x_np, y_np = dataset.get_batch(indices)
            x = _tensor_from_numpy(x_np, device, pin_memory=False, non_blocking=True)
            rnn_pred = rnn(x).detach().cpu().numpy()
            mamba_pred = mamba(x).detach().cpu().numpy()
            for row in range(rnn_pred.shape[0]):
                targets.append(y_np[row, ignore_start:, 1].astype(np.float64, copy=True) / y_train_soma_scale)
                spikes.append((y_np[row, ignore_start:, 0] > 0.5).astype(bool, copy=True))
                rnn_preds.append(rnn_pred[row, ignore_start:, 1].astype(np.float64, copy=True))
                mamba_preds.append(mamba_pred[row, ignore_start:, 1].astype(np.float64, copy=True))
    _synchronize_device(device)

    rnn_mv = _affine_calibrate_trace_list(rnn_preds, targets)
    mamba_mv = _affine_calibrate_trace_list(mamba_preds, targets)
    target = np.concatenate(targets)
    spike_flat = np.concatenate(spikes)
    rnn_flat = np.concatenate(rnn_mv)
    mamba_flat = np.concatenate(mamba_mv)
    trace_ids = np.concatenate([np.full(trace.size, index, dtype=np.int32) for index, trace in enumerate(targets)])
    rnn_resid = rnn_flat - target
    mamba_resid = mamba_flat - target
    gap = np.square(mamba_resid) - np.square(rnn_resid)
    target_diff = _concat_diffs(targets)
    rnn_diff = _concat_diffs(rnn_mv)
    mamba_diff = _concat_diffs(mamba_mv)

    masks = _analysis_masks(target, target_diff, spike_flat, trace_ids)
    localization_rows = [_mask_gap_row(name, mask, gap, rnn_resid, mamba_resid) for name, mask in masks.items()]
    compression = _compression_metrics(target, rnn_flat, mamba_flat, target_diff, rnn_diff, mamba_diff)
    peak_rows = _peak_rows(targets, spikes, rnn_mv, mamba_mv)
    peak_summary = _peak_summary(peak_rows)
    global_metrics = {
        "samples": sample_count,
        "timesteps": int(target.size),
        "rnn_rmse_mv": _rmse(rnn_resid),
        "mamba_rmse_mv": _rmse(mamba_resid),
        "mamba_minus_rnn_rmse_mv": _rmse(mamba_resid) - _rmse(rnn_resid),
        "rnn_mae_mv": float(np.mean(np.abs(rnn_resid))),
        "mamba_mae_mv": float(np.mean(np.abs(mamba_resid))),
        "rnn_corr": _corr(rnn_flat, target),
        "mamba_corr": _corr(mamba_flat, target),
        "total_mse_gap_sum": float(np.sum(gap)),
    }
    report = {
        "global": global_metrics,
        "compression": compression,
        "localization_rows": localization_rows,
        "peak_summary": peak_summary,
        "hypothesis_readout": _hypothesis_readout(global_metrics, compression, localization_rows, peak_summary),
        "paths": {
            "data": str(data_path),
            "rnn_checkpoint": str(rnn_checkpoint),
            "mamba_checkpoint": str(mamba_checkpoint),
        },
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "failure_mode_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_rows(output_path / "localization.csv", localization_rows)
    _write_rows(output_path / "peaks.csv", peak_rows)
    _write_summary(output_path / "summary.csv", report)
    _write_gap_svg(output_path / "gap_localization.svg", localization_rows)
    _write_peak_svg(output_path / "peak_amplitude_error.svg", peak_summary)
    return report


def _analysis_masks(target: np.ndarray, target_diff: np.ndarray, spikes: np.ndarray, trace_ids: np.ndarray) -> dict[str, np.ndarray]:
    abs_diff = np.abs(target_diff)
    high_voltage_p90 = target >= np.quantile(target, 0.9)
    high_voltage_p99 = target >= np.quantile(target, 0.99)
    high_dvdt_p90 = abs_diff >= np.quantile(abs_diff, 0.9)
    high_dvdt_p99 = abs_diff >= np.quantile(abs_diff, 0.99)
    near_spike_5 = _near_event_mask(spikes, trace_ids, 5)
    near_spike_25 = _near_event_mask(spikes, trace_ids, 25)
    return {
        "all": np.ones_like(target, dtype=bool),
        "near_target_spike_5ms": near_spike_5,
        "near_target_spike_25ms": near_spike_25,
        "far_from_target_spike_gt25ms": ~near_spike_25,
        "target_high_voltage_p90": high_voltage_p90,
        "target_high_voltage_p99": high_voltage_p99,
        "target_high_abs_dvdt_p90": high_dvdt_p90,
        "target_high_abs_dvdt_p99": high_dvdt_p99,
        "union_spike5_or_high_voltage_or_high_dvdt": near_spike_5 | high_voltage_p90 | high_dvdt_p90,
    }


def _mask_gap_row(name: str, mask: np.ndarray, gap: np.ndarray, rnn_resid: np.ndarray, mamba_resid: np.ndarray) -> dict[str, float | int | str]:
    selected_gap = gap[mask]
    total_positive_gap = float(np.sum(np.maximum(gap, 0.0)))
    positive_gap = float(np.sum(np.maximum(selected_gap, 0.0)))
    fraction = float(mask.mean())
    gap_fraction = positive_gap / total_positive_gap if total_positive_gap > 0 else float("nan")
    return {
        "region": name,
        "count": int(mask.sum()),
        "fraction_timesteps": fraction,
        "rnn_rmse_mv": _rmse(rnn_resid[mask]),
        "mamba_rmse_mv": _rmse(mamba_resid[mask]),
        "mamba_minus_rnn_rmse_mv": _rmse(mamba_resid[mask]) - _rmse(rnn_resid[mask]),
        "mean_mse_gap": float(np.mean(selected_gap)) if selected_gap.size else float("nan"),
        "positive_gap_fraction": gap_fraction,
        "gap_enrichment_vs_timestep_fraction": gap_fraction / fraction if fraction > 0 else float("nan"),
        "mamba_residual_bias_mv": float(np.mean(mamba_resid[mask])) if mask.any() else float("nan"),
        "rnn_residual_bias_mv": float(np.mean(rnn_resid[mask])) if mask.any() else float("nan"),
    }


def _compression_metrics(
    target: np.ndarray,
    rnn: np.ndarray,
    mamba: np.ndarray,
    target_diff: np.ndarray,
    rnn_diff: np.ndarray,
    mamba_diff: np.ndarray,
) -> dict[str, float]:
    return {
        "rnn_total_variation_ratio": float(np.mean(np.abs(rnn_diff)) / np.mean(np.abs(target_diff))),
        "mamba_total_variation_ratio": float(np.mean(np.abs(mamba_diff)) / np.mean(np.abs(target_diff))),
        "rnn_dvdt_std_ratio": float(np.std(rnn_diff) / np.std(target_diff)),
        "mamba_dvdt_std_ratio": float(np.std(mamba_diff) / np.std(target_diff)),
        "rnn_dvdt_corr": _corr(rnn_diff, target_diff),
        "mamba_dvdt_corr": _corr(mamba_diff, target_diff),
        "rnn_p99_minus_median_ratio": _upper_range_ratio(rnn, target),
        "mamba_p99_minus_median_ratio": _upper_range_ratio(mamba, target),
        "rnn_p01_to_p99_range_ratio": _range_ratio(rnn, target),
        "mamba_p01_to_p99_range_ratio": _range_ratio(mamba, target),
        "mamba_peak_bias_high_voltage_p99_mv": float(np.mean((mamba - target)[target >= np.quantile(target, 0.99)])),
        "rnn_peak_bias_high_voltage_p99_mv": float(np.mean((rnn - target)[target >= np.quantile(target, 0.99)])),
    }


def _peak_rows(
    targets: list[np.ndarray],
    spikes: list[np.ndarray],
    rnn: list[np.ndarray],
    mamba: list[np.ndarray],
) -> list[dict[str, float | int | str]]:
    rows = []
    for trace_index, target in enumerate(targets):
        spike_indices = np.flatnonzero(spikes[trace_index])
        for spike_index in spike_indices:
            start = max(0, spike_index - 5)
            stop = min(target.size, spike_index + 11)
            target_window = target[start:stop]
            rnn_window = rnn[trace_index][start:stop]
            mamba_window = mamba[trace_index][start:stop]
            target_peak_index = int(np.argmax(target_window))
            rnn_peak_index = int(np.argmax(rnn_window))
            mamba_peak_index = int(np.argmax(mamba_window))
            target_peak = float(target_window[target_peak_index])
            rnn_peak = float(rnn_window[rnn_peak_index])
            mamba_peak = float(mamba_window[mamba_peak_index])
            rows.append(
                {
                    "trace_index": trace_index,
                    "spike_index": int(spike_index),
                    "target_peak_mv": target_peak,
                    "rnn_peak_mv": rnn_peak,
                    "mamba_peak_mv": mamba_peak,
                    "rnn_peak_error_mv": rnn_peak - target_peak,
                    "mamba_peak_error_mv": mamba_peak - target_peak,
                    "rnn_peak_lag_ms": rnn_peak_index - target_peak_index,
                    "mamba_peak_lag_ms": mamba_peak_index - target_peak_index,
                }
            )
    return rows


def _peak_summary(rows: list[dict[str, float | int | str]]) -> dict[str, float | int]:
    if not rows:
        return {"spike_windows": 0}
    rnn_errors = np.asarray([float(row["rnn_peak_error_mv"]) for row in rows], dtype=np.float64)
    mamba_errors = np.asarray([float(row["mamba_peak_error_mv"]) for row in rows], dtype=np.float64)
    rnn_lags = np.asarray([float(row["rnn_peak_lag_ms"]) for row in rows], dtype=np.float64)
    mamba_lags = np.asarray([float(row["mamba_peak_lag_ms"]) for row in rows], dtype=np.float64)
    return {
        "spike_windows": len(rows),
        "rnn_mean_peak_error_mv": float(np.mean(rnn_errors)),
        "mamba_mean_peak_error_mv": float(np.mean(mamba_errors)),
        "rnn_median_peak_error_mv": float(np.median(rnn_errors)),
        "mamba_median_peak_error_mv": float(np.median(mamba_errors)),
        "rnn_fraction_peak_underestimated": float(np.mean(rnn_errors < 0.0)),
        "mamba_fraction_peak_underestimated": float(np.mean(mamba_errors < 0.0)),
        "rnn_mean_abs_peak_lag_ms": float(np.mean(np.abs(rnn_lags))),
        "mamba_mean_abs_peak_lag_ms": float(np.mean(np.abs(mamba_lags))),
    }


def _hypothesis_readout(
    global_metrics: dict[str, float | int],
    compression: dict[str, float],
    localization_rows: list[dict[str, float | int | str]],
    peak_summary: dict[str, float | int],
) -> dict[str, Any]:
    row_by_name = {str(row["region"]): row for row in localization_rows}
    far_gap = float(row_by_name["far_from_target_spike_gt25ms"]["mamba_minus_rnn_rmse_mv"])
    near_gap = float(row_by_name["near_target_spike_5ms"]["mamba_minus_rnn_rmse_mv"])
    union_enrichment = float(row_by_name["union_spike5_or_high_voltage_or_high_dvdt"]["gap_enrichment_vs_timestep_fraction"])
    smoothing_signal = float(compression["mamba_total_variation_ratio"] - compression["rnn_total_variation_ratio"])
    peak_error_delta = float(peak_summary.get("mamba_mean_peak_error_mv", 0.0)) - float(peak_summary.get("rnn_mean_peak_error_mv", 0.0))
    return {
        "localized_failure_supported": bool(union_enrichment > 1.5 and near_gap > far_gap),
        "global_failure_supported": bool(far_gap > 0.0),
        "oversmoothing_supported": bool(smoothing_signal < 0.0 or peak_error_delta < 0.0),
        "near_vs_far_gap_delta_mv": near_gap - far_gap,
        "far_from_spike_gap_mv": far_gap,
        "union_gap_enrichment": union_enrichment,
        "mamba_minus_rnn_total_variation_ratio": smoothing_signal,
        "mamba_minus_rnn_mean_peak_error_mv": peak_error_delta,
        "interpretation": (
            "Localized failure is supported when the Mamba-RNN gap is enriched near spikes/high-dVdt/high-voltage events. "
            "Global failure is supported when Mamba is also worse far from spikes. Oversmoothing is supported when Mamba has lower "
            "total variation/dVdt scale or stronger peak underestimation."
        ),
    }


def _near_event_mask(events: np.ndarray, trace_ids: np.ndarray, window: int) -> np.ndarray:
    result = np.zeros_like(events, dtype=bool)
    for trace_id in np.unique(trace_ids):
        indices = np.flatnonzero(trace_ids == trace_id)
        event_indices = np.flatnonzero(events[indices])
        for event_index in event_indices:
            start = max(0, event_index - window)
            stop = min(indices.size, event_index + window + 1)
            result[indices[start:stop]] = True
    return result


def _affine_calibrate_trace_list(predictions: list[np.ndarray], targets_mv: list[np.ndarray]) -> list[np.ndarray]:
    pred_all = np.concatenate(predictions)
    target_all = np.concatenate(targets_mv)
    pred_std = float(pred_all.std())
    if pred_std == 0.0:
        raise ValueError("cannot affine-calibrate zero-variance predictions")
    calibrated = (pred_all - pred_all.mean()) / pred_std * target_all.std() + target_all.mean()
    result = []
    offset = 0
    for trace in predictions:
        result.append(calibrated[offset : offset + trace.size].copy())
        offset += trace.size
    return result


def _concat_diffs(traces: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.concatenate(([0.0], np.diff(trace))) for trace in traces])


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _upper_range_ratio(pred: np.ndarray, target: np.ndarray) -> float:
    pred_range = float(np.quantile(pred, 0.99) - np.quantile(pred, 0.5))
    target_range = float(np.quantile(target, 0.99) - np.quantile(target, 0.5))
    return pred_range / target_range if target_range else float("nan")


def _range_ratio(pred: np.ndarray, target: np.ndarray) -> float:
    pred_range = float(np.quantile(pred, 0.99) - np.quantile(pred, 0.01))
    target_range = float(np.quantile(target, 0.99) - np.quantile(target, 0.01))
    return pred_range / target_range if target_range else float("nan")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    rows = []
    for section, payload in report.items():
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, (int, float, str, bool)):
                    rows.append({"section": section, "metric": key, "value": value})
    _write_rows(path, rows)


def _write_gap_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    values = [float(row["mamba_minus_rnn_rmse_mv"]) for row in rows]
    labels = [str(row["region"]) for row in rows]
    _write_simple_bar_svg(path, labels, values, "Mamba minus RNN RMSE by region", "delta RMSE mV")


def _write_peak_svg(path: Path, summary: dict[str, Any]) -> None:
    labels = ["RNN mean peak error", "Mamba mean peak error", "RNN abs lag", "Mamba abs lag"]
    values = [
        float(summary.get("rnn_mean_peak_error_mv", 0.0)),
        float(summary.get("mamba_mean_peak_error_mv", 0.0)),
        float(summary.get("rnn_mean_abs_peak_lag_ms", 0.0)),
        float(summary.get("mamba_mean_abs_peak_lag_ms", 0.0)),
    ]
    _write_simple_bar_svg(path, labels, values, "Spike-window peak error and lag", "mV or ms")


def _write_simple_bar_svg(path: Path, labels: list[str], values: list[float], title: str, y_label: str) -> None:
    width, height = 980, 460
    left, right, top, bottom = 76, 28, 48, 140
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min = min(0.0, min(values))
    y_max = max(0.0, max(values))
    if y_min == y_max:
        y_max = y_min + 1.0
    bar_gap = plot_w / max(len(values), 1)
    bar_w = bar_gap * 0.68
    zero_y = top + (y_max - 0.0) / (y_max - y_min) * plot_h
    svg = [_svg_header(width, height), f'<text x="{left}" y="28" class="title">{_esc(title)}</text>']
    svg.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width-right}" y2="{zero_y:.1f}" class="axis"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    svg.append(f'<text x="20" y="{top + plot_h / 2}" class="ylabel" transform="rotate(-90 20 {top + plot_h / 2})">{_esc(y_label)}</text>')
    for index, value in enumerate(values):
        x = left + index * bar_gap + (bar_gap - bar_w) / 2
        y = top + (y_max - value) / (y_max - y_min) * plot_h
        h = abs(y - zero_y)
        fill = "#b33f3f" if value > 0 else "#1f77b4"
        svg.append(f'<rect x="{x:.1f}" y="{min(y, zero_y):.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{fill}"/>')
        svg.append(f'<text x="{x + bar_w / 2:.1f}" y="{min(y, zero_y) - 5:.1f}" class="tick">{value:.3g}</text>')
        svg.append(f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 18}" class="xtick">{_esc(labels[index])}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#1f2933}.title{font-size:18px;font-weight:700}'
        '.axis{stroke:#3f4a54;stroke-width:1.2}.tick{font-size:11px;text-anchor:middle}.xtick{font-size:9px;text-anchor:middle}.ylabel{font-size:13px;font-weight:600}</style>'
    )


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
