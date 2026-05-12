from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from bnn_sim.data import ShardedNeuronIODataset
from bnn_sim.evaluate import _first_window_starts, _metric_mask, _synchronize_device, _tensor_from_numpy
from bnn_sim.metrics import binary_auc
from bnn_sim.official.constants import DEFAULT_Y_TRAIN_SOMA_SCALE


def diagnose_dataset(
    model: torch.nn.Module,
    dataset: ShardedNeuronIODataset,
    *,
    output_dir: str | Path,
    batch_size: int = 8,
    device: str = "cpu",
    y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE,
    ignore_start: int = 0,
    mask_mode: str = "ignore-start",
    stitch_burn_in: int = 150,
    soma_affine_calibration: bool = False,
    max_samples: int | None = None,
    pin_memory: bool = False,
    non_blocking: bool = True,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sample_count = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
    first_window_starts = _first_window_starts(dataset) if mask_mode == "official-overlap" else {}

    model = model.to(device)
    model.eval()
    pred_soma_chunks: list[np.ndarray] = []
    target_soma_chunks: list[np.ndarray] = []
    spike_score_chunks: list[np.ndarray] = []
    spike_label_chunks: list[np.ndarray] = []
    event_count_chunks: list[np.ndarray] = []
    trace_ids: list[int] = []
    trace_lengths: list[int] = []

    _synchronize_device(device)
    start_time = time.perf_counter()
    eval_context = torch.no_grad() if getattr(model, "learn_memory_tau", False) else torch.inference_mode()
    with eval_context:
        for start in range(0, sample_count, batch_size):
            batch_indices = range(start, min(start + batch_size, sample_count))
            if hasattr(dataset, "get_batch"):
                x_np, y_np = dataset.get_batch(batch_indices)
            else:
                samples = [dataset[i] for i in batch_indices]
                x_np = np.stack([sample.inputs for sample in samples]).astype(np.float32, copy=False)
                y_np = np.stack([sample.targets for sample in samples]).astype(np.float32, copy=False)
            sample_ids = (
                dataset.get_sample_ids(batch_indices)
                if hasattr(dataset, "get_sample_ids")
                else [dataset[i].sample_id for i in batch_indices]
            )
            x = _tensor_from_numpy(x_np, device, pin_memory=pin_memory, non_blocking=non_blocking)
            pred = model(x).detach().cpu().numpy()
            mask = _metric_mask(
                sample_ids,
                sequence_length=pred.shape[1],
                ignore_start=ignore_start,
                mask_mode=mask_mode,
                stitch_burn_in=stitch_burn_in,
                first_window_starts=first_window_starts,
            )
            for row in range(pred.shape[0]):
                row_mask = mask[row]
                length = int(row_mask.sum())
                pred_soma_chunks.append(pred[row, :, 1][row_mask].astype(np.float64, copy=True))
                target_soma_chunks.append(y_np[row, :, 1][row_mask].astype(np.float64, copy=True))
                spike_score_chunks.append(pred[row, :, 0][row_mask].astype(np.float64, copy=True))
                spike_label_chunks.append(y_np[row, :, 0][row_mask].astype(np.float64, copy=True))
                event_count_chunks.append(np.count_nonzero(x_np[row], axis=-1)[row_mask].astype(np.float64, copy=True))
                trace_ids.extend([start + row] * length)
                trace_lengths.append(length)

    pred_soma = np.concatenate(pred_soma_chunks)
    target_soma = np.concatenate(target_soma_chunks)
    spike_scores = np.concatenate(spike_score_chunks)
    spike_labels = np.concatenate(spike_label_chunks)
    event_counts = np.concatenate(event_count_chunks)
    trace_ids_array = np.asarray(trace_ids, dtype=np.int32)
    if soma_affine_calibration:
        pred_soma = _affine_calibrate(pred_soma, target_soma)
    pred_soma_mv = pred_soma / y_train_soma_scale
    target_soma_mv = target_soma / y_train_soma_scale
    residual_mv = pred_soma_mv - target_soma_mv
    _synchronize_device(device)
    total_seconds = time.perf_counter() - start_time

    global_metrics = _global_voltage_metrics(pred_soma_mv, target_soma_mv, residual_mv)
    high_voltage_threshold = float(np.quantile(target_soma_mv, 0.9))
    high_voltage_mask = target_soma_mv >= high_voltage_threshold
    global_metrics.update(
        {
            "samples": sample_count,
            "timesteps": int(residual_mv.size),
            "total_seconds": total_seconds,
            "samples_per_second": sample_count / total_seconds if total_seconds > 0 else float("inf"),
            "soma_affine_calibration": bool(soma_affine_calibration),
            "metric_ignore_start": int(ignore_start),
            "metric_mask_mode": mask_mode,
            "near_threshold_proxy_target_soma_p90_mv": high_voltage_threshold,
            "near_threshold_proxy_rmse_mv": _rmse(residual_mv[high_voltage_mask]),
        }
    )
    voltage_rows = _bin_metric_rows(
        "target_soma_mv",
        target_soma_mv,
        residual_mv,
        bins=np.quantile(target_soma_mv, np.linspace(0.0, 1.0, 6)),
    )
    slope = np.concatenate([_within_trace_diff(chunk / y_train_soma_scale) for chunk in target_soma_chunks])
    pred_slope = np.concatenate([_within_trace_diff(chunk / y_train_soma_scale) for chunk in pred_soma_chunks])
    if soma_affine_calibration:
        pred_slope = _within_trace_values(pred_soma_mv, trace_ids_array, op="diff")
    abs_slope = np.abs(slope)
    slope_rows = _bin_metric_rows(
        "abs_target_dvdt_mv_per_ms",
        abs_slope,
        residual_mv,
        bins=np.quantile(abs_slope, np.array([0.0, 0.5, 0.9, 0.99, 1.0])),
    )
    event_rows = _bin_metric_rows(
        "input_event_count",
        event_counts,
        residual_mv,
        bins=_event_bins(event_counts),
    )
    post_event_rows = _post_event_metrics(residual_mv, event_counts > 0, trace_ids_array)
    spike_proximity_rows = _spike_proximity_metrics(residual_mv, spike_labels, trace_ids_array)
    spike_metrics = _spike_metrics(spike_scores, spike_labels, trace_ids_array)
    autocorr_rows = _residual_autocorrelation(residual_mv, trace_ids_array, lags=(1, 2, 5, 10, 25, 50, 100))
    slope_metrics = {
        "target_pred_dvdt_corr": _corr(pred_slope, slope),
        "high_abs_dvdt_threshold_mv_per_ms": float(np.quantile(abs_slope, 0.9)),
        "high_abs_dvdt_rmse_mv": _rmse(residual_mv[abs_slope >= np.quantile(abs_slope, 0.9)]),
    }
    report: dict[str, Any] = {
        "global": global_metrics,
        "slope": slope_metrics,
        "spike": spike_metrics,
        "voltage_regime_rows": voltage_rows,
        "slope_regime_rows": slope_rows,
        "event_regime_rows": event_rows,
        "post_event_rows": post_event_rows,
        "spike_proximity_rows": spike_proximity_rows,
        "residual_autocorrelation_rows": autocorr_rows,
        "interpretation_note": (
            "These diagnostics are surrogate-fidelity probes, not direct claims about biological acceptability. "
            "Near-spike, high-slope, event-conditioned, and autocorrelated errors are often more consequential than global RMSE."
        ),
    }
    _write_json(output_path / "diagnostics.json", report)
    _write_rows(output_path / "voltage_regime_rmse.csv", voltage_rows)
    _write_rows(output_path / "slope_regime_rmse.csv", slope_rows)
    _write_rows(output_path / "event_regime_rmse.csv", event_rows)
    _write_rows(output_path / "post_event_rmse.csv", post_event_rows)
    _write_rows(output_path / "spike_proximity_rmse.csv", spike_proximity_rows)
    _write_rows(output_path / "residual_autocorrelation.csv", autocorr_rows)
    _write_summary_csv(output_path / "summary.csv", report)
    _write_bar_svg(output_path / "voltage_regime_rmse.svg", voltage_rows, "target_soma_mv", "rmse_mv", "RMSE by target voltage regime")
    _write_bar_svg(output_path / "slope_regime_rmse.svg", slope_rows, "abs_target_dvdt_mv_per_ms", "rmse_mv", "RMSE by voltage-slope regime")
    _write_bar_svg(output_path / "event_regime_rmse.svg", event_rows, "input_event_count", "rmse_mv", "RMSE by input event load")
    _write_bar_svg(output_path / "post_event_rmse.svg", post_event_rows, "post_input_event_window", "rmse_mv", "RMSE after input events")
    _write_line_svg(output_path / "residual_autocorrelation.svg", autocorr_rows, "lag_ms", "autocorrelation", "Residual autocorrelation")
    _write_bar_svg(output_path / "spike_timing.svg", spike_metrics["timing_windows"], "window_ms", "pred_precision", "Spike timing precision by tolerance")
    return report


def _affine_calibrate(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_std = float(prediction.std())
    if pred_std == 0.0:
        raise ValueError("cannot affine-calibrate soma predictions with zero standard deviation")
    return (prediction - prediction.mean()) / pred_std * target.std() + target.mean()


def _global_voltage_metrics(prediction_mv: np.ndarray, target_mv: np.ndarray, residual_mv: np.ndarray) -> dict[str, float]:
    return {
        "soma_rmse_mv": _rmse(residual_mv),
        "soma_mae_mv": float(np.mean(np.abs(residual_mv))),
        "soma_bias_mv": float(np.mean(residual_mv)),
        "soma_residual_std_mv": float(np.std(residual_mv)),
        "target_soma_std_mv": float(np.std(target_mv)),
        "rmse_over_target_std": float(_rmse(residual_mv) / np.std(target_mv)) if np.std(target_mv) > 0 else float("nan"),
        "soma_corr": _corr(prediction_mv, target_mv),
    }


def _bin_metric_rows(name: str, values: np.ndarray, residual_mv: np.ndarray, *, bins: np.ndarray) -> list[dict[str, float | int | str]]:
    edges = _unique_edges(bins)
    rows: list[dict[str, float | int | str]] = []
    if len(edges) < 2:
        rows.append(_row_for_mask(name, "all", values, residual_mv, np.ones_like(values, dtype=bool)))
        return rows
    for index in range(len(edges) - 1):
        lo = edges[index]
        hi = edges[index + 1]
        if index == len(edges) - 2:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        rows.append(_row_for_mask(name, f"[{lo:.4g}, {hi:.4g}]", values, residual_mv, mask))
    return rows


def _row_for_mask(name: str, label: str, values: np.ndarray, residual_mv: np.ndarray, mask: np.ndarray) -> dict[str, float | int | str]:
    selected = residual_mv[mask]
    selected_values = values[mask]
    return {
        name: label,
        "count": int(selected.size),
        "fraction": float(selected.size / residual_mv.size) if residual_mv.size else 0.0,
        "value_mean": float(np.mean(selected_values)) if selected_values.size else float("nan"),
        "rmse_mv": _rmse(selected),
        "mae_mv": float(np.mean(np.abs(selected))) if selected.size else float("nan"),
        "bias_mv": float(np.mean(selected)) if selected.size else float("nan"),
    }


def _spike_proximity_metrics(residual_mv: np.ndarray, spike_labels: np.ndarray, trace_ids: np.ndarray) -> list[dict[str, float | int | str]]:
    rows = []
    for window in (1, 3, 5, 10, 25):
        near = _near_spike_mask(spike_labels > 0.5, trace_ids, window)
        rows.append(_row_for_mask("target_spike_distance", f"<= {window} ms", np.zeros_like(residual_mv), residual_mv, near))
    far = ~_near_spike_mask(spike_labels > 0.5, trace_ids, 25)
    rows.append(_row_for_mask("target_spike_distance", "> 25 ms", np.zeros_like(residual_mv), residual_mv, far))
    return rows


def _post_event_metrics(residual_mv: np.ndarray, events: np.ndarray, trace_ids: np.ndarray) -> list[dict[str, float | int | str]]:
    rows = []
    for window in (1, 3, 5, 10, 25):
        after_event = _after_event_mask(events, trace_ids, window)
        rows.append(_row_for_mask("post_input_event_window", f"<= {window} ms", np.zeros_like(residual_mv), residual_mv, after_event))
    quiet = ~_after_event_mask(events, trace_ids, 25)
    rows.append(_row_for_mask("post_input_event_window", "> 25 ms", np.zeros_like(residual_mv), residual_mv, quiet))
    return rows


def _spike_metrics(scores: np.ndarray, labels: np.ndarray, trace_ids: np.ndarray) -> dict[str, Any]:
    positive = labels > 0.5
    threshold_zero_pred = scores >= 0.0
    rate_threshold = _rate_matched_threshold(scores, int(positive.sum()))
    rate_pred = scores >= rate_threshold
    result = {
        "auc": binary_auc(scores, labels),
        "target_spike_count": int(positive.sum()),
        "target_spike_rate_per_1000_steps": float(1000.0 * positive.mean()),
        "threshold_zero": _classification_metrics(threshold_zero_pred, positive),
        "rate_matched_threshold": float(rate_threshold),
        "rate_matched": _classification_metrics(rate_pred, positive),
        "count_error_per_trace": _count_error_per_trace(rate_pred, positive, trace_ids),
        "timing_windows": [],
    }
    for window in (1, 3, 5, 10, 25):
        pred_near_target = _near_spike_mask(positive, trace_ids, window)[rate_pred]
        target_near_pred = _near_spike_mask(rate_pred, trace_ids, window)[positive]
        result["timing_windows"].append(
            {
                "window_ms": str(window),
                "pred_precision": float(pred_near_target.mean()) if pred_near_target.size else float("nan"),
                "target_recall": float(target_near_pred.mean()) if target_near_pred.size else float("nan"),
            }
        )
    return result


def _classification_metrics(predicted: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    tp = int(np.logical_and(predicted, labels).sum())
    fp = int(np.logical_and(predicted, ~labels).sum())
    fn = int(np.logical_and(~predicted, labels).sum())
    tn = int(np.logical_and(~predicted, ~labels).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_spike_count": int(predicted.sum()),
        "predicted_spike_rate_per_1000_steps": float(1000.0 * predicted.mean()),
    }


def _count_error_per_trace(predicted: np.ndarray, labels: np.ndarray, trace_ids: np.ndarray) -> dict[str, float]:
    errors = []
    for trace_id in np.unique(trace_ids):
        mask = trace_ids == trace_id
        errors.append(float(predicted[mask].sum() - labels[mask].sum()))
    error_array = np.asarray(errors, dtype=np.float64)
    return {
        "mean_error": float(error_array.mean()),
        "mean_abs_error": float(np.abs(error_array).mean()),
        "rmse": _rmse(error_array),
    }


def _residual_autocorrelation(residual_mv: np.ndarray, trace_ids: np.ndarray, *, lags: tuple[int, ...]) -> list[dict[str, float | int]]:
    rows = []
    for lag in lags:
        numerator = 0.0
        denominator = 0.0
        count = 0
        for trace_id in np.unique(trace_ids):
            trace = residual_mv[trace_ids == trace_id]
            if trace.size <= lag:
                continue
            centered = trace - trace.mean()
            numerator += float(np.dot(centered[:-lag], centered[lag:]))
            denominator += float(np.dot(centered, centered))
            count += int(trace.size - lag)
        rows.append(
            {
                "lag_ms": int(lag),
                "autocorrelation": numerator / denominator if denominator > 0 else float("nan"),
                "pairs": count,
            }
        )
    return rows


def _near_spike_mask(spikes: np.ndarray, trace_ids: np.ndarray, window: int) -> np.ndarray:
    result = np.zeros_like(spikes, dtype=bool)
    for trace_id in np.unique(trace_ids):
        indices = np.flatnonzero(trace_ids == trace_id)
        local_spikes = np.flatnonzero(spikes[indices])
        for spike_index in local_spikes:
            start = max(0, spike_index - window)
            stop = min(indices.size, spike_index + window + 1)
            result[indices[start:stop]] = True
    return result


def _after_event_mask(events: np.ndarray, trace_ids: np.ndarray, window: int) -> np.ndarray:
    result = np.zeros_like(events, dtype=bool)
    for trace_id in np.unique(trace_ids):
        indices = np.flatnonzero(trace_ids == trace_id)
        local_events = np.flatnonzero(events[indices])
        for event_index in local_events:
            stop = min(indices.size, event_index + window + 1)
            result[indices[event_index:stop]] = True
    return result


def _rate_matched_threshold(scores: np.ndarray, positive_count: int) -> float:
    if positive_count <= 0:
        return float("inf")
    if positive_count >= scores.size:
        return float("-inf")
    return float(np.partition(scores, scores.size - positive_count)[scores.size - positive_count])


def _within_trace_diff(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    return np.concatenate(([0.0], np.diff(values)))


def _within_trace_values(values: np.ndarray, trace_ids: np.ndarray, *, op: str) -> np.ndarray:
    pieces = []
    for trace_id in np.unique(trace_ids):
        trace = values[trace_ids == trace_id]
        if op == "diff":
            pieces.append(_within_trace_diff(trace))
        else:
            raise ValueError(f"unknown op: {op}")
    return np.concatenate(pieces)


def _event_bins(event_counts: np.ndarray) -> np.ndarray:
    max_value = float(event_counts.max()) if event_counts.size else 0.0
    quantiles = np.quantile(event_counts, np.array([0.0, 0.5, 0.9, 0.99, 1.0]))
    fallback = np.array([0.0, 1.0, 3.0, 10.0, max(11.0, max_value)])
    return quantiles if len(_unique_edges(quantiles)) >= 3 else fallback


def _unique_edges(values: np.ndarray) -> np.ndarray:
    edges = np.unique(np.asarray(values, dtype=np.float64))
    if edges.size > 1:
        edges[0] = np.nextafter(edges[0], -np.inf)
        edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def _rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(path: Path, report: dict[str, Any]) -> None:
    rows = []
    for section, metrics in report.items():
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    rows.append({"section": section, "metric": key, "value": value})
    _write_rows(path, rows)


def _write_bar_svg(path: Path, rows: list[dict[str, Any]], label_key: str, value_key: str, title: str) -> None:
    width, height = 820, 420
    margin_left, margin_right, margin_top, margin_bottom = 72, 24, 48, 96
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    values = np.asarray([float(row[value_key]) for row in rows], dtype=np.float64)
    max_value = float(np.nanmax(values)) if values.size else 1.0
    max_value = max(max_value, 1e-9)
    bar_w = plot_w / max(len(rows), 1) * 0.72
    gap = plot_w / max(len(rows), 1)
    svg = [_svg_header(width, height), f'<text x="{margin_left}" y="28" class="title">{_esc(title)}</text>']
    svg.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" class="axis"/>')
    svg.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis"/>')
    for index, row in enumerate(rows):
        value = float(row[value_key])
        x = margin_left + index * gap + (gap - bar_w) / 2
        h = 0.0 if math.isnan(value) else value / max_value * plot_h
        y = margin_top + plot_h - h
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#3568a8"/>')
        svg.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" class="tick">{value:.3g}</text>')
        svg.append(f'<text x="{x + bar_w / 2:.1f}" y="{margin_top + plot_h + 18}" class="xtick">{_esc(str(row[label_key]))}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def _write_line_svg(path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str, title: str) -> None:
    width, height = 820, 420
    margin_left, margin_right, margin_top, margin_bottom = 72, 24, 48, 64
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    xs = np.asarray([float(row[x_key]) for row in rows], dtype=np.float64)
    ys = np.asarray([float(row[y_key]) for row in rows], dtype=np.float64)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = min(0.0, float(np.nanmin(ys))), max(0.0, float(np.nanmax(ys)))
    if y_min == y_max:
        y_max = y_min + 1.0
    points = []
    for x, y in zip(xs, ys):
        px = margin_left + (x - x_min) / max(x_max - x_min, 1e-9) * plot_w
        py = margin_top + (y_max - y) / (y_max - y_min) * plot_h
        points.append((px, py))
    commands = " ".join(("M" if index == 0 else "L") + f"{x:.1f},{y:.1f}" for index, (x, y) in enumerate(points))
    svg = [_svg_header(width, height), f'<text x="{margin_left}" y="28" class="title">{_esc(title)}</text>']
    svg.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" class="axis"/>')
    svg.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis"/>')
    svg.append(f'<path d="{commands}" fill="none" stroke="#b33f3f" stroke-width="2.5"/>')
    for x, y in points:
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="white" stroke="#b33f3f" stroke-width="2"/>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#1f2933}.title{font-size:18px;font-weight:700}'
        '.axis{stroke:#3f4a54;stroke-width:1.2}.tick{font-size:12px;text-anchor:middle}.xtick{font-size:10px;text-anchor:middle}</style>'
    )


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
