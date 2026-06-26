from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from axosim.checkpoint import load_checkpoint
from axosim.data import ShardedNeuronIODataset
from axosim.evaluate import _synchronize_device, _tensor_from_numpy
from axosim.official.constants import DEFAULT_Y_TRAIN_SOMA_SCALE


def compare_trace_predictions(
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

    sample_ids: list[str] = []
    target_soma: list[np.ndarray] = []
    target_spikes: list[np.ndarray] = []
    rnn_soma: list[np.ndarray] = []
    mamba_soma: list[np.ndarray] = []
    rnn_spike_prob: list[np.ndarray] = []
    mamba_spike_prob: list[np.ndarray] = []

    _synchronize_device(device)
    with torch.inference_mode():
        for start in range(0, sample_count, batch_size):
            indices = range(start, min(start + batch_size, sample_count))
            x_np, y_np = dataset.get_batch(indices)
            x = _tensor_from_numpy(x_np, device, pin_memory=False, non_blocking=True)
            rnn_pred = rnn(x).detach().cpu().numpy()
            mamba_pred = mamba(x).detach().cpu().numpy()
            ids = dataset.get_sample_ids(indices)
            for row, sample_id in enumerate(ids):
                sample_ids.append(sample_id)
                target_soma.append(y_np[row, :, 1].astype(np.float64, copy=True))
                target_spikes.append(y_np[row, :, 0].astype(np.float64, copy=True))
                rnn_soma.append(rnn_pred[row, :, 1].astype(np.float64, copy=True))
                mamba_soma.append(mamba_pred[row, :, 1].astype(np.float64, copy=True))
                rnn_spike_prob.append(_sigmoid(rnn_pred[row, :, 0]).astype(np.float64, copy=True))
                mamba_spike_prob.append(_sigmoid(mamba_pred[row, :, 0]).astype(np.float64, copy=True))
    _synchronize_device(device)

    rnn_soma = _calibrate_trace_list(rnn_soma, target_soma)
    mamba_soma = _calibrate_trace_list(mamba_soma, target_soma)
    target_soma_mv = [trace / y_train_soma_scale for trace in target_soma]
    rnn_soma_mv = [trace / y_train_soma_scale for trace in rnn_soma]
    mamba_soma_mv = [trace / y_train_soma_scale for trace in mamba_soma]

    rows = []
    for index, sample_id in enumerate(sample_ids):
        mask = _mask_for_trace(len(target_soma_mv[index]), ignore_start)
        rnn_error = rnn_soma_mv[index][mask] - target_soma_mv[index][mask]
        mamba_error = mamba_soma_mv[index][mask] - target_soma_mv[index][mask]
        target_tail = target_soma_mv[index][mask]
        rows.append(
            {
                "index": index,
                "sample_id": sample_id,
                "rnn_rmse_mv": _rmse(rnn_error),
                "mamba_rmse_mv": _rmse(mamba_error),
                "mamba_minus_rnn_rmse_mv": _rmse(mamba_error) - _rmse(rnn_error),
                "target_max_mv": float(np.max(target_tail)),
                "target_spike_count": int(np.sum(target_spikes[index][mask] > 0.5)),
            }
        )

    selected = _select_rows(rows)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for label, row in selected.items():
        idx = int(row["index"])
        svg = _trace_svg(
            sample_id=str(row["sample_id"]),
            target_soma_mv=target_soma_mv[idx],
            rnn_soma_mv=rnn_soma_mv[idx],
            mamba_soma_mv=mamba_soma_mv[idx],
            target_spikes=target_spikes[idx],
            rnn_spike_prob=rnn_spike_prob[idx],
            mamba_spike_prob=mamba_spike_prob[idx],
            ignore_start=ignore_start,
            title=f"{label}: {row['sample_id']}",
        )
        (output_path / f"{label}.svg").write_text(svg, encoding="utf-8")

    summary = {
        "data_path": str(data_path),
        "rnn_checkpoint": str(rnn_checkpoint),
        "mamba_checkpoint": str(mamba_checkpoint),
        "sample_count": sample_count,
        "ignore_start": ignore_start,
        "selected": selected,
        "mean_rnn_rmse_mv": float(np.mean([row["rnn_rmse_mv"] for row in rows])),
        "mean_mamba_rmse_mv": float(np.mean([row["mamba_rmse_mv"] for row in rows])),
    }
    (output_path / "trace_selection.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_rows_csv(output_path / "trace_metrics.csv", rows)
    return summary


def _calibrate_trace_list(predictions: list[np.ndarray], targets: list[np.ndarray]) -> list[np.ndarray]:
    pred_all = np.concatenate(predictions)
    target_all = np.concatenate(targets)
    pred_std = float(pred_all.std())
    if pred_std == 0.0:
        raise ValueError("cannot affine-calibrate zero-variance predictions")
    calibrated_all = (pred_all - pred_all.mean()) / pred_std * target_all.std() + target_all.mean()
    result = []
    offset = 0
    for trace in predictions:
        result.append(calibrated_all[offset : offset + trace.size].copy())
        offset += trace.size
    return result


def _select_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ordered_gap = sorted(rows, key=lambda row: row["mamba_minus_rnn_rmse_mv"], reverse=True)
    ordered_spikes = sorted(rows, key=lambda row: (row["target_spike_count"], row["target_max_mv"]), reverse=True)
    ordered_voltage = sorted(rows, key=lambda row: row["target_max_mv"], reverse=True)
    ordered_total = sorted(rows, key=lambda row: row["rnn_rmse_mv"] + row["mamba_rmse_mv"])
    median = ordered_total[len(ordered_total) // 2]
    selected = {
        "largest_mamba_gap": ordered_gap[0],
        "highest_spike_count": ordered_spikes[0],
        "highest_voltage": ordered_voltage[0],
        "representative_median_error": median,
    }
    # Avoid duplicate plots when one trace satisfies multiple criteria.
    used = set()
    deduped = {}
    for label, row in selected.items():
        if row["index"] in used:
            continue
        deduped[label] = row
        used.add(row["index"])
    return deduped


def _trace_svg(
    *,
    sample_id: str,
    target_soma_mv: np.ndarray,
    rnn_soma_mv: np.ndarray,
    mamba_soma_mv: np.ndarray,
    target_spikes: np.ndarray,
    rnn_spike_prob: np.ndarray,
    mamba_spike_prob: np.ndarray,
    ignore_start: int,
    title: str,
) -> str:
    width, height = 1100, 620
    left, right, top, bottom = 72, 28, 48, 58
    voltage_h, spike_h, gap = 360, 110, 42
    plot_w = width - left - right
    voltage_top = top
    spike_top = top + voltage_h + gap
    start = min(ignore_start, len(target_soma_mv) - 1)
    time = np.arange(start, len(target_soma_mv))
    target = target_soma_mv[start:]
    rnn = rnn_soma_mv[start:]
    mamba = mamba_soma_mv[start:]
    y_min = float(min(target.min(), rnn.min(), mamba.min()))
    y_max = float(max(target.max(), rnn.max(), mamba.max()))
    pad = max(0.5, 0.08 * (y_max - y_min))
    y_min -= pad
    y_max += pad

    def sx(t: np.ndarray | float) -> np.ndarray | float:
        return left + (np.asarray(t) - start) / max(len(target_soma_mv) - 1 - start, 1) * plot_w

    def sy(v: np.ndarray | float) -> np.ndarray | float:
        return voltage_top + (y_max - np.asarray(v)) / max(y_max - y_min, 1e-9) * voltage_h

    svg = [_svg_header(width, height)]
    svg.append(f'<text x="{left}" y="28" class="title">{_esc(title)}</text>')
    svg.append(f'<text x="{left}" y="44" class="subtitle">sample { _esc(sample_id) }; plotted after ignore_start={ignore_start} ms</text>')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = voltage_top + frac * voltage_h
        value = y_max - frac * (y_max - y_min)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        svg.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{value:.1f}</text>')
    svg.append(_polyline(time, target, sx, sy, "#111827", 2.5))
    svg.append(_polyline(time, rnn, sx, sy, "#1f77b4", 1.8))
    svg.append(_polyline(time, mamba, sx, sy, "#d62728", 1.8))
    spike_indices = np.flatnonzero(target_spikes[start:] > 0.5) + start
    for spike_time in spike_indices:
        x = float(sx(spike_time))
        svg.append(f'<line x1="{x:.1f}" y1="{voltage_top}" x2="{x:.1f}" y2="{voltage_top + voltage_h}" class="spike"/>')
    svg.append(f'<text x="{width-right-310}" y="{voltage_top+18}" class="legend">target</text><line x1="{width-right-370}" y1="{voltage_top+14}" x2="{width-right-320}" y2="{voltage_top+14}" stroke="#111827" stroke-width="3"/>')
    svg.append(f'<text x="{width-right-210}" y="{voltage_top+18}" class="legend">RNN</text><line x1="{width-right-260}" y1="{voltage_top+14}" x2="{width-right-220}" y2="{voltage_top+14}" stroke="#1f77b4" stroke-width="3"/>')
    svg.append(f'<text x="{width-right-105}" y="{voltage_top+18}" class="legend">Mamba</text><line x1="{width-right-165}" y1="{voltage_top+14}" x2="{width-right-115}" y2="{voltage_top+14}" stroke="#d62728" stroke-width="3"/>')
    svg.append(f'<line x1="{left}" y1="{voltage_top + voltage_h}" x2="{width-right}" y2="{voltage_top + voltage_h}" class="axis"/>')
    svg.append(f'<line x1="{left}" y1="{voltage_top}" x2="{left}" y2="{voltage_top + voltage_h}" class="axis"/>')
    svg.append(f'<text x="20" y="{voltage_top + voltage_h / 2}" class="ylabel" transform="rotate(-90 20 {voltage_top + voltage_h / 2})">soma mV</text>')

    def spy(p: np.ndarray | float) -> np.ndarray | float:
        return spike_top + (1.0 - np.asarray(p)) * spike_h

    svg.append(_polyline(time, rnn_spike_prob[start:], sx, spy, "#1f77b4", 1.7))
    svg.append(_polyline(time, mamba_spike_prob[start:], sx, spy, "#d62728", 1.7))
    for spike_time in spike_indices:
        x = float(sx(spike_time))
        svg.append(f'<line x1="{x:.1f}" y1="{spike_top}" x2="{x:.1f}" y2="{spike_top + spike_h}" class="spike"/>')
    svg.append(f'<line x1="{left}" y1="{spike_top + spike_h}" x2="{width-right}" y2="{spike_top + spike_h}" class="axis"/>')
    svg.append(f'<line x1="{left}" y1="{spike_top}" x2="{left}" y2="{spike_top + spike_h}" class="axis"/>')
    svg.append(f'<text x="22" y="{spike_top + spike_h / 2}" class="ylabel" transform="rotate(-90 22 {spike_top + spike_h / 2})">spike p</text>')
    svg.append(f'<text x="{left + plot_w/2}" y="{height-18}" class="xlabel">time (ms)</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def _polyline(time: np.ndarray, values: np.ndarray, sx, sy, color: str, width: float) -> str:
    xs = sx(time)
    ys = sy(values)
    points = " ".join(f"{float(x):.1f},{float(y):.1f}" for x, y in zip(xs, ys))
    return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'


def _mask_for_trace(length: int, ignore_start: int) -> np.ndarray:
    mask = np.ones(length, dtype=bool)
    if ignore_start:
        mask[:ignore_start] = False
    return mask


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#1f2933}.title{font-size:18px;font-weight:700}'
        '.subtitle{font-size:12px;fill:#5f6f7c}.axis{stroke:#3f4a54;stroke-width:1.2}.grid{stroke:#e4e8ee;stroke-width:1}'
        '.spike{stroke:#9aa6b2;stroke-width:0.7;opacity:0.42}.tick{font-size:11px}.legend{font-size:13px}.ylabel,.xlabel{font-size:13px;font-weight:600}</style>'
    )


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
