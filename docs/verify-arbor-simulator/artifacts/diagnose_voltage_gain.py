#!/usr/bin/env python3
"""Diagnose soma-voltage gain/bias errors for a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from bnn_sim.checkpoint import load_checkpoint
from bnn_sim.data import ShardedNeuronIODataset
from bnn_sim.metrics import binary_auc
from bnn_sim.official.constants import DEFAULT_Y_TRAIN_SOMA_BIAS, DEFAULT_Y_TRAIN_SOMA_SCALE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--ignore-start", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--soma-scale", type=float, default=DEFAULT_Y_TRAIN_SOMA_SCALE)
    parser.add_argument("--soma-bias-mv", type=float, default=DEFAULT_Y_TRAIN_SOMA_BIAS)
    args = parser.parse_args()

    started = time.perf_counter()
    dataset = ShardedNeuronIODataset(args.data, cache_shards=1)
    sample_count = min(len(dataset), args.max_samples or len(dataset))
    inputs, targets = dataset.get_batch(range(sample_count))
    model, metadata = load_checkpoint(args.checkpoint, map_location="cpu")
    model = model.to(args.device).eval()

    predictions = []
    with torch.inference_mode():
        for start in range(0, sample_count, args.batch_size):
            batch = torch.from_numpy(inputs[start : start + args.batch_size]).float().to(args.device)
            predictions.append(model(batch).detach().cpu().numpy())
    pred = np.concatenate(predictions, axis=0).astype(np.float32, copy=False)

    report = diagnose(
        pred,
        targets.astype(np.float32, copy=False),
        inputs.astype(np.float32, copy=False),
        ignore_start=args.ignore_start,
        soma_scale=args.soma_scale,
        soma_bias_mv=args.soma_bias_mv,
    )
    report["metadata"] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "elapsed_seconds": time.perf_counter() - started,
        "data": args.data,
        "checkpoint": args.checkpoint,
        "checkpoint_metadata": metadata,
        "samples": sample_count,
        "ignore_start": args.ignore_start,
        "device": args.device,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(compact(report), indent=2, sort_keys=True))
    return 0


def diagnose(
    predictions: np.ndarray,
    targets: np.ndarray,
    inputs: np.ndarray,
    *,
    ignore_start: int,
    soma_scale: float,
    soma_bias_mv: float,
) -> dict[str, Any]:
    pred = predictions[:, ignore_start:, :]
    tgt = targets[:, ignore_start:, :]
    inp = inputs[:, ignore_start:, :]
    pred_mv = pred[..., 1] / soma_scale + soma_bias_mv
    target_mv = tgt[..., 1] / soma_scale + soma_bias_mv
    residual = pred_mv - target_mv
    affine = fit_affine(pred_mv.reshape(-1), target_mv.reshape(-1))
    pred_affine = affine["slope"] * pred_mv + affine["intercept_mv"]
    residual_affine = pred_affine - target_mv
    spike_labels = tgt[..., 0] > 0.5
    spike_scores = pred[..., 0]
    matched_threshold = prevalence_matched_threshold(spike_scores, int(spike_labels.sum()))
    event_counts = np.count_nonzero(inp, axis=-1)
    high_event_threshold = float(np.quantile(event_counts, 0.95)) if np.count_nonzero(event_counts) else None
    return {
        "global": {
            "rmse_mv": rmse(residual),
            "mae_mv": float(np.mean(np.abs(residual))),
            "affine_calibrated_rmse_mv": rmse(residual_affine),
            "affine_calibrated_mae_mv": float(np.mean(np.abs(residual_affine))),
            "rmse_gain_gap_mv": rmse(residual) - rmse(residual_affine),
            "pred_mean_mv": float(np.mean(pred_mv)),
            "target_mean_mv": float(np.mean(target_mv)),
            "pred_std_mv": float(np.std(pred_mv)),
            "target_std_mv": float(np.std(target_mv)),
            "pred_target_corr": safe_corr(pred_mv.reshape(-1), target_mv.reshape(-1)),
            "affine_slope": affine["slope"],
            "affine_intercept_mv": affine["intercept_mv"],
            "spike_auc": float(binary_auc(spike_scores, spike_labels.astype(np.float32))),
            "target_spike_count": int(spike_labels.sum()),
            "pred_spike_count_threshold_0": int((spike_scores > 0.0).sum()),
            "pred_spike_count_prevalence_matched": int((spike_scores >= matched_threshold).sum()),
            "prevalence_matched_threshold": float(matched_threshold),
        },
        "voltage_quantile_bins": voltage_bin_rows(target_mv, residual, residual_affine),
        "event_drive": {
            "event_count_p95_threshold": high_event_threshold,
            "event_fraction": float(np.count_nonzero(inp) / max(1, inp.size)),
        },
    }


def fit_affine(pred_mv: np.ndarray, target_mv: np.ndarray) -> dict[str, float]:
    pred_var = float(np.var(pred_mv))
    if pred_var == 0.0:
        return {"slope": 0.0, "intercept_mv": float(np.mean(target_mv))}
    cov = float(np.mean((pred_mv - np.mean(pred_mv)) * (target_mv - np.mean(target_mv))))
    slope = cov / pred_var
    intercept = float(np.mean(target_mv) - slope * np.mean(pred_mv))
    return {"slope": float(slope), "intercept_mv": intercept}


def voltage_bin_rows(target_mv: np.ndarray, residual: np.ndarray, residual_affine: np.ndarray) -> list[dict[str, Any]]:
    edges = np.quantile(target_mv.reshape(-1), [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    labels = ["q0_25", "q25_50", "q50_75", "q75_90", "q90_100"]
    rows = []
    for index, label in enumerate(labels):
        mask = (target_mv >= edges[index]) & (target_mv < edges[index + 1])
        if index == len(labels) - 1:
            mask = (target_mv >= edges[index]) & (target_mv <= edges[index + 1])
        rows.append(
            {
                "label": label,
                "target_low_mv": float(edges[index]),
                "target_high_mv": float(edges[index + 1]),
                "count": int(mask.sum()),
                "rmse_mv": rmse(residual[mask]) if mask.any() else None,
                "affine_calibrated_rmse_mv": rmse(residual_affine[mask]) if mask.any() else None,
            }
        )
    return rows


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def prevalence_matched_threshold(scores: np.ndarray, positive_count: int) -> float:
    flat = np.asarray(scores).reshape(-1)
    if positive_count <= 0:
        return float("inf")
    if positive_count >= flat.size:
        return float("-inf")
    return float(np.partition(flat, flat.size - positive_count)[flat.size - positive_count])


def compact(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "global": report["global"],
        "voltage_q90_100": report["voltage_quantile_bins"][-1],
        "output": report["metadata"]["checkpoint"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
