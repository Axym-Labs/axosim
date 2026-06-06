#!/usr/bin/env python3
"""Run the current CoreNEURON/Hay NEURO-BENCH MVP comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from bnn_sim.benchmark import BenchmarkSuite
from bnn_sim.checkpoint import load_checkpoint
from bnn_sim.data import ShardedNeuronIODataset


SOMA_SCALE = 0.1
SOMA_BIAS_MV = -67.7


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-data", default="data/hay-coreneuron-source-val16-exc0925-20260605/val")
    parser.add_argument("--calibration-data", default="data/hay-coreneuron-source-train32-exc0925-20260605/train")
    parser.add_argument("--m4-event-dropout-pair", default="docs/verify-arbor-simulator/artifacts/coreneuron_event_dropout_pair_20260605/event_dropout_pair.npz")
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--output-json", default="docs/verify-arbor-simulator/artifacts/coreneuron_neuro_bench_mvp_20260605/results.json")
    parser.add_argument("--output-md", default="docs/verify-arbor-simulator/artifacts/coreneuron_neuro_bench_mvp_20260605/report.md")
    parser.add_argument("--max-val-samples", type=int, default=16)
    parser.add_argument("--max-calibration-samples", type=int, default=32)
    parser.add_argument("--ignore-start", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    started = time.perf_counter()
    val_inputs, val_targets, val_ids = load_dataset(args.val_data, args.max_val_samples)
    cal_inputs, cal_targets, _ = load_dataset(args.calibration_data, args.max_calibration_samples)
    m4_pair = load_m4_pair(args.m4_event_dropout_pair)

    models: dict[str, Any] = {}
    for item in args.checkpoint:
        name, path = parse_checkpoint_arg(item)
        model, metadata = load_checkpoint(path, map_location="cpu")
        model = model.to(args.device).eval()

        def model_fn(x: np.ndarray, *, model=model) -> np.ndarray:
            with torch.inference_mode():
                tensor = torch.from_numpy(x).float().to(args.device)
                return model(tensor).detach().cpu().numpy()

        calibration_predictions = model_fn(cal_inputs)
        threshold, threshold_summary = learn_prevalence_threshold(
            calibration_predictions,
            cal_targets,
            args.ignore_start,
        )
        suite = BenchmarkSuite()
        report = suite.evaluate_model(
            model_fn,
            val_inputs,
            val_targets,
            soma_scale=SOMA_SCALE,
            soma_bias_mv=SOMA_BIAS_MV,
            ignore_start=args.ignore_start,
            run_perturbation=False,
            model_name=name,
            model_config={
                "checkpoint": path,
                "checkpoint_metadata": metadata,
                "calibrated_spike_threshold": threshold_summary,
            },
            dataset_path=args.val_data,
            trace_ids=val_ids,
            calibrated_spike_threshold=threshold,
            m4_paired_intervention={
                "baseline_inputs": m4_pair["baseline_inputs"],
                "baseline_targets": m4_pair["baseline_targets"],
                "intervention_inputs": m4_pair["intervention_inputs"],
                "intervention_targets": m4_pair["intervention_targets"],
                "intervention_name": "coreneuron_hay_event_dropout",
                "trace_ids": m4_pair["trace_ids"],
                "model_spike_threshold": threshold,
            },
        )
        models[name] = {
            "checkpoint": path,
            "parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "calibrated_spike_threshold": threshold_summary,
            "benchmark": report,
        }

    payload = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "elapsed_seconds": time.perf_counter() - started,
            "device": args.device,
            "val_data": args.val_data,
            "calibration_data": args.calibration_data,
            "m4_event_dropout_pair": args.m4_event_dropout_pair,
            "ignore_start": args.ignore_start,
            "max_val_samples": args.max_val_samples,
            "max_calibration_samples": args.max_calibration_samples,
        },
        "teacher_m4_event_dropout": BenchmarkSuite().evaluate_model(
            lambda x: np.zeros((x.shape[0], x.shape[1], 2), dtype=np.float32),
            val_inputs[:1],
            val_targets[:1],
            soma_scale=SOMA_SCALE,
            soma_bias_mv=SOMA_BIAS_MV,
            ignore_start=args.ignore_start,
            run_perturbation=False,
            m4_paired_intervention={
                "baseline_inputs": m4_pair["baseline_inputs"],
                "baseline_targets": m4_pair["baseline_targets"],
                "intervention_inputs": m4_pair["intervention_inputs"],
                "intervention_targets": m4_pair["intervention_targets"],
                "intervention_name": "coreneuron_hay_event_dropout",
                "trace_ids": m4_pair["trace_ids"],
                "compare_model": False,
            },
        )["m4_intervention"],
        "models": models,
        "limitations": [
            f"M4 event-dropout uses {m4_pair['baseline_inputs'].shape[0]} paired traces from one intervention type.",
            "Spike thresholds are learned on the calibration split and frozen for validation/M4.",
            "This MVP report covers M1/M1b/M2 plus one M4 intervention; M3/M5/M6 remain deferred by spec.",
        ],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.output_md).write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(compact_summary(payload), indent=2, sort_keys=True))
    return 0


def load_dataset(path: str | Path, max_samples: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    dataset = ShardedNeuronIODataset(path, cache_shards=1)
    n = min(max_samples, len(dataset))
    indices = list(range(n))
    inputs, targets = dataset.get_batch(indices)
    ids = dataset.get_sample_ids(indices)
    return inputs.astype(np.float32, copy=False), targets.astype(np.float32, copy=False), ids


def load_m4_pair(path: str | Path) -> dict[str, Any]:
    pair_path = Path(path)
    if not pair_path.exists():
        raise FileNotFoundError(pair_path)
    with np.load(pair_path) as data:
        return {
            "baseline_inputs": data["baseline_inputs"].astype(np.float32, copy=False),
            "baseline_targets": data["baseline_targets"].astype(np.float32, copy=False),
            "intervention_inputs": data["intervention_inputs"].astype(np.float32, copy=False),
            "intervention_targets": data["intervention_targets"].astype(np.float32, copy=False),
            "trace_ids": [str(x) for x in data["sample_ids"].tolist()],
        }


def parse_checkpoint_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"--checkpoint must be NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    if not name:
        raise ValueError("checkpoint name must not be empty")
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return name, path


def learn_prevalence_threshold(predictions: np.ndarray, targets: np.ndarray, ignore_start: int) -> tuple[float, dict[str, Any]]:
    pred = predictions[:, ignore_start:, :]
    tgt = targets[:, ignore_start:, :]
    logits = pred[..., 0].reshape(-1)
    target_spikes = (tgt[..., 0] > 0.5).reshape(-1)
    positive_count = int(target_spikes.sum())
    if positive_count <= 0:
        threshold = float("inf")
    elif positive_count >= logits.size:
        threshold = float("-inf")
    else:
        threshold = float(np.partition(logits, logits.size - positive_count)[logits.size - positive_count])
    calibrated = logits >= threshold
    return threshold, {
        "method": "prevalence",
        "threshold": threshold,
        "target_spike_count": positive_count,
        "calibrated_pred_spike_count": int(calibrated.sum()),
        "calibration_samples": int(predictions.shape[0]),
        "semantics": "global logit threshold learned by matching spike prevalence on calibration traces",
    }


def compact_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for name, row in payload["models"].items():
        report = row["benchmark"]
        m1 = report["m1_trace_fidelity"]
        m2 = report["m2_regimes"]
        m4 = report["m4_intervention"]["model_response"]
        rows[name] = {
            "rmse_mv": m1["soma_rmse_mv"],
            "state_weighted_trace_distance_mv": m1["state_weighted_trace_distance_mv"],
            "dynamic_distance": m1["dynamic_distance"],
            "auc": m1["spike_auc"],
            "threshold": row["calibrated_spike_threshold"]["threshold"],
            "count_mae_calibrated": m1["spike_calibrated_threshold"]["count_mae_per_trace"],
            "timing_f1_calibrated": m2["spike_timing_calibrated_threshold"]["f1_at_2steps"],
            "near_spike_rmse": next(
                r["rmse"] for r in m2["mvp_regimes"] if r["regime_label"] == "near_spike"
            ),
            "m4_soma_delta_rmse_mv": m4["soma_delta_rmse_mv"],
            "m4_soma_delta_rmse_ci95": [
                m4["soma_delta_rmse_per_trace"]["ci95_low"],
                m4["soma_delta_rmse_per_trace"]["ci95_high"],
            ],
            "m4_spike_count_delta_mae": m4["spike_count_delta_mae_per_trace"],
            "m4_spike_count_delta_mae_ci95": [
                m4["spike_count_delta_abs_error_per_trace"]["ci95_low"],
                m4["spike_count_delta_abs_error_per_trace"]["ci95_high"],
            ],
            "m4_direction_agreement": m4["direction_agreement"],
        }
    return {
        "metadata": payload["metadata"],
        "teacher_m4": payload["teacher_m4_event_dropout"]["teacher_response"],
        "models": rows,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    teacher = payload["teacher_m4_event_dropout"]["teacher_response"]
    lines = [
        "# CoreNEURON/Hay NEURO-BENCH MVP",
        "",
        "## Teacher M4 Event Dropout",
        "",
        f"- Baseline event fraction: `{teacher['baseline_event_fraction']:.6f}`",
        f"- Intervention event fraction: `{teacher['intervention_event_fraction']:.6f}`",
        f"- Event fraction ratio: `{teacher['event_fraction_ratio']:.4f}`",
        f"- Delta spike rate / 1000 steps: `{teacher['delta_spike_rate_per_1000_steps']:.4f}`",
        f"- Delta soma mean mV: `{teacher['delta_soma_mean_mv']:.4f}`",
        f"- Direction: `{teacher['teacher_direction']}`",
        f"- Response detected: `{teacher['response_detected']}`",
        "",
        "## Model Smoke Comparison",
        "",
        "| model | RMSE mV | state-weighted mV | dynamic distance | AUC | threshold | count MAE calibrated | timing F1 calibrated | near-spike RMSE | M4 soma-delta RMSE | M4 spike-delta MAE | M4 soma/spike direction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, row in payload["models"].items():
        report = row["benchmark"]
        m1 = report["m1_trace_fidelity"]
        m2 = report["m2_regimes"]
        m4 = report["m4_intervention"]["model_response"]
        near = next(r["rmse"] for r in m2["mvp_regimes"] if r["regime_label"] == "near_spike")
        direction = m4["direction_agreement"]
        soma_interval = m4["soma_delta_rmse_per_trace"]
        spike_interval = m4["spike_count_delta_abs_error_per_trace"]
        soma_cell = _format_ci_cell(m4["soma_delta_rmse_mv"], soma_interval)
        spike_cell = _format_ci_cell(m4["spike_count_delta_mae_per_trace"], spike_interval)
        lines.append(
            f"| `{name}` | `{m1['soma_rmse_mv']:.4f}` | `{m1['state_weighted_trace_distance_mv']:.4f}` | "
            f"`{m1['dynamic_distance']:.4f}` | `{m1['spike_auc']:.4f}` | "
            f"`{row['calibrated_spike_threshold']['threshold']:.4f}` | "
            f"`{m1['spike_calibrated_threshold']['count_mae_per_trace']:.4f}` | "
            f"`{m2['spike_timing_calibrated_threshold']['f1_at_2steps']:.4f}` | "
            f"`{near:.4f}` | {soma_cell} | "
            f"{spike_cell} | "
            f"`{direction['soma_mean']}/{direction['spike_count']}` |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in payload["limitations"])
    lines.append("")
    return "\n".join(lines)


def _format_ci_cell(value: float, interval: dict[str, Any]) -> str:
    low = interval.get("ci95_low")
    high = interval.get("ci95_high")
    if low is None or high is None:
        return f"`{value:.4f}`"
    return f"`{value:.4f}` (`{low:.4f}`, `{high:.4f}`)"


if __name__ == "__main__":
    raise SystemExit(main())
