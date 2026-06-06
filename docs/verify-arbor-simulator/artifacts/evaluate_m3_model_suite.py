#!/usr/bin/env python3
"""Evaluate source-trained models on held-out M3 morphology pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ARTIFACTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ARTIFACTS))

from bnn_sim.benchmark import BenchmarkSuite  # noqa: E402
from bnn_sim.checkpoint import load_checkpoint  # noqa: E402
from run_coreneuron_neuro_bench_mvp import load_dataset, learn_prevalence_threshold  # noqa: E402


SOMA_SCALE = 0.1
SOMA_BIAS_MV = -67.7


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", default="docs/verify-arbor-simulator/artifacts/m3_normroute_suite_20260606")
    parser.add_argument("--calibration-data", default="data/hay-coreneuron-source-train32-exc0925-20260605/train")
    parser.add_argument("--max-calibration-samples", type=int, default=32)
    parser.add_argument("--ignore-start", type=int, default=500)
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--output-dir", default="docs/verify-arbor-simulator/artifacts/m3_model_suite_20260606")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    started = time.perf_counter()
    pair_root = Path(args.pair_root)
    pair_paths = sorted(pair_root.glob("specimen_*_event_dropout_simbase*/intervention_pair.npz"))
    if not pair_paths:
        raise FileNotFoundError(f"no event-dropout M3 pairs found under {pair_root}")
    cal_inputs, cal_targets, _ = load_dataset(args.calibration_data, args.max_calibration_samples)
    checkpoints = [_parse_checkpoint_arg(item) for item in args.checkpoint]

    pair_rows: list[dict[str, Any]] = []
    for pair_path in pair_paths:
        specimen = pair_path.parent.name.split("_")[1]
        pair = _load_pair(pair_path)
        model_rows = {}
        for name, checkpoint_path in checkpoints:
            model, metadata = load_checkpoint(checkpoint_path, map_location="cpu")
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
            report = BenchmarkSuite().evaluate_model(
                model_fn,
                pair["baseline_inputs"],
                pair["baseline_targets"],
                soma_scale=SOMA_SCALE,
                soma_bias_mv=SOMA_BIAS_MV,
                ignore_start=args.ignore_start,
                run_perturbation=False,
                model_name=name,
                model_config={
                    "checkpoint": checkpoint_path,
                    "checkpoint_metadata": metadata,
                    "calibrated_spike_threshold": threshold_summary,
                },
                dataset_path=str(pair_path),
                trace_ids=pair["trace_ids"],
                calibrated_spike_threshold=threshold,
                m4_paired_intervention={
                    "baseline_inputs": pair["baseline_inputs"],
                    "baseline_targets": pair["baseline_targets"],
                    "intervention_inputs": pair["intervention_inputs"],
                    "intervention_targets": pair["intervention_targets"],
                    "intervention_name": f"m3_specimen_{specimen}_event_dropout",
                    "trace_ids": pair["trace_ids"],
                    "model_spike_threshold": threshold,
                },
            )
            model_rows[name] = _summarize_model(report, checkpoint_path, threshold_summary)
        pair_rows.append({
            "specimen": specimen,
            "pair_path": str(pair_path),
            "n_traces": int(pair["baseline_inputs"].shape[0]),
            "models": model_rows,
        })

    payload = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "elapsed_seconds": time.perf_counter() - started,
            "device": args.device,
            "pair_root": str(pair_root),
            "calibration_data": args.calibration_data,
            "ignore_start": args.ignore_start,
            "semantics": "M3 source-trained model transfer over held-out morphology event-dropout pairs",
        },
        "pairs": pair_rows,
        "aggregate": _aggregate(pair_rows),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({"metadata": payload["metadata"], "aggregate": payload["aggregate"]}, indent=2, sort_keys=True))
    return 0


def _parse_checkpoint_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"--checkpoint must be NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    if not name:
        raise ValueError("checkpoint name must not be empty")
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return name, path


def _load_pair(path: Path) -> dict[str, Any]:
    with np.load(path) as data:
        return {
            "baseline_inputs": data["baseline_inputs"].astype(np.float32, copy=False),
            "baseline_targets": data["baseline_targets"].astype(np.float32, copy=False),
            "intervention_inputs": data["intervention_inputs"].astype(np.float32, copy=False),
            "intervention_targets": data["intervention_targets"].astype(np.float32, copy=False),
            "trace_ids": [str(x) for x in data["sample_ids"].tolist()],
        }


def _summarize_model(report: dict[str, Any], checkpoint_path: str, threshold_summary: dict[str, Any]) -> dict[str, Any]:
    m1 = report["m1_trace_fidelity"]
    m2 = report["m2_regimes"]
    m4 = report["m4_intervention"]["model_response"]
    return {
        "checkpoint": checkpoint_path,
        "threshold": threshold_summary["threshold"],
        "rmse_mv": m1["soma_rmse_mv"],
        "trace_shape_distance": m1["trace_shape_distance"],
        "state_trace_shape_distance": m1["state_trace_shape_distance"],
        "spike_train_distance": m1["spike_train_distance"],
        "auc": m1["spike_auc"],
        "count_mae_calibrated": m1["spike_calibrated_threshold"]["count_mae_per_trace"],
        "timing_f1_calibrated": m2["spike_timing_calibrated_threshold"]["f1_at_2steps"],
        "event_dropout_soma_delta_rmse_mv": m4["soma_delta_rmse_mv"],
        "event_dropout_spike_delta_mae": m4["spike_count_delta_mae_per_trace"],
        "direction_agreement": m4["direction_agreement"],
    }


def _aggregate(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_names = sorted({name for pair in pair_rows for name in pair["models"]})
    out = {}
    fields = [
        "rmse_mv",
        "trace_shape_distance",
        "state_trace_shape_distance",
        "spike_train_distance",
        "count_mae_calibrated",
        "event_dropout_soma_delta_rmse_mv",
        "event_dropout_spike_delta_mae",
    ]
    for name in model_names:
        rows = [pair["models"][name] for pair in pair_rows if name in pair["models"]]
        out[name] = {
            f"{field}_mean": float(np.mean([row[field] for row in rows]))
            for field in fields
        }
        out[name]["direction_agreement_all"] = bool(
            all(
                row["direction_agreement"].get("soma_mean") and row["direction_agreement"].get("spike_count")
                for row in rows
            )
        )
    return out


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# M3 Held-Out Morphology Model Suite",
        "",
        "| specimen | model | RMSE mV | trace-shape | spike-train dist | count MAE | soma-delta RMSE | spike-delta MAE | direction |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for pair in payload["pairs"]:
        for name, row in pair["models"].items():
            direction = row["direction_agreement"]
            lines.append(
                f"| `{pair['specimen']}` | `{name}` | `{row['rmse_mv']:.4f}` | "
                f"`{row['trace_shape_distance']:.4f}` | `{row['spike_train_distance']:.4f}` | "
                f"`{row['count_mae_calibrated']:.4f}` | `{row['event_dropout_soma_delta_rmse_mv']:.4f}` | "
                f"`{row['event_dropout_spike_delta_mae']:.4f}` | "
                f"`{direction.get('soma_mean')}/{direction.get('spike_count')}` |"
            )
    lines.extend(["", "## Aggregate", ""])
    lines.append("| model | mean RMSE mV | mean trace-shape | mean spike-train dist | mean count MAE | mean M3 soma-delta RMSE | mean M3 spike-delta MAE | all directions |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for name, row in payload["aggregate"].items():
        lines.append(
            f"| `{name}` | `{row['rmse_mv_mean']:.4f}` | `{row['trace_shape_distance_mean']:.4f}` | "
            f"`{row['spike_train_distance_mean']:.4f}` | `{row['count_mae_calibrated_mean']:.4f}` | "
            f"`{row['event_dropout_soma_delta_rmse_mv_mean']:.4f}` | "
            f"`{row['event_dropout_spike_delta_mae_mean']:.4f}` | `{row['direction_agreement_all']}` |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
