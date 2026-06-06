#!/usr/bin/env python3
"""Evaluate models across the source CoreNEURON/Hay M4 intervention battery."""

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
DEFAULT_INTERVENTIONS = ("event_dropout", "exc_dropout", "inh_dropout", "temporal_jitter", "site_silence")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-template", default="docs/verify-arbor-simulator/artifacts/coreneuron_m4_{intervention}_16_20260606/intervention_pair.npz")
    parser.add_argument("--calibration-data", default="data/hay-coreneuron-source-train32-exc0925-20260605/train")
    parser.add_argument("--max-calibration-samples", type=int, default=32)
    parser.add_argument("--ignore-start", type=int, default=500)
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--intervention", action="append", choices=DEFAULT_INTERVENTIONS)
    parser.add_argument("--output-dir", default="docs/verify-arbor-simulator/artifacts/m4_model_suite_20260606")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    interventions = tuple(args.intervention or DEFAULT_INTERVENTIONS)
    started = time.perf_counter()
    cal_inputs, cal_targets, _ = load_dataset(args.calibration_data, args.max_calibration_samples)
    checkpoints = [_parse_checkpoint_arg(item) for item in args.checkpoint]
    pairs = [(name, _load_pair(Path(args.pair_template.format(intervention=name)))) for name in interventions]

    model_cache = {}
    for model_name, checkpoint_path in checkpoints:
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
        model_cache[model_name] = {
            "checkpoint": checkpoint_path,
            "metadata": metadata,
            "threshold": threshold,
            "threshold_summary": threshold_summary,
            "model_fn": model_fn,
        }

    intervention_rows: list[dict[str, Any]] = []
    for intervention_name, pair in pairs:
        teacher = BenchmarkSuite().evaluate_model(
            lambda x: np.zeros((x.shape[0], x.shape[1], 2), dtype=np.float32),
            pair["baseline_inputs"][:1],
            pair["baseline_targets"][:1],
            soma_scale=SOMA_SCALE,
            soma_bias_mv=SOMA_BIAS_MV,
            ignore_start=args.ignore_start,
            run_perturbation=False,
            m4_paired_intervention={
                "baseline_inputs": pair["baseline_inputs"],
                "baseline_targets": pair["baseline_targets"],
                "intervention_inputs": pair["intervention_inputs"],
                "intervention_targets": pair["intervention_targets"],
                "intervention_name": f"m4_{intervention_name}",
                "trace_ids": pair["trace_ids"],
                "compare_model": False,
            },
        )["m4_intervention"]["teacher_response"]
        model_rows = {}
        for model_name, row in model_cache.items():
            report = BenchmarkSuite().evaluate_model(
                row["model_fn"],
                pair["baseline_inputs"],
                pair["baseline_targets"],
                soma_scale=SOMA_SCALE,
                soma_bias_mv=SOMA_BIAS_MV,
                ignore_start=args.ignore_start,
                run_perturbation=False,
                model_name=model_name,
                model_config={
                    "checkpoint": row["checkpoint"],
                    "checkpoint_metadata": row["metadata"],
                    "calibrated_spike_threshold": row["threshold_summary"],
                },
                dataset_path=str(pair["path"]),
                trace_ids=pair["trace_ids"],
                calibrated_spike_threshold=row["threshold"],
                m4_paired_intervention={
                    "baseline_inputs": pair["baseline_inputs"],
                    "baseline_targets": pair["baseline_targets"],
                    "intervention_inputs": pair["intervention_inputs"],
                    "intervention_targets": pair["intervention_targets"],
                    "intervention_name": f"m4_{intervention_name}",
                    "trace_ids": pair["trace_ids"],
                    "model_spike_threshold": row["threshold"],
                },
            )
            model_rows[model_name] = _summarize_model(report, row["threshold_summary"])
        intervention_rows.append({
            "intervention": intervention_name,
            "pair_path": str(pair["path"]),
            "n_traces": int(pair["baseline_inputs"].shape[0]),
            "teacher": teacher,
            "models": model_rows,
        })

    payload = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "elapsed_seconds": time.perf_counter() - started,
            "device": args.device,
            "calibration_data": args.calibration_data,
            "ignore_start": args.ignore_start,
            "interventions": interventions,
            "semantics": "source CoreNEURON/Hay M4 model behavior battery",
        },
        "interventions": intervention_rows,
        "aggregate": _aggregate(intervention_rows),
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
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return {
            "path": path,
            "baseline_inputs": data["baseline_inputs"].astype(np.float32, copy=False),
            "baseline_targets": data["baseline_targets"].astype(np.float32, copy=False),
            "intervention_inputs": data["intervention_inputs"].astype(np.float32, copy=False),
            "intervention_targets": data["intervention_targets"].astype(np.float32, copy=False),
            "trace_ids": [str(x) for x in data["sample_ids"].tolist()],
        }


def _summarize_model(report: dict[str, Any], threshold_summary: dict[str, Any]) -> dict[str, Any]:
    m4 = report["m4_intervention"]["model_response"]
    return {
        "threshold": threshold_summary["threshold"],
        "soma_delta_rmse_mv": m4["soma_delta_rmse_mv"],
        "soma_delta_mae_mv": m4["soma_delta_mae_mv"],
        "soma_delta_gain": m4["soma_delta_gain"],
        "soma_delta_correlation": m4["soma_delta_correlation"],
        "spike_count_delta_mae_per_trace": m4["spike_count_delta_mae_per_trace"],
        "spike_count_delta_gain": m4["spike_count_delta_gain"],
        "spike_count_delta_teacher_mean": m4["spike_count_delta_teacher_mean"],
        "spike_count_delta_model_mean": m4["spike_count_delta_model_mean"],
        "direction_agreement": m4["direction_agreement"],
    }


def _aggregate(intervention_rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_names = sorted({name for row in intervention_rows for name in row["models"]})
    out = {}
    for model_name in model_names:
        rows = [intervention["models"][model_name] for intervention in intervention_rows]
        out[model_name] = {
            "soma_delta_rmse_mv_mean": float(np.mean([row["soma_delta_rmse_mv"] for row in rows])),
            "spike_count_delta_mae_mean": float(np.mean([row["spike_count_delta_mae_per_trace"] for row in rows])),
            "soma_direction_agreement_fraction": float(np.mean([row["direction_agreement"]["soma_mean"] for row in rows])),
            "spike_direction_agreement_fraction": float(np.mean([row["direction_agreement"]["spike_count"] for row in rows])),
            "soma_gain_abs_error_mean": _mean_gain_abs_error([row["soma_delta_gain"] for row in rows]),
            "spike_gain_abs_error_mean": _mean_gain_abs_error([row["spike_count_delta_gain"] for row in rows]),
        }
    return out


def _mean_gain_abs_error(values: list[float | None]) -> float | None:
    finite = [abs(value - 1.0) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# M4 Structured Intervention Model Suite",
        "",
        "## Teacher Sanity",
        "",
        "| intervention | traces | soma delta mV | spike-count delta | direction |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in payload["interventions"]:
        teacher = row["teacher"]
        lines.append(
            f"| `{row['intervention']}` | `{row['n_traces']}` | `{teacher['delta_soma_mean_mv']:.4f}` | "
            f"`{teacher['delta_spike_count_trace_mean']['mean']:.4f}` | `{teacher['teacher_direction']}` |"
        )
    lines.extend([
        "",
        "## Model Behavior",
        "",
        "| intervention | model | soma RMSE | soma gain | spike MAE | spike gain | direction |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for intervention in payload["interventions"]:
        for model_name, row in intervention["models"].items():
            direction = row["direction_agreement"]
            lines.append(
                f"| `{intervention['intervention']}` | `{model_name}` | `{row['soma_delta_rmse_mv']:.4f}` | "
                f"{_fmt(row['soma_delta_gain'])} | `{row['spike_count_delta_mae_per_trace']:.4f}` | "
                f"{_fmt(row['spike_count_delta_gain'])} | `{direction['soma_mean']}/{direction['spike_count']}` |"
            )
    lines.extend(["", "## Aggregate", ""])
    lines.append("| model | mean soma RMSE | mean spike MAE | soma direction frac | spike direction frac | mean soma gain error | mean spike gain error |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model_name, row in payload["aggregate"].items():
        lines.append(
            f"| `{model_name}` | `{row['soma_delta_rmse_mv_mean']:.4f}` | `{row['spike_count_delta_mae_mean']:.4f}` | "
            f"`{row['soma_direction_agreement_fraction']:.3f}` | `{row['spike_direction_agreement_fraction']:.3f}` | "
            f"{_fmt(row['soma_gain_abs_error_mean'])} | {_fmt(row['spike_gain_abs_error_mean'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "`NA`" if value is None else f"`{value:.4f}`"


if __name__ == "__main__":
    raise SystemExit(main())
