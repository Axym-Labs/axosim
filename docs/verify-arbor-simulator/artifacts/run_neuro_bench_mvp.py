#!/usr/bin/env python3
"""Run the NEURO-BENCH MVP smoke validation.

This script uses the paired low/high input Arbor teacher traces as the M4
event-rate intervention sanity check, then compares available checkpoints on
both ordinary held-out M1/M1b/M2 metrics and paired M4 delta preservation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from bnn_sim.benchmark import BenchmarkSuite, compute_paired_event_intervention_metrics
from bnn_sim.mamba_pytorch import SimpleMambaModel
from bnn_sim.model import BranchELM, BranchELMConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-data", default="data/arbor-allen-nmda-source-medium-20260510/val")
    parser.add_argument("--low-input-data", default="runs/verify_t3_low_input")
    parser.add_argument("--high-input-data", default="runs/verify_t3_high_input")
    parser.add_argument("--output-json", default="docs/verify-arbor-simulator/artifacts/neuro_bench_mvp_smoke_results.json")
    parser.add_argument("--output-md", default="docs/verify-arbor-simulator/artifacts/neuro_bench_mvp_smoke_report.md")
    parser.add_argument("--n-val-samples", type=int, default=8)
    parser.add_argument("--ignore-start", type=int, default=500)
    parser.add_argument("--device", default=default_torch_device())
    parser.add_argument("--skip-mamba", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    val_inputs, val_targets, val_ids = load_dataset(args.val_data, limit=args.n_val_samples)
    low_inputs, low_targets, low_ids = load_dataset(args.low_input_data)
    high_inputs, high_targets, high_ids = load_dataset(args.high_input_data)
    pair_ids = paired_trace_ids(low_ids, high_ids)

    teacher_m4 = compute_paired_event_intervention_metrics(
        None,
        low_inputs,
        low_targets,
        high_inputs,
        high_targets,
        intervention_name="event_rate_shift_low_to_high",
        soma_scale=0.1,
        ignore_start=args.ignore_start,
        trace_ids=pair_ids,
    )
    models = available_models(skip_mamba=args.skip_mamba)
    model_reports: dict[str, Any] = {}
    for name, model in models.items():
        model = model.to(args.device).eval()

        def model_fn(x: np.ndarray, *, model=model) -> np.ndarray:
            with torch.inference_mode():
                tensor = torch.from_numpy(x).float().to(args.device)
                return model(tensor).detach().cpu().numpy()

        suite = BenchmarkSuite()
        model_reports[name] = {
            "checkpoint": getattr(model, "_checkpoint_path", None),
            "parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "mvp_trace_report": suite.evaluate_model(
                model_fn,
                val_inputs,
                val_targets,
                soma_scale=0.1,
                ignore_start=args.ignore_start,
                run_perturbation=False,
                model_name=name,
                dataset_path=args.val_data,
                trace_ids=val_ids,
            ),
            "m4_event_rate_shift": compute_paired_event_intervention_metrics(
                model_fn,
                low_inputs,
                low_targets,
                high_inputs,
                high_targets,
                intervention_name="event_rate_shift_low_to_high",
                soma_scale=0.1,
                ignore_start=args.ignore_start,
                trace_ids=pair_ids,
            ),
        }

    report = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": time.perf_counter() - start,
            "device": args.device,
            "val_data": args.val_data,
            "low_input_data": args.low_input_data,
            "high_input_data": args.high_input_data,
            "ignore_start": args.ignore_start,
            "n_val_samples": int(val_inputs.shape[0]),
            "n_intervention_pairs": int(low_inputs.shape[0]),
        },
        "teacher_m4_event_rate_shift": teacher_m4,
        "models": model_reports,
        "interpretation": interpret_teacher_response(teacher_m4),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(compact_summary(report), indent=2, sort_keys=True))
    return 0


def load_dataset(path: str | Path, *, limit: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    root = Path(path)
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no npz shards found in {root}")
    inputs = []
    targets = []
    ids: list[str] = []
    for shard in files:
        with np.load(shard) as data:
            inputs.append(data["inputs"])
            targets.append(data["targets"])
            if "sample_ids" in data:
                ids.extend(str(x) for x in data["sample_ids"].tolist())
            else:
                ids.extend(f"{shard.stem}:{i}" for i in range(data["inputs"].shape[0]))
    input_array = np.concatenate(inputs, axis=0)
    target_array = np.concatenate(targets, axis=0)
    if limit is not None:
        input_array = input_array[:limit]
        target_array = target_array[:limit]
        ids = ids[:limit]
    return input_array, target_array, ids


def paired_trace_ids(low_ids: list[str], high_ids: list[str]) -> list[str]:
    n = min(len(low_ids), len(high_ids))
    return [f"low={low_ids[i]} | high={high_ids[i]}" for i in range(n)]


def available_models(*, skip_mamba: bool) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {}
    branch_ckpt = Path("runs/branch_elm_30_trained.pt")
    if branch_ckpt.exists():
        model = load_branch_elm(branch_ckpt)
        model._checkpoint_path = str(branch_ckpt)  # type: ignore[attr-defined]
        models["branch_elm_30_trained"] = model
    mamba_ckpt = Path("runs/mamba_best_trained.pt")
    if not skip_mamba and mamba_ckpt.exists():
        model = load_mamba(mamba_ckpt)
        model._checkpoint_path = str(mamba_ckpt)  # type: ignore[attr-defined]
        models["mamba_pytorch_best"] = model
    if not models:
        raise FileNotFoundError("no supported checkpoints found under runs/")
    return models


def default_torch_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
        if arch not in torch.cuda.get_arch_list():
            return "cpu"
    except Exception:
        return "cpu"
    return "cuda"


def load_branch_elm(path: Path) -> BranchELM:
    checkpoint = torch.load(path, map_location="cpu")
    config = BranchELMConfig(**checkpoint["config"])
    model = BranchELM(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def load_mamba(path: Path) -> SimpleMambaModel:
    checkpoint = torch.load(path, map_location="cpu")
    model = SimpleMambaModel(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def interpret_teacher_response(report: dict[str, Any]) -> dict[str, Any]:
    response = report["teacher_response"]
    event_delta = response["event_fraction_delta"]
    spike_delta = response["delta_spike_rate_per_1000_steps"]
    soma_delta = response["delta_soma_mean_mv"]
    warning = None
    if event_delta > 0 and spike_delta < 0:
        warning = (
            "Higher event density reduced teacher spike rate in this small pair. "
            "Treat M4 as teacher-fidelity/stress testing, not biological robustness, "
            "until the teacher protocol is calibrated."
        )
    return {
        "teacher_response_detected": bool(response["response_detected"]),
        "event_fraction_delta": event_delta,
        "delta_spike_rate_per_1000_steps": spike_delta,
        "delta_soma_mean_mv": soma_delta,
        "warning": warning,
    }


def compact_summary(report: dict[str, Any]) -> dict[str, Any]:
    model_summary = {}
    for name, row in report["models"].items():
        m1 = row["mvp_trace_report"]["m1_trace_fidelity"]
        m4 = row["m4_event_rate_shift"]["model_response"]
        model_summary[name] = {
            "m1_soma_rmse_mv": m1["soma_rmse_mv"],
            "m1_spike_auc": m1["spike_auc"],
            "m4_soma_delta_rmse_mv": m4["soma_delta_rmse_mv"],
            "m4_spike_count_delta_mae": m4["spike_count_delta_mae_per_trace"],
        }
    return {
        "metadata": report["metadata"],
        "teacher_interpretation": report["interpretation"],
        "models": model_summary,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# NEURO-BENCH MVP Smoke Report",
        "",
        "## Teacher M4 Event-Rate Shift",
    ]
    teacher = report["teacher_m4_event_rate_shift"]["teacher_response"]
    lines.extend(
        [
            f"- Baseline event fraction: `{teacher['baseline_event_fraction']:.6f}`",
            f"- Intervention event fraction: `{teacher['intervention_event_fraction']:.6f}`",
            f"- Delta spike rate / 1000 steps: `{teacher['delta_spike_rate_per_1000_steps']:.4f}`",
            f"- Delta soma mean mV: `{teacher['delta_soma_mean_mv']:.4f}`",
            f"- Direction: `{teacher['teacher_direction']}`",
            f"- Warning: {report['interpretation']['warning'] or 'none'}",
            "",
            "## Model Smoke Comparison",
            "",
            "| model | M1 RMSE mV | M1 AUC | M4 soma-delta RMSE mV | M4 spike-count-delta MAE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in report["models"].items():
        m1 = row["mvp_trace_report"]["m1_trace_fidelity"]
        m4 = row["m4_event_rate_shift"]["model_response"]
        lines.append(
            f"| `{name}` | {m1['soma_rmse_mv']:.4f} | {m1['spike_auc']:.4f} | "
            f"{m4['soma_delta_rmse_mv']:.4f} | {m4['spike_count_delta_mae_per_trace']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This is an MVP smoke comparison on a small intervention pair.",
            "- The M4 teacher response is not yet a biological-validity claim.",
            "- If Mamba is included, this uses the slow pure-PyTorch checkpoint available in `runs/`.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
