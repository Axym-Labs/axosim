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
from bnn_sim.checkpoint import load_checkpoint
from bnn_sim.mamba_pytorch import SimpleMambaModel
from bnn_sim.model import BranchELM, BranchELMConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-data", default="data/arbor-calibrated-nmda-proxy-20260604/val")
    parser.add_argument("--low-input-data", default="runs/verify_t3_low_input")
    parser.add_argument("--high-input-data", default="runs/verify_t3_high_input")
    parser.add_argument("--event-dropout-pair", default="docs/verify-arbor-simulator/artifacts/event_dropout_pair/event_dropout_pair.npz")
    parser.add_argument(
        "--extra-checkpoint",
        action="append",
        default=[],
        help="additional checkpoint as NAME=PATH; supports current bnn_sim checkpoint format",
    )
    parser.add_argument("--output-json", default="docs/verify-arbor-simulator/artifacts/neuro_bench_mvp_smoke_results.json")
    parser.add_argument("--output-md", default="docs/verify-arbor-simulator/artifacts/neuro_bench_mvp_smoke_report.md")
    parser.add_argument("--n-val-samples", type=int, default=8)
    parser.add_argument("--ignore-start", type=int, default=500)
    parser.add_argument("--device", default=default_torch_device())
    parser.add_argument("--skip-mamba", action="store_true")
    parser.add_argument("--skip-m4", action="store_true", help="run M1/M1b/M2 only")
    parser.add_argument(
        "--allow-mixed-teacher-m4",
        action="store_true",
        help="allow M4 pairs from a different teacher/data source than --val-data; stress-test only",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    val_inputs, val_targets, val_ids = load_dataset(args.val_data, limit=args.n_val_samples)
    run_m4 = (not args.skip_m4) and args.allow_mixed_teacher_m4
    low_inputs = low_targets = high_inputs = high_targets = None
    pair_ids: list[str] | None = None
    dropout_pair = None
    if run_m4:
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
        dropout_pair = load_event_dropout_pair(args.event_dropout_pair)
        dropout_teacher_m4 = None
        if dropout_pair is not None:
            dropout_teacher_m4 = compute_paired_event_intervention_metrics(
                None,
                dropout_pair["baseline_inputs"],
                dropout_pair["baseline_targets"],
                dropout_pair["intervention_inputs"],
                dropout_pair["intervention_targets"],
                intervention_name="event_dropout",
                soma_scale=0.1,
                ignore_start=args.ignore_start,
                trace_ids=dropout_pair["trace_ids"],
            )
    else:
        teacher_m4 = not_run_m4(args)
        dropout_teacher_m4 = None
    models = available_models(skip_mamba=args.skip_mamba, extra_checkpoints=args.extra_checkpoint)
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
        }
        if run_m4:
            assert low_inputs is not None and low_targets is not None and high_inputs is not None and high_targets is not None
            model_reports[name]["m4_event_rate_shift"] = compute_paired_event_intervention_metrics(
                model_fn,
                low_inputs,
                low_targets,
                high_inputs,
                high_targets,
                intervention_name="event_rate_shift_low_to_high",
                soma_scale=0.1,
                ignore_start=args.ignore_start,
                trace_ids=pair_ids,
            )
        if run_m4 and dropout_pair is not None:
            model_reports[name]["m4_event_dropout"] = compute_paired_event_intervention_metrics(
                model_fn,
                dropout_pair["baseline_inputs"],
                dropout_pair["baseline_targets"],
                dropout_pair["intervention_inputs"],
                dropout_pair["intervention_targets"],
                intervention_name="event_dropout",
                soma_scale=0.1,
                ignore_start=args.ignore_start,
                trace_ids=dropout_pair["trace_ids"],
            )

    report = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": time.perf_counter() - start,
            "device": args.device,
            "val_data": args.val_data,
            "low_input_data": args.low_input_data,
            "high_input_data": args.high_input_data,
            "m4_status": "mixed_teacher_stress_test" if run_m4 else "not_run",
            "ignore_start": args.ignore_start,
            "n_val_samples": int(val_inputs.shape[0]),
            "n_intervention_pairs": int(low_inputs.shape[0]) if low_inputs is not None else 0,
        },
        "teacher_m4_event_rate_shift": teacher_m4,
        "teacher_m4_event_dropout": dropout_teacher_m4,
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


def load_event_dropout_pair(path: str | Path) -> dict[str, Any] | None:
    pair_path = Path(path)
    if not pair_path.exists():
        return None
    with np.load(pair_path) as data:
        trace_ids = [str(x) for x in data["sample_ids"].tolist()] if "sample_ids" in data else None
        return {
            "baseline_inputs": data["baseline_inputs"],
            "baseline_targets": data["baseline_targets"],
            "intervention_inputs": data["intervention_inputs"],
            "intervention_targets": data["intervention_targets"],
            "trace_ids": trace_ids,
        }


def paired_trace_ids(low_ids: list[str], high_ids: list[str]) -> list[str]:
    n = min(len(low_ids), len(high_ids))
    return [f"low={low_ids[i]} | high={high_ids[i]}" for i in range(n)]


def not_run_m4(args: argparse.Namespace) -> dict[str, Any]:
    reason = "skipped by --skip-m4" if args.skip_m4 else "skipped to avoid mixing calibrated-Arbor M1 with old-teacher M4 pairs"
    return {
        "status": "not_run",
        "kind": "paired_event_intervention",
        "reason": reason,
        "teacher_response": {
            "response_detected": False,
            "event_fraction_delta": None,
            "delta_spike_rate_per_1000_steps": None,
            "delta_soma_mean_mv": None,
            "teacher_direction": "not_run",
        },
    }


def available_models(*, skip_mamba: bool, extra_checkpoints: list[str]) -> dict[str, torch.nn.Module]:
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
    for item in extra_checkpoints:
        if "=" not in item:
            raise ValueError(f"--extra-checkpoint must be NAME=PATH, got {item!r}")
        name, path_text = item.split("=", 1)
        model, _ = load_checkpoint(path_text, map_location="cpu")
        model._checkpoint_path = path_text  # type: ignore[attr-defined]
        models[name] = model
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
    if report.get("status") == "not_run":
        return {
            "teacher_response_detected": False,
            "event_fraction_delta": None,
            "delta_spike_rate_per_1000_steps": None,
            "delta_soma_mean_mv": None,
            "warning": report.get("reason", "M4 not run"),
        }
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
        model_summary[name] = {
            "m1_soma_rmse_mv": m1["soma_rmse_mv"],
            "m1_spike_auc": m1["spike_auc"],
        }
        if "m4_event_rate_shift" in row:
            m4 = row["m4_event_rate_shift"]["model_response"]
            model_summary[name]["m4_soma_delta_rmse_mv"] = m4["soma_delta_rmse_mv"]
            model_summary[name]["m4_spike_count_delta_mae"] = m4["spike_count_delta_mae_per_trace"]
        if "m4_event_dropout" in row:
            dropout = row["m4_event_dropout"]["model_response"]
            model_summary[name]["dropout_soma_delta_rmse_mv"] = dropout["soma_delta_rmse_mv"]
            model_summary[name]["dropout_spike_count_delta_mae"] = dropout["spike_count_delta_mae_per_trace"]
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
    teacher_report = report["teacher_m4_event_rate_shift"]
    teacher = teacher_report["teacher_response"]
    if teacher_report.get("status") == "not_run":
        lines.extend([f"- Not run: {teacher_report['reason']}", "", "## Teacher M4 Event Dropout", ""])
    else:
        lines.extend(
            [
                f"- Baseline event fraction: `{teacher['baseline_event_fraction']:.6f}`",
                f"- Intervention event fraction: `{teacher['intervention_event_fraction']:.6f}`",
                f"- Delta spike rate / 1000 steps: `{teacher['delta_spike_rate_per_1000_steps']:.4f}`",
                f"- Delta soma mean mV: `{teacher['delta_soma_mean_mv']:.4f}`",
                f"- Direction: `{teacher['teacher_direction']}`",
                f"- Warning: {report['interpretation']['warning'] or 'none'}",
                "",
                "## Teacher M4 Event Dropout",
                "",
            ]
        )
    dropout_teacher = report.get("teacher_m4_event_dropout")
    if dropout_teacher is None:
        lines.append("- Not run: no paired dropout artifact was found.")
    else:
        dropout = dropout_teacher["teacher_response"]
        lines.extend(
            [
                f"- Baseline event fraction: `{dropout['baseline_event_fraction']:.6f}`",
                f"- Intervention event fraction: `{dropout['intervention_event_fraction']:.6f}`",
                f"- Delta spike rate / 1000 steps: `{dropout['delta_spike_rate_per_1000_steps']:.4f}`",
                f"- Delta soma mean mV: `{dropout['delta_soma_mean_mv']:.4f}`",
                f"- Direction: `{dropout['teacher_direction']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Model Smoke Comparison",
            "",
            "| model | M1 RMSE mV | M1 AUC | rate soma-delta RMSE | rate spike-delta MAE | dropout soma-delta RMSE | dropout spike-delta MAE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in report["models"].items():
        m1 = row["mvp_trace_report"]["m1_trace_fidelity"]
        m4 = row.get("m4_event_rate_shift", {}).get("model_response")
        dropout = row.get("m4_event_dropout", {}).get("model_response")
        m4_soma = f"{m4['soma_delta_rmse_mv']:.4f}" if m4 else "N/A"
        m4_spike = f"{m4['spike_count_delta_mae_per_trace']:.4f}" if m4 else "N/A"
        dropout_soma = f"{dropout['soma_delta_rmse_mv']:.4f}" if dropout else "N/A"
        dropout_spike = f"{dropout['spike_count_delta_mae_per_trace']:.4f}" if dropout else "N/A"
        lines.append(
            f"| `{name}` | {m1['soma_rmse_mv']:.4f} | {m1['spike_auc']:.4f} | "
            f"{m4_soma} | {m4_spike} | "
            f"{dropout_soma} | {dropout_spike} |"
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
