#!/usr/bin/env python3
"""Evaluate trained checkpoints on the NEURO-BENCH MVP over NMDA traces.

This is intentionally separate from the older smoke runner, whose default M4
pairs come from Arbor. Here the default is M1/M1b/M2 on the corrected
NeuronIO/NMDA validation split, plus dependency-free SVG trace plots for sanity
inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
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
    parser.add_argument("--val-data", default="data/full-trace-shards/nmda-val")
    parser.add_argument("--output-dir", default="docs/verify-arbor-simulator/artifacts/nmda_neuro_bench_eval_20260604")
    parser.add_argument("--checkpoint", action="append", default=[], help="NAME=PATH")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--ignore-start", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot-count", type=int, default=3)
    parser.add_argument("--report-title", default="NEURO-BENCH Evaluation")
    args = parser.parse_args()

    if not args.checkpoint:
        raise ValueError("provide at least one --checkpoint NAME=PATH")

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    inputs, targets, trace_ids = load_traces(args.val_data, args.max_samples)
    teacher_stats = trace_stats(targets, inputs)

    models: dict[str, Any] = {}
    predictions_by_model: dict[str, np.ndarray] = {}
    for item in args.checkpoint:
        name, path = parse_checkpoint_arg(item)
        model, metadata = load_checkpoint(path, map_location="cpu")
        model = model.to(args.device).eval()
        predictions = predict_in_batches(model, inputs, args.device, args.eval_batch_size)
        suite = BenchmarkSuite()
        report = suite.evaluate_model(
            lambda x, predictions=predictions: predictions,
            inputs,
            targets,
            soma_scale=SOMA_SCALE,
            ignore_start=args.ignore_start,
            run_perturbation=False,
            model_name=name,
            model_config={"checkpoint": path, "checkpoint_metadata": metadata},
            dataset_path=args.val_data,
            trace_ids=trace_ids,
        )
        sanity = prediction_sanity(predictions, targets, inputs, args.ignore_start)
        trace_errors = per_trace_errors(predictions, targets, args.ignore_start)
        plot_paths = write_trace_plots(
            plots_dir,
            name=name,
            predictions=predictions,
            targets=targets,
            inputs=inputs,
            trace_ids=trace_ids,
            trace_errors=trace_errors,
            plot_count=args.plot_count,
            ignore_start=args.ignore_start,
        )
        models[name] = {
            "checkpoint": path,
            "checkpoint_metadata": metadata,
            "parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "benchmark": report,
            "sanity": sanity,
            "trace_error_summary": summarize_trace_errors(trace_errors, trace_ids),
            "plots": [str(path.relative_to(output_dir)) for path in plot_paths],
        }
        predictions_by_model[name] = predictions

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "elapsed_seconds": time.perf_counter() - started,
        "output_dir": str(output_dir),
        "report_title": args.report_title,
        "device": args.device,
        "val_data": args.val_data,
        "max_samples": args.max_samples,
        "loaded_samples": int(inputs.shape[0]),
        "trace_shape": list(inputs.shape),
        "ignore_start": args.ignore_start,
        "teacher_stats": teacher_stats,
        "models": models,
        "plots_produced": any(row["plots"] for row in models.values()),
        "limitations": [
            "This evaluator runs M1/M1b/M2 only; it does not mix in Arbor M4 intervention pairs.",
            "Trace plots are diagnostic sanity plots, not publication figures.",
            "Metrics are over the selected validation subset unless max_samples covers the full split.",
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(compact_summary(summary), indent=2, sort_keys=True))
    return 0


def parse_checkpoint_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"checkpoint must be NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    if not name:
        raise ValueError("checkpoint name must not be empty")
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return name, path


def load_traces(path: str | Path, max_samples: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    dataset = ShardedNeuronIODataset(path, cache_shards=1)
    n = min(max_samples, len(dataset))
    indices = list(range(n))
    inputs, targets = dataset.get_batch(indices)
    trace_ids = dataset.get_sample_ids(indices)
    return inputs.astype(np.float32, copy=False), targets.astype(np.float32, copy=False), trace_ids


def predict_in_batches(model: torch.nn.Module, inputs: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], batch_size):
            batch = torch.from_numpy(inputs[start : start + batch_size]).float().to(device)
            pred = model(batch).detach().cpu().numpy()
            outputs.append(pred.astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0)


def soma_mv(array: np.ndarray) -> np.ndarray:
    return array[..., 1] / SOMA_SCALE + SOMA_BIAS_MV


def trace_stats(targets: np.ndarray, inputs: np.ndarray) -> dict[str, Any]:
    soma = soma_mv(targets)
    spikes = targets[..., 0] > 0.5
    return {
        "event_fraction": float(np.count_nonzero(inputs) / max(1, inputs.size)),
        "spike_count": int(spikes.sum()),
        "spike_rate_per_1000_steps": float(1000.0 * spikes.mean()),
        "soma_mean_mv": float(soma.mean()),
        "soma_std_mv": float(soma.std()),
        "soma_min_mv": float(soma.min()),
        "soma_max_mv": float(soma.max()),
    }


def prediction_sanity(predictions: np.ndarray, targets: np.ndarray, inputs: np.ndarray, ignore_start: int) -> dict[str, Any]:
    pred = predictions[:, ignore_start:, :]
    tgt = targets[:, ignore_start:, :]
    pred_soma = soma_mv(predictions)[:, ignore_start:]
    target_soma = soma_mv(targets)[:, ignore_start:]
    logits = pred[..., 0]
    pred_spikes = logits > 0.0
    target_spikes = tgt[..., 0] > 0.5
    matched_threshold = prevalence_matched_threshold(logits, int(target_spikes.sum()))
    matched_spikes = logits >= matched_threshold
    finite = bool(np.isfinite(predictions).all())
    return {
        "all_finite": finite,
        "pred_soma_mean_mv": float(pred_soma.mean()),
        "pred_soma_std_mv": float(pred_soma.std()),
        "pred_soma_min_mv": float(pred_soma.min()),
        "pred_soma_max_mv": float(pred_soma.max()),
        "target_soma_mean_mv": float(target_soma.mean()),
        "target_soma_std_mv": float(target_soma.std()),
        "pred_spike_count_threshold_0": int(pred_spikes.sum()),
        "pred_spike_count_prevalence_matched": int(matched_spikes.sum()),
        "prevalence_matched_threshold": float(matched_threshold),
        "target_spike_count": int(target_spikes.sum()),
        "spike_logit_p01_p50_p99": [float(x) for x in np.quantile(logits, [0.01, 0.5, 0.99])],
        "event_fraction": float(np.count_nonzero(inputs[:, ignore_start:, :]) / max(1, inputs[:, ignore_start:, :].size)),
        "flags": sanity_flags(finite, pred_soma, target_soma, pred_spikes, target_spikes),
    }


def sanity_flags(
    finite: bool,
    pred_soma: np.ndarray,
    target_soma: np.ndarray,
    pred_spikes: np.ndarray,
    target_spikes: np.ndarray,
) -> list[str]:
    flags: list[str] = []
    if not finite:
        flags.append("non_finite_predictions")
    if pred_soma.min() < -150 or pred_soma.max() > 80:
        flags.append("implausible_voltage_range")
    if pred_soma.std() < 0.05 * max(target_soma.std(), 1e-6):
        flags.append("near_constant_voltage")
    if target_spikes.sum() > 0 and pred_spikes.sum() == 0:
        flags.append("threshold_zero_predicts_no_spikes")
    if pred_spikes.sum() > 20 * max(target_spikes.sum(), 1):
        flags.append("too_many_thresholded_predicted_spikes")
    return flags


def prevalence_matched_threshold(scores: np.ndarray, positive_count: int) -> float:
    flat = np.asarray(scores).reshape(-1)
    if positive_count <= 0:
        return float("inf")
    if positive_count >= flat.size:
        return float("-inf")
    return float(np.partition(flat, flat.size - positive_count)[flat.size - positive_count])


def per_trace_errors(predictions: np.ndarray, targets: np.ndarray, ignore_start: int) -> np.ndarray:
    pred = predictions[:, ignore_start:, :]
    tgt = targets[:, ignore_start:, :]
    residual_mv = (pred[..., 1] - tgt[..., 1]) / SOMA_SCALE
    return np.sqrt(np.mean(residual_mv**2, axis=1))


def summarize_trace_errors(errors: np.ndarray, trace_ids: list[str]) -> dict[str, Any]:
    order = np.argsort(errors)
    return {
        "min_rmse_mv": float(errors[order[0]]),
        "median_rmse_mv": float(np.median(errors)),
        "max_rmse_mv": float(errors[order[-1]]),
        "best_trace_id": trace_ids[int(order[0])],
        "worst_trace_id": trace_ids[int(order[-1])],
    }


def write_trace_plots(
    plots_dir: Path,
    *,
    name: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    inputs: np.ndarray,
    trace_ids: list[str],
    trace_errors: np.ndarray,
    plot_count: int,
    ignore_start: int,
) -> list[Path]:
    if plot_count <= 0:
        return []
    order = np.argsort(trace_errors)
    candidates = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    for idx in order[-plot_count:]:
        if int(idx) not in candidates:
            candidates.append(int(idx))
    paths = []
    for trace_index in candidates[: max(plot_count, 3)]:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        path = plots_dir / f"{safe_name}_trace{trace_index:03d}.svg"
        path.write_text(
            render_trace_svg(
                predictions[trace_index],
                targets[trace_index],
                inputs[trace_index],
                title=f"{name} | trace {trace_index} | RMSE {trace_errors[trace_index]:.3f} mV",
                trace_id=trace_ids[trace_index],
                ignore_start=ignore_start,
            ),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def render_trace_svg(
    prediction: np.ndarray,
    target: np.ndarray,
    inputs: np.ndarray,
    *,
    title: str,
    trace_id: str,
    ignore_start: int,
) -> str:
    width, height = 1100, 420
    left, right, top, bottom = 60, 20, 45, 40
    plot_w = width - left - right
    soma_top, soma_h = 70, 230
    spike_y0, spike_y1 = 330, 380
    pred_mv = soma_mv(prediction)
    target_mv = soma_mv(target)
    lo = float(min(pred_mv[ignore_start:].min(), target_mv[ignore_start:].min()))
    hi = float(max(pred_mv[ignore_start:].max(), target_mv[ignore_start:].max()))
    pad = max(1.0, 0.05 * (hi - lo))
    lo -= pad
    hi += pad
    max_points = 900
    start = ignore_start
    step = max(1, (prediction.shape[0] - start) // max_points)
    xs = np.arange(start, prediction.shape[0], step)

    def xy(points: np.ndarray) -> str:
        coords = []
        for x in xs:
            sx = left + (x - start) / max(1, prediction.shape[0] - 1 - start) * plot_w
            sy = soma_top + (hi - points[x]) / max(1e-6, hi - lo) * soma_h
            coords.append(f"{sx:.2f},{sy:.2f}")
        return " ".join(coords)

    target_poly = xy(target_mv)
    pred_poly = xy(pred_mv)
    target_spikes = np.flatnonzero(target[:, 0] > 0.5)
    pred_spikes = np.flatnonzero(prediction[:, 0] > 0.0)
    event_counts = np.count_nonzero(inputs, axis=1)
    event_idx = np.flatnonzero(event_counts > 0)
    event_sample = event_idx[:: max(1, event_idx.size // 250)] if event_idx.size else np.asarray([], dtype=int)

    def xcoord(t: int) -> float:
        return left + (t - start) / max(1, prediction.shape[0] - 1 - start) * plot_w

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="24" font-family="sans-serif" font-size="16">{escape_xml(title)}</text>',
        f'<text x="{left}" y="43" font-family="sans-serif" font-size="11" fill="#555">{escape_xml(trace_id[:145])}</text>',
        f'<rect x="{left}" y="{soma_top}" width="{plot_w}" height="{soma_h}" fill="#fafafa" stroke="#ddd"/>',
        f'<text x="8" y="{soma_top + 15}" font-family="sans-serif" font-size="11" fill="#555">{hi:.1f} mV</text>',
        f'<text x="8" y="{soma_top + soma_h}" font-family="sans-serif" font-size="11" fill="#555">{lo:.1f} mV</text>',
        f'<polyline points="{target_poly}" fill="none" stroke="#111827" stroke-width="1.4"/>',
        f'<polyline points="{pred_poly}" fill="none" stroke="#dc2626" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{spike_y0}" x2="{left + plot_w}" y2="{spike_y0}" stroke="#ddd"/>',
        f'<line x1="{left}" y1="{spike_y1}" x2="{left + plot_w}" y2="{spike_y1}" stroke="#ddd"/>',
        f'<text x="{left}" y="{height - 12}" font-family="sans-serif" font-size="12" fill="#555">black=teacher voltage, red=prediction, blue ticks=teacher spikes, orange ticks=predicted spikes, gray ticks=input-event samples</text>',
    ]
    for t in event_sample:
        if t >= start:
            x = xcoord(int(t))
            elements.append(f'<line x1="{x:.2f}" y1="{spike_y0}" x2="{x:.2f}" y2="{spike_y1}" stroke="#9ca3af" stroke-width="0.4" opacity="0.35"/>')
    for t in target_spikes:
        if t >= start:
            x = xcoord(int(t))
            elements.append(f'<line x1="{x:.2f}" y1="{spike_y0}" x2="{x:.2f}" y2="{spike_y1}" stroke="#2563eb" stroke-width="1.3"/>')
    for t in pred_spikes:
        if t >= start:
            x = xcoord(int(t))
            elements.append(f'<line x1="{x:.2f}" y1="{spike_y0 + 8}" x2="{x:.2f}" y2="{spike_y1 - 8}" stroke="#f97316" stroke-width="0.8" opacity="0.7"/>')
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for name, row in summary["models"].items():
        m1 = row["benchmark"]["m1_trace_fidelity"]
        m1b = row["benchmark"]["m1b_feature_fidelity"]["features"]
        m2 = row["benchmark"]["m2_regimes"]
        near = regime_rmse(m2, "near_spike")
        rows[name] = {
            "checkpoint": row["checkpoint"],
            "m1_rmse_mv": m1["soma_rmse_mv"],
            "m1_auc": m1["spike_auc"],
            "spike_count_mae_threshold_zero": m1["spike_threshold_zero"]["count_mae_per_trace"],
            "spike_count_mae_prevalence_matched": m1["spike_prevalence_matched"]["count_mae_per_trace"],
            "spike_timing_f1_prevalence_matched": row["benchmark"]["m2_regimes"]["spike_timing_prevalence_matched"]["f1_at_2steps"],
            "near_spike_rmse_mv": near,
            "sanity_flags": row["sanity"]["flags"],
            "plots": row["plots"],
        }
    return {
        "output": str(Path(summary["output_dir"])),
        "loaded_samples": summary["loaded_samples"],
        "plots_produced": summary["plots_produced"],
        "models": rows,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary.get('report_title', 'NEURO-BENCH Evaluation')}",
        "",
        f"- Dataset: `{summary['val_data']}`",
        f"- Samples: `{summary['loaded_samples']}`",
        f"- Ignore start: `{summary['ignore_start']}`",
        f"- Plots produced: `{summary['plots_produced']}`",
        "",
        "## Teacher Subset",
        "",
        "| event fraction | spike rate / 1000 steps | soma mean mV | soma std mV | soma min/max mV |",
        "|---:|---:|---:|---:|---:|",
    ]
    t = summary["teacher_stats"]
    lines.append(
        f"| {t['event_fraction']:.6f} | {t['spike_rate_per_1000_steps']:.4f} | "
        f"{t['soma_mean_mv']:.4f} | {t['soma_std_mv']:.4f} | {t['soma_min_mv']:.2f}/{t['soma_max_mv']:.2f} |"
    )
    lines.extend([
        "",
        "## Models",
        "",
        "| model | params | M1 RMSE mV | M1 AUC | spike-count MAE @0 | spike-count MAE rate-matched | timing F1 rate-matched | near-spike RMSE | worst trace RMSE | sanity flags | plots |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for name, row in summary["models"].items():
        m1 = row["benchmark"]["m1_trace_fidelity"]
        m2 = row["benchmark"]["m2_regimes"]
        near = regime_rmse(m2, "near_spike")
        timing_matched = m2["spike_timing_prevalence_matched"]["f1_at_2steps"]
        flags = ", ".join(row["sanity"]["flags"]) or "none"
        plots = ", ".join(f"[plot {i + 1}]({p})" for i, p in enumerate(row["plots"]))
        lines.append(
            f"| `{name}` | {row['parameter_count']} | {m1['soma_rmse_mv']:.4f} | {m1['spike_auc']:.4f} | "
            f"{m1['spike_threshold_zero']['count_mae_per_trace']:.4f} | "
            f"{m1['spike_prevalence_matched']['count_mae_per_trace']:.4f} | "
            f"{timing_matched:.4f} | {format_optional_float(near)} | "
            f"{row['trace_error_summary']['max_rmse_mv']:.4f} | {flags} | {plots} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.append("")
    return "\n".join(lines)


def regime_rmse(m2: dict[str, Any], label: str) -> float | None:
    for row in m2.get("mvp_regimes", []):
        if row.get("regime_label") == label and row.get("rmse") is not None:
            return float(row["rmse"])
    return None


def format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
