from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from axobench.metrics import compute_core_metric_set, soma_rmse, spike_auc
from axobench.benchmark.plots import (
    apply_paper_plot_style,
    plot_phase_plane_surface,
    plot_state_embedding_surface,
)
from axosim.checkpoint import load_checkpoint
from axosim.data import ShardedNeuronIODataset
from axosim.evaluate import _tensor_from_numpy


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate AxoSim checkpoints with AxoBench core metrics.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ignore-start", type=int, default=500)
    parser.add_argument("--skip-trace", action="store_true")
    parser.add_argument("--skip-state-plots", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = ShardedNeuronIODataset(args.data, cache_shards=2)
    n_samples = min(args.samples, len(dataset))
    indices = list(range(n_samples))
    inputs, targets, raw_soma = _load_arrays(dataset, indices, batch_size=args.batch_size)

    rows = []
    prediction_cache = {}
    for item in args.checkpoint:
        name, path = _parse_checkpoint_arg(item)
        predictions = _predict_checkpoint(path, inputs, batch_size=args.batch_size, device=args.device)
        prediction_cache[name] = predictions
        core = compute_core_metric_set(
            predictions,
            targets,
            raw_soma_mv=raw_soma,
            ignore_start=args.ignore_start,
            spike_tolerances_steps=(0, 1, 2, 3, 4, 5),
        )
        old_rmse_mv = _clipped_rmse_mv(predictions[:, args.ignore_start :], targets[:, args.ignore_start :])
        rows.append(
            {
                "model": name,
                **_duration_metadata(name),
                "samples": n_samples,
                "old_clipped_rmse_mv": old_rmse_mv,
                "old_spike_auc": spike_auc(predictions[:, args.ignore_start :], targets[:, args.ignore_start :]),
                "raw_rmse_mv": core["voltage"]["voltage_rmse_mv"],
                "voltage_sera_mv": core["voltage_sera_mv"],
                "exact_spike_f1": _f1_at_tolerance(core, 0),
                "f1_area_0_5ms": core["spike_f1_area"],
                "spike_threshold": core["spike_threshold"],
                "spike_threshold_kind": core["spike_threshold_kind"],
            }
        )

    _write_json(output / "summary.json", {"rows": rows, "samples": n_samples, "ignore_start": args.ignore_start})
    _write_table(output / "table01_branch_elm_weighting_probe.md", rows)
    if not args.skip_trace:
        _plot_trace(output / "fig01_branch_elm_weighting_trace", targets, raw_soma, prediction_cache, ignore_start=args.ignore_start)
    if any(row.get("presentations") is not None for row in rows):
        _plot_duration_curves(output / "fig04_branch_elm_duration_probe", rows)
    if not args.skip_state_plots:
        plot_phase_plane_surface(
            output=output / "fig02_branch_elm_phase_plane_voltage_error",
            targets=targets,
            predictions_by_model=prediction_cache,
            raw_soma_mv=raw_soma,
            ignore_start=args.ignore_start,
            statistic="voltage_rmse",
        )
        plot_phase_plane_surface(
            output=output / "fig03_teacher_phase_plane_spike_rate",
            targets=targets,
            raw_soma_mv=raw_soma,
            ignore_start=args.ignore_start,
            statistic="teacher_spike_rate",
        )
        plot_state_embedding_surface(
            output=output / "fig05_branch_elm_state_embedding_voltage_error",
            targets=targets,
            predictions_by_model=prediction_cache,
            inputs=inputs,
            raw_soma_mv=raw_soma,
            ignore_start=args.ignore_start,
            statistic="voltage_rmse",
            reducer="umap",
        )
        plot_state_embedding_surface(
            output=output / "fig06_teacher_state_embedding_spike_rate",
            targets=targets,
            inputs=inputs,
            raw_soma_mv=raw_soma,
            ignore_start=args.ignore_start,
            statistic="teacher_spike_rate",
            reducer="umap",
        )
    return 0


def _load_arrays(
    dataset: ShardedNeuronIODataset,
    indices: list[int],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    input_batches = []
    target_batches = []
    raw_batches = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        x, y = dataset.get_batch(batch_indices)
        input_batches.append(np.asarray(x, dtype=np.float32))
        target_batches.append(np.asarray(y, dtype=np.float32))
        raw = dataset.get_optional_array_batch("soma_mv", batch_indices)
        if raw is not None:
            raw_batches.append(np.asarray(raw, dtype=np.float32))
    inputs = np.concatenate(input_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0)
    raw_soma = np.concatenate(raw_batches, axis=0) if raw_batches else None
    return inputs, targets, raw_soma


def _predict_checkpoint(path: Path, inputs: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    model, _metadata = load_checkpoint(path, map_location=device)
    model = model.to(device).eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], batch_size):
            x = _tensor_from_numpy(
                inputs[start : start + batch_size],
                device,
                pin_memory=False,
                non_blocking=True,
            )
            pred = model(x).detach().cpu().numpy()
            predictions.append(pred.astype(np.float32, copy=False))
    return np.concatenate(predictions, axis=0)


def _clipped_rmse_mv(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(soma_rmse(prediction, target) / 0.1)


def _f1_at_tolerance(core: dict, tolerance: int) -> float | None:
    for row in core["spike"]["rows"]:
        if int(row["tolerance_steps"]) == tolerance:
            return None if row["f1"] is None else float(row["f1"])
    return None


def _plot_trace(
    output_base: Path,
    targets: np.ndarray,
    raw_soma: np.ndarray | None,
    predictions: dict[str, np.ndarray],
    *,
    ignore_start: int,
) -> None:
    apply_paper_plot_style()
    spike_counts = targets[:, ignore_start:, 0].sum(axis=1)
    sample = int(np.argmax(spike_counts))
    start = ignore_start
    stop = min(targets.shape[1], start + 700)
    if spike_counts[sample] > 0:
        spike_positions = np.flatnonzero(targets[sample, :, 0] > 0.5)
        first_spike = int(spike_positions[spike_positions >= ignore_start][0]) if np.any(spike_positions >= ignore_start) else start
        start = max(ignore_start, first_spike - 220)
        stop = min(targets.shape[1], first_spike + 480)
    time = np.arange(start, stop)
    target_soma = raw_soma[sample, start:stop] if raw_soma is not None else targets[sample, start:stop, 1] / 0.1 - 67.7

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.plot(time, target_soma, color="#111111", linewidth=1.2, label="teacher")
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for color, (name, pred) in zip(colors, predictions.items(), strict=False):
        pred_soma = pred[sample, start:stop, 1] / 0.1 - 67.7
        ax.plot(time, pred_soma, color=color, linewidth=0.95, label=_display_name(name))
    spike_times = np.flatnonzero(targets[sample, start:stop, 0] > 0.5) + start
    for spike_time in spike_times:
        ax.axvline(spike_time, color="#777777", linewidth=0.55, alpha=0.45)
    ax.set_xlabel("time step")
    ax.set_ylabel("soma voltage (mV)")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=min(len(predictions) + 1, 4),
        frameon=False,
        handlelength=2.0,
        columnspacing=1.1,
    )
    ax.grid(True, alpha=0.18)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for suffix in ("svg", "png", "pdf"):
        fig.savefig(output_base.with_suffix(f".{suffix}"), dpi=240)
    plt.close(fig)


def _plot_duration_curves(output_base: Path, rows: list[dict]) -> None:
    apply_paper_plot_style()
    duration_rows = [row for row in rows if row.get("presentations") is not None]
    if not duration_rows:
        return
    objectives = list(dict.fromkeys(str(row["objective"]) for row in duration_rows))
    colors = {"uniform MSE": "#0072B2", "SERA weighted": "#D55E00"}
    metrics = [
        ("voltage_sera_mv", "Voltage SERA (mV)", "lower"),
        ("raw_rmse_mv", "raw RMSE (mV)", "lower"),
        ("old_clipped_rmse_mv", "clipped RMSE (mV)", "lower"),
        ("exact_spike_f1", "exact Spike F1", "higher"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.1), sharex=True)
    for ax, (metric, label, direction) in zip(axes.flat, metrics, strict=False):
        for objective in objectives:
            series = sorted((row for row in duration_rows if row["objective"] == objective), key=lambda row: int(row["presentations"]))
            x = [row["presentations"] / 1000.0 for row in series]
            y = [row[metric] for row in series]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.15,
                markersize=3.0,
                color=colors.get(objective, "#555555"),
                label=objective,
            )
        ax.text(0.04, 0.92, label, transform=ax.transAxes, ha="left", va="top", fontsize=6.8)
        ax.text(0.96, 0.08, direction, transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color="#555555")
        ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("presentations (thousands)")
    axes[0, 1].legend(loc="best", frameon=False)
    fig.tight_layout()
    for suffix in ("svg", "png", "pdf"):
        fig.savefig(output_base.with_suffix(f".{suffix}"), dpi=240)
    plt.close(fig)


def _write_table(path: Path, rows: list[dict]) -> None:
    has_duration = any(row.get("presentations") is not None for row in rows)
    headers = []
    if has_duration:
        headers.extend(["objective", "presentations"])
    headers.extend(
        [
            "model",
            "old clipped RMSE (mV)",
            "old spike AUC",
            "raw RMSE (mV)",
            "Voltage SERA (mV)",
            "exact Spike F1",
            "F1 area 0-5 ms",
        ]
    )
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cells = []
        if has_duration:
            cells.extend(
                [
                    str(row.get("objective", "")),
                    "" if row.get("presentations") is None else str(row["presentations"]),
                ]
            )
        cells.extend(
            [
                str(row["model"]),
                f"{row['old_clipped_rmse_mv']:.3f}",
                f"{row['old_spike_auc']:.4f}",
                f"{row['raw_rmse_mv']:.3f}",
                f"{row['voltage_sera_mv']:.3f}",
                "NA" if row["exact_spike_f1"] is None else f"{row['exact_spike_f1']:.3f}",
                f"{row['f1_area_0_5ms']:.3f}",
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _duration_metadata(name: str) -> dict[str, str | int | None]:
    objective = None
    if name.startswith("baseline"):
        objective = "uniform MSE"
    elif name.startswith("sera1p0") or name.startswith("sera_weighted_1p0"):
        objective = "SERA weighted"
    elif name.startswith("sera_weighted_0p5"):
        objective = "SERA weighted 0.5"
    presentations = None
    suffix = name.rsplit("_", 1)[-1]
    if suffix.isdigit():
        presentations = int(suffix)
    return {"objective": objective, "presentations": presentations}


def _display_name(name: str) -> str:
    return {
        "baseline": "baseline",
        "sera_weighted_0p5": "SERA weighted 0.5",
        "sera_weighted_1p0": "SERA weighted 1.0",
    }.get(name, name.replace("_", " "))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_checkpoint_arg(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise ValueError("--checkpoint must be NAME=PATH")
    name, path = item.split("=", 1)
    return name, Path(path)


if __name__ == "__main__":
    raise SystemExit(main())
