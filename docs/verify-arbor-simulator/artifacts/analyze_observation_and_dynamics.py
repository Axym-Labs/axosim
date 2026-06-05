#!/usr/bin/env python3
"""Compare NeuronIO-style observation conventions and teacher dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bnn_sim.data import ShardedNeuronIODataset
from bnn_sim.official.constants import DEFAULT_Y_TRAIN_SOMA_BIAS, DEFAULT_Y_TRAIN_SOMA_SCALE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmda", required=True)
    parser.add_argument("--arbor-raw", required=True)
    parser.add_argument("--arbor-clipped", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--ignore-start", type=int, default=500)
    parser.add_argument("--clip-thresholds", default="-65,-60,-55,-50,-45")
    parser.add_argument("--soma-scale", type=float, default=DEFAULT_Y_TRAIN_SOMA_SCALE)
    parser.add_argument("--soma-bias-mv", type=float, default=DEFAULT_Y_TRAIN_SOMA_BIAS)
    args = parser.parse_args()

    thresholds = [float(item) for item in args.clip_thresholds.split(",") if item]
    nmda = load_view(args.nmda, max_samples=args.max_samples, soma_scale=args.soma_scale, soma_bias_mv=args.soma_bias_mv)
    arbor_raw = load_view(
        args.arbor_raw,
        max_samples=args.max_samples,
        soma_scale=args.soma_scale,
        soma_bias_mv=args.soma_bias_mv,
        prefer_raw_soma=True,
    )
    arbor_clipped = load_view(
        args.arbor_clipped,
        max_samples=args.max_samples,
        soma_scale=args.soma_scale,
        soma_bias_mv=args.soma_bias_mv,
    )

    report = {
        "metadata": {
            "nmda": args.nmda,
            "arbor_raw": args.arbor_raw,
            "arbor_clipped": args.arbor_clipped,
            "max_samples": args.max_samples,
            "ignore_start": args.ignore_start,
            "clip_thresholds": thresholds,
        },
        "views": {
            "nmda": summarize_view(nmda, ignore_start=args.ignore_start),
            "arbor_raw": summarize_view(arbor_raw, ignore_start=args.ignore_start),
            "arbor_clipped": summarize_view(arbor_clipped, ignore_start=args.ignore_start),
        },
        "clipping_distortion": {
            "arbor_raw_to_minus55": clipping_distortion(arbor_raw, clip_mv=-55.0, ignore_start=args.ignore_start),
            "threshold_sweep": [
                {
                    "clip_mv": threshold,
                    **clipping_distortion(arbor_raw, clip_mv=threshold, ignore_start=args.ignore_start),
                }
                for threshold in thresholds
            ],
        },
        "dynamic_comparison": {
            "nmda_vs_arbor_clipped": compare_dynamics(nmda, arbor_clipped, ignore_start=args.ignore_start),
            "nmda_vs_arbor_raw": compare_dynamics(nmda, arbor_raw, ignore_start=args.ignore_start),
        },
        "interpretation": interpretation(),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(compact(report), indent=2, sort_keys=True))
    return 0


def load_view(
    path: str,
    *,
    max_samples: int | None,
    soma_scale: float,
    soma_bias_mv: float,
    prefer_raw_soma: bool = False,
) -> dict[str, np.ndarray]:
    dataset = ShardedNeuronIODataset(path, cache_shards=1)
    n = min(len(dataset), max_samples or len(dataset))
    inputs, targets = dataset.get_batch(range(n))
    soma_mv = targets[..., 1].astype(np.float32) / soma_scale + soma_bias_mv
    if prefer_raw_soma:
        raw = raw_soma_batch(dataset, range(n))
        if raw is not None:
            soma_mv = raw.astype(np.float32, copy=False)
    return {
        "inputs": inputs.astype(np.float32, copy=False),
        "spikes": (targets[..., 0] > 0.5),
        "soma_mv": soma_mv,
    }


def raw_soma_batch(dataset: ShardedNeuronIODataset, indices: range) -> np.ndarray | None:
    rows = []
    for index in indices:
        shard_index, sample_index, _sample_id = dataset._index[index]
        shard = dataset.shards[shard_index]
        path = shard.path or (dataset.root / shard.name)
        with np.load(path) as data:
            if "soma_mv" not in data:
                return None
            rows.append(data["soma_mv"][sample_index])
    return np.stack(rows).astype(np.float32, copy=False)


def masked(view: dict[str, np.ndarray], ignore_start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        view["inputs"][:, ignore_start:, :],
        view["spikes"][:, ignore_start:],
        view["soma_mv"][:, ignore_start:],
    )


def summarize_view(view: dict[str, np.ndarray], *, ignore_start: int) -> dict[str, Any]:
    inputs, spikes, soma = masked(view, ignore_start)
    event_counts = np.count_nonzero(inputs, axis=-1)
    dvdt = np.diff(soma, axis=1, prepend=soma[:, :1])
    return {
        "samples": int(soma.shape[0]),
        "timesteps_after_mask": int(soma.shape[1]),
        "event_fraction": float(np.count_nonzero(inputs) / max(1, inputs.size)),
        "event_count_p50_p95_p99": [float(x) for x in np.quantile(event_counts, [0.5, 0.95, 0.99])],
        "spike_rate_per_1000_steps": float(1000.0 * np.mean(spikes)),
        "spike_count": int(spikes.sum()),
        "soma_mean_mv": float(np.mean(soma)),
        "soma_std_mv": float(np.std(soma)),
        "soma_min_mv": float(np.min(soma)),
        "soma_max_mv": float(np.max(soma)),
        "soma_quantiles_mv": [float(x) for x in np.quantile(soma, [0.01, 0.1, 0.5, 0.9, 0.99])],
        "abs_dvdt_quantiles_mv_per_step": [float(x) for x in np.quantile(np.abs(dvdt), [0.5, 0.9, 0.95, 0.99])],
        "spike_triggered": spike_triggered_summary(soma, spikes),
        "high_drive_kernel": high_drive_kernel(soma, event_counts),
    }


def clipping_distortion(view: dict[str, np.ndarray], *, clip_mv: float, ignore_start: int) -> dict[str, Any]:
    _, spikes, soma = masked(view, ignore_start)
    clipped = np.minimum(soma, clip_mv)
    delta = clipped - soma
    affected = soma > clip_mv
    near = dilate(spikes, before=5, after=10)
    return {
        "affected_fraction": float(np.mean(affected)),
        "affected_fraction_near_spike": float(np.mean(affected[near])) if near.any() else None,
        "global_rmse_mv": float(np.sqrt(np.mean(delta**2))),
        "global_mae_mv": float(np.mean(np.abs(delta))),
        "near_spike_rmse_mv": float(np.sqrt(np.mean(delta[near] ** 2))) if near.any() else None,
        "max_removed_mv": float(np.max(np.maximum(soma - clip_mv, 0.0))),
    }


def compare_dynamics(a: dict[str, np.ndarray], b: dict[str, np.ndarray], *, ignore_start: int) -> dict[str, Any]:
    a_summary = summarize_view(a, ignore_start=ignore_start)
    b_summary = summarize_view(b, ignore_start=ignore_start)
    a_sta = np.asarray(a_summary["spike_triggered"]["mean_waveform_mv"], dtype=np.float64)
    b_sta = np.asarray(b_summary["spike_triggered"]["mean_waveform_mv"], dtype=np.float64)
    sta_rmse = None
    if a_sta.size and b_sta.size and a_sta.shape == b_sta.shape:
        sta_rmse = float(np.sqrt(np.mean((a_sta - b_sta) ** 2)))
    return {
        "delta_spike_rate_per_1000_steps": b_summary["spike_rate_per_1000_steps"] - a_summary["spike_rate_per_1000_steps"],
        "delta_soma_mean_mv": b_summary["soma_mean_mv"] - a_summary["soma_mean_mv"],
        "delta_soma_std_mv": b_summary["soma_std_mv"] - a_summary["soma_std_mv"],
        "spike_triggered_waveform_rmse_mv": sta_rmse,
        "spike_triggered_peak_delta_mv": (
            b_summary["spike_triggered"]["mean_peak_mv"] - a_summary["spike_triggered"]["mean_peak_mv"]
            if a_summary["spike_triggered"]["mean_peak_mv"] is not None
            and b_summary["spike_triggered"]["mean_peak_mv"] is not None
            else None
        ),
        "high_drive_kernel_rmse_mv": kernel_rmse(
            a_summary["high_drive_kernel"]["mean_relative_mv"],
            b_summary["high_drive_kernel"]["mean_relative_mv"],
        ),
    }


def spike_triggered_summary(soma: np.ndarray, spikes: np.ndarray, *, before: int = 20, after: int = 80) -> dict[str, Any]:
    snippets = []
    for row in range(soma.shape[0]):
        for index in np.flatnonzero(spikes[row]):
            if index - before < 0 or index + after >= soma.shape[1]:
                continue
            snippets.append(soma[row, index - before : index + after + 1])
    if not snippets:
        return {
            "count": 0,
            "window_before": before,
            "window_after": after,
            "mean_waveform_mv": [],
            "mean_peak_mv": None,
            "mean_pre_spike_mv": None,
            "mean_post_20_mv": None,
        }
    arr = np.stack(snippets)
    mean = arr.mean(axis=0)
    return {
        "count": int(arr.shape[0]),
        "window_before": before,
        "window_after": after,
        "mean_waveform_mv": [float(x) for x in mean],
        "mean_peak_mv": float(mean.max()),
        "mean_pre_spike_mv": float(mean[before - 1]),
        "mean_post_20_mv": float(mean[before + 20]),
    }


def high_drive_kernel(soma: np.ndarray, event_counts: np.ndarray, *, before: int = 10, after: int = 50) -> dict[str, Any]:
    threshold = float(np.quantile(event_counts, 0.95))
    candidates = event_counts >= threshold
    if threshold <= 0.0:
        candidates = event_counts > 0
    snippets = []
    for row in range(soma.shape[0]):
        for index in np.flatnonzero(candidates[row]):
            if index - before < 0 or index + after >= soma.shape[1]:
                continue
            snippet = soma[row, index - before : index + after + 1]
            snippets.append(snippet - snippet[before])
    if not snippets:
        return {
            "count": 0,
            "event_count_threshold": threshold,
            "window_before": before,
            "window_after": after,
            "mean_relative_mv": [],
        }
    mean = np.stack(snippets).mean(axis=0)
    return {
        "count": int(len(snippets)),
        "event_count_threshold": threshold,
        "window_before": before,
        "window_after": after,
        "mean_relative_mv": [float(x) for x in mean],
        "peak_relative_mv": float(mean.max()),
        "trough_relative_mv": float(mean.min()),
    }


def dilate(mask: np.ndarray, *, before: int, after: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for row in range(mask.shape[0]):
        for index in np.flatnonzero(mask[row]):
            out[row, max(0, index - before) : min(mask.shape[1], index + after + 1)] = True
    return out


def kernel_rmse(a: list[float], b: list[float]) -> float | None:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.size == 0 or arr_b.size == 0 or arr_a.shape != arr_b.shape:
        return None
    return float(np.sqrt(np.mean((arr_a - arr_b) ** 2)))


def interpretation() -> list[str]:
    return [
        "Hard clipping is an observation convention: it discards raw action-potential peak amplitude while preserving the spike-label channel.",
        "It is defensible for NeuronIO comparability only if raw AP-shape fidelity is reported separately on the raw-soma view.",
        "For simulator calibration, match dynamic features such as spike-triggered waveform, dV/dt distribution, and high-drive response kernels rather than only mean/std/spike rate.",
    ]


def compact(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "views": {
            name: {
                "spike_rate_per_1000_steps": row["spike_rate_per_1000_steps"],
                "soma_mean_mv": row["soma_mean_mv"],
                "soma_std_mv": row["soma_std_mv"],
                "soma_max_mv": row["soma_max_mv"],
            }
            for name, row in report["views"].items()
        },
        "arbor_clip_minus55": report["clipping_distortion"]["arbor_raw_to_minus55"],
        "nmda_vs_arbor_clipped": report["dynamic_comparison"]["nmda_vs_arbor_clipped"],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Observation And Dynamics Comparison",
        "",
        "## Dataset Views",
        "",
        "| view | spike / 1000 steps | soma mean/std mV | soma min/max mV | |dV/dt| p95/p99 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in report["views"].items():
        dvdt = row["abs_dvdt_quantiles_mv_per_step"]
        lines.append(
            f"| `{name}` | {row['spike_rate_per_1000_steps']:.4f} | "
            f"{row['soma_mean_mv']:.4f}/{row['soma_std_mv']:.4f} | "
            f"{row['soma_min_mv']:.2f}/{row['soma_max_mv']:.2f} | {dvdt[2]:.4f}/{dvdt[3]:.4f} |"
        )

    distortion = report["clipping_distortion"]["arbor_raw_to_minus55"]
    lines.extend(
        [
            "",
            "## Clipping Distortion",
            "",
            f"- Arbor raw -> `-55 mV` clip affects `{100 * distortion['affected_fraction']:.4f}%` of masked timesteps.",
            f"- Global raw-vs-clipped distortion RMSE: `{distortion['global_rmse_mv']:.4f} mV`; near-spike distortion RMSE: `{distortion['near_spike_rmse_mv']:.4f} mV`.",
            f"- Maximum removed AP peak amplitude above threshold: `{distortion['max_removed_mv']:.2f} mV`.",
            "",
            "| clip mV | affected % | global RMSE mV | near-spike RMSE mV | max removed mV |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["clipping_distortion"]["threshold_sweep"]:
        lines.append(
            f"| {row['clip_mv']:.1f} | {100 * row['affected_fraction']:.4f} | "
            f"{row['global_rmse_mv']:.4f} | {row['near_spike_rmse_mv']:.4f} | {row['max_removed_mv']:.2f} |"
        )

    dyn = report["dynamic_comparison"]["nmda_vs_arbor_clipped"]
    lines.extend(
        [
            "",
            "## Dynamic Gap After Clipping",
            "",
            f"- Arbor clipped minus NMDA spike-rate delta: `{dyn['delta_spike_rate_per_1000_steps']:.4f}` spikes / 1000 steps.",
            f"- Arbor clipped minus NMDA soma mean/std deltas: `{dyn['delta_soma_mean_mv']:.4f}` / `{dyn['delta_soma_std_mv']:.4f} mV`.",
            f"- Spike-triggered average waveform RMSE: `{dyn['spike_triggered_waveform_rmse_mv']:.4f} mV`.",
            f"- High-drive response-kernel RMSE: `{dyn['high_drive_kernel_rmse_mv']:.4f} mV`.",
            "",
            "## Interpretation",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["interpretation"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
