#!/usr/bin/env python3
"""Quantify how target observation transforms change Arbor soma dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bnn_sim.data import ShardedNeuronIODataset
from bnn_sim.neuronio_raw import DEFAULT_Y_TRAIN_SOMA_BIAS, DEFAULT_Y_TRAIN_SOMA_SCALE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="raw Arbor split containing soma_mv")
    parser.add_argument("--reference", required=True, help="NeuronIO/NMDA split for dynamic targets")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--ignore-start", type=int, default=500)
    parser.add_argument("--clip-mv", type=float, default=-55.0)
    parser.add_argument("--lowpass-tau-steps", default="2,3,5,8,10,15")
    parser.add_argument("--slew-limits-mv-per-step", default="0.5,0.7,1.0,1.5,2.0,3.0")
    args = parser.parse_args()

    arbor_raw = load_soma(args.data, args.max_samples, prefer_raw=True)
    reference = load_soma(args.reference, args.max_samples, prefer_raw=False)
    raw_masked = arbor_raw[:, args.ignore_start :]
    ref_masked = reference[:, args.ignore_start :]
    clipped = np.minimum(arbor_raw, args.clip_mv)

    transforms: list[dict[str, Any]] = []
    transforms.append(summarize_transform("clip_only", clipped, arbor_raw, ref_masked, args.ignore_start))
    for tau in parse_float_list(args.lowpass_tau_steps):
        transformed = lowpass(clipped, tau_steps=tau)
        transforms.append(
            summarize_transform(f"clip_lowpass_tau{tau:g}", transformed, arbor_raw, ref_masked, args.ignore_start)
        )
    for limit in parse_float_list(args.slew_limits_mv_per_step):
        transformed = slew_limit(clipped, limit_mv_per_step=limit)
        transforms.append(
            summarize_transform(f"clip_slew_limit{limit:g}", transformed, arbor_raw, ref_masked, args.ignore_start)
        )

    ref_summary = summarize_soma(ref_masked)
    raw_summary = summarize_soma(raw_masked)
    report = {
        "metadata": {
            "data": args.data,
            "reference": args.reference,
            "max_samples": args.max_samples,
            "ignore_start": args.ignore_start,
            "clip_mv": args.clip_mv,
        },
        "reference_summary": ref_summary,
        "arbor_raw_summary": raw_summary,
        "transforms": transforms,
        "interpretation": [
            "A target transform is only defensible if it corresponds to an observation convention, not if it merely hides simulator dynamics.",
            "Low-pass or slew transforms that match NeuronIO dV/dt while strongly shrinking voltage variance should be treated as distortive unless NeuronIO measurement filtering justifies them.",
            "For Arbor replacement data, keep raw soma voltage available even when a clipped training target is used for historical Branch-ELM comparability.",
        ],
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(compact(report), indent=2, sort_keys=True))
    return 0


def load_soma(path: str, max_samples: int, *, prefer_raw: bool) -> np.ndarray:
    dataset = ShardedNeuronIODataset(path, cache_shards=1)
    n = min(len(dataset), max_samples)
    _inputs, targets = dataset.get_batch(range(n))
    if prefer_raw:
        raw = raw_soma_batch(dataset, range(n))
        if raw is not None:
            return raw.astype(np.float32, copy=False)
    return targets[..., 1].astype(np.float32) / DEFAULT_Y_TRAIN_SOMA_SCALE + DEFAULT_Y_TRAIN_SOMA_BIAS


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


def parse_float_list(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item]


def lowpass(soma: np.ndarray, *, tau_steps: float) -> np.ndarray:
    alpha = float(np.exp(-1.0 / max(tau_steps, 1e-6)))
    out = np.empty_like(soma)
    out[:, 0] = soma[:, 0]
    for t in range(1, soma.shape[1]):
        out[:, t] = alpha * out[:, t - 1] + (1.0 - alpha) * soma[:, t]
    return out


def slew_limit(soma: np.ndarray, *, limit_mv_per_step: float) -> np.ndarray:
    out = np.empty_like(soma)
    out[:, 0] = soma[:, 0]
    for t in range(1, soma.shape[1]):
        delta = np.clip(soma[:, t] - out[:, t - 1], -limit_mv_per_step, limit_mv_per_step)
        out[:, t] = out[:, t - 1] + delta
    return out


def summarize_transform(
    name: str,
    transformed_full: np.ndarray,
    raw_full: np.ndarray,
    reference_masked: np.ndarray,
    ignore_start: int,
) -> dict[str, Any]:
    transformed = transformed_full[:, ignore_start:]
    raw = raw_full[:, ignore_start:]
    summary = summarize_soma(transformed)
    raw_delta = transformed - raw
    ref_summary = summarize_soma(reference_masked)
    score = (
        abs(summary["std_mv"] - ref_summary["std_mv"])
        + abs(summary["abs_dvdt_p95_mv_per_step"] - ref_summary["abs_dvdt_p95_mv_per_step"])
        + abs(summary["abs_dvdt_p99_mv_per_step"] - ref_summary["abs_dvdt_p99_mv_per_step"])
    )
    return {
        "name": name,
        **summary,
        "raw_distortion_rmse_mv": float(np.sqrt(np.mean(raw_delta**2))),
        "raw_distortion_mae_mv": float(np.mean(np.abs(raw_delta))),
        "score_vs_reference_std_dvdt": float(score),
    }


def summarize_soma(soma: np.ndarray) -> dict[str, float]:
    dvdt = np.diff(soma, axis=1, prepend=soma[:, :1])
    return {
        "mean_mv": float(np.mean(soma)),
        "std_mv": float(np.std(soma)),
        "min_mv": float(np.min(soma)),
        "max_mv": float(np.max(soma)),
        "abs_dvdt_p95_mv_per_step": float(np.quantile(np.abs(dvdt), 0.95)),
        "abs_dvdt_p99_mv_per_step": float(np.quantile(np.abs(dvdt), 0.99)),
    }


def compact(report: dict[str, Any]) -> dict[str, Any]:
    transforms = sorted(report["transforms"], key=lambda row: row["score_vs_reference_std_dvdt"])[:5]
    return {
        "reference": report["reference_summary"],
        "arbor_raw": report["arbor_raw_summary"],
        "top_by_std_dvdt_score": transforms,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Observation Transform Sensitivity",
        "",
        "## Summaries",
        "",
        "| view | mean mV | std mV | min/max mV | |dV/dt| p95/p99 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in (("reference", report["reference_summary"]), ("arbor_raw", report["arbor_raw_summary"])):
        lines.append(
            f"| `{name}` | {row['mean_mv']:.4f} | {row['std_mv']:.4f} | "
            f"{row['min_mv']:.2f}/{row['max_mv']:.2f} | "
            f"{row['abs_dvdt_p95_mv_per_step']:.4f}/{row['abs_dvdt_p99_mv_per_step']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Transforms",
            "",
            "| transform | std mV | max mV | |dV/dt| p95/p99 | raw RMSE mV | score |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(report["transforms"], key=lambda item: item["score_vs_reference_std_dvdt"]):
        lines.append(
            f"| `{row['name']}` | {row['std_mv']:.4f} | {row['max_mv']:.2f} | "
            f"{row['abs_dvdt_p95_mv_per_step']:.4f}/{row['abs_dvdt_p99_mv_per_step']:.4f} | "
            f"{row['raw_distortion_rmse_mv']:.4f} | {row['score_vs_reference_std_dvdt']:.4f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in report["interpretation"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
