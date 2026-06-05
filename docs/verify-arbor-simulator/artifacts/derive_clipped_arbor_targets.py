#!/usr/bin/env python3
"""Derive NeuronIO-compatible clipped targets from an Arbor dataset.

The source dataset must contain raw `soma_mv` in each shard. Inputs, spike
labels, sample ids, heterogeneity, and raw soma traces are copied unchanged.
Only `targets[..., 1]` is replaced by the clipped-and-normalized soma voltage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from bnn_sim.data import ShardedNeuronIODataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clip-mv", type=float, default=-55.0)
    parser.add_argument("--offset-mv", type=float, default=-67.7)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    split_reports = {}
    for split in sorted(path.name for path in source.iterdir() if path.is_dir()):
        split_reports[split] = derive_split(
            source / split,
            output / split,
            clip_mv=args.clip_mv,
            offset_mv=args.offset_mv,
            scale=args.scale,
        )

    root_reports = {}
    for name in ("calibrated_teacher_report.json", "generation_summary.json"):
        src = source / name
        if src.exists():
            root_reports[name] = json.loads(src.read_text(encoding="utf-8"))

    report = {
        "source": str(source),
        "output": str(output),
        "target_transform": {
            "kind": "neuronio_compatible_soma_clip",
            "clip_mv": args.clip_mv,
            "offset_mv": args.offset_mv,
            "scale": args.scale,
            "formula": "targets[...,1] = (minimum(raw_soma_mv, clip_mv) - offset_mv) * scale",
        },
        "split_reports": split_reports,
        "source_reports": root_reports,
        "interpretation": [
            "This is a target-convention view over the same Arbor simulations, not a new simulator run.",
            "Use this for comparability to historical NeuronIO metrics; use the source raw-soma dataset for AP-shape fidelity.",
        ],
    }
    (output / "clipped_target_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "clipped_target_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(output), "split_reports": split_reports}, indent=2, sort_keys=True))
    return 0


def derive_split(source: Path, output: Path, *, clip_mv: float, offset_mv: float, scale: float) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_manifest = dict(manifest)
    output_manifest["target_transform"] = {
        "kind": "neuronio_compatible_soma_clip",
        "source": str(source),
        "clip_mv": clip_mv,
        "offset_mv": offset_mv,
        "scale": scale,
    }

    for shard_entry in manifest.get("shards", []):
        file_name = shard_entry["file"]
        with np.load(source / file_name) as data:
            if "soma_mv" not in data:
                raise ValueError(f"{source / file_name} does not contain raw soma_mv")
            arrays = {name: data[name] for name in data.files}
        soma_mv = arrays["soma_mv"].astype(np.float32, copy=False)
        targets = arrays["targets"].astype(np.float32, copy=True)
        targets[..., 1] = ((np.minimum(soma_mv, clip_mv) - offset_mv) * scale).astype(np.float32)
        arrays["targets"] = targets
        np.savez_compressed(output / file_name, **arrays)

    (output / "manifest.json").write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stats = dataset_stats(output, offset_mv=offset_mv, scale=scale)
    (output / "generation_report.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stats


def dataset_stats(path: Path, *, offset_mv: float, scale: float) -> dict[str, float | int]:
    dataset = ShardedNeuronIODataset(path, cache_shards=1)
    total_events = 0
    total_inputs = 0
    total_spikes = 0.0
    total_time = 0
    soma_values = []
    raw_values = []
    for index in range(len(dataset)):
        sample = dataset[index]
        total_events += int(np.count_nonzero(sample.inputs))
        total_inputs += int(sample.inputs.size)
        total_spikes += float(sample.targets[:, 0].sum())
        total_time += int(sample.targets.shape[0])
        soma_values.append(sample.targets[:, 1].astype(np.float64) / scale + offset_mv)
        with np.load(path / dataset.shards[dataset._index[index][0]].name) as data:
            raw_values.append(data["soma_mv"][dataset._index[index][1]].astype(np.float64))
    soma = np.concatenate(soma_values)
    raw = np.concatenate(raw_values)
    return {
        "samples": len(dataset),
        "time_steps": int(total_time / max(1, len(dataset))),
        "event_fraction": float(total_events / max(1, total_inputs)),
        "spike_rate_hz": float(1000.0 * total_spikes / max(1, total_time)),
        "clipped_soma_mean_mv": float(soma.mean()),
        "clipped_soma_std_mv": float(soma.std()),
        "clipped_soma_min_mv": float(soma.min()),
        "clipped_soma_max_mv": float(soma.max()),
        "raw_soma_mean_mv": float(raw.mean()),
        "raw_soma_std_mv": float(raw.std()),
        "raw_soma_max_mv": float(raw.max()),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Clipped Arbor Target Report",
        "",
        f"- Source: `{report['source']}`",
        f"- Output: `{report['output']}`",
        f"- Clip: `{report['target_transform']['clip_mv']} mV`",
        "",
        "| split | samples | event fraction | spike rate Hz | clipped soma mean/std | clipped min/max | raw max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, stats in report["split_reports"].items():
        lines.append(
            f"| `{split}` | {stats['samples']} | {stats['event_fraction']:.6f} | {stats['spike_rate_hz']:.4f} | "
            f"{stats['clipped_soma_mean_mv']:.4f}/{stats['clipped_soma_std_mv']:.4f} | "
            f"{stats['clipped_soma_min_mv']:.2f}/{stats['clipped_soma_max_mv']:.2f} | {stats['raw_soma_max_mv']:.2f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in report["interpretation"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
