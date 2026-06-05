#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bnn_sim.data import ShardedNeuronIODataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare teacher dataset summary statistics.")
    parser.add_argument("--dataset", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    report = {name: _stats(Path(path)) for name, path in args.dataset}
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _stats(path: Path) -> dict[str, Any]:
    dataset = ShardedNeuronIODataset(path, cache_shards=1)
    total_steps = 0
    total_events = 0
    total_spikes = 0
    soma_sum = 0.0
    soma_square_sum = 0.0
    soma_min = float("inf")
    soma_max = float("-inf")
    input_dim = 0
    for shard in dataset.shards:
        with np.load(shard.path, mmap_mode="r") as data:
            inputs = data["inputs"]
            targets = data["targets"]
            input_dim = int(inputs.shape[-1])
            soma_mv = targets[..., 1].astype(np.float64) / 0.1 - 67.7
            total_events += int(np.count_nonzero(inputs))
            total_spikes += int(targets[..., 0].sum())
            total_steps += int(targets.shape[0] * targets.shape[1])
            soma_sum += float(soma_mv.sum())
            soma_square_sum += float(np.square(soma_mv).sum())
            soma_min = min(soma_min, float(soma_mv.min()))
            soma_max = max(soma_max, float(soma_mv.max()))
    soma_mean = soma_sum / total_steps
    soma_var = soma_square_sum / total_steps - soma_mean * soma_mean
    return {
        "samples": len(dataset),
        "input_dim": input_dim,
        "event_fraction": total_events / max(total_steps * input_dim, 1),
        "spike_rate_hz": 1000.0 * total_spikes / max(total_steps, 1),
        "soma_mean_mv_from_targets": soma_mean,
        "soma_std_mv_from_targets": float(np.sqrt(max(soma_var, 0.0))),
        "soma_min_mv_from_targets": soma_min,
        "soma_max_mv_from_targets": soma_max,
    }


if __name__ == "__main__":
    raise SystemExit(main())
