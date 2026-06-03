#!/usr/bin/env python3
"""Create and evaluate a paired Arbor teacher event-dropout intervention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from bnn_sim.arbor_sim import ArborHeteroConfig, simulate_arbor_targets_for_inputs
from bnn_sim.benchmark import compute_paired_event_intervention_metrics, make_event_dropout_inputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="docs/verify-arbor-simulator/artifacts/teacher_rate_sweep/nominal")
    parser.add_argument("--output-dir", default="docs/verify-arbor-simulator/artifacts/event_dropout_pair")
    parser.add_argument("--dropout-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--gpu", action="store_true", help="use Arbor GPU id 0 instead of the CPU context")
    args = parser.parse_args()

    start = time.perf_counter()
    source = Path(args.source)
    inputs, targets, heterogeneity, names, sample_ids = load_source(source, args.samples)
    dropout_inputs, drop_mask = make_event_dropout_inputs(
        inputs,
        dropout_probability=args.dropout_probability,
        seed=args.seed,
    )
    config = ArborHeteroConfig(
        input_dim=int(inputs.shape[2]),
        num_branch=45,
        time_steps=int(inputs.shape[1]),
        integration_dt_ms=0.05,
        arbor_gpu_id=0 if args.gpu else None,
        sim_batch_size=min(4, int(inputs.shape[0])),
    )
    simulated = simulate_arbor_targets_for_inputs(
        config,
        inputs=dropout_inputs,
        heterogeneity=heterogeneity,
        heterogeneity_names=names,
    )
    teacher_report = compute_paired_event_intervention_metrics(
        None,
        inputs,
        targets,
        dropout_inputs,
        simulated["targets"],
        intervention_name=f"event_dropout_p{args.dropout_probability:g}",
        soma_scale=0.1,
        ignore_start=min(500, max(0, inputs.shape[1] - 1)),
        trace_ids=sample_ids,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "event_dropout_pair.npz",
        baseline_inputs=inputs,
        baseline_targets=targets,
        intervention_inputs=dropout_inputs,
        intervention_targets=simulated["targets"],
        intervention_soma_mv=simulated["soma_mv"],
        dropped_event_mask=drop_mask,
        sample_ids=np.asarray(sample_ids),
    )
    report = {
        "metadata": {
            "source": str(source),
            "elapsed_seconds": time.perf_counter() - start,
            "dropout_probability": args.dropout_probability,
            "seed": args.seed,
            "samples": int(inputs.shape[0]),
            "time_steps": int(inputs.shape[1]),
            "dropped_events": int(drop_mask.sum()),
            "baseline_events": int(np.count_nonzero(inputs)),
        },
        "teacher_event_dropout": teacher_report,
    }
    (output_dir / "event_dropout_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "event_dropout_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["teacher_event_dropout"]["teacher_response"], indent=2, sort_keys=True))
    return 0


def load_source(root: Path, limit: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no source shards found in {root}")
    inputs, targets, heterogeneity = [], [], []
    names: list[str] | None = None
    sample_ids: list[str] = []
    for shard in files:
        with np.load(shard) as data:
            inputs.append(data["inputs"])
            targets.append(data["targets"])
            heterogeneity.append(data["heterogeneity"])
            names = [str(x) for x in data["heterogeneity_names"].tolist()]
            sample_ids.extend(str(x) for x in data["sample_ids"].tolist())
    x = np.concatenate(inputs, axis=0)
    y = np.concatenate(targets, axis=0)
    h = np.concatenate(heterogeneity, axis=0)
    if limit is not None:
        x, y, h, sample_ids = x[:limit], y[:limit], h[:limit], sample_ids[:limit]
    return x, y, h, names or [], sample_ids


def render_markdown(report: dict) -> str:
    response = report["teacher_event_dropout"]["teacher_response"]
    meta = report["metadata"]
    return "\n".join(
        [
            "# Event Dropout Teacher Pair",
            "",
            f"- Source: `{meta['source']}`",
            f"- Dropout probability: `{meta['dropout_probability']}`",
            f"- Dropped events: `{meta['dropped_events']}` / `{meta['baseline_events']}`",
            f"- Baseline event fraction: `{response['baseline_event_fraction']:.6f}`",
            f"- Intervention event fraction: `{response['intervention_event_fraction']:.6f}`",
            f"- Delta spike rate / 1000 steps: `{response['delta_spike_rate_per_1000_steps']:.4f}`",
            f"- Delta soma mean mV: `{response['delta_soma_mean_mv']:.4f}`",
            f"- Direction: `{response['teacher_direction']}`",
            f"- Response detected: `{response['response_detected']}`",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
