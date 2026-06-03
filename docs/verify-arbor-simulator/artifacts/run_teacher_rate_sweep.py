#!/usr/bin/env python3
"""Generate a small paired Arbor input-rate sweep for M4 sanity checking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from bnn_sim.arbor_sim import ArborHeteroConfig, write_arbor_hetero_dataset


PROTOCOLS = {
    "low": {"base_event_rate_hz": 0.5, "burst_probability": 0.005, "burst_event_rate_hz": 20.0},
    "nominal": {"base_event_rate_hz": 2.0, "burst_probability": 0.015, "burst_event_rate_hz": 35.0},
    "high": {"base_event_rate_hz": 10.0, "burst_probability": 0.05, "burst_event_rate_hz": 50.0},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="docs/verify-arbor-simulator/artifacts/teacher_rate_sweep")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--time-steps", type=int, default=3000)
    parser.add_argument("--input-dim", type=int, default=1278)
    parser.add_argument("--branches", type=int, default=45)
    parser.add_argument("--integration-dt-ms", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--sim-batch-size", type=int, default=4)
    parser.add_argument("--arbor-gpu-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    reports = {}
    for name, protocol in PROTOCOLS.items():
        out = output_root / name
        report_path = out / "generation_report.json"
        if args.reuse_existing and report_path.exists():
            reports[name] = json.loads(report_path.read_text(encoding="utf-8"))
            continue
        config = ArborHeteroConfig(
            input_dim=args.input_dim,
            num_branch=args.branches,
            time_steps=args.time_steps,
            integration_dt_ms=args.integration_dt_ms,
            base_event_rate_hz=protocol["base_event_rate_hz"],
            burst_probability=protocol["burst_probability"],
            burst_event_rate_hz=protocol["burst_event_rate_hz"],
            arbor_gpu_id=None if args.cpu_only else args.arbor_gpu_id,
            sim_batch_size=args.sim_batch_size,
        )
        reports[name] = write_arbor_hetero_dataset(
            out,
            samples=args.samples,
            shards=1,
            seed=args.seed,
            config=config,
        )

    summary = {
        "metadata": {
            "elapsed_seconds": time.perf_counter() - start,
            "samples": args.samples,
            "time_steps": args.time_steps,
            "seed": args.seed,
            "protocols": PROTOCOLS,
        },
        "reports": reports,
        "observed": {name: observed_metrics(Path(report["output"])) for name, report in reports.items()},
    }
    summary["interpretation"] = interpret(summary["observed"])
    summary_path = output_root / "rate_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "rate_sweep_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["interpretation"], indent=2, sort_keys=True))
    return 0


def observed_metrics(root: Path) -> dict[str, float]:
    inputs, targets, soma_mv = [], [], []
    for shard in sorted(root.glob("*.npz")):
        with np.load(shard) as data:
            inputs.append(data["inputs"])
            targets.append(data["targets"])
            soma_mv.append(data["soma_mv"])
    x = np.concatenate(inputs, axis=0)
    y = np.concatenate(targets, axis=0)
    soma = np.concatenate(soma_mv, axis=0)
    return {
        "event_fraction": float(np.count_nonzero(x) / x.size),
        "spike_rate_hz": float(1000.0 * np.mean(y[..., 0] > 0.5)),
        "soma_mean_mv": float(np.mean(soma)),
        "soma_std_mv": float(np.std(soma)),
        "samples": int(x.shape[0]),
        "time_steps": int(x.shape[1]),
    }


def interpret(observed: dict[str, dict[str, float]]) -> dict[str, object]:
    ordered = [observed[name] for name in ("low", "nominal", "high")]
    event_monotone = ordered[0]["event_fraction"] < ordered[1]["event_fraction"] < ordered[2]["event_fraction"]
    spike_monotone = ordered[0]["spike_rate_hz"] <= ordered[1]["spike_rate_hz"] <= ordered[2]["spike_rate_hz"]
    soma_monotone = ordered[0]["soma_mean_mv"] <= ordered[1]["soma_mean_mv"] <= ordered[2]["soma_mean_mv"]
    return {
        "event_fraction_monotone": event_monotone,
        "spike_rate_monotone_increasing": spike_monotone,
        "soma_mean_monotone_increasing": soma_monotone,
        "warning": None if spike_monotone else "Spike rate is not monotone increasing with balanced E/I event-rate protocol.",
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# Arbor Teacher Rate Sweep",
        "",
        "| protocol | event fraction | spike rate Hz | soma mean mV | soma std mV |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("low", "nominal", "high"):
        row = summary["observed"][name]
        lines.append(
            f"| `{name}` | {row['event_fraction']:.6f} | {row['spike_rate_hz']:.4f} | "
            f"{row['soma_mean_mv']:.4f} | {row['soma_std_mv']:.4f} |"
        )
    lines.extend(["", f"Interpretation: `{json.dumps(summary['interpretation'], sort_keys=True)}`", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
