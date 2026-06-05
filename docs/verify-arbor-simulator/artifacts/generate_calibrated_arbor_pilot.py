#!/usr/bin/env python3
"""Generate a calibrated Arbor pilot split and report teacher statistics.

This uses the best current NMDA-regime calibration candidate. It is a pilot
replacement dataset for simulator-fidelity validation, not a final full-size
training corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bnn_sim.arbor_sim import ArborHeteroConfig, write_arbor_hetero_splits
from bnn_sim.data import ShardedNeuronIODataset


TARGET_NMDA_VAL10 = {
    "event_fraction": 0.01061581297681925,
    "spike_rate_hz": 0.8815104166666666,
    "soma_mean_mv": -67.61582763850524,
    "soma_std_mv": 3.1696798142366305,
    "soma_max_mv": -55.0,
}


CALIBRATED_PARAMS = {
    "soma_cm_scale": 1.0,
    "dend_cm_scale": 1.0,
    "apic_cm_scale": 1.0,
    "axon_cm_scale": 1.0,
    "axial_rL_scale": 1.0,
    "v_init_mv": -76.0,
    "temperature_celsius": 34.0,
    "soma_pas_g_scale": 1.0,
    "dend_pas_g_scale": 1.0,
    "apic_pas_g_scale": 1.0,
    "axon_pas_g_scale": 1.0,
    "soma_e_pas_shift_mv": 3.0,
    "dend_e_pas_shift_mv": 3.0,
    "apic_e_pas_shift_mv": 3.0,
    "axon_e_pas_shift_mv": 3.0,
    "soma_nav_scale": 1.0,
    "axon_nav_scale": 1.0,
    "dend_nav_scale": 1.0,
    "apic_nav_scale": 1.0,
    "soma_kv_scale": 1.0,
    "axon_kv_scale": 1.0,
    "dend_kv_scale": 1.0,
    "apic_kv_scale": 1.0,
    "dend_ih_scale": 1.0,
    "apic_ih_scale": 1.0,
    "dend_im_scale": 1.0,
    "apic_im_scale": 1.0,
    "soma_ca_scale": 1.0,
    "dend_ca_scale": 1.0,
    "apic_ca_scale": 1.0,
    "soma_sk_scale": 1.0,
    "axon_sk_scale": 1.0,
    "ca_decay_scale": 1.0,
    "ca_gamma_scale": 1.0,
    "exc_tau1_ms": 0.5,
    "exc_tau2_ms": 5.0,
    "inh_tau1_ms": 1.0,
    "inh_tau2_ms": 12.0,
    "ampa_weight_scale": 0.0012 * 2.0,
    "nmda_weight_scale": 0.00015 * 2.0,
    "inh_weight_scale": 0.006 * 1.0,
    "ei_balance": 0.0,
    "input_rate_scale": 1.0,
    "burstiness_scale": 1.0,
    "synchrony_scale": 1.0,
    "apic_synapse_fraction": 0.55,
    "spike_threshold_mv": -32.0,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/arbor-calibrated-nmda-proxy-20260604")
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=32)
    parser.add_argument("--shards-per-split", type=int, default=1)
    parser.add_argument("--time-steps", type=int, default=6000)
    parser.add_argument("--branches", type=int, default=45)
    parser.add_argument("--input-protocol", choices=["arbor_poisson", "neuronio_smoothed"], default="arbor_poisson")
    parser.add_argument("--synapse-routing", choices=["branch", "distributed"], default="branch")
    parser.add_argument("--distributed-sites-per-polarity", type=int)
    parser.add_argument("--spike-label-mode", choices=["threshold_crossing", "local_peak"], default="threshold_crossing")
    parser.add_argument("--spike-peak-threshold-mv", type=float, default=-25.0)
    parser.add_argument("--base-event-rate-hz", type=float, default=10.0)
    parser.add_argument("--burst-probability", type=float, default=0.0)
    parser.add_argument("--integration-dt-ms", type=float, default=0.05)
    parser.add_argument("--sim-batch-size", type=int, default=8)
    parser.add_argument("--arbor-gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260604)
    args = parser.parse_args()

    config = ArborHeteroConfig(
        time_steps=args.time_steps,
        num_branch=args.branches,
        input_protocol=args.input_protocol,
        synapse_routing=args.synapse_routing,
        distributed_sites_per_polarity=args.distributed_sites_per_polarity,
        spike_label_mode=args.spike_label_mode,
        spike_peak_threshold_mv=args.spike_peak_threshold_mv,
        base_event_rate_hz=args.base_event_rate_hz,
        burst_probability=args.burst_probability,
        integration_dt_ms=args.integration_dt_ms,
        arbor_gpu_id=args.arbor_gpu_id,
        sim_batch_size=args.sim_batch_size,
    )
    output = Path(args.output)
    summary = write_arbor_hetero_splits(
        output,
        split_samples={"train": args.train_samples, "val": args.val_samples},
        shards_per_split=args.shards_per_split,
        seed=args.seed,
        config=config,
        param_overrides=CALIBRATED_PARAMS,
    )
    stats = {
        split: dataset_stats(output / split)
        for split in ("train", "val")
    }
    report = {
        "target_nmda_val10": TARGET_NMDA_VAL10,
        "config": config.__dict__,
        "param_overrides": CALIBRATED_PARAMS,
        "generation_summary": summary,
        "split_stats": stats,
        "interpretation": [
            "Arbor targets are raw soma voltages normalized with NeuronIO affine scale; unlike historical NMDA, AP peaks are not clipped at -55 mV.",
            "This pilot fixes heterogeneity to calibrated nominal physiology so teacher fidelity can be tested before reintroducing heterogeneous cells.",
            "The `neuronio_smoothed` input protocol is an Arbor-morphology approximation of the Beniaguev/NeuronIO smoothed rate-field stimulus, not an exact NeuronIO simulator port.",
        ],
    }
    (output / "calibrated_teacher_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "calibrated_teacher_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(output), "split_stats": stats}, indent=2, sort_keys=True))
    return 0


def dataset_stats(path: Path) -> dict[str, float | int]:
    dataset = ShardedNeuronIODataset(path, cache_shards=1)
    total_events = 0
    total_inputs = 0
    total_spikes = 0.0
    total_time = 0
    soma_values: list[np.ndarray] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        total_events += int(np.count_nonzero(sample.inputs))
        total_inputs += int(sample.inputs.size)
        total_spikes += float(sample.targets[:, 0].sum())
        total_time += int(sample.targets.shape[0])
        soma_values.append(sample.targets[:, 1].astype(np.float64) / 0.1 + (-67.7))
    soma = np.concatenate(soma_values)
    return {
        "samples": len(dataset),
        "time_steps": int(total_time / max(1, len(dataset))),
        "event_fraction": float(total_events / max(1, total_inputs)),
        "spike_rate_hz": float(1000.0 * total_spikes / max(1, total_time)),
        "soma_mean_mv": float(soma.mean()),
        "soma_std_mv": float(soma.std()),
        "soma_min_mv": float(soma.min()),
        "soma_max_mv": float(soma.max()),
    }


def render_markdown(report: dict[str, Any]) -> str:
    target = report["target_nmda_val10"]
    lines = [
        "# Calibrated Arbor Pilot Teacher Report",
        "",
        "## Target",
        "",
        f"- NMDA val10 event fraction: `{target['event_fraction']:.6f}`",
        f"- NMDA val10 spike rate: `{target['spike_rate_hz']:.4f} Hz`",
        f"- NMDA val10 soma mean/std: `{target['soma_mean_mv']:.4f} / {target['soma_std_mv']:.4f} mV`",
        "",
        "## Pilot Split Stats",
        "",
        "| split | samples | event fraction | spike rate Hz | soma mean mV | soma std mV | soma min/max mV |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, stats in report["split_stats"].items():
        lines.append(
            f"| `{split}` | {stats['samples']} | {stats['event_fraction']:.6f} | {stats['spike_rate_hz']:.4f} | "
            f"{stats['soma_mean_mv']:.4f} | {stats['soma_std_mv']:.4f} | {stats['soma_min_mv']:.2f}/{stats['soma_max_mv']:.2f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in report["interpretation"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
