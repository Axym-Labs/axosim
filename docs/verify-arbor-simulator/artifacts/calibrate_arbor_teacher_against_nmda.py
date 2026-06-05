#!/usr/bin/env python3
"""Run small Arbor teacher calibration sweeps against NMDA/NeuronIO stats.

This is a teacher-fidelity diagnostic, not a surrogate benchmark. The goal is
to find whether any lightweight Arbor protocol sits in the same basic regime as
the historical NMDA dataset before spending training compute on surrogates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from bnn_sim.arbor_sim import ArborHeteroConfig, write_arbor_hetero_dataset
from bnn_sim.data import ShardedNeuronIODataset


TARGET_NMDA_VAL10 = {
    "event_fraction": 0.01061581297681925,
    "spike_rate_hz": 0.8815104166666666,
    "soma_mean_mv": -67.61582763850524,
    "soma_std_mv": 3.1696798142366305,
    "clipped_soma_max_mv": -55.0,
    "abs_dvdt_p95_mv_per_step": 0.4375,
    "abs_dvdt_p99_mv_per_step": 0.6875,
}


NOMINAL_FIXED_PARAMS = {
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
    "soma_e_pas_shift_mv": 0.0,
    "dend_e_pas_shift_mv": 0.0,
    "apic_e_pas_shift_mv": 0.0,
    "axon_e_pas_shift_mv": 0.0,
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
    "ampa_weight_scale": 0.0012,
    "nmda_weight_scale": 0.00015,
    "inh_weight_scale": 0.006,
    "ei_balance": 0.0,
    "input_rate_scale": 1.0,
    "burstiness_scale": 1.0,
    "synchrony_scale": 1.0,
    "apic_synapse_fraction": 0.55,
    "spike_threshold_mv": -32.0,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/verify-arbor-simulator/artifacts/arbor_teacher_calibration_20260604")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--time-steps", type=int, default=3000)
    parser.add_argument("--input-dim", type=int, default=1278)
    parser.add_argument("--branches", type=int, default=45)
    parser.add_argument("--integration-dt-ms", type=float, default=0.05)
    parser.add_argument("--input-protocol", choices=["arbor_poisson", "neuronio_smoothed"], default="arbor_poisson")
    parser.add_argument("--synapse-routing", choices=["branch", "distributed"], default="branch")
    parser.add_argument("--distributed-sites-per-polarity", type=int)
    parser.add_argument("--spike-label-mode", choices=["threshold_crossing", "local_peak"], default="threshold_crossing")
    parser.add_argument("--spike-peak-threshold-mv", type=float, default=-25.0)
    parser.add_argument("--sim-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--arbor-gpu-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--base-event-rates", default="0.05,0.1,0.25,0.5,1.0,2.0")
    parser.add_argument("--burst-probabilities", default="0.0,0.002,0.005,0.015")
    parser.add_argument("--weight-scales", default="0.02,0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--inhibition-scales", default="1.0,2.0,4.0")
    parser.add_argument("--e-pas-shifts", default="0.0", help="comma-separated passive reversal shifts in mV applied to all regions")
    parser.add_argument("--v-init-mvs", default="-76.0", help="comma-separated initial voltages in mV")
    parser.add_argument("--max-runs", type=int, default=0, help="optional cap after grid construction; 0 means all")
    parser.add_argument("--clean", action="store_true", help="remove output directory before running")
    args = parser.parse_args()

    output = Path(args.output)
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    base_event_rates = _parse_float_list(args.base_event_rates)
    burst_probabilities = _parse_float_list(args.burst_probabilities)
    weight_scales = _parse_float_list(args.weight_scales)
    inhibition_scales = _parse_float_list(args.inhibition_scales)
    e_pas_shifts = _parse_float_list(args.e_pas_shifts)
    v_init_mvs = _parse_float_list(args.v_init_mvs)
    grid = list(itertools.product(base_event_rates, burst_probabilities, weight_scales, inhibition_scales, e_pas_shifts, v_init_mvs))
    if args.max_runs:
        grid = grid[: args.max_runs]

    results: list[dict[str, Any]] = []
    for index, (base_rate, burst_probability, weight_scale, inhibition_scale, e_pas_shift, v_init_mv) in enumerate(grid):
        run_dir = output / f"run_{index:04d}"
        overrides = dict(NOMINAL_FIXED_PARAMS)
        overrides["v_init_mv"] = v_init_mv
        for region in ("soma", "dend", "apic", "axon"):
            overrides[f"{region}_e_pas_shift_mv"] = e_pas_shift
        overrides["ampa_weight_scale"] = NOMINAL_FIXED_PARAMS["ampa_weight_scale"] * weight_scale
        overrides["nmda_weight_scale"] = NOMINAL_FIXED_PARAMS["nmda_weight_scale"] * weight_scale
        overrides["inh_weight_scale"] = NOMINAL_FIXED_PARAMS["inh_weight_scale"] * inhibition_scale
        config = ArborHeteroConfig(
            input_dim=args.input_dim,
            num_branch=args.branches,
            time_steps=args.time_steps,
            integration_dt_ms=args.integration_dt_ms,
            input_protocol=args.input_protocol,
            synapse_routing=args.synapse_routing,
            distributed_sites_per_polarity=args.distributed_sites_per_polarity,
            spike_label_mode=args.spike_label_mode,
            spike_peak_threshold_mv=args.spike_peak_threshold_mv,
            base_event_rate_hz=base_rate,
            burst_probability=burst_probability,
            burst_event_rate_hz=35.0,
            arbor_gpu_id=None if args.cpu_only else args.arbor_gpu_id,
            arbor_threads=8 if args.cpu_only else 1,
            sim_batch_size=args.sim_batch_size,
        )
        report = write_arbor_hetero_dataset(
            run_dir,
            samples=args.samples,
            shards=1,
            seed=args.seed + index,
            config=config,
            param_overrides=overrides,
        )
        stats = _dataset_stats(run_dir)
        row = {
            "index": index,
            "output": str(run_dir),
            "sweep": {
                "base_event_rate_hz": base_rate,
                "burst_probability": burst_probability,
                "weight_scale": weight_scale,
                "inhibition_scale": inhibition_scale,
                "e_pas_shift_mv": e_pas_shift,
                "v_init_mv": v_init_mv,
            },
            "config": asdict(config),
            "param_overrides": overrides,
            "generation_report": report,
            "stats": stats,
            "objective": _objective(stats),
        }
        results.append(row)
        _write_outputs(output, results, args)


def _parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def _dataset_stats(path: Path) -> dict[str, float | int]:
    dataset = ShardedNeuronIODataset(path, cache_shards=1)
    total_events = 0
    total_spikes = 0.0
    soma_values: list[np.ndarray] = []
    total_input = 0
    total_time = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        total_events += int(np.count_nonzero(sample.inputs))
        total_input += int(sample.inputs.size)
        total_spikes += float(sample.targets[:, 0].sum())
        soma_mv = sample.targets[:, 1].astype(np.float64) / 0.1 + (-67.7)
        soma_values.append(soma_mv)
        total_time += int(sample.targets.shape[0])
    soma = np.concatenate(soma_values)
    clipped = np.minimum(soma, TARGET_NMDA_VAL10["clipped_soma_max_mv"])
    dvdt = np.diff(clipped.reshape(len(dataset), -1), axis=1, prepend=clipped.reshape(len(dataset), -1)[:, :1])
    return {
        "samples": len(dataset),
        "time_steps": int(total_time / max(len(dataset), 1)),
        "event_fraction": total_events / max(total_input, 1),
        "spike_rate_hz": 1000.0 * total_spikes / max(total_time, 1),
        "soma_mean_mv": float(soma.mean()),
        "soma_std_mv": float(soma.std()),
        "soma_min_mv": float(soma.min()),
        "soma_max_mv": float(soma.max()),
        "clipped_soma_mean_mv": float(clipped.mean()),
        "clipped_soma_std_mv": float(clipped.std()),
        "clipped_soma_min_mv": float(clipped.min()),
        "clipped_soma_max_mv": float(clipped.max()),
        "clipped_abs_dvdt_p95_mv_per_step": float(np.quantile(np.abs(dvdt), 0.95)),
        "clipped_abs_dvdt_p99_mv_per_step": float(np.quantile(np.abs(dvdt), 0.99)),
    }


def _objective(stats: dict[str, float | int]) -> float:
    spike = abs(np.log10((float(stats["spike_rate_hz"]) + 0.05) / (TARGET_NMDA_VAL10["spike_rate_hz"] + 0.05)))
    event = abs(np.log10((float(stats["event_fraction"]) + 1e-6) / (TARGET_NMDA_VAL10["event_fraction"] + 1e-6)))
    mean = abs(float(stats["clipped_soma_mean_mv"]) - TARGET_NMDA_VAL10["soma_mean_mv"]) / 10.0
    std = abs(float(stats["clipped_soma_std_mv"]) - TARGET_NMDA_VAL10["soma_std_mv"]) / 5.0
    dvdt95 = abs(
        np.log10(
            (float(stats["clipped_abs_dvdt_p95_mv_per_step"]) + 1e-3)
            / (TARGET_NMDA_VAL10["abs_dvdt_p95_mv_per_step"] + 1e-3)
        )
    )
    dvdt99 = abs(
        np.log10(
            (float(stats["clipped_abs_dvdt_p99_mv_per_step"]) + 1e-3)
            / (TARGET_NMDA_VAL10["abs_dvdt_p99_mv_per_step"] + 1e-3)
        )
    )
    return float(2.0 * spike + event + mean + std + 0.25 * dvdt95 + 0.25 * dvdt99)


def _write_outputs(output: Path, results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    ranked = sorted(results, key=lambda row: row["objective"])
    payload = {
        "target": TARGET_NMDA_VAL10,
        "args": vars(args),
        "results": ranked,
    }
    (output / "calibration_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Arbor Teacher Calibration Against NMDA",
        "",
        f"Target: `{json.dumps(TARGET_NMDA_VAL10, sort_keys=True)}`",
        "",
        "| rank | objective | run | base Hz | weight | inhibition | e_pas shift | v_init | event fraction | spike rate Hz | clipped mean mV | clipped std mV | clipped |dV/dt| p95/p99 |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked[:25], start=1):
        stats = row["stats"]
        sweep = row.get("sweep", {})
        lines.append(
            "| {rank} | {objective:.4f} | `{run}` | {base:.3g} | {weight:.3g} | {inh:.3g} | {epas:.3g} | {vinit:.3g} | {event:.6f} | {spike:.4f} | {mean:.4f} | {std:.4f} | {dvdt95:.4f}/{dvdt99:.4f} |".format(
                rank=rank,
                objective=float(row["objective"]),
                run=Path(row["output"]).name,
                base=float(sweep.get("base_event_rate_hz", row["config"]["base_event_rate_hz"])),
                weight=float(sweep.get("weight_scale", float("nan"))),
                inh=float(sweep.get("inhibition_scale", float("nan"))),
                epas=float(sweep.get("e_pas_shift_mv", float("nan"))),
                vinit=float(sweep.get("v_init_mv", float("nan"))),
                event=float(stats["event_fraction"]),
                spike=float(stats["spike_rate_hz"]),
                mean=float(stats["clipped_soma_mean_mv"]),
                std=float(stats["clipped_soma_std_mv"]),
                dvdt95=float(stats["clipped_abs_dvdt_p95_mv_per_step"]),
                dvdt99=float(stats["clipped_abs_dvdt_p99_mv_per_step"]),
            )
        )
    (output / "calibration_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
