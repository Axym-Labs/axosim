#!/usr/bin/env python3
"""Generate NeuronIO-style shards from the original Hay/NEURON source teacher."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from run_hay_neuron_driven_probe import (
    attach_synapses,
    collect_segments,
    load_mechanisms,
    sample_neuronio_inputs,
    schedule_events,
)


SOMA_BIAS_MV = -67.7
SOMA_SCALE = 0.1
SOMA_CLIP_THRESHOLD_MV = -55.0
SPIKE_PEAK_THRESHOLD_MV = -25.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="/tmp/neuron-as-deep-net/L5PC_NEURON_simulation")
    parser.add_argument("--output-root", default="data/hay-neuron-source-sanity-20260605")
    parser.add_argument("--train-samples", type=int, default=4)
    parser.add_argument("--val-samples", type=int, default=2)
    parser.add_argument("--duration-ms", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--v-init-mv", type=float, default=-76.0)
    parser.add_argument("--use-cvode", action="store_true")
    args = parser.parse_args()

    summary = write_hay_splits(
        output_root=Path(args.output_root).resolve(),
        source_root=Path(args.source_root).resolve(),
        split_samples={"train": args.train_samples, "val": args.val_samples},
        duration_ms=args.duration_ms,
        seed=args.seed,
        v_init_mv=args.v_init_mv,
        use_cvode=args.use_cvode,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def write_hay_splits(
    *,
    output_root: Path,
    source_root: Path,
    split_samples: dict[str, int],
    duration_ms: int,
    seed: int,
    v_init_mv: float,
    use_cvode: bool,
) -> dict[str, Any]:
    from neuron import h
    import neuron

    start = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)
    os.chdir(source_root)
    load_mechanisms(h, source_root)
    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    h.load_file("L5PCbiophys5b.hoc")
    h.load_file("L5PCtemplate_2.hoc")

    split_reports: dict[str, Any] = {}
    for split_index, (split_name, samples) in enumerate(split_samples.items()):
        if samples <= 0:
            continue
        split_reports[split_name] = write_hay_dataset(
            output_dir=output_root / split_name,
            h=h,
            neuron_version=getattr(neuron, "__version__", "unknown"),
            source_root=source_root,
            samples=samples,
            duration_ms=duration_ms,
            seed=seed + 1009 * split_index,
            v_init_mv=v_init_mv,
            use_cvode=use_cvode,
        )

    summary = {
        "backend": "neuron",
        "dataset_kind": "hay-l5pc-source-neuronio-compatible",
        "output_root": str(output_root),
        "source_root": str(source_root),
        "duration_ms": int(duration_ms),
        "seed": int(seed),
        "v_init_mv": float(v_init_mv),
        "use_cvode": bool(use_cvode),
        "splits": split_reports,
        "elapsed_seconds": time.perf_counter() - start,
        "target_policy": {
            "channel_layout": "first 639 excitatory channels are +1, second 639 inhibitory channels are -1",
            "spike_labels": f"high-resolution local soma peaks above {SPIKE_PEAK_THRESHOLD_MV} mV, binned to 1 ms",
            "soma": f"low-resolution soma clipped at {SOMA_CLIP_THRESHOLD_MV} mV then normalized as (v - {SOMA_BIAS_MV}) * {SOMA_SCALE}",
        },
    }
    (output_root / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def write_hay_dataset(
    *,
    output_dir: Path,
    h,
    neuron_version: str,
    source_root: Path,
    samples: int,
    duration_ms: int,
    seed: int,
    v_init_mv: float,
    use_cvode: bool,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = np.zeros((samples, duration_ms, 1278), dtype=np.int8)
    targets = np.zeros((samples, duration_ms, 2), dtype=np.float32)
    soma_mv = np.zeros((samples, duration_ms), dtype=np.float32)
    sample_ids = np.empty(samples, dtype="<U96")
    rows: list[dict[str, Any]] = []

    for sample_index in range(samples):
        sample_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        sample_rng = np.random.default_rng(sample_seed)
        trace = simulate_one_trace(
            h=h,
            sample_rng=sample_rng,
            duration_ms=duration_ms,
            v_init_mv=v_init_mv,
            use_cvode=use_cvode,
        )
        inputs[sample_index] = trace["inputs"]
        targets[sample_index] = trace["targets"]
        soma_mv[sample_index] = trace["soma_mv"]
        sample_ids[sample_index] = f"hay_l5pc_source:{sample_index:06d}:seed{sample_seed}"
        row = summarize_written_trace(sample_index, trace["inputs"], trace["targets"], trace["soma_mv"])
        row["sample_seed"] = sample_seed
        rows.append(row)

    shard_path = output_dir / "hay_neuron_source_00000.npz"
    np.savez_compressed(
        shard_path,
        inputs=inputs,
        targets=targets,
        sample_ids=sample_ids,
        soma_mv=soma_mv,
    )
    manifest = {
        "version": 1,
        "dataset_kind": "hay-l5pc-source-neuronio-compatible",
        "simulator": {
            "backend": "neuron",
            "neuron_version": neuron_version,
            "source_root": str(source_root),
            "morphology": "morphologies/cell1.asc",
            "biophysics": "L5PCbiophys5b.hoc",
            "template": "L5PCtemplate_2.hoc",
            "mechanism_library": str(source_root / "x86_64" / ".libs" / "libnrnmech.so"),
            "use_cvode": bool(use_cvode),
            "v_init_mv": float(v_init_mv),
        },
        "shards": [
            {
                "file": shard_path.name,
                "count": int(samples),
                "sample_ids": sample_ids.tolist(),
            }
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = {
        "samples": int(samples),
        "duration_ms": int(duration_ms),
        "seed": int(seed),
        "shard": str(shard_path),
        "rows": rows,
        "aggregate": aggregate(rows),
    }
    (output_dir / "generation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def simulate_one_trace(*, h, sample_rng: np.random.Generator, duration_ms: int, v_init_mv: float, use_cvode: bool) -> dict[str, np.ndarray]:
    cell = h.L5PCtemplate("morphologies/cell1.asc")
    cvode = h.CVode()
    cvode.active(1 if use_cvode else 0)
    h.v_init = v_init_mv
    segments, segment_types, segment_lengths = collect_segments(cell)
    for section in list(cell.dend) + list(cell.apic):
        section.vshift_Ih = 0
    cell.soma[0].vshift_Ih = 0

    inputs_exc, inputs_inh = sample_neuronio_inputs(sample_rng, duration_ms, segment_types, segment_lengths)
    netcons, synapses = attach_synapses(h, segments, inputs_exc, inputs_inh)
    t_vec = h.Vector()
    soma_vec = h.Vector()
    t_vec.record(h._ref_t)
    soma_vec.record(cell.soma[0](0.5)._ref_v)

    h.finitialize(v_init_mv)
    schedule_events(netcons)
    h.continuerun(duration_ms)
    t = np.asarray(t_vec, dtype=np.float64)
    soma = np.asarray(soma_vec, dtype=np.float64)
    low_time = np.arange(0, duration_ms, dtype=np.float64)
    high_time = np.arange(0, duration_ms, 0.125, dtype=np.float64)
    low_soma = np.interp(low_time, t, soma).astype(np.float32)
    high_soma = np.interp(high_time, t, soma)

    signed_inputs = np.zeros((duration_ms, 1278), dtype=np.int8)
    signed_inputs[:, :639] = inputs_exc.T.astype(np.int8, copy=False)
    signed_inputs[:, 639:] = -inputs_inh.T.astype(np.int8, copy=False)
    spike_labels = local_peak_spike_labels(high_time, high_soma, duration_ms)
    clipped_soma = np.minimum(low_soma, SOMA_CLIP_THRESHOLD_MV)
    normalized_soma = ((clipped_soma - SOMA_BIAS_MV) * SOMA_SCALE).astype(np.float32)
    targets = np.stack([spike_labels, normalized_soma], axis=1).astype(np.float32)

    del netcons, synapses, cell
    return {
        "inputs": signed_inputs,
        "targets": targets,
        "soma_mv": low_soma,
    }


def local_peak_spike_labels(high_time: np.ndarray, high_soma: np.ndarray, duration_ms: int) -> np.ndarray:
    rising_before = np.hstack(([False], high_soma[1:] - high_soma[:-1] > 0))
    falling_after = np.hstack((high_soma[1:] - high_soma[:-1] < 0, [False]))
    peaks = rising_before & falling_after & (high_soma > SPIKE_PEAK_THRESHOLD_MV)
    spike_bins = np.floor(high_time[peaks]).astype(np.int64)
    spike_bins = spike_bins[(spike_bins >= 0) & (spike_bins < duration_ms)]
    labels = np.zeros(duration_ms, dtype=np.float32)
    labels[np.unique(spike_bins)] = 1.0
    return labels


def summarize_written_trace(sample_index: int, inputs: np.ndarray, targets: np.ndarray, soma_mv: np.ndarray) -> dict[str, Any]:
    clipped = np.minimum(soma_mv.astype(np.float64), SOMA_CLIP_THRESHOLD_MV)
    dvdt = np.diff(clipped, prepend=clipped[:1])
    return {
        "sample_index": int(sample_index),
        "event_fraction": float(np.count_nonzero(inputs) / inputs.size),
        "exc_fraction": float((inputs > 0).sum() / (inputs.shape[0] * 639)),
        "inh_fraction": float((inputs < 0).sum() / (inputs.shape[0] * 639)),
        "spike_count": int(targets[:, 0].sum()),
        "spike_rate_hz": float(1000.0 * targets[:, 0].sum() / inputs.shape[0]),
        "soma_mean_mv": float(soma_mv.mean()),
        "soma_std_mv": float(soma_mv.std()),
        "soma_min_mv": float(soma_mv.min()),
        "soma_max_mv": float(soma_mv.max()),
        "clipped_soma_mean_mv": float(clipped.mean()),
        "clipped_soma_std_mv": float(clipped.std()),
        "clipped_abs_dvdt_p95": float(np.quantile(np.abs(dvdt), 0.95)),
        "clipped_abs_dvdt_p99": float(np.quantile(np.abs(dvdt), 0.99)),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = [key for key in rows[0] if key not in {"sample_index", "sample_seed"}] if rows else []
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        out[key] = {"mean": float(values.mean()), "std": float(values.std())}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
