#!/usr/bin/env python3
"""Generate NeuronIO-style shards with the CoreNEURON-compatible Hay path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ARTIFACTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARTIFACTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_hay_neuron_dataset import (  # noqa: E402
    SOMA_BIAS_MV,
    SOMA_CLIP_THRESHOLD_MV,
    SOMA_SCALE,
    SPIKE_PEAK_THRESHOLD_MV,
    aggregate,
    local_peak_spike_labels,
    summarize_written_trace,
)
from run_coreneuron_hay_probe import attach_deterministic_synapses, load_probe_mechanisms  # noqa: E402
from run_hay_neuron_driven_probe import collect_segments, sample_neuronio_inputs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="/tmp/neuron-as-deep-net/L5PC_NEURON_simulation")
    parser.add_argument("--mechanism-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--backend", choices=["neuron", "coreneuron"], default="coreneuron")
    parser.add_argument("--coreneuron-gpu", action="store_true")
    parser.add_argument("--train-samples", type=int, default=4)
    parser.add_argument("--val-samples", type=int, default=4)
    parser.add_argument("--duration-ms", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--v-init-mv", type=float, default=-76.0)
    parser.add_argument("--exc-event-keep-prob", type=float, default=1.0)
    parser.add_argument("--inh-event-keep-prob", type=float, default=1.0)
    parser.add_argument("--exc-gmax-scale", type=float, default=1.0)
    parser.add_argument("--inh-gmax-scale", type=float, default=1.0)
    args = parser.parse_args()

    validate_keep_prob(args.exc_event_keep_prob, "exc-event-keep-prob")
    validate_keep_prob(args.inh_event_keep_prob, "inh-event-keep-prob")
    validate_positive(args.exc_gmax_scale, "exc-gmax-scale")
    validate_positive(args.inh_gmax_scale, "inh-gmax-scale")
    output_root = Path(args.output_root).resolve()
    summary = write_splits(
        output_root=output_root,
        source_root=Path(args.source_root).resolve(),
        mechanism_root=Path(args.mechanism_root).resolve(),
        backend=args.backend,
        coreneuron_gpu=args.coreneuron_gpu,
        split_samples={"train": args.train_samples, "val": args.val_samples},
        duration_ms=args.duration_ms,
        seed=args.seed,
        v_init_mv=args.v_init_mv,
        exc_event_keep_prob=args.exc_event_keep_prob,
        inh_event_keep_prob=args.inh_event_keep_prob,
        exc_gmax_scale=args.exc_gmax_scale,
        inh_gmax_scale=args.inh_gmax_scale,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def write_splits(
    *,
    output_root: Path,
    source_root: Path,
    mechanism_root: Path,
    backend: str,
    coreneuron_gpu: bool,
    split_samples: dict[str, int],
    duration_ms: int,
    seed: int,
    v_init_mv: float,
    exc_event_keep_prob: float,
    inh_event_keep_prob: float,
    exc_gmax_scale: float,
    inh_gmax_scale: float,
) -> dict[str, Any]:
    from neuron import coreneuron, h
    import neuron

    if backend == "coreneuron":
        coreneuron.enable = True
        coreneuron.gpu = bool(coreneuron_gpu)
        coreneuron.verbose = 0
    else:
        coreneuron.enable = False
        coreneuron.gpu = False

    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)
    os.chdir(mechanism_root)
    load_probe_mechanisms(h, mechanism_root)
    hoc_lib = Path(os.environ.get("NEURONHOME", "")) / "share" / "nrn" / "lib" / "hoc"
    if hoc_lib.exists():
        h.load_file(str(hoc_lib / "stdrun.hoc"))
        h.load_file(str(hoc_lib / "import3d.hoc"))
    else:
        h.load_file("stdrun.hoc")
        h.load_file("import3d.hoc")
    h.load_file(str(source_root / "L5PCbiophys5b.hoc"))
    h.load_file(str(source_root / "L5PCtemplate_2.hoc"))
    h.dt = 0.025
    h.steps_per_ms = 40
    h.secondorder = 0
    h.CVode().active(0)
    h.CVode().cache_efficient(1)

    split_reports: dict[str, Any] = {}
    for split_index, (split_name, samples) in enumerate(split_samples.items()):
        if samples <= 0:
            continue
        split_reports[split_name] = write_split(
            output_dir=output_root / split_name,
            h=h,
            neuron_version=getattr(neuron, "__version__", "unknown"),
            source_root=source_root,
            mechanism_root=mechanism_root,
            backend=backend,
            coreneuron_gpu=coreneuron_gpu,
            samples=samples,
            duration_ms=duration_ms,
            seed=seed + 1009 * split_index,
            v_init_mv=v_init_mv,
            exc_event_keep_prob=exc_event_keep_prob,
            inh_event_keep_prob=inh_event_keep_prob,
            exc_gmax_scale=exc_gmax_scale,
            inh_gmax_scale=inh_gmax_scale,
        )

    summary = {
        "backend": backend,
        "coreneuron_gpu": bool(coreneuron_gpu),
        "dataset_kind": "hay-l5pc-source-coreneuron-compatible",
        "output_root": str(output_root),
        "source_root": str(source_root),
        "mechanism_root": str(mechanism_root),
        "duration_ms": int(duration_ms),
        "seed": int(seed),
        "v_init_mv": float(v_init_mv),
        "splits": split_reports,
        "elapsed_seconds": time.perf_counter() - started,
        "target_policy": {
            "channel_layout": "first 639 excitatory channels are +1, second 639 inhibitory channels are -1",
            "spike_labels": f"high-resolution local soma peaks above {SPIKE_PEAK_THRESHOLD_MV} mV, binned to 1 ms",
            "soma": f"low-resolution soma clipped at {SOMA_CLIP_THRESHOLD_MV} mV then normalized as (v - {SOMA_BIAS_MV}) * {SOMA_SCALE}",
            "synapses": "DetAMPANMDA/DetGABA deterministic subsets matching the source-teacher Use=1, Dep=0, Fac=0, u0=0 policy",
            "event_keep_prob": {
                "excitatory": float(exc_event_keep_prob),
                "inhibitory": float(inh_event_keep_prob),
            },
            "gmax_scale": {
                "excitatory": float(exc_gmax_scale),
                "inhibitory": float(inh_gmax_scale),
            },
        },
    }
    (output_root / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def write_split(
    *,
    output_dir: Path,
    h,
    neuron_version: str,
    source_root: Path,
    mechanism_root: Path,
    backend: str,
    coreneuron_gpu: bool,
    samples: int,
    duration_ms: int,
    seed: int,
    v_init_mv: float,
    exc_event_keep_prob: float,
    inh_event_keep_prob: float,
    exc_gmax_scale: float,
    inh_gmax_scale: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = np.zeros((samples, duration_ms, 1278), dtype=np.int8)
    targets = np.zeros((samples, duration_ms, 2), dtype=np.float32)
    soma_mv = np.zeros((samples, duration_ms), dtype=np.float32)
    sample_ids = np.empty(samples, dtype="<U112")
    rows: list[dict[str, Any]] = []

    for sample_index in range(samples):
        sample_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        sample_rng = np.random.default_rng(sample_seed)
        trace = simulate_one_trace(
            h=h,
            source_root=source_root,
            backend=backend,
            sample_rng=sample_rng,
            duration_ms=duration_ms,
            v_init_mv=v_init_mv,
            exc_event_keep_prob=exc_event_keep_prob,
            inh_event_keep_prob=inh_event_keep_prob,
            exc_gmax_scale=exc_gmax_scale,
            inh_gmax_scale=inh_gmax_scale,
        )
        inputs[sample_index] = trace["inputs"]
        targets[sample_index] = trace["targets"]
        soma_mv[sample_index] = trace["soma_mv"]
        sample_ids[sample_index] = f"hay_l5pc_{backend}:det_syn:{sample_index:06d}:seed{sample_seed}"
        row = summarize_written_trace(sample_index, trace["inputs"], trace["targets"], trace["soma_mv"])
        row["sample_seed"] = sample_seed
        rows.append(row)

    shard_path = output_dir / "hay_coreneuron_source_00000.npz"
    np.savez_compressed(
        shard_path,
        inputs=inputs,
        targets=targets,
        sample_ids=sample_ids,
        soma_mv=soma_mv,
    )
    manifest = {
        "version": 1,
        "dataset_kind": "hay-l5pc-source-coreneuron-compatible",
        "simulator": {
            "backend": backend,
            "coreneuron_gpu": bool(coreneuron_gpu),
            "neuron_version": neuron_version,
            "source_root": str(source_root),
            "mechanism_root": str(mechanism_root),
            "morphology": "morphologies/cell1.asc",
            "biophysics": "L5PCbiophys5b.hoc",
            "template": "L5PCtemplate_2.hoc",
            "fixed_step_dt_ms": 0.025,
            "v_init_mv": float(v_init_mv),
            "event_delivery": "PatternStim + ParallelContext.gid_connect",
            "event_keep_prob": {
                "excitatory": float(exc_event_keep_prob),
                "inhibitory": float(inh_event_keep_prob),
            },
            "gmax_scale": {
                "excitatory": float(exc_gmax_scale),
                "inhibitory": float(inh_gmax_scale),
            },
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


def simulate_one_trace(
    *,
    h,
    source_root: Path,
    backend: str,
    sample_rng: np.random.Generator,
    duration_ms: int,
    v_init_mv: float,
    exc_event_keep_prob: float,
    inh_event_keep_prob: float,
    exc_gmax_scale: float,
    inh_gmax_scale: float,
) -> dict[str, np.ndarray]:
    cell = h.L5PCtemplate(str(source_root / "morphologies" / "cell1.asc"))
    h.CVode().active(0)
    h.v_init = v_init_mv
    segments, segment_types, segment_lengths = collect_segments(cell)
    for section in list(cell.dend) + list(cell.apic):
        section.vshift_Ih = 0
    cell.soma[0].vshift_Ih = 0

    inputs_exc, inputs_inh = sample_neuronio_inputs(sample_rng, duration_ms, segment_types, segment_lengths)
    inputs_exc = apply_event_keep(sample_rng, inputs_exc, exc_event_keep_prob)
    inputs_inh = apply_event_keep(sample_rng, inputs_inh, inh_event_keep_prob)
    pc = h.ParallelContext()
    netcons, synapses, stim_refs = attach_deterministic_synapses(
        h,
        pc,
        segments,
        inputs_exc,
        inputs_inh,
        exc_gmax_scale=exc_gmax_scale,
        inh_gmax_scale=inh_gmax_scale,
    )
    t_vec = h.Vector()
    soma_vec = h.Vector()
    t_vec.record(h._ref_t)
    soma_vec.record(cell.soma[0](0.5)._ref_v)

    h.finitialize(v_init_mv)
    if backend == "coreneuron":
        pc.psolve(duration_ms)
    else:
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

    del netcons, synapses, stim_refs, cell
    return {
        "inputs": signed_inputs,
        "targets": targets,
        "soma_mv": low_soma,
    }


def validate_keep_prob(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def validate_positive(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


def apply_event_keep(rng: np.random.Generator, events: np.ndarray, keep_prob: float) -> np.ndarray:
    if keep_prob >= 1.0:
        return events
    if keep_prob <= 0.0:
        return np.zeros_like(events, dtype=bool)
    return events & (rng.random(events.shape) < keep_prob)


if __name__ == "__main__":
    raise SystemExit(main())
