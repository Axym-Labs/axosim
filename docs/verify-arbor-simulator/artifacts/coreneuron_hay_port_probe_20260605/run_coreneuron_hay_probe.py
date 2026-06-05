#!/usr/bin/env python3
"""Compare deterministic-synapse Hay/NEURON fixed-step and CoreNEURON runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import time

import numpy as np

ARTIFACTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARTIFACTS))

from run_hay_neuron_driven_probe import collect_segments, sample_neuronio_inputs, summarize_trace  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="/tmp/neuron-as-deep-net/L5PC_NEURON_simulation")
    parser.add_argument("--mechanism-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--backend", choices=["neuron", "coreneuron"], default="neuron")
    parser.add_argument("--coreneuron-gpu", action="store_true")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--duration-ms", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--v-init-mv", type=float, default=-76.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    result = run_probe(
        source_root=Path(args.source_root).resolve(),
        mechanism_root=Path(args.mechanism_root).resolve(),
        backend=args.backend,
        coreneuron_gpu=args.coreneuron_gpu,
        samples=args.samples,
        duration_ms=args.duration_ms,
        seed=args.seed,
        v_init_mv=args.v_init_mv,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_probe(
    *,
    source_root: Path,
    mechanism_root: Path,
    backend: str,
    coreneuron_gpu: bool,
    samples: int,
    duration_ms: int,
    seed: int,
    v_init_mv: float,
) -> dict:
    from neuron import h
    import neuron
    from neuron import coreneuron

    if backend == "coreneuron":
        coreneuron.enable = True
        coreneuron.gpu = bool(coreneuron_gpu)
        coreneuron.verbose = 1
    else:
        coreneuron.enable = False
        coreneuron.gpu = False

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
    h.CVode().cache_efficient(1)

    rng = np.random.default_rng(seed)
    rows = []
    started = time.perf_counter()
    for sample_index in range(samples):
        sample_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        sample_rng = np.random.default_rng(sample_seed)
        rows.append(
            run_one(
                h=h,
                source_root=source_root,
                backend=backend,
                sample_index=sample_index,
                sample_rng=sample_rng,
                duration_ms=duration_ms,
                v_init_mv=v_init_mv,
            )
        )
        rows[-1]["sample_seed"] = sample_seed

    return {
        "backend": backend,
        "coreneuron_gpu": bool(coreneuron_gpu),
        "source_root": str(source_root),
        "mechanism_root": str(mechanism_root),
        "neuron_version": getattr(neuron, "__version__", "unknown"),
        "samples": int(samples),
        "duration_ms": int(duration_ms),
        "seed": int(seed),
        "v_init_mv": float(v_init_mv),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
        "aggregate": aggregate(rows),
        "synapse_policy": "DetAMPANMDA/DetGABA deterministic subset of ProbAMPANMDA2/ProbUDFsyn2 with Use=1, Dep=0, Fac=0, u0=0",
    }


def load_probe_mechanisms(h, mechanism_root: Path) -> None:
    for candidate in (mechanism_root / "x86_64" / ".libs" / "libnrnmech.so", mechanism_root / "x86_64" / "libnrnmech.so"):
        if candidate.exists():
            try:
                h.nrn_load_dll(str(candidate))
            except RuntimeError as error:
                if "user defined name already exists" not in str(error):
                    raise
            return
    raise FileNotFoundError(f"NEURON mechanism library not found under {mechanism_root / 'x86_64'}")


def run_one(*, h, source_root: Path, backend: str, sample_index: int, sample_rng: np.random.Generator, duration_ms: int, v_init_mv: float) -> dict:
    cell = h.L5PCtemplate(str(source_root / "morphologies" / "cell1.asc"))
    h.CVode().active(0)
    h.v_init = v_init_mv
    segments, segment_types, segment_lengths = collect_segments(cell)
    for section in list(cell.dend) + list(cell.apic):
        section.vshift_Ih = 0
    cell.soma[0].vshift_Ih = 0

    inputs_exc, inputs_inh = sample_neuronio_inputs(sample_rng, duration_ms, segment_types, segment_lengths)
    pc = h.ParallelContext()
    netcons, synapses, stim_refs = attach_deterministic_synapses(h, pc, segments, inputs_exc, inputs_inh)
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
    low_soma = np.interp(low_time, t, soma)
    high_time = np.arange(0, duration_ms, 0.125, dtype=np.float64)
    high_soma = np.interp(high_time, t, soma)
    row = summarize_trace(sample_index, inputs_exc, inputs_inh, low_soma, high_soma)
    row["recorded_points"] = int(t.size)
    row["final_time_ms"] = float(t[-1]) if t.size else None

    del netcons, synapses, stim_refs, cell
    return row


def attach_deterministic_synapses(
    h,
    pc,
    segments: list,
    inputs_exc: np.ndarray,
    inputs_inh: np.ndarray,
    *,
    exc_gmax_scale: float = 1.0,
    inh_gmax_scale: float = 1.0,
) -> tuple[list, list, list]:
    netcons = []
    synapses = []
    event_times = []
    event_gids = []
    next_gid = 1
    for index, segment in enumerate(segments):
        ex_syn = h.DetAMPANMDA(segment)
        ex_syn.tau_r_AMPA = 0.3
        ex_syn.tau_d_AMPA = 3.0
        ex_syn.tau_r_NMDA = 2.0
        ex_syn.tau_d_NMDA = 70.0
        ex_syn.gmax = 0.0004 * exc_gmax_scale
        ex_syn.e = 0
        ex_gid = next_gid
        next_gid += 1
        ex_nc = pc.gid_connect(ex_gid, ex_syn)
        ex_nc.weight[0] = 1
        ex_times = np.flatnonzero(inputs_exc[index]).astype(np.float64)
        event_times.extend(float(x) for x in ex_times)
        event_gids.extend(ex_gid for _ in range(ex_times.size))
        netcons.append(ex_nc)
        synapses.append(ex_syn)

        inh_syn = h.DetGABA(segment)
        inh_syn.tau_r = 0.2
        inh_syn.tau_d = 8.0
        inh_syn.gmax = 0.001 * inh_gmax_scale
        inh_syn.e = -80
        inh_gid = next_gid
        next_gid += 1
        inh_nc = pc.gid_connect(inh_gid, inh_syn)
        inh_nc.weight[0] = 1
        inh_times = np.flatnonzero(inputs_inh[index]).astype(np.float64)
        event_times.extend(float(x) for x in inh_times)
        event_gids.extend(inh_gid for _ in range(inh_times.size))
        netcons.append(inh_nc)
        synapses.append(inh_syn)

    order = np.argsort(np.asarray(event_times, dtype=np.float64), kind="stable") if event_times else np.asarray([], dtype=np.int64)
    sorted_times = np.asarray(event_times, dtype=np.float64)[order] if event_times else np.asarray([], dtype=np.float64)
    sorted_gids = np.asarray(event_gids, dtype=np.int32)[order] if event_gids else np.asarray([], dtype=np.int32)
    stim = h.PatternStim()
    t_vec = h.Vector(sorted_times)
    gid_vec = h.Vector(sorted_gids)
    stim.play(t_vec, gid_vec)
    stim_refs = [stim, t_vec, gid_vec]
    return netcons, synapses, stim_refs


def aggregate(rows: list[dict]) -> dict:
    keys = [key for key in rows[0] if key not in {"sample_index", "sample_seed", "recorded_points", "final_time_ms"}] if rows else []
    out = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        out[key] = {"mean": float(values.mean()), "std": float(values.std())}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
