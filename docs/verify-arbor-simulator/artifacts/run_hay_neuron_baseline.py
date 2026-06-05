#!/usr/bin/env python3
"""Run small direct NEURON/Hay baseline diagnostics.

This uses the original Beniaguev/Hay morphology and HOC files from the local
`neuron_as_deep_net` clone. It is meant to measure source-teacher behavior
directly instead of inferring it from Arbor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="/tmp/neuron-as-deep-net/L5PC_NEURON_simulation")
    parser.add_argument("--duration-ms", type=float, default=3000.0)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--v-init-mv", type=float, default=-76.0)
    parser.add_argument("--use-cvode", action="store_true")
    parser.add_argument("--output", default="docs/verify-arbor-simulator/artifacts/hay_neuron_baseline_20260605.json")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_baseline(source_root, args.duration_ms, args.dt_ms, args.v_init_mv, args.use_cvode)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_baseline(source_root: Path, duration_ms: float, dt_ms: float, v_init_mv: float, use_cvode: bool) -> dict:
    from neuron import h
    import neuron

    os.chdir(source_root)
    dll_candidates = [
        source_root / "x86_64" / ".libs" / "libnrnmech.so",
        source_root / "x86_64" / "libnrnmech.so",
    ]
    dll_loaded = None
    for candidate in dll_candidates:
        if candidate.exists():
            h.nrn_load_dll(str(candidate))
            dll_loaded = str(candidate)
            break
    if dll_loaded is None:
        raise FileNotFoundError(f"compiled NEURON mechanisms not found under {source_root / 'x86_64'}")

    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    h.load_file("L5PCbiophys5b.hoc")
    h.load_file("L5PCtemplate_2.hoc")
    cell = h.L5PCtemplate("morphologies/cell1.asc")

    cvode = h.CVode()
    cvode.active(1 if use_cvode else 0)
    h.dt = dt_ms
    h.tstop = duration_ms
    h.v_init = v_init_mv

    t_vec = h.Vector()
    soma_vec = h.Vector()
    t_vec.record(h._ref_t)
    soma_vec.record(cell.soma[0](0.5)._ref_v)

    h.finitialize(v_init_mv)
    h.continuerun(duration_ms)
    t = np.asarray(t_vec, dtype=np.float64)
    soma = np.asarray(soma_vec, dtype=np.float64)
    if soma.size == 0:
        raise ValueError("no soma samples recorded")
    last_mask = t >= max(0.0, duration_ms - 500.0)
    peaks = np.zeros(soma.shape, dtype=bool)
    if soma.size >= 3:
        peaks[1:-1] = (soma[1:-1] > soma[:-2]) & (soma[1:-1] >= soma[2:]) & (soma[1:-1] > -25.0)
    threshold_cross = np.zeros(soma.shape, dtype=bool)
    threshold_cross[1:] = (soma[1:] >= -32.0) & (soma[:-1] < -32.0)
    return {
        "source_root": str(source_root),
        "dll_loaded": dll_loaded,
        "neuron_version": getattr(neuron, "__version__", "unknown"),
        "duration_ms": float(duration_ms),
        "dt_ms": float(dt_ms),
        "use_cvode": bool(use_cvode),
        "v_init_mv": float(v_init_mv),
        "samples": int(soma.size),
        "first_mv": float(soma[0]),
        "final_mv": float(soma[-1]),
        "mean_all_mv": float(soma.mean()),
        "mean_last_500_mv": float(soma[last_mask].mean()),
        "std_last_500_mv": float(soma[last_mask].std()),
        "min_mv": float(soma.min()),
        "max_mv": float(soma.max()),
        "local_peak_spikes_above_minus25": int(peaks.sum()),
        "threshold_crossings_above_minus32": int(threshold_cross.sum()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
