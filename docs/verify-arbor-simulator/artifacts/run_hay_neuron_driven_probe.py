#!/usr/bin/env python3
"""Run tiny driven NEURON/Hay probes with the original NeuronIO protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

import numpy as np


RATE_INTERVAL_OPTIONS_MS = (25, 30, 35, 40, 45, 55, 60, 65, 70, 75, 80, 85, 90, 100, 150, 200, 300, 450)
SMOOTHING_SIGMA_OPTIONS_MS = (25, 30, 35, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500, 600)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="/tmp/neuron-as-deep-net/L5PC_NEURON_simulation")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--duration-ms", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--v-init-mv", type=float, default=-76.0)
    parser.add_argument("--use-cvode", action="store_true")
    parser.add_argument("--output", default="docs/verify-arbor-simulator/artifacts/hay_neuron_driven_probe_20260605.json")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_probe(
        source_root=Path(args.source_root).resolve(),
        samples=args.samples,
        duration_ms=args.duration_ms,
        seed=args.seed,
        v_init_mv=args.v_init_mv,
        use_cvode=args.use_cvode,
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_probe(source_root: Path, samples: int, duration_ms: int, seed: int, v_init_mv: float, use_cvode: bool) -> dict:
    from neuron import h
    import neuron

    rng = np.random.default_rng(seed)
    os.chdir(source_root)
    load_mechanisms(h, source_root)
    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    h.load_file("L5PCbiophys5b.hoc")
    h.load_file("L5PCtemplate_2.hoc")

    rows = []
    for sample_index in range(samples):
        cell = h.L5PCtemplate("morphologies/cell1.asc")
        cvode = h.CVode()
        cvode.active(1 if use_cvode else 0)
        h.v_init = v_init_mv
        segments, segment_types, segment_lengths = collect_segments(cell)
        for section in list(cell.dend) + list(cell.apic):
            section.vshift_Ih = 0
        cell.soma[0].vshift_Ih = 0

        inputs_exc, inputs_inh = sample_neuronio_inputs(rng, duration_ms, segment_types, segment_lengths)
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
        low_soma = np.interp(low_time, t, soma)
        high_time = np.arange(0, duration_ms, 0.125, dtype=np.float64)
        high_soma = np.interp(high_time, t, soma)
        rows.append(summarize_trace(sample_index, inputs_exc, inputs_inh, low_soma, high_soma))

        # Keep references until after simulation, then release before next cell.
        del netcons, synapses, cell

    return {
        "source_root": str(source_root),
        "neuron_version": getattr(neuron, "__version__", "unknown"),
        "duration_ms": int(duration_ms),
        "samples": int(samples),
        "seed": int(seed),
        "use_cvode": bool(use_cvode),
        "v_init_mv": float(v_init_mv),
        "rows": rows,
        "aggregate": aggregate(rows),
    }


def load_mechanisms(h, source_root: Path) -> None:
    for candidate in (source_root / "x86_64" / ".libs" / "libnrnmech.so", source_root / "x86_64" / "libnrnmech.so"):
        if candidate.exists():
            h.nrn_load_dll(str(candidate))
            return
    raise FileNotFoundError(f"compiled mechanisms not found under {source_root / 'x86_64'}")


def collect_segments(cell) -> tuple[list, np.ndarray, np.ndarray]:
    basal_sections = [cell.dend[i] for i in range(len(cell.dend))]
    apical_sections = [cell.apic[i] for i in range(len(cell.apic))]
    sections = basal_sections + apical_sections
    types = ["basal"] * len(basal_sections) + ["apical"] * len(apical_sections)
    segments = []
    segment_types = []
    segment_lengths = []
    for section, section_type in zip(sections, types):
        for segment in section:
            segments.append(segment)
            segment_types.append(section_type)
            segment_lengths.append(float(section.L) / float(section.nseg))
    return segments, np.asarray(segment_types, dtype=str), np.asarray(segment_lengths, dtype=np.float64)


def sample_neuronio_inputs(rng: np.random.Generator, duration_ms: int, segment_types: np.ndarray, segment_lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    exc = np.zeros((segment_types.shape[0], duration_ms), dtype=bool)
    inh = np.zeros_like(exc)
    for label, exc_range, diff_range in (
        ("basal", (0.0, 800.0), (-600.0, 200.0)),
        ("apical", (0.0, 800.0), (-600.0, 200.0)),
    ):
        mask = segment_types == label
        ex_rates, inh_rates = sample_region_rates(rng, duration_ms, segment_lengths[mask], exc_range, diff_range)
        exc[mask] = rng.random(ex_rates.shape) < rng.exponential(scale=np.clip(ex_rates, 0.0, None))
        inh[mask] = rng.random(inh_rates.shape) < rng.exponential(scale=np.clip(inh_rates, 0.0, None))
    return exc, inh


def sample_region_rates(
    rng: np.random.Generator,
    duration_ms: int,
    lengths: np.ndarray,
    exc_range: tuple[float, float],
    diff_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    hold_ms = int(RATE_INTERVAL_OPTIONS_MS[int(rng.integers(0, len(RATE_INTERVAL_OPTIONS_MS)))])
    hold_ms += int(rng.integers(-20, 21))
    hold_ms = max(1, hold_ms)
    sigma_ms = int(SMOOTHING_SIGMA_OPTIONS_MS[int(rng.integers(0, len(SMOOTHING_SIGMA_OPTIONS_MS)))])
    sigma_ms += int(rng.integers(-20, 21))
    sigma_ms = max(1, sigma_ms)
    coarse_steps = int(np.ceil(duration_ms / hold_ms))
    exc_counts = rng.uniform(exc_range[0], exc_range[1], size=coarse_steps)
    inh_low = np.maximum(0.0, exc_counts + diff_range[0])
    inh_high = np.maximum(inh_low, exc_counts + diff_range[1])
    inh_counts = rng.uniform(inh_low, inh_high)
    adjusted_lengths = lengths + 10.0
    total_length = max(float(adjusted_lengths.sum()), 1e-12)
    exc_rates = adjusted_lengths[:, None] * (exc_counts[None, :] / (total_length * 100.0))
    inh_rates = adjusted_lengths[:, None] * (inh_counts[None, :] / (total_length * 100.0))
    exc_rates *= rng.uniform(0.5, 1.5, size=exc_rates.shape)
    inh_rates *= rng.uniform(0.5, 1.5, size=inh_rates.shape)
    exc_full = np.repeat(exc_rates, hold_ms, axis=1)[:, :duration_ms]
    inh_full = np.repeat(inh_rates, hold_ms, axis=1)[:, :duration_ms]
    return smooth_rows(exc_full, sigma_ms), smooth_rows(inh_full, sigma_ms)


def smooth_rows(rates: np.ndarray, sigma_ms: int) -> np.ndarray:
    width = max(3, int(1 + 7 * sigma_ms))
    if width % 2 == 0:
        width += 1
    offsets = np.arange(width, dtype=np.float64) - width // 2
    kernel = np.exp(-0.5 * np.square(offsets / float(sigma_ms)))
    kernel /= kernel.sum()
    rows = []
    for row in rates:
        convolved = np.convolve(row, kernel, mode="same")
        if convolved.shape[0] != row.shape[0]:
            start = max(0, (convolved.shape[0] - row.shape[0]) // 2)
            convolved = convolved[start : start + row.shape[0]]
        rows.append(convolved)
    return np.vstack(rows)


def attach_synapses(h, segments: list, inputs_exc: np.ndarray, inputs_inh: np.ndarray) -> tuple[list[tuple[object, np.ndarray]], list]:
    netcons = []
    synapses = []
    for index, segment in enumerate(segments):
        ex_syn = h.ProbAMPANMDA2(segment)
        ex_syn.tau_r_AMPA = 0.3
        ex_syn.tau_d_AMPA = 3.0
        ex_syn.tau_r_NMDA = 2.0
        ex_syn.tau_d_NMDA = 70.0
        ex_syn.gmax = 0.0004
        ex_syn.e = 0
        ex_syn.Use = 1
        ex_syn.u0 = 0
        ex_syn.Dep = 0
        ex_syn.Fac = 0
        ex_nc = h.NetCon(None, ex_syn)
        ex_nc.delay = 0
        ex_nc.weight[0] = 1
        netcons.append((ex_nc, np.flatnonzero(inputs_exc[index]).astype(np.float64)))
        synapses.append(ex_syn)

        inh_syn = h.ProbUDFsyn2(segment)
        inh_syn.tau_r = 0.2
        inh_syn.tau_d = 8.0
        inh_syn.gmax = 0.001
        inh_syn.e = -80
        inh_syn.Use = 1
        inh_syn.u0 = 0
        inh_syn.Dep = 0
        inh_syn.Fac = 0
        inh_nc = h.NetCon(None, inh_syn)
        inh_nc.delay = 0
        inh_nc.weight[0] = 1
        netcons.append((inh_nc, np.flatnonzero(inputs_inh[index]).astype(np.float64)))
        synapses.append(inh_syn)
    return netcons, synapses


def schedule_events(netcons: list[tuple[object, np.ndarray]]) -> None:
    for netcon, times in netcons:
        for time in times:
            netcon.event(float(time))


def summarize_trace(sample_index: int, inputs_exc: np.ndarray, inputs_inh: np.ndarray, low_soma: np.ndarray, high_soma: np.ndarray) -> dict:
    rising_before = np.hstack(([False], high_soma[1:] - high_soma[:-1] > 0))
    falling_after = np.hstack((high_soma[1:] - high_soma[:-1] < 0, [False]))
    spikes = rising_before & falling_after & (high_soma > -25.0)
    clipped = np.minimum(low_soma, -55.0)
    dvdt = np.diff(clipped, prepend=clipped[:1])
    return {
        "sample_index": int(sample_index),
        "event_fraction": float((inputs_exc.sum() + inputs_inh.sum()) / (inputs_exc.size + inputs_inh.size)),
        "exc_fraction": float(inputs_exc.sum() / inputs_exc.size),
        "inh_fraction": float(inputs_inh.sum() / inputs_inh.size),
        "spike_count": int(spikes.sum()),
        "spike_rate_hz": float(1000.0 * spikes.sum() / low_soma.shape[0]),
        "soma_mean_mv": float(low_soma.mean()),
        "soma_std_mv": float(low_soma.std()),
        "soma_min_mv": float(low_soma.min()),
        "soma_max_mv": float(low_soma.max()),
        "clipped_soma_mean_mv": float(clipped.mean()),
        "clipped_soma_std_mv": float(clipped.std()),
        "clipped_abs_dvdt_p95": float(np.quantile(np.abs(dvdt), 0.95)),
        "clipped_abs_dvdt_p99": float(np.quantile(np.abs(dvdt), 0.99)),
    }


def aggregate(rows: list[dict]) -> dict:
    keys = [key for key in rows[0] if key != "sample_index"] if rows else []
    out = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        out[key] = {"mean": float(values.mean()), "std": float(values.std())}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
