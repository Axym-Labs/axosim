#!/usr/bin/env python3
"""Create a paired CoreNEURON/Hay teacher structured intervention."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

# CoreNEURON GPU uses the NVHPC/OpenACC runtime.  Import it before package
# modules that may import torch/CUDA; otherwise GPU discovery can fail later in
# embedded CoreNEURON even when the device is present.
from neuron import coreneuron as _early_coreneuron  # noqa: F401
from neuron import h as _early_h  # noqa: F401

ARTIFACTS = Path(__file__).resolve().parent
CORENEURON_PROBE = ARTIFACTS / "coreneuron_hay_port_probe_20260605"
DEFAULT_SOURCE_ROOT = ARTIFACTS / "source_models" / "neuron_as_deep_net" / "L5PC_NEURON_simulation"
sys.path.insert(0, str(ARTIFACTS))
sys.path.insert(0, str(CORENEURON_PROBE))

from bnn_sim.benchmark import compute_paired_event_intervention_metrics, make_structured_intervention_inputs  # noqa: E402
from bnn_sim.data import ShardedNeuronIODataset  # noqa: E402
from generate_hay_neuron_dataset import (  # noqa: E402
    SOMA_BIAS_MV,
    SOMA_CLIP_THRESHOLD_MV,
    SOMA_SCALE,
    local_peak_spike_labels,
)
from run_coreneuron_hay_probe import attach_deterministic_synapses, load_probe_mechanisms  # noqa: E402
from run_hay_neuron_driven_probe import collect_segments  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--mechanism-root", default=str(CORENEURON_PROBE))
    parser.add_argument("--morphology", default="morphologies/cell1.asc")
    parser.add_argument("--template-hoc", default="L5PCtemplate_2.hoc")
    parser.add_argument("--source-data", default="data/hay-coreneuron-source-val16-exc0925-20260605/val")
    parser.add_argument("--output-dir", default="docs/verify-arbor-simulator/artifacts/coreneuron_event_dropout_pair_20260605")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument(
        "--intervention",
        choices=["event_dropout", "exc_dropout", "inh_dropout", "temporal_jitter", "site_silence"],
        default="event_dropout",
    )
    parser.add_argument("--dropout-probability", type=float, default=0.35)
    parser.add_argument("--jitter-steps", type=int, default=2)
    parser.add_argument("--site-start-fraction", type=float, default=0.0)
    parser.add_argument("--site-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--backend", choices=["neuron", "coreneuron"], default="coreneuron")
    parser.add_argument("--coreneuron-gpu", action="store_true")
    parser.add_argument("--v-init-mv", type=float, default=-76.0)
    parser.add_argument("--exc-gmax-scale", type=float, default=1.0)
    parser.add_argument("--inh-gmax-scale", type=float, default=1.0)
    parser.add_argument(
        "--simulate-baseline",
        action="store_true",
        help="Simulate baseline targets under the requested morphology/template instead of using source-data targets.",
    )
    parser.add_argument(
        "--input-routing",
        choices=["ordered_truncate", "normalized_by_type"],
        default="ordered_truncate",
        help=(
            "How to map source E/I site channels onto the instantiated morphology. "
            "ordered_truncate preserves legacy behavior; normalized_by_type maps "
            "within basal/apical compartments by normalized arclength."
        ),
    )
    args = parser.parse_args()

    if not 0.0 <= args.dropout_probability <= 1.0:
        raise ValueError("--dropout-probability must be in [0, 1]")

    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    source_root = Path(args.source_root).resolve()
    mechanism_root = Path(args.mechanism_root).resolve()
    morphology_path = _resolve_source_path(source_root, args.morphology)
    template_hoc = _resolve_source_path(source_root, args.template_hoc)
    baseline_inputs, baseline_targets, trace_ids = load_source(args.source_data, args.samples)
    intervention_inputs, change_mask, intervention_meta = make_structured_intervention_inputs(
        baseline_inputs,
        intervention=args.intervention,
        dropout_probability=args.dropout_probability,
        jitter_steps=args.jitter_steps,
        site_start_fraction=args.site_start_fraction,
        site_fraction=args.site_fraction,
        seed=args.seed,
    )
    baseline_targets_for_report = baseline_targets
    baseline_soma_mv = None
    baseline_input_routing = None
    if args.simulate_baseline:
        baseline_targets_for_report, baseline_soma_mv, baseline_input_routing = simulate_intervention_targets(
            baseline_inputs,
            source_root=source_root,
            mechanism_root=mechanism_root,
            morphology_path=morphology_path,
            template_hoc=template_hoc,
            backend=args.backend,
            coreneuron_gpu=args.coreneuron_gpu,
            v_init_mv=args.v_init_mv,
            exc_gmax_scale=args.exc_gmax_scale,
            inh_gmax_scale=args.inh_gmax_scale,
            input_routing=args.input_routing,
        )
    simulated_targets, simulated_soma_mv, input_routing = simulate_intervention_targets(
        intervention_inputs,
        source_root=source_root,
        mechanism_root=mechanism_root,
        morphology_path=morphology_path,
        template_hoc=template_hoc,
        backend=args.backend,
        coreneuron_gpu=args.coreneuron_gpu,
        v_init_mv=args.v_init_mv,
        exc_gmax_scale=args.exc_gmax_scale,
        inh_gmax_scale=args.inh_gmax_scale,
        input_routing=args.input_routing,
    )
    teacher_report = compute_paired_event_intervention_metrics(
        None,
        baseline_inputs,
        baseline_targets_for_report,
        intervention_inputs,
        simulated_targets,
        intervention_name=f"coreneuron_hay_{args.intervention}",
        soma_scale=SOMA_SCALE,
        soma_bias_mv=SOMA_BIAS_MV,
        ignore_start=min(500, max(0, baseline_inputs.shape[1] - 1)),
        trace_ids=trace_ids,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "intervention_pair.npz",
        baseline_inputs=baseline_inputs,
        baseline_targets=baseline_targets_for_report,
        source_baseline_targets=baseline_targets,
        intervention_inputs=intervention_inputs,
        intervention_targets=simulated_targets,
        baseline_soma_mv=np.asarray([] if baseline_soma_mv is None else baseline_soma_mv),
        intervention_soma_mv=simulated_soma_mv,
        changed_event_mask=change_mask,
        sample_ids=np.asarray(trace_ids),
    )
    # Keep the historical filename for existing scripts.
    np.savez_compressed(
        output_dir / "event_dropout_pair.npz",
        baseline_inputs=baseline_inputs,
        baseline_targets=baseline_targets_for_report,
        source_baseline_targets=baseline_targets,
        intervention_inputs=intervention_inputs,
        intervention_targets=simulated_targets,
        baseline_soma_mv=np.asarray([] if baseline_soma_mv is None else baseline_soma_mv),
        intervention_soma_mv=simulated_soma_mv,
        changed_event_mask=change_mask,
        sample_ids=np.asarray(trace_ids),
    )
    report = {
        "metadata": {
            "source_data": args.source_data,
            "source_root": args.source_root,
            "mechanism_root": args.mechanism_root,
            "morphology": str(morphology_path),
            "template_hoc": str(template_hoc),
            "backend": args.backend,
            "coreneuron_gpu": bool(args.coreneuron_gpu),
            "elapsed_seconds": time.perf_counter() - started,
            "intervention": args.intervention,
            "intervention_parameters": intervention_meta,
            "seed": args.seed,
            "samples": int(baseline_inputs.shape[0]),
            "time_steps": int(baseline_inputs.shape[1]),
            "changed_events": int(change_mask.sum()),
            "baseline_events": int(np.count_nonzero(baseline_inputs)),
            "intervention_events": int(np.count_nonzero(intervention_inputs)),
            "input_routing": input_routing,
            "baseline_input_routing": baseline_input_routing,
            "simulate_baseline": bool(args.simulate_baseline),
            "baseline_target_policy": (
                "baseline targets simulated under requested morphology/template"
                if args.simulate_baseline
                else "baseline targets loaded from source-data dataset"
            ),
            "target_policy": "same clipped/normalized soma and local-peak spike labels as accepted CoreNEURON/Hay data",
        },
        "teacher_intervention": teacher_report,
        "teacher_event_dropout": teacher_report,
    }
    (output_dir / "intervention_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "intervention_report.md").write_text(render_markdown(report), encoding="utf-8")
    (output_dir / "event_dropout_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "event_dropout_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["teacher_intervention"]["teacher_response"], indent=2, sort_keys=True))
    return 0


def _resolve_source_path(source_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (source_root / path).resolve()


def load_source(path: str | Path, samples: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    dataset = ShardedNeuronIODataset(path, cache_shards=1)
    n = min(samples, len(dataset))
    inputs, targets = dataset.get_batch(list(range(n)))
    trace_ids = dataset.get_sample_ids(list(range(n)))
    return inputs.astype(np.int8, copy=False), targets.astype(np.float32, copy=False), trace_ids


def simulate_intervention_targets(
    signed_inputs: np.ndarray,
    *,
    source_root: Path,
    mechanism_root: Path,
    morphology_path: Path,
    template_hoc: Path,
    backend: str,
    coreneuron_gpu: bool,
    v_init_mv: float,
    exc_gmax_scale: float,
    inh_gmax_scale: float,
    input_routing: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from neuron import coreneuron, h

    if backend == "coreneuron":
        coreneuron.enable = True
        coreneuron.gpu = bool(coreneuron_gpu)
        coreneuron.verbose = 0
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
    h.load_file(str(template_hoc))
    h.dt = 0.025
    h.steps_per_ms = 40
    h.secondorder = 0
    h.CVode().active(0)
    h.CVode().cache_efficient(1)
    source_route = _build_source_site_route(h, source_root)

    traces = simulate_signed_trace_batch(
        h=h,
        source_root=source_root,
        morphology_path=morphology_path,
        backend=backend,
        signed_inputs_batch=signed_inputs,
        v_init_mv=v_init_mv,
        exc_gmax_scale=exc_gmax_scale,
        inh_gmax_scale=inh_gmax_scale,
        input_routing=input_routing,
        source_route=source_route,
    )
    targets = np.stack([trace["targets"] for trace in traces]).astype(np.float32, copy=False)
    soma_mv = np.stack([trace["soma_mv"] for trace in traces]).astype(np.float32, copy=False)
    routing_rows = [trace["input_routing"] for trace in traces]
    return targets, soma_mv, _summarize_input_routing(routing_rows)


def simulate_one_signed_trace(
    *,
    h,
    source_root: Path,
    morphology_path: Path,
    backend: str,
    signed_inputs: np.ndarray,
    v_init_mv: float,
    exc_gmax_scale: float,
    inh_gmax_scale: float,
    input_routing: str,
    source_route: dict[str, Any],
) -> dict[str, np.ndarray]:
    duration_ms = int(signed_inputs.shape[0])
    cell = h.L5PCtemplate(str(morphology_path))
    h.CVode().active(0)
    h.v_init = v_init_mv
    segments, segment_types, segment_lengths = collect_segments(cell)
    for section in list(cell.dend) + list(cell.apic):
        section.vshift_Ih = 0
    cell.soma[0].vshift_Ih = 0

    inputs_exc, inputs_inh, input_routing_summary = _route_inputs_to_segments(
        signed_inputs,
        segment_types,
        segment_lengths,
        routing=input_routing,
        source_route=source_route,
    )
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
    spike_labels = local_peak_spike_labels(high_time, high_soma, duration_ms)
    clipped_soma = np.minimum(low_soma, SOMA_CLIP_THRESHOLD_MV)
    normalized_soma = ((clipped_soma - SOMA_BIAS_MV) * SOMA_SCALE).astype(np.float32)
    targets = np.stack([spike_labels, normalized_soma], axis=1).astype(np.float32)

    del netcons, synapses, stim_refs, cell
    return {"targets": targets, "soma_mv": low_soma, "input_routing": input_routing_summary}


def simulate_signed_trace_batch(
    *,
    h,
    source_root: Path,
    morphology_path: Path,
    backend: str,
    signed_inputs_batch: np.ndarray,
    v_init_mv: float,
    exc_gmax_scale: float,
    inh_gmax_scale: float,
    input_routing: str,
    source_route: dict[str, Any],
) -> list[dict[str, np.ndarray]]:
    if signed_inputs_batch.ndim != 3:
        raise ValueError(f"signed_inputs_batch must have shape (samples, time, channels), got {signed_inputs_batch.shape}")
    duration_ms = int(signed_inputs_batch.shape[1])
    pc = h.ParallelContext()
    try:
        pc.gid_clear()
    except Exception:
        pass
    h.CVode().active(0)
    h.v_init = v_init_mv

    records = []
    refs = []
    event_sink: list[tuple[float, int]] = []
    next_gid = 1
    for signed_inputs in signed_inputs_batch:
        cell = h.L5PCtemplate(str(morphology_path))
        segments, segment_types, segment_lengths = collect_segments(cell)
        for section in list(cell.dend) + list(cell.apic):
            section.vshift_Ih = 0
        cell.soma[0].vshift_Ih = 0
        inputs_exc, inputs_inh, input_routing_summary = _route_inputs_to_segments(
            signed_inputs,
            segment_types,
            segment_lengths,
            routing=input_routing,
            source_route=source_route,
        )
        netcons, synapses, stim_refs = attach_deterministic_synapses(
            h,
            pc,
            segments,
            inputs_exc,
            inputs_inh,
            exc_gmax_scale=exc_gmax_scale,
            inh_gmax_scale=inh_gmax_scale,
            gid_start=next_gid,
            event_sink=event_sink,
        )
        next_gid += 2 * len(segments)
        t_vec = h.Vector()
        soma_vec = h.Vector()
        t_vec.record(h._ref_t)
        soma_vec.record(cell.soma[0](0.5)._ref_v)
        records.append({"t_vec": t_vec, "soma_vec": soma_vec, "input_routing": input_routing_summary})
        refs.append((cell, netcons, synapses, stim_refs, t_vec, soma_vec))

    if event_sink:
        order = np.argsort(np.asarray([item[0] for item in event_sink], dtype=np.float64), kind="stable")
        sorted_times = np.asarray([item[0] for item in event_sink], dtype=np.float64)[order]
        sorted_gids = np.asarray([item[1] for item in event_sink], dtype=np.int32)[order]
    else:
        sorted_times = np.asarray([], dtype=np.float64)
        sorted_gids = np.asarray([], dtype=np.int32)
    stim = h.PatternStim()
    t_events = h.Vector(sorted_times)
    gid_events = h.Vector(sorted_gids)
    stim.play(t_events, gid_events)
    refs.append((stim, t_events, gid_events))

    h.finitialize(v_init_mv)
    if backend == "coreneuron":
        pc.psolve(duration_ms)
    else:
        h.continuerun(duration_ms)

    traces = []
    for record in records:
        t = np.asarray(record["t_vec"], dtype=np.float64)
        soma = np.asarray(record["soma_vec"], dtype=np.float64)
        low_time = np.arange(0, duration_ms, dtype=np.float64)
        high_time = np.arange(0, duration_ms, 0.125, dtype=np.float64)
        low_soma = np.interp(low_time, t, soma).astype(np.float32)
        high_soma = np.interp(high_time, t, soma)
        spike_labels = local_peak_spike_labels(high_time, high_soma, duration_ms)
        clipped_soma = np.minimum(low_soma, SOMA_CLIP_THRESHOLD_MV)
        normalized_soma = ((clipped_soma - SOMA_BIAS_MV) * SOMA_SCALE).astype(np.float32)
        targets = np.stack([spike_labels, normalized_soma], axis=1).astype(np.float32)
        traces.append({"targets": targets, "soma_mv": low_soma, "input_routing": record["input_routing"]})

    del refs
    return traces


def _build_source_site_route(h, source_root: Path) -> dict[str, Any]:
    source_cell = h.L5PCtemplate(str(source_root / "morphologies" / "cell1.asc"))
    try:
        _, source_types, source_lengths = collect_segments(source_cell)
        return _segment_route(source_types, source_lengths)
    finally:
        del source_cell


def _segment_route(segment_types: np.ndarray, segment_lengths: np.ndarray) -> dict[str, Any]:
    segment_types = np.asarray(segment_types, dtype=str)
    segment_lengths = np.asarray(segment_lengths, dtype=np.float64)
    positions = np.zeros(segment_types.shape[0], dtype=np.float64)
    for label in np.unique(segment_types):
        mask = segment_types == label
        lengths = np.maximum(segment_lengths[mask], 1e-12)
        cumulative = np.cumsum(lengths) - 0.5 * lengths
        positions[mask] = cumulative / max(float(lengths.sum()), 1e-12)
    return {
        "types": segment_types,
        "lengths": segment_lengths,
        "positions": positions,
        "count_by_type": {str(label): int(np.count_nonzero(segment_types == label)) for label in np.unique(segment_types)},
    }


def _route_inputs_to_segments(
    signed_inputs: np.ndarray,
    segment_types: np.ndarray,
    segment_lengths: np.ndarray,
    *,
    routing: str,
    source_route: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Route source E/I site channels onto the current morphology segments.

    The original Hay/NeuronIO protocol has one excitatory and one inhibitory
    channel per source segment.  Held-out morphologies can have fewer segments;
    for the M3 smoke we preserve source channel order and use the first
    `segment_count` sites, recording the truncation explicitly.  More segments
    than source sites would require a new interpolation/routing policy and is
    therefore rejected.
    """
    if signed_inputs.ndim != 2:
        raise ValueError(f"signed_inputs must have shape (time, channels), got {signed_inputs.shape}")
    if signed_inputs.shape[1] % 2 != 0:
        raise ValueError(f"signed_inputs channel count must split evenly into E/I sites, got {signed_inputs.shape[1]}")
    source_sites = signed_inputs.shape[1] // 2
    segment_count = int(np.asarray(segment_types).shape[0])
    if source_sites != int(source_route["types"].shape[0]):
        raise ValueError(
            "source route metadata does not match signed input sites; "
            f"source_route_sites={source_route['types'].shape[0]}, input_sites={source_sites}"
        )
    if routing == "ordered_truncate" and segment_count > source_sites:
        raise ValueError(
            "morphology has more segments than available source input sites; "
            f"segments={segment_count}, source_sites={source_sites}. "
            "Define an explicit interpolation/routing policy before using this morphology."
        )
    source_indices = (
        np.arange(segment_count, dtype=np.int64)
        if routing == "ordered_truncate"
        else _normalized_source_indices(segment_types, segment_lengths, source_route)
    )
    inputs_exc = (signed_inputs[:, :source_sites] > 0).T[source_indices]
    inputs_inh = (signed_inputs[:, source_sites:] < 0).T[source_indices]
    unique_source = int(np.unique(source_indices).size)
    if routing == "ordered_truncate":
        policy = "ordered_source_site_routing" if segment_count == source_sites else "ordered_source_site_truncation"
    else:
        policy = "normalized_by_type_arclength"
    return inputs_exc, inputs_inh, {
        "policy": policy,
        "source_sites_per_sign": int(source_sites),
        "morphology_segments": int(segment_count),
        "used_sites_per_sign": int(segment_count),
        "unique_source_sites_per_sign": unique_source,
        "duplicated_source_sites_per_sign": int(segment_count - unique_source),
        "unused_source_sites_per_sign": int(source_sites - unique_source),
        "target_count_by_type": _count_by_type(segment_types),
        "source_count_by_type": source_route["count_by_type"],
        "defensibility_note": (
            "Exact source-channel routing is preserved only when morphology segment count matches source sites. "
            "normalized_by_type_arclength is a morphology-normalized smoke policy: each target basal/apical "
            "segment receives the nearest source site from the same compartment by normalized arclength."
        ),
    }


def _normalized_source_indices(
    target_types: np.ndarray,
    target_lengths: np.ndarray,
    source_route: dict[str, Any],
) -> np.ndarray:
    target_route = _segment_route(target_types, target_lengths)
    source_types = source_route["types"]
    source_positions = source_route["positions"]
    indices = np.zeros(target_route["types"].shape[0], dtype=np.int64)
    for index, (label, position) in enumerate(zip(target_route["types"], target_route["positions"])):
        candidates = np.flatnonzero(source_types == label)
        if candidates.size == 0:
            candidates = np.arange(source_types.shape[0], dtype=np.int64)
        nearest = candidates[np.argmin(np.abs(source_positions[candidates] - position))]
        indices[index] = int(nearest)
    return indices


def _count_by_type(segment_types: np.ndarray) -> dict[str, int]:
    arr = np.asarray(segment_types, dtype=str)
    return {str(label): int(np.count_nonzero(arr == label)) for label in np.unique(arr)}


def _summarize_input_routing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "not_run"}
    keys = (
        "source_sites_per_sign",
        "morphology_segments",
        "used_sites_per_sign",
        "unique_source_sites_per_sign",
        "duplicated_source_sites_per_sign",
        "unused_source_sites_per_sign",
    )
    summary = {
        "status": "recorded",
        "policy": rows[0]["policy"],
        "n_traces": len(rows),
        "all_traces_identical": all(row == rows[0] for row in rows),
        "defensibility_note": rows[0]["defensibility_note"],
    }
    for key in keys:
        values = [int(row[key]) for row in rows]
        summary[key] = {
            "min": int(min(values)),
            "max": int(max(values)),
            "mean": float(np.mean(values)),
        }
    return summary


def render_markdown(report: dict[str, Any]) -> str:
    response = report["teacher_intervention"]["teacher_response"]
    meta = report["metadata"]
    return "\n".join(
        [
            "# CoreNEURON/Hay Structured Teacher Pair",
            "",
            f"- Source data: `{meta['source_data']}`",
            f"- Backend: `{meta['backend']}`",
            f"- Intervention: `{meta['intervention']}`",
            f"- Baseline target policy: `{meta['baseline_target_policy']}`",
            f"- Changed events: `{meta['changed_events']}` / `{meta['baseline_events']}`",
            f"- Intervention events: `{meta['intervention_events']}`",
            f"- Input routing policy: `{meta['input_routing']['policy']}`",
            f"- Used source sites per sign: `{meta['input_routing']['used_sites_per_sign']['mean']:.1f}`",
            f"- Unused source sites per sign: `{meta['input_routing']['unused_source_sites_per_sign']['mean']:.1f}`",
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
