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
    args = parser.parse_args()

    if not 0.0 <= args.dropout_probability <= 1.0:
        raise ValueError("--dropout-probability must be in [0, 1]")

    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
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
    simulated_targets, simulated_soma_mv = simulate_intervention_targets(
        intervention_inputs,
        source_root=Path(args.source_root).resolve(),
        mechanism_root=Path(args.mechanism_root).resolve(),
        morphology_path=_resolve_source_path(Path(args.source_root).resolve(), args.morphology),
        template_hoc=_resolve_source_path(Path(args.source_root).resolve(), args.template_hoc),
        backend=args.backend,
        coreneuron_gpu=args.coreneuron_gpu,
        v_init_mv=args.v_init_mv,
        exc_gmax_scale=args.exc_gmax_scale,
        inh_gmax_scale=args.inh_gmax_scale,
    )
    teacher_report = compute_paired_event_intervention_metrics(
        None,
        baseline_inputs,
        baseline_targets,
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
        baseline_targets=baseline_targets,
        intervention_inputs=intervention_inputs,
        intervention_targets=simulated_targets,
        intervention_soma_mv=simulated_soma_mv,
        changed_event_mask=change_mask,
        sample_ids=np.asarray(trace_ids),
    )
    # Keep the historical filename for existing scripts.
    np.savez_compressed(
        output_dir / "event_dropout_pair.npz",
        baseline_inputs=baseline_inputs,
        baseline_targets=baseline_targets,
        intervention_inputs=intervention_inputs,
        intervention_targets=simulated_targets,
        intervention_soma_mv=simulated_soma_mv,
        changed_event_mask=change_mask,
        sample_ids=np.asarray(trace_ids),
    )
    report = {
        "metadata": {
            "source_data": args.source_data,
            "source_root": args.source_root,
            "mechanism_root": args.mechanism_root,
            "morphology": str(_resolve_source_path(Path(args.source_root).resolve(), args.morphology)),
            "template_hoc": str(_resolve_source_path(Path(args.source_root).resolve(), args.template_hoc)),
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
) -> tuple[np.ndarray, np.ndarray]:
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

    targets = np.zeros((signed_inputs.shape[0], signed_inputs.shape[1], 2), dtype=np.float32)
    soma_mv = np.zeros((signed_inputs.shape[0], signed_inputs.shape[1]), dtype=np.float32)
    for index in range(signed_inputs.shape[0]):
        trace = simulate_one_signed_trace(
            h=h,
            source_root=source_root,
            morphology_path=morphology_path,
            backend=backend,
            signed_inputs=signed_inputs[index],
            v_init_mv=v_init_mv,
            exc_gmax_scale=exc_gmax_scale,
            inh_gmax_scale=inh_gmax_scale,
        )
        targets[index] = trace["targets"]
        soma_mv[index] = trace["soma_mv"]
    return targets, soma_mv


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
) -> dict[str, np.ndarray]:
    duration_ms = int(signed_inputs.shape[0])
    cell = h.L5PCtemplate(str(morphology_path))
    h.CVode().active(0)
    h.v_init = v_init_mv
    segments, _, _ = collect_segments(cell)
    for section in list(cell.dend) + list(cell.apic):
        section.vshift_Ih = 0
    cell.soma[0].vshift_Ih = 0

    inputs_exc = (signed_inputs[:, :639] > 0).T
    inputs_inh = (signed_inputs[:, 639:] < 0).T
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
    return {"targets": targets, "soma_mv": low_soma}


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
            f"- Changed events: `{meta['changed_events']}` / `{meta['baseline_events']}`",
            f"- Intervention events: `{meta['intervention_events']}`",
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
