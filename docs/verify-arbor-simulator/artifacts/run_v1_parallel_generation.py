#!/usr/bin/env python3
"""Generate v1 NeuronIO-like morphology shards with parallel NEURON workers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

# CoreNEURON GPU uses the NVHPC/OpenACC runtime.  Import it before package
# modules that may import torch/CUDA; otherwise the worker can later abort with
# "Enabled GPU execution but couldn't find NVIDIA GPU!".
from neuron import coreneuron as _early_coreneuron  # noqa: F401
from neuron import h as _early_h  # noqa: F401


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "docs" / "verify-arbor-simulator" / "artifacts"
CORENEURON_PROBE = ARTIFACTS / "coreneuron_hay_port_probe_20260605"
SWC_MECH_ROOT = Path(os.environ.get("BNN_SIM_SWC_MECH_ROOT", ARTIFACTS / "hay_swc_template")).resolve()
SOURCE_ROOT = ARTIFACTS / "source_models" / "neuron_as_deep_net" / "L5PC_NEURON_simulation"
TEMPLATE_HOC = Path(os.environ.get("BNN_SIM_SWC_TEMPLATE_HOC", SWC_MECH_ROOT / "L5PCtemplate_swc.hoc")).resolve()
DEFAULT_PYTHON = Path(".venv/bin/python")

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ARTIFACTS))
sys.path.insert(0, str(CORENEURON_PROBE))

from bnn_sim.benchmark import make_structured_intervention_inputs  # noqa: E402
from bnn_sim.benchmark.dataset_schema import INTERVENTION_CONDITIONS, build_v1_dataset_manifest_template, provisional_v1_morphologies  # noqa: E402
from generate_hay_neuron_dataset import SOMA_BIAS_MV, SOMA_CLIP_THRESHOLD_MV, SOMA_SCALE, local_peak_spike_labels  # noqa: E402
from run_coreneuron_event_dropout_pair import _build_source_site_route, _summarize_input_routing, simulate_signed_trace_batch  # noqa: E402
from run_coreneuron_hay_probe import load_probe_mechanisms  # noqa: E402
from run_hay_neuron_driven_probe import collect_segments, sample_neuronio_inputs  # noqa: E402
from generate_coreneuron_hay_dataset import apply_event_keep  # noqa: E402


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.worker:
        code = worker_main(args)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    return orchestrator_main(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ARTIFACTS / "v1_dataset_smoke_20260606")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--samples-per-shard", type=int, default=128)
    parser.add_argument("--duration-ms", type=int, default=6000)
    parser.add_argument("--backend", choices=["neuron", "coreneuron"], default="neuron")
    parser.add_argument("--coreneuron-gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--exc-event-keep-prob", type=float, default=0.925)
    parser.add_argument("--inh-event-keep-prob", type=float, default=1.0)
    parser.add_argument("--train-samples-per-morphology", type=int, default=0)
    parser.add_argument("--val-samples-per-morphology", type=int, default=1)
    parser.add_argument("--test-samples-per-morphology", type=int, default=0)
    parser.add_argument("--private-test-samples-per-morphology", type=int, default=0)
    parser.add_argument("--intervention-samples-per-morphology", type=int, default=1)
    parser.add_argument("--intervention", action="append", choices=INTERVENTION_CONDITIONS)
    parser.add_argument("--morphology", action="append", help="Specimen id to include; defaults to all v1 morphologies")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-kind", choices=["split", "intervention"], help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-morphology-id", help=argparse.SUPPRESS)
    parser.add_argument("--worker-morphology-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-split", help=argparse.SUPPRESS)
    parser.add_argument("--worker-intervention", choices=INTERVENTION_CONDITIONS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-samples", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-shard-index", type=int, help=argparse.SUPPRESS)
    return parser


def orchestrator_main(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    if args.force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    morphologies = selected_morphologies(args.morphology)
    interventions = tuple(args.intervention or INTERVENTION_CONDITIONS)
    tasks = build_tasks(args, output_root, morphologies, interventions)
    started = time.perf_counter()
    run_tasks(tasks, args.parallelism)
    write_split_manifests(output_root)
    summary = {
        "kind": "v1_parallel_generation",
        "output_root": str(output_root),
        "backend": args.backend,
        "coreneuron_gpu": bool(args.coreneuron_gpu),
        "duration_ms": int(args.duration_ms),
        "parallelism": int(args.parallelism),
        "elapsed_seconds": time.perf_counter() - started,
        "morphologies": [morphology["id"] for morphology in morphologies],
        "interventions": list(interventions),
        "tasks": len(tasks),
        "samples": sum(task["samples"] for task in tasks),
        "source_root": str(SOURCE_ROOT),
        "mechanism_root": str(SWC_MECH_ROOT),
        "template_hoc": str(TEMPLATE_HOC),
        "note": (
            "SWC morphology path uses the Hay-compatible SWC template/mechanisms. "
            "CoreNEURON GPU is supported for actual v1 SWC morphologies when the durable GPU runtime "
            "and SWC mechanism root are on PATH/PYTHONPATH/LD_LIBRARY_PATH."
        ),
    }
    (output_root / "generation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "dataset_manifest_template.json").write_text(
        json.dumps(build_v1_dataset_manifest_template(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def selected_morphologies(selected: list[str] | None) -> list[dict[str, str]]:
    wanted = {int(item) for item in selected} if selected else None
    rows = []
    for morphology in provisional_v1_morphologies():
        if wanted is not None and morphology.specimen_id not in wanted:
            continue
        rows.append(
            {
                "id": f"specimen_{morphology.specimen_id}",
                "path": str((ROOT / morphology.swc_path).resolve()),
            }
        )
    if not rows:
        raise ValueError("no selected morphologies")
    return rows


def build_tasks(
    args: argparse.Namespace,
    output_root: Path,
    morphologies: list[dict[str, str]],
    interventions: tuple[str, ...],
) -> list[dict[str, Any]]:
    if args.samples_per_shard <= 0:
        raise ValueError(f"--samples-per-shard must be positive, got {args.samples_per_shard}")
    split_counts = {
        "train": args.train_samples_per_morphology,
        "val": args.val_samples_per_morphology,
        "test": args.test_samples_per_morphology,
        "private-test": args.private_test_samples_per_morphology,
    }
    tasks = []
    shard_index = 0
    for morphology in morphologies:
        for split, samples in split_counts.items():
            if samples <= 0:
                continue
            for offset in range(0, samples, args.samples_per_shard):
                shard_samples = min(args.samples_per_shard, samples - offset)
                tasks.append(
                    task_payload(
                        args,
                        "split",
                        output_root / split / f"{morphology['id']}_{shard_index:05d}.npz",
                        morphology,
                        split,
                        None,
                        shard_samples,
                        shard_index,
                    )
                )
                shard_index += 1
        for intervention in interventions:
            if args.intervention_samples_per_morphology <= 0:
                continue
            for offset in range(0, args.intervention_samples_per_morphology, args.samples_per_shard):
                shard_samples = min(args.samples_per_shard, args.intervention_samples_per_morphology - offset)
                tasks.append(
                    task_payload(
                        args,
                        "intervention",
                        output_root / "interventions" / intervention / f"{morphology['id']}_intervention_pair_{shard_index:05d}.npz",
                        morphology,
                        "",
                        intervention,
                        shard_samples,
                        shard_index,
                    )
                )
                shard_index += 1
    return tasks


def task_payload(
    args: argparse.Namespace,
    kind: str,
    output: Path,
    morphology: dict[str, str],
    split: str,
    intervention: str | None,
    samples: int,
    shard_index: int,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "output": str(output),
        "morphology_id": morphology["id"],
        "morphology_path": morphology["path"],
        "split": split,
        "intervention": intervention,
        "samples": int(samples),
        "shard_index": int(shard_index),
        "duration_ms": int(args.duration_ms),
        "backend": args.backend,
        "coreneuron_gpu": bool(args.coreneuron_gpu),
        "seed": int(args.seed + 1009 * shard_index),
        "exc_event_keep_prob": float(args.exc_event_keep_prob),
        "inh_event_keep_prob": float(args.inh_event_keep_prob),
        "python": str(args.python),
    }


def run_tasks(tasks: list[dict[str, Any]], parallelism: int) -> None:
    running: list[tuple[subprocess.Popen, dict[str, Any]]] = []
    pending = list(tasks)
    while pending or running:
        while pending and len(running) < parallelism:
            task = pending.pop(0)
            cmd = [
                task["python"],
                str(Path(__file__).resolve()),
                "--worker",
                "--worker-kind",
                task["kind"],
                "--worker-output",
                task["output"],
                "--worker-morphology-id",
                task["morphology_id"],
                "--worker-morphology-path",
                task["morphology_path"],
                "--worker-split",
                task["split"],
                "--worker-samples",
                str(task["samples"]),
                "--worker-shard-index",
                str(task["shard_index"]),
                "--duration-ms",
                str(task["duration_ms"]),
                "--backend",
                task["backend"],
                "--seed",
                str(task["seed"]),
                "--exc-event-keep-prob",
                str(task["exc_event_keep_prob"]),
                "--inh-event-keep-prob",
                str(task["inh_event_keep_prob"]),
            ]
            if task["intervention"]:
                cmd += ["--worker-intervention", task["intervention"]]
            if task["coreneuron_gpu"]:
                cmd += ["--coreneuron-gpu"]
            env = os.environ.copy()
            env["PYTHONPATH"] = f"src:{ARTIFACTS}:{CORENEURON_PROBE}:{env.get('PYTHONPATH', '')}"
            running.append((subprocess.Popen(cmd, cwd=ROOT, env=env), task))
        next_running = []
        for proc, task in running:
            status = proc.poll()
            if status is None:
                next_running.append((proc, task))
            elif status != 0:
                raise RuntimeError(f"task failed with status {status}: {task}")
        running = next_running
        if running:
            time.sleep(0.25)


def worker_main(args: argparse.Namespace) -> int:
    if args.worker_samples is None or args.worker_output is None:
        raise ValueError("worker requires --worker-samples and --worker-output")
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    setup = prepare_neuron(args.backend, args.coreneuron_gpu)
    sample_rows = []
    baseline_inputs = []
    baseline_targets = []
    baseline_soma = []
    intervention_inputs = []
    intervention_targets = []
    intervention_soma = []
    changed_masks = []
    sample_ids = []
    for sample_index in range(args.worker_samples):
        sample_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        signed_inputs = sample_signed_inputs(
            setup["source_segment_types"],
            setup["source_segment_lengths"],
            duration_ms=args.duration_ms,
            seed=sample_seed,
            exc_event_keep_prob=args.exc_event_keep_prob,
            inh_event_keep_prob=args.inh_event_keep_prob,
        )
        baseline_inputs.append(signed_inputs)
        sample_id = f"{args.worker_morphology_id}:{args.worker_kind}:{args.worker_split or args.worker_intervention}:{args.worker_shard_index:05d}:{sample_index:04d}:seed{sample_seed}"
        sample_ids.append(sample_id)
        if args.worker_kind == "intervention":
            intervention_batch, changed_mask, _ = make_structured_intervention_inputs(
                signed_inputs[None, ...],
                intervention=args.worker_intervention,
                dropout_probability=0.35,
                jitter_steps=2,
                site_start_fraction=0.0,
                site_fraction=0.25,
                seed=args.seed + sample_index,
            )
            intervention_inputs.append(intervention_batch[0])
            changed_masks.append(changed_mask[0])

    baseline_traces = simulate_signed_trace_batch(
        h=setup["h"],
        source_root=SOURCE_ROOT,
        morphology_path=args.worker_morphology_path.resolve(),
        backend=args.backend,
        signed_inputs_batch=np.stack(baseline_inputs).astype(np.int8, copy=False),
        v_init_mv=-76.0,
        exc_gmax_scale=1.0,
        inh_gmax_scale=1.0,
        input_routing="normalized_by_type",
        source_route=setup["source_route"],
    )
    for baseline in baseline_traces:
        baseline_targets.append(baseline["targets"])
        baseline_soma.append(baseline["soma_mv"])
        sample_rows.append(baseline["input_routing"])

    if args.worker_kind == "intervention":
        intervention_traces = simulate_signed_trace_batch(
            h=setup["h"],
            source_root=SOURCE_ROOT,
            morphology_path=args.worker_morphology_path.resolve(),
            backend=args.backend,
            signed_inputs_batch=np.stack(intervention_inputs).astype(np.int8, copy=False),
            v_init_mv=-76.0,
            exc_gmax_scale=1.0,
            inh_gmax_scale=1.0,
            input_routing="normalized_by_type",
            source_route=setup["source_route"],
        )
        for intervention in intervention_traces:
            intervention_targets.append(intervention["targets"])
            intervention_soma.append(intervention["soma_mv"])

    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker_kind == "split":
        np.savez_compressed(
            args.worker_output,
            inputs=np.stack(baseline_inputs).astype(np.int8, copy=False),
            targets=np.stack(baseline_targets).astype(np.float32, copy=False),
            soma_mv=np.stack(baseline_soma).astype(np.float32, copy=False),
            sample_ids=np.asarray(sample_ids),
        )
    else:
        np.savez_compressed(
            args.worker_output,
            baseline_inputs=np.stack(baseline_inputs).astype(np.int8, copy=False),
            baseline_targets=np.stack(baseline_targets).astype(np.float32, copy=False),
            baseline_soma_mv=np.stack(baseline_soma).astype(np.float32, copy=False),
            intervention_inputs=np.stack(intervention_inputs).astype(np.int8, copy=False),
            intervention_targets=np.stack(intervention_targets).astype(np.float32, copy=False),
            intervention_soma_mv=np.stack(intervention_soma).astype(np.float32, copy=False),
            changed_event_mask=np.stack(changed_masks).astype(bool, copy=False),
            sample_ids=np.asarray(sample_ids),
        )
    metadata = {
        "kind": args.worker_kind,
        "backend": args.backend,
        "coreneuron_gpu": bool(args.coreneuron_gpu),
        "duration_ms": int(args.duration_ms),
        "elapsed_seconds": time.perf_counter() - started,
        "morphology_id": args.worker_morphology_id,
        "morphology_path": str(args.worker_morphology_path),
        "samples": int(args.worker_samples),
        "shard_index": int(args.worker_shard_index),
        "input_routing": _summarize_input_routing(sample_rows),
    }
    args.worker_output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def prepare_neuron(backend: str, coreneuron_gpu: bool) -> dict[str, Any]:
    from neuron import coreneuron, h

    if backend == "coreneuron":
        coreneuron.enable = True
        coreneuron.gpu = bool(coreneuron_gpu)
        if coreneuron_gpu:
            coreneuron.cell_permute = int(os.environ.get("BNN_SIM_CORENEURON_CELL_PERMUTE", "1"))
        if os.environ.get("BNN_SIM_CORENEURON_NUM_GPUS"):
            coreneuron.num_gpus = int(os.environ["BNN_SIM_CORENEURON_NUM_GPUS"])
        coreneuron.verbose = 0
    else:
        coreneuron.enable = False
        coreneuron.gpu = False
    os.chdir(SWC_MECH_ROOT)
    load_probe_mechanisms(h, SWC_MECH_ROOT)
    hoc_lib = Path(os.environ.get("NEURONHOME", "")) / "share" / "nrn" / "lib" / "hoc"
    if hoc_lib.exists():
        h.load_file(str(hoc_lib / "stdrun.hoc"))
        h.load_file(str(hoc_lib / "import3d.hoc"))
    else:
        h.load_file("stdrun.hoc")
        h.load_file("import3d.hoc")
    h.load_file(str(SOURCE_ROOT / "L5PCbiophys5b.hoc"))
    h.load_file(str(TEMPLATE_HOC))
    h.dt = 0.025
    h.steps_per_ms = 40
    h.secondorder = 0
    h.CVode().active(0)
    h.CVode().cache_efficient(1)
    source_cell = h.L5PCtemplate(str(SOURCE_ROOT / "morphologies" / "cell1.asc"))
    _, source_segment_types, source_segment_lengths = collect_segments(source_cell)
    del source_cell
    return {
        "h": h,
        "source_route": _build_source_site_route(h, SOURCE_ROOT),
        "source_segment_types": source_segment_types,
        "source_segment_lengths": source_segment_lengths,
    }


def sample_signed_inputs(
    source_segment_types: np.ndarray,
    source_segment_lengths: np.ndarray,
    *,
    duration_ms: int,
    seed: int,
    exc_event_keep_prob: float,
    inh_event_keep_prob: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    inputs_exc, inputs_inh = sample_neuronio_inputs(rng, duration_ms, source_segment_types, source_segment_lengths)
    inputs_exc = apply_event_keep(rng, inputs_exc, exc_event_keep_prob)
    inputs_inh = apply_event_keep(rng, inputs_inh, inh_event_keep_prob)
    signed_inputs = np.zeros((duration_ms, 2 * inputs_exc.shape[0]), dtype=np.int8)
    signed_inputs[:, : inputs_exc.shape[0]] = inputs_exc.T.astype(np.int8, copy=False)
    signed_inputs[:, inputs_exc.shape[0] :] = -inputs_inh.T.astype(np.int8, copy=False)
    return signed_inputs


def write_split_manifests(output_root: Path) -> None:
    for split in ("train", "val", "test", "private-test"):
        write_npz_manifest(output_root / split)
    for intervention in INTERVENTION_CONDITIONS:
        write_pair_manifest(output_root / "interventions" / intervention)


def write_npz_manifest(root: Path) -> None:
    if not root.exists():
        return
    entries = []
    for shard in sorted(root.glob("*.npz")):
        with np.load(shard) as data:
            sample_ids = [str(item) for item in data["sample_ids"]]
            entries.append({"file": shard.name, "count": int(data["inputs"].shape[0]), "sample_ids": sample_ids})
    if entries:
        (root / "manifest.json").write_text(json.dumps({"version": 1, "shards": entries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_pair_manifest(root: Path) -> None:
    if not root.exists():
        return
    entries = []
    for shard in sorted({*root.glob("intervention_pair_*.npz"), *root.glob("*_intervention_pair_*.npz")}):
        with np.load(shard) as data:
            entries.append({"file": shard.name, "count": int(data["baseline_inputs"].shape[0]), "sample_ids": [str(item) for item in data["sample_ids"]]})
    if entries:
        (root / "manifest.json").write_text(json.dumps({"version": 1, "pair_shards": entries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
