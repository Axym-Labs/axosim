from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bnn_sim.checkpoint import load_checkpoint
from bnn_sim.data import (
    DeterministicWindowDataset,
    OfficialStyleFullTraceWindowDataset,
    RandomFullTraceWindowDataset,
    ShardedNeuronIODataset,
    repack_shards_as_npy,
    write_demo_shards,
)
from bnn_sim.diagnostics import diagnose_dataset
from bnn_sim.experiments import (
    append_registry_entry,
    apply_preset,
    build_registry_entry,
    list_presets,
    resolve_registry_path,
)
from bnn_sim.evaluate import evaluate_dataset, write_metrics
from bnn_sim.inference_benchmark import (
    benchmark_inference_matrix,
    parse_int_list,
    write_inference_benchmark,
)
from bnn_sim.arbor_sim import (
    ArborHeteroConfig,
    parse_split_samples,
    write_arbor_hetero_dataset,
    write_arbor_hetero_splits,
)
from bnn_sim.benchmark.profiles import (
    build_axis_plan,
    estimate_profile_cost,
    get_benchmark_profile,
    list_benchmark_profiles,
)
from bnn_sim.benchmark.dataset_schema import (
    build_diagnostic_suite_catalog,
    build_v1_dataset_manifest_template,
)
from bnn_sim.benchmark.diagnostic_rows import DiagnosticContext
from bnn_sim.benchmark.bundle import run_v1_benchmark_bundle
from bnn_sim.benchmark.runner import (
    run_paired_intervention_on_npz,
    run_paired_intervention_on_selector,
    run_teacher_qc_on_selector,
)
from bnn_sim.benchmark.reports import write_diagnostic_summary
from bnn_sim.benchmark.selectors import V1DatasetLayout
from bnn_sim.model import BranchELM, BranchELMConfig
from bnn_sim.neuronio_raw import convert_neuronio_pickles
from bnn_sim.official import create_official_elm, load_official_elm_config, load_official_state_dict
from bnn_sim.rnn_variants import BranchTraceRNN, BranchTraceRNNConfig
from bnn_sim.train import train_and_save

DEFAULT_OFFICIAL_MODEL_CONFIG = "configs/branch_elm_30_official.json"
DEFAULT_MAMBA_OFFICIAL_MODEL_CONFIG = "configs/branch_mamba_official_45x64.json"
MODEL_KIND_CHOICES = [
    "baseline",
    "official",
    "mamba-official",
    "mamba-pytorch",
    "branch-mamba-pytorch",
    "branch-trace-rnn",
]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="bnn-sim")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("make-demo-data", help="write deterministic demo NPZ shards")
    demo.add_argument("--output", required=True)
    demo.add_argument("--samples", type=int, default=8)
    demo.add_argument("--shards", type=int, default=1)
    demo.add_argument("--time-steps", type=int, default=32)
    demo.add_argument("--input-dim", type=int, default=64)
    demo.add_argument("--seed", type=int, default=0)

    arbor_data = subparsers.add_parser(
        "make-arbor-hetero-data",
        help="write Arbor-backed heterogeneous cable-cell shards for distillation",
    )
    arbor_data.add_argument("--config", help="JSON config with make-arbor-hetero-data option names")
    arbor_data.add_argument("--output")
    arbor_data.add_argument("--samples", type=int, default=128)
    arbor_data.add_argument("--shards", type=int, default=1)
    arbor_data.add_argument("--splits", help="comma-separated NAME:SAMPLES pairs, e.g. train:128,val:32,test:32")
    arbor_data.add_argument("--shards-per-split", type=int, default=1)
    arbor_data.add_argument("--time-steps", type=int, default=6000)
    arbor_data.add_argument("--input-dim", type=int, default=1278)
    arbor_data.add_argument("--branches", type=int, default=45)
    arbor_data.add_argument("--integration-dt-ms", type=float, default=0.025)
    arbor_data.add_argument("--input-protocol", choices=["arbor_poisson", "neuronio_smoothed"], default="arbor_poisson")
    arbor_data.add_argument("--synapse-routing", choices=["branch", "distributed"], default="branch")
    arbor_data.add_argument("--distributed-sites-per-polarity", type=int)
    arbor_data.add_argument("--spike-label-mode", choices=["threshold_crossing", "local_peak"], default="threshold_crossing")
    arbor_data.add_argument("--spike-peak-threshold-mv", type=float, default=-25.0)
    arbor_data.add_argument("--base-event-rate-hz", type=float, default=2.0)
    arbor_data.add_argument("--burst-probability", type=float, default=0.015)
    arbor_data.add_argument("--burst-event-rate-hz", type=float, default=35.0)
    arbor_data.add_argument("--arbor-gpu-id", type=int, default=0, help="GPU id for Arbor context; ignored with --cpu-only")
    arbor_data.add_argument("--cpu-only", action="store_true", help="use Arbor CPU context instead of CUDA")
    arbor_data.add_argument("--arbor-threads", type=int, default=1)
    arbor_data.add_argument("--sim-batch-size", type=int, default=32, help="independent cells per Arbor simulation batch")
    arbor_data.add_argument("--seed", type=int, default=0)
    arbor_data.add_argument(
        "--param-overrides",
        help="JSON object or path to a JSON object overriding Arbor heterogeneity parameters for all samples",
    )
    arbor_data.add_argument("--report", help="optional path for a copy of the generation report")

    convert = subparsers.add_parser("convert-neuronio", help="convert raw NeuronIO pickle files to deterministic shards")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--shard-size", type=int, default=128)
    convert.add_argument("--window-size", type=int)
    convert.add_argument("--window-stride", type=int)
    convert.add_argument("--ignore-start", type=int, default=0)

    repack = subparsers.add_parser("repack-shards", help="repack shards as sliceable .npy arrays")
    repack.add_argument("--input", required=True)
    repack.add_argument("--output", required=True)
    repack.add_argument("--input-dtype", choices=["int8", "float32"], default="int8")

    preset_list = subparsers.add_parser("list-presets", help="list named train/evaluate presets")

    profile = subparsers.add_parser("benchmark-profile", help="print a NEURO-BENCH axis/default profile")
    profile.add_argument("--name", default="paper-default", choices=list_benchmark_profiles())
    profile.add_argument("--plan", action="store_true", help="print the expanded axis-composition plan")
    profile.add_argument("--cost", action="store_true", help="print only the local capacity estimate")

    dataset_schema = subparsers.add_parser(
        "dataset-schema",
        help="print the v1 dataset manifest template or diagnostic suite catalog",
    )
    dataset_schema.add_argument("--kind", choices=["manifest", "diagnostics"], default="manifest")

    benchmark_rows = subparsers.add_parser(
        "benchmark-rows",
        help="write dataframe-first v1 benchmark diagnostic rows",
    )
    benchmark_rows.add_argument("--mode", choices=["teacher-qc", "model", "intervention"], default="teacher-qc")
    benchmark_rows.add_argument("--dataset-root", required=True)
    benchmark_rows.add_argument("--selector", choices=["val", "intervention"], default="val")
    benchmark_rows.add_argument("--intervention", default="event_dropout")
    benchmark_rows.add_argument("--pair-path", help="paired intervention .npz path, for pilot artifacts outside a v1 layout")
    benchmark_rows.add_argument("--output", required=True)
    benchmark_rows.add_argument("--summary-output", help="optional compact benchmark summary path (.md or .json)")
    benchmark_rows.add_argument("--suites", default="teacher-qc")
    benchmark_rows.add_argument("--morphology-id", default="unknown")
    benchmark_rows.add_argument("--teacher-id", default="teacher")
    benchmark_rows.add_argument("--max-samples", type=int)
    benchmark_rows.add_argument("--cache-shards", type=int, default=1)
    benchmark_rows.add_argument("--ignore-start", type=int, default=500)
    benchmark_rows.add_argument("--soma-scale", type=float, default=0.1)
    benchmark_rows.add_argument("--soma-bias-mv", type=float, default=-67.7)
    benchmark_rows.add_argument("--input-dim", type=int, default=1278)
    benchmark_rows.add_argument("--memory-units", type=int, default=30)
    benchmark_rows.add_argument("--branches", type=int, default=32)
    benchmark_rows.add_argument("--model-kind", choices=MODEL_KIND_CHOICES, default="baseline")
    benchmark_rows.add_argument("--model-config", default=DEFAULT_OFFICIAL_MODEL_CONFIG)
    benchmark_rows.add_argument("--checkpoint")
    benchmark_rows.add_argument("--device", default="cpu")
    benchmark_rows.add_argument("--batch-size", type=int, default=4)
    benchmark_rows.add_argument("--model-id", default="model")
    benchmark_rows.add_argument("--calibrated-spike-threshold", type=float)
    benchmark_rows.add_argument("--compare-model", action="store_true", help="include model response rows for intervention mode")

    benchmark_v1 = subparsers.add_parser(
        "benchmark-v1",
        help="run the promoted v1 benchmark bundle on val plus paired interventions",
    )
    benchmark_v1.add_argument("--dataset-root", required=True)
    benchmark_v1.add_argument("--output-dir", required=True)
    benchmark_v1.add_argument("--suite", choices=["teacher-qc", "paper-summary", "full"], default="paper-summary")
    benchmark_v1.add_argument("--intervention-pair-glob", action="append", default=[])
    benchmark_v1.add_argument("--morphology-id", default="unknown")
    benchmark_v1.add_argument("--teacher-id", default="teacher")
    benchmark_v1.add_argument("--max-val-samples", type=int)
    benchmark_v1.add_argument("--max-intervention-samples", type=int)
    benchmark_v1.add_argument("--cache-shards", type=int, default=1)
    benchmark_v1.add_argument("--ignore-start", type=int, default=500)
    benchmark_v1.add_argument("--soma-scale", type=float, default=0.1)
    benchmark_v1.add_argument("--soma-bias-mv", type=float, default=-67.7)
    benchmark_v1.add_argument("--checkpoint")
    benchmark_v1.add_argument("--device", default="cpu")
    benchmark_v1.add_argument("--batch-size", type=int, default=4)
    benchmark_v1.add_argument("--model-id", default="model")
    benchmark_v1.add_argument("--calibrated-spike-threshold", type=float)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a model with the corrected full-trace metric path")
    evaluate.add_argument("--preset", choices=sorted(list_presets()["evaluate"]))
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--input-dim", type=int, default=1278)
    evaluate.add_argument("--memory-units", type=int, default=30)
    evaluate.add_argument("--branches", type=int, default=32)
    evaluate.add_argument("--model-kind", choices=MODEL_KIND_CHOICES, default="baseline", help="use 'official' for the paper-style model, 'mamba-official' for the official mamba-ssm backend, or 'branch-trace-rnn' for the RNN comparison path; 'baseline' is legacy/debug")
    evaluate.add_argument("--model-config", default=DEFAULT_OFFICIAL_MODEL_CONFIG)
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--window-size", type=int)
    evaluate.add_argument("--window-stride", type=int)
    evaluate.add_argument("--ignore-start", type=int, default=0)
    evaluate.add_argument("--cache-shards", type=int, default=1)
    evaluate.add_argument("--soma-units", choices=["millivolts", "normalized"], default="millivolts")
    evaluate.add_argument("--metric-ignore-start", type=int, default=0)
    evaluate.add_argument("--metric-mask-mode", choices=["ignore-start", "official-overlap"], default="ignore-start", help=argparse.SUPPRESS)
    evaluate.add_argument("--metric-stitch-burn-in", type=int, default=150, help=argparse.SUPPRESS)
    evaluate.add_argument("--soma-affine-calibration", action="store_true")
    evaluate.add_argument("--pin-memory", action="store_true")
    evaluate.add_argument("--blocking-transfer", action="store_false", dest="non_blocking")
    evaluate.add_argument("--registry", help="append a one-line JSONL record; defaults to <output-dir>/experiment-registry.jsonl")
    evaluate.add_argument("--no-registry", action="store_const", const="off", dest="registry")
    evaluate.set_defaults(non_blocking=True)

    diagnose = subparsers.add_parser("diagnose", help="write biology-oriented surrogate fidelity diagnostics")
    diagnose.add_argument("--preset", choices=sorted(list_presets()["evaluate"]))
    diagnose.add_argument("--data", required=True)
    diagnose.add_argument("--output-dir", required=True)
    diagnose.add_argument("--input-dim", type=int, default=1278)
    diagnose.add_argument("--memory-units", type=int, default=30)
    diagnose.add_argument("--branches", type=int, default=32)
    diagnose.add_argument("--model-kind", choices=MODEL_KIND_CHOICES, default="baseline")
    diagnose.add_argument("--model-config", default=DEFAULT_OFFICIAL_MODEL_CONFIG)
    diagnose.add_argument("--batch-size", type=int, default=8)
    diagnose.add_argument("--seed", type=int, default=0)
    diagnose.add_argument("--device", default="cpu")
    diagnose.add_argument("--checkpoint")
    diagnose.add_argument("--window-size", type=int)
    diagnose.add_argument("--window-stride", type=int)
    diagnose.add_argument("--ignore-start", type=int, default=0)
    diagnose.add_argument("--cache-shards", type=int, default=1)
    diagnose.add_argument("--metric-ignore-start", type=int, default=0)
    diagnose.add_argument("--metric-mask-mode", choices=["ignore-start", "official-overlap"], default="ignore-start", help=argparse.SUPPRESS)
    diagnose.add_argument("--metric-stitch-burn-in", type=int, default=150, help=argparse.SUPPRESS)
    diagnose.add_argument("--soma-affine-calibration", action="store_true")
    diagnose.add_argument("--max-samples", type=int)
    diagnose.add_argument("--pin-memory", action="store_true")
    diagnose.add_argument("--blocking-transfer", action="store_false", dest="non_blocking")
    diagnose.set_defaults(non_blocking=True)

    train = subparsers.add_parser("train", help="train a model with the current experiment workflow")
    train.add_argument("--preset", choices=sorted(list_presets()["train"]))
    train.add_argument("--data", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--metrics", required=True)
    train.add_argument("--input-dim", type=int, default=1278)
    train.add_argument("--memory-units", type=int, default=30)
    train.add_argument("--branches", type=int, default=32)
    train.add_argument("--model-kind", choices=MODEL_KIND_CHOICES, default="baseline", help="use 'official' for the paper-style model, 'mamba-official' for the official mamba-ssm backend, or 'branch-trace-rnn' for the RNN comparison path; 'baseline' is legacy/debug")
    train.add_argument("--model-config", default=DEFAULT_OFFICIAL_MODEL_CONFIG)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=5e-4)
    train.add_argument("--burn-in", type=int, default=0)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cpu")
    train.add_argument("--window-size", type=int)
    train.add_argument("--window-stride", type=int)
    train.add_argument("--ignore-start", type=int, default=0)
    train.add_argument(
        "--window-sampling",
        choices=["deterministic", "random-full-trace", "official-full-trace"],
        default="random-full-trace",
    )
    train.add_argument("--epoch-samples", type=int)
    train.add_argument("--shard-reuse-batches", type=int, default=1)
    train.add_argument("--file-load-fraction", type=float, default=0.3)
    train.add_argument("--full-trace-length", type=int)
    train.add_argument("--shuffle-mode", choices=["sample", "shard", "none"], default="shard")
    train.add_argument("--cache-shards", type=int, default=1)
    train.add_argument("--val-data")
    train.add_argument("--val-batch-size", type=int)
    train.add_argument("--val-cache-shards", type=int, default=1)
    train.add_argument("--val-window-size", type=int)
    train.add_argument("--val-window-stride", type=int)
    train.add_argument("--val-ignore-start", type=int, default=0)
    train.add_argument("--val-soma-units", choices=["millivolts", "normalized"], default="millivolts")
    train.add_argument("--val-metric-ignore-start", type=int, default=500)
    train.add_argument("--val-metric-mask-mode", choices=["ignore-start", "official-overlap"], default="ignore-start", help=argparse.SUPPRESS)
    train.add_argument("--val-metric-stitch-burn-in", type=int, default=150, help=argparse.SUPPRESS)
    train.add_argument("--val-soma-affine-calibration", action="store_true", default=True)
    train.add_argument("--no-val-soma-affine-calibration", action="store_false", dest="val_soma_affine_calibration")
    train.add_argument("--best-checkpoint")
    train.add_argument("--max-train-batches", type=int)
    train.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    train.add_argument("--lr-schedule-steps", type=int)
    train.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--l1-lambda", type=float, default=0.0)
    train.add_argument("--spike-loss-weight", type=float, default=0.5)
    train.add_argument("--soma-loss-weight", type=float, default=0.5)
    train.add_argument("--sparse-soma-loss-weight", type=float, default=0.0)
    train.add_argument("--sparse-soma-high-voltage-quantile", type=float, default=0.9)
    train.add_argument("--sparse-soma-high-dvdt-quantile", type=float, default=0.9)
    train.add_argument("--sparse-soma-input-event-quantile", type=float, default=0.95)
    train.add_argument("--sparse-soma-spike-window", type=int, default=5)
    train.add_argument("--sparse-soma-post-event-window", type=int, default=5)
    train.add_argument("--soma-slope-loss-weight", type=float, default=0.0)
    train.add_argument("--grad-clip-norm", type=float, default=0.0)
    train.add_argument("--train-update-log-interval", type=int, default=0)
    train.add_argument("--train-update-log")
    train.add_argument("--prefetch-batches", type=int, default=0)
    train.add_argument("--pin-memory", action="store_true")
    train.add_argument("--blocking-transfer", action="store_false", dest="non_blocking")
    train.add_argument("--registry", help="append a one-line JSONL record; defaults to <metrics-dir>/experiment-registry.jsonl")
    train.add_argument("--no-registry", action="store_const", const="off", dest="registry")
    train.set_defaults(non_blocking=True)

    benchmark = subparsers.add_parser("benchmark", help="time model forward passes on random input")
    benchmark.add_argument("--input-dim", type=int, default=1278)
    benchmark.add_argument("--memory-units", type=int, default=30)
    benchmark.add_argument("--branches", type=int, default=32)
    benchmark.add_argument("--model-kind", choices=MODEL_KIND_CHOICES, default="baseline", help="use 'official' for the paper-style model, 'mamba-official' for the official mamba-ssm backend, or 'branch-trace-rnn' for the RNN comparison path; 'baseline' is legacy/debug")
    benchmark.add_argument("--model-config", default=DEFAULT_OFFICIAL_MODEL_CONFIG)
    benchmark.add_argument("--batch-size", type=int, default=1)
    benchmark.add_argument("--time-steps", type=int, default=500)
    benchmark.add_argument("--runs", type=int, default=10)
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--compile", action="store_true", dest="compile_model")
    benchmark.add_argument("--output")

    inference_benchmark = subparsers.add_parser(
        "inference-benchmark",
        help="benchmark deployment-oriented inference throughput across batch and sequence scales",
    )
    inference_benchmark.add_argument("--input-dim", type=int, default=1278)
    inference_benchmark.add_argument("--memory-units", type=int, default=30)
    inference_benchmark.add_argument("--branches", type=int, default=32)
    inference_benchmark.add_argument("--model-kind", choices=MODEL_KIND_CHOICES, default="baseline")
    inference_benchmark.add_argument("--model-config", default=DEFAULT_OFFICIAL_MODEL_CONFIG)
    inference_benchmark.add_argument("--checkpoint")
    inference_benchmark.add_argument("--batch-sizes", default="1,8,32,128")
    inference_benchmark.add_argument("--time-steps", default="500,1000,6000")
    inference_benchmark.add_argument("--runs", type=int, default=20)
    inference_benchmark.add_argument("--warmup-runs", type=int, default=5)
    inference_benchmark.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="float32")
    inference_benchmark.add_argument("--device", default="cpu")
    inference_benchmark.add_argument("--compile", action="store_true", dest="compile_model")
    inference_benchmark.add_argument("--accuracy-metrics")
    inference_benchmark.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    explicit_dests = _explicit_option_dests(parser, argv, args.command)
    if args.command in {"train", "evaluate"}:
        apply_preset(args.command, getattr(args, "preset", None), args, explicit_dests)
    elif args.command == "diagnose":
        apply_preset("evaluate", getattr(args, "preset", None), args, explicit_dests)

    if args.command == "list-presets":
        print(json.dumps(list_presets(), indent=2, sort_keys=True))
        return 0

    if args.command == "benchmark-profile":
        if args.cost:
            payload = estimate_profile_cost(args.name)
        elif args.plan:
            payload = build_axis_plan(args.name)
        else:
            payload = get_benchmark_profile(args.name).to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "dataset-schema":
        if args.kind == "manifest":
            payload = build_v1_dataset_manifest_template()
        else:
            payload = build_diagnostic_suite_catalog()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "benchmark-rows":
        try:
            layout = V1DatasetLayout(args.dataset_root)
            selector = (
                layout.default_validation()
                if args.selector == "val"
                else layout.intervention(args.intervention)
            )
            context = DiagnosticContext(
                suites=tuple(item.strip() for item in args.suites.split(",") if item.strip()),
                model_id="teacher" if args.mode == "teacher-qc" else args.model_id,
                teacher_id=args.teacher_id,
                morphology_id=args.morphology_id,
                split=selector.split or "",
                condition=selector.condition,
                intervention=selector.intervention,
                ignore_start=args.ignore_start,
                soma_scale=args.soma_scale,
                soma_bias_mv=args.soma_bias_mv,
                calibrated_spike_threshold=args.calibrated_spike_threshold,
            )
            model_fn = None
            if args.mode in {"model", "intervention"} and (
                args.mode == "model" or args.compare_model or args.checkpoint
            ):
                if args.checkpoint:
                    model = _load_checkpoint_for_evaluate(args)
                else:
                    model = _build_model_from_args(args)
                model_fn = _numpy_model_predict_fn(model, device=args.device, batch_size=args.batch_size)

            if args.mode == "teacher-qc":
                rows = run_teacher_qc_on_selector(
                    selector,
                    context=context,
                    max_samples=args.max_samples,
                    output=args.output,
                    cache_shards=args.cache_shards,
                )
            elif args.mode == "model":
                from bnn_sim.benchmark.runner import run_model_diagnostics_on_selector

                rows = run_model_diagnostics_on_selector(
                    model_fn,
                    selector,
                    context=context,
                    max_samples=args.max_samples,
                    output=args.output,
                    cache_shards=args.cache_shards,
                )
            else:
                if args.pair_path:
                    rows = run_paired_intervention_on_npz(
                        args.pair_path,
                        model_fn=model_fn,
                        context=context,
                        max_samples=args.max_samples,
                        output=args.output,
                    )
                else:
                    rows = run_paired_intervention_on_selector(
                        selector,
                        model_fn=model_fn,
                        context=context,
                        max_samples=args.max_samples,
                        output=args.output,
                    )
        except (FileNotFoundError, ValueError) as error:
            print(str(error))
            return 2
        if args.summary_output:
            try:
                write_diagnostic_summary(rows, args.summary_output)
            except ValueError as error:
                print(str(error))
                return 2
        print(json.dumps({"rows": len(rows), "output": args.output}, indent=2, sort_keys=True))
        return 0

    if args.command == "benchmark-v1":
        try:
            model_fn = None
            if args.checkpoint:
                model = _load_checkpoint_for_evaluate(args)
                model_fn = _numpy_model_predict_fn(model, device=args.device, batch_size=args.batch_size)
            pair_paths = _expand_globs(args.intervention_pair_glob)
            manifest = run_v1_benchmark_bundle(
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                base_context=DiagnosticContext(
                    model_id=args.model_id,
                    teacher_id=args.teacher_id,
                    morphology_id=args.morphology_id,
                    ignore_start=args.ignore_start,
                    soma_scale=args.soma_scale,
                    soma_bias_mv=args.soma_bias_mv,
                    calibrated_spike_threshold=args.calibrated_spike_threshold,
                ),
                model_fn=model_fn,
                model_id=args.model_id if model_fn is not None else None,
                intervention_pair_paths=pair_paths,
                suite=args.suite,
                max_val_samples=args.max_val_samples,
                max_intervention_samples=args.max_intervention_samples,
                cache_shards=args.cache_shards,
            )
        except (FileNotFoundError, ValueError) as error:
            print(str(error))
            return 2
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    if args.command == "make-demo-data":
        write_demo_shards(
            args.output,
            shard_count=args.shards,
            samples_per_shard=args.samples,
            time_steps=args.time_steps,
            input_dim=args.input_dim,
            seed=args.seed,
        )
        return 0

    if args.command == "make-arbor-hetero-data":
        try:
            _apply_arbor_data_config(args, explicit_dests)
            if not args.output:
                raise ValueError("--output is required unless provided by --config")
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
                base_event_rate_hz=args.base_event_rate_hz,
                burst_probability=args.burst_probability,
                burst_event_rate_hz=args.burst_event_rate_hz,
                arbor_threads=args.arbor_threads,
                arbor_gpu_id=_resolve_arbor_gpu_id(args, explicit_dests),
                sim_batch_size=args.sim_batch_size,
            )
            if args.splits:
                report = write_arbor_hetero_splits(
                    args.output,
                    split_samples=parse_split_samples(args.splits),
                    shards_per_split=args.shards_per_split,
                    seed=args.seed,
                    config=config,
                    param_overrides=_parse_param_overrides(args.param_overrides),
                )
            else:
                report = write_arbor_hetero_dataset(
                    args.output,
                    samples=args.samples,
                    shards=args.shards,
                    seed=args.seed,
                    config=config,
                    param_overrides=_parse_param_overrides(args.param_overrides),
                )
        except ValueError as error:
            print(str(error))
            return 2
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.command == "convert-neuronio":
        try:
            written = convert_neuronio_pickles(
                args.input,
                args.output,
                shard_size=args.shard_size,
                window_size=args.window_size,
                window_stride=args.window_stride,
                ignore_start=args.ignore_start,
            )
        except (FileNotFoundError, ValueError) as error:
            print(str(error))
            return 2
        print(f"wrote_shards={len(written)}")
        return 0

    if args.command == "repack-shards":
        try:
            input_dtype = np.float32 if args.input_dtype == "float32" else np.int8
            written = repack_shards_as_npy(args.input, args.output, input_dtype=input_dtype)
        except (FileNotFoundError, ValueError) as error:
            print(str(error))
            return 2
        print(f"wrote_shards={len(written)}")
        return 0

    if args.command == "evaluate":
        torch.manual_seed(args.seed)
        dataset = _build_dataset(
            args.data,
            window_size=args.window_size,
            window_stride=args.window_stride,
            ignore_start=args.ignore_start,
            cache_shards=args.cache_shards,
        )
        if args.checkpoint:
            model = _load_checkpoint_for_evaluate(args)
        else:
            model = _build_model_from_args(args)
        metrics = evaluate_dataset(
            model,
            dataset,
            batch_size=args.batch_size,
            device=args.device,
            soma_units=args.soma_units,
            ignore_start=args.metric_ignore_start,
            mask_mode=args.metric_mask_mode,
            stitch_burn_in=args.metric_stitch_burn_in,
            soma_affine_calibration=args.soma_affine_calibration,
            pin_memory=args.pin_memory,
            non_blocking=args.non_blocking,
        )
        write_metrics(metrics, args.output)
        registry_path = resolve_registry_path("evaluate", registry=args.registry, output_path=args.output)
        if registry_path is not None:
            append_registry_entry(
                registry_path,
                build_registry_entry(
                    command="evaluate",
                    preset=args.preset,
                    args=args,
                    metrics_path=args.output,
                    metrics=metrics,
                    checkpoint_path=args.checkpoint,
                ),
            )
        return 0

    if args.command == "diagnose":
        torch.manual_seed(args.seed)
        dataset = _build_dataset(
            args.data,
            window_size=args.window_size,
            window_stride=args.window_stride,
            ignore_start=args.ignore_start,
            cache_shards=args.cache_shards,
        )
        if args.checkpoint:
            model = _load_checkpoint_for_evaluate(args)
        else:
            model = _build_model_from_args(args)
        diagnose_dataset(
            model,
            dataset,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
            ignore_start=args.metric_ignore_start,
            mask_mode=args.metric_mask_mode,
            stitch_burn_in=args.metric_stitch_burn_in,
            soma_affine_calibration=args.soma_affine_calibration,
            max_samples=args.max_samples,
            pin_memory=args.pin_memory,
            non_blocking=args.non_blocking,
        )
        return 0

    if args.command == "train":
        torch.manual_seed(args.seed)
        dataset = _build_dataset(
            args.data,
            shuffle=args.shuffle_mode != "none",
            shuffle_mode="shard" if args.shuffle_mode == "none" else args.shuffle_mode,
            seed=args.seed,
            window_size=args.window_size,
            window_stride=args.window_stride,
            ignore_start=args.ignore_start,
            cache_shards=args.cache_shards,
            window_sampling=args.window_sampling,
            epoch_samples=args.epoch_samples,
            batch_size=args.batch_size,
            shard_reuse_batches=args.shard_reuse_batches,
            file_load_fraction=args.file_load_fraction,
            sequence_length=args.full_trace_length,
        )
        validation_dataset = None
        if args.val_data:
            validation_dataset = _build_dataset(
                args.val_data,
                window_size=args.val_window_size,
                window_stride=args.val_window_stride,
                ignore_start=args.val_ignore_start,
                cache_shards=args.val_cache_shards,
            )
        model = _build_model_from_args(args)
        best_checkpoint = args.best_checkpoint
        if validation_dataset is not None and best_checkpoint is None:
            checkpoint_path = Path(args.checkpoint)
            best_checkpoint = str(checkpoint_path.with_name(f"{checkpoint_path.stem}-best{checkpoint_path.suffix}"))
        metrics = train_and_save(
            model,
            dataset,
            checkpoint_path=args.checkpoint,
            metrics_path=args.metrics,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            burn_in=args.burn_in,
            device=args.device,
            seed=args.seed,
            validation_dataset=validation_dataset,
            validation_batch_size=args.val_batch_size,
            best_checkpoint_path=best_checkpoint,
            validation_soma_units=args.val_soma_units,
            validation_metric_ignore_start=args.val_metric_ignore_start,
            validation_metric_mask_mode=args.val_metric_mask_mode,
            validation_metric_stitch_burn_in=args.val_metric_stitch_burn_in,
            validation_soma_affine_calibration=args.val_soma_affine_calibration,
            max_train_batches=args.max_train_batches,
            lr_schedule=args.lr_schedule,
            lr_schedule_steps=args.lr_schedule_steps,
            optimizer_name=args.optimizer,
            weight_decay=args.weight_decay,
            l1_lambda=args.l1_lambda,
            spike_loss_weight=args.spike_loss_weight,
            soma_loss_weight=args.soma_loss_weight,
            sparse_soma_loss_weight=args.sparse_soma_loss_weight,
            sparse_soma_high_voltage_quantile=args.sparse_soma_high_voltage_quantile,
            sparse_soma_high_dvdt_quantile=args.sparse_soma_high_dvdt_quantile,
            sparse_soma_input_event_quantile=args.sparse_soma_input_event_quantile,
            sparse_soma_spike_window=args.sparse_soma_spike_window,
            sparse_soma_post_event_window=args.sparse_soma_post_event_window,
            soma_slope_loss_weight=args.soma_slope_loss_weight,
            grad_clip_norm=args.grad_clip_norm,
            update_log_interval=args.train_update_log_interval,
            update_log_path=args.train_update_log,
            prefetch_batches=args.prefetch_batches,
            pin_memory=args.pin_memory,
            non_blocking=args.non_blocking,
        )
        registry_path = resolve_registry_path("train", registry=args.registry, output_path=args.metrics)
        if registry_path is not None:
            append_registry_entry(
                registry_path,
                build_registry_entry(
                    command="train",
                    preset=args.preset,
                    args=args,
                    metrics_path=args.metrics,
                    metrics=metrics,
                    checkpoint_path=args.checkpoint,
                ),
            )
        return 0

    if args.command == "benchmark":
        torch.manual_seed(0)
        model = _build_model_from_args(args).to(args.device).eval()
        if args.compile_model:
            model = torch.compile(model)
        input_dim = int(getattr(model, "num_input", args.input_dim))
        x = torch.randn(args.batch_size, args.time_steps, input_dim, device=args.device)
        with torch.inference_mode():
            model(x)
            start = time.perf_counter()
            for _ in range(args.runs):
                model(x)
            elapsed = time.perf_counter() - start
        report = {
            "batch_size": args.batch_size,
            "input_dim": input_dim,
            "model_kind": args.model_kind,
            "time_steps": args.time_steps,
            "runs": args.runs,
            "mean_forward_seconds": elapsed / args.runs,
            "compiled": bool(args.compile_model),
            "device": args.device,
        }
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            print(f"mean_forward_seconds={report['mean_forward_seconds']:.6f}")
        return 0

    if args.command == "inference-benchmark":
        torch.manual_seed(0)
        if args.checkpoint:
            model, _ = load_checkpoint(args.checkpoint, map_location=args.device)
        else:
            model = _build_model_from_args(args)
        input_dim = int(getattr(model, "num_input", args.input_dim))
        try:
            batch_sizes = parse_int_list(args.batch_sizes)
            time_steps = parse_int_list(args.time_steps)
            report = benchmark_inference_matrix(
                model,
                batch_sizes=batch_sizes,
                time_steps=time_steps,
                input_dim=input_dim,
                device=args.device,
                precision=args.precision,
                warmup_runs=args.warmup_runs,
                runs=args.runs,
                compile_model=args.compile_model,
                accuracy_metrics_path=args.accuracy_metrics,
            )
        except (FileNotFoundError, ValueError) as error:
            print(str(error))
            return 2
        write_inference_benchmark(report, args.output)
        return 0

    raise AssertionError(f"unhandled command {args.command}")


def _build_dataset(
    data_path: str,
    *,
    shuffle: bool = False,
    shuffle_mode: str = "sample",
    seed: int = 0,
    window_size: int | None = None,
    window_stride: int | None = None,
    ignore_start: int = 0,
    cache_shards: int = 0,
    window_sampling: str = "deterministic",
    epoch_samples: int | None = None,
    batch_size: int = 8,
    shard_reuse_batches: int = 1,
    file_load_fraction: float = 0.3,
    sequence_length: int | None = None,
):
    dataset = ShardedNeuronIODataset(
        data_path,
        shuffle=shuffle,
        shuffle_mode=shuffle_mode,  # type: ignore[arg-type]
        seed=seed,
        cache_shards=cache_shards,
    )
    if window_size is None:
        return dataset
    if window_sampling == "random-full-trace":
        return RandomFullTraceWindowDataset(
            dataset,
            window_size=window_size,
            start_offset=ignore_start,
            samples_per_epoch=epoch_samples,
            batch_size=batch_size,
            shard_reuse_batches=shard_reuse_batches,
            sequence_length=sequence_length,
            seed=seed,
        )
    if window_sampling == "official-full-trace":
        return OfficialStyleFullTraceWindowDataset(
            dataset,
            window_size=window_size,
            start_offset=ignore_start,
            samples_per_epoch=epoch_samples,
            batch_size=batch_size,
            file_load_fraction=file_load_fraction,
            sequence_length=sequence_length,
            seed=seed,
        )
    if window_sampling != "deterministic":
        raise ValueError("window_sampling must be 'deterministic', 'random-full-trace', or 'official-full-trace'")
    return DeterministicWindowDataset(
        dataset,
        window_size=window_size,
        stride=window_stride,
        start_offset=ignore_start,
    )


def _build_model_from_args(args):
    if args.model_kind == "official":
        return create_official_elm(load_official_elm_config(args.model_config))
    if args.model_kind == "branch-trace-rnn":
        return BranchTraceRNN(BranchTraceRNNConfig(**load_official_elm_config(args.model_config)))
    if args.model_kind == "mamba-official":
        from bnn_sim.mamba_official import BranchOfficialMamba, BranchOfficialMambaConfig

        config_path = args.model_config if args.model_config != DEFAULT_OFFICIAL_MODEL_CONFIG else DEFAULT_MAMBA_OFFICIAL_MODEL_CONFIG
        return BranchOfficialMamba(BranchOfficialMambaConfig(**load_official_elm_config(config_path)))
    if args.model_kind == "branch-mamba-pytorch":
        from bnn_sim.mamba_official import BranchOfficialMambaConfig, BranchPyTorchMamba

        config_path = args.model_config if args.model_config != DEFAULT_OFFICIAL_MODEL_CONFIG else DEFAULT_MAMBA_OFFICIAL_MODEL_CONFIG
        return BranchPyTorchMamba(BranchOfficialMambaConfig(**load_official_elm_config(config_path)))
    if args.model_kind == "mamba-pytorch":
        from bnn_sim.mamba_pytorch import SimpleMambaModel

        config_path = Path(args.model_config)
        if args.model_config != DEFAULT_OFFICIAL_MODEL_CONFIG and config_path.exists():
            return SimpleMambaModel(**load_official_elm_config(config_path))
        return SimpleMambaModel(input_dim=args.input_dim)
    config = BranchELMConfig(input_dim=args.input_dim, memory_units=args.memory_units, num_branches=args.branches)
    return BranchELM(config)


def _load_checkpoint_for_evaluate(args):
    try:
        model, _ = load_checkpoint(args.checkpoint, map_location=args.device)
        return model
    except (KeyError, ValueError):
        if args.model_kind != "official":
            raise
        model = create_official_elm(load_official_elm_config(args.model_config))
        state_dict = load_official_state_dict(args.checkpoint, map_location=args.device)
        model.load_state_dict(state_dict, strict=True)
        return model


def _numpy_model_predict_fn(model, *, device: str, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model = model.to(device).eval()

    def predict(inputs: np.ndarray) -> np.ndarray:
        outputs = []
        with torch.inference_mode():
            for start in range(0, inputs.shape[0], batch_size):
                batch = torch.as_tensor(inputs[start : start + batch_size], dtype=torch.float32, device=device)
                outputs.append(model(batch).detach().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)

    return predict


def _expand_globs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"intervention pair glob matched no files: {pattern}")
        paths.extend(Path(match) for match in matches)
    return paths


def _apply_arbor_data_config(args, explicit_dests: set[str]) -> None:
    config_path = getattr(args, "config", None)
    if not config_path:
        return
    path = Path(config_path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"simulator config not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid simulator config JSON in {path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("simulator config must be a JSON object")
    allowed = {
        "output",
        "samples",
        "shards",
        "splits",
        "shards_per_split",
        "time_steps",
        "input_dim",
        "branches",
        "integration_dt_ms",
        "input_protocol",
        "synapse_routing",
        "distributed_sites_per_polarity",
        "spike_label_mode",
        "spike_peak_threshold_mv",
        "base_event_rate_hz",
        "burst_probability",
        "burst_event_rate_hz",
        "arbor_gpu_id",
        "cpu_only",
        "arbor_threads",
        "sim_batch_size",
        "seed",
        "param_overrides",
        "report",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"unknown simulator config keys: {', '.join(unknown)}")
    for key, value in config.items():
        if key not in explicit_dests:
            setattr(args, key, value)


def _parse_param_overrides(value: str | dict[str, object] | None) -> dict[str, float] | None:
    if value is None:
        return None
    raw: object
    if isinstance(value, dict):
        raw = value
    else:
        text = str(value)
        path = Path(text)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("param_overrides must be a JSON object")
    overrides: dict[str, float] = {}
    for key, item in raw.items():
        if not isinstance(key, str):
            raise ValueError("param_overrides keys must be strings")
        try:
            overrides[key] = float(item)
        except (TypeError, ValueError) as error:
            raise ValueError(f"param_overrides value for {key!r} must be numeric") from error
    return overrides or None


def _resolve_arbor_gpu_id(args, explicit_dests: set[str]) -> int | None:
    if args.cpu_only:
        return None
    if "arbor_gpu_id" in explicit_dests:
        return args.arbor_gpu_id
    try:
        import arbor as A
    except ModuleNotFoundError:
        return args.arbor_gpu_id
    if A.config().get("gpu") != "cuda":
        return None
    return args.arbor_gpu_id


def _explicit_option_dests(parser: argparse.ArgumentParser, argv: list[str], command: str) -> set[str]:
    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for option_string in getattr(action, "option_strings", ()):
            option_to_dest[option_string] = action.dest
    subparser_action = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    for action in subparser_action.choices[command]._actions:
        for option_string in getattr(action, "option_strings", ()):
            option_to_dest[option_string] = action.dest

    explicit_dests: set[str] = set()
    for token in argv:
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest is not None:
            explicit_dests.add(dest)
    return explicit_dests


if __name__ == "__main__":
    raise SystemExit(main())
