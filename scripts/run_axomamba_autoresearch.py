#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AXOBENCH_ROOT = PROJECT_ROOT.parent / "AxoBench"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(AXOBENCH_ROOT / "src"))

from axobench.benchmark.iteration_evaluator import (  # noqa: E402
    EvaluationConfig,
    create_model_evaluation_cache,
    evaluate_model_against_cache,
    load_model_evaluation_cache,
    write_evaluation_report,
    write_model_comparison,
    write_model_evaluation_cache,
)
from axosim.autoresearch import (  # noqa: E402
    AXOMAMBA_DEFAULT_BURN_IN,
    benchmark_model_inference,
    benchmark_model_training_step,
    FrozenWindowDataset,
    PanelCandidate,
    TrainingObjective,
    create_frozen_window_schedule,
    freeze_named_parameters,
    initialize_morphology_parameter_rows_from_mean,
    initialize_named_parameters,
    load_named_buffer_artifact,
    offset_named_parameters,
    practical_candidate_gate,
    resolve_presentation_budget,
    set_trainable_parameter_names,
    train_candidate_panel,
)
from axosim.axobench_iteration import make_checkpoint_predictor  # noqa: E402
from axosim.axomamba import AxoMamba, AxoMambaConfig  # noqa: E402
from axosim.data import ShardedNeuronIODataset  # noqa: E402
from axosim.checkpoint import load_checkpoint  # noqa: E402
from axosim.structured import (  # noqa: E402
    average_repeated_module_state,
    balanced_proxy_compress_mamba_state,
    factorize_morphology_synapse_gain_state,
    initialize_factorized_linears_from_state,
    initialize_shared_tcn_adapters_from_pretrained_state,
    share_morphology_synapse_gain_state,
)


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "v1-neuronio-like-gpu-20260611"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "axomamba.json"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs" / "axomamba-autoresearch"
DEFAULT_DIAGNOSTICS_ROOT = DEFAULT_RUN_ROOT / "diagnostics"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one manifest-defined AxoMamba training and diagnostics iteration.",
        allow_abbrev=False,
    )
    parser.add_argument("manifest", help="JSON iteration manifest")
    parser.add_argument("--long", action="store_true", help="force the 300k presentation budget")
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = resolve_plan(manifest, force_long=args.long)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    result = run_iteration(
        manifest,
        manifest_path=manifest_path,
        force_long=args.long,
        diagnostics_only=args.diagnostics_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def resolve_plan(manifest: dict[str, Any], *, force_long: bool) -> dict[str, Any]:
    iteration = int(manifest["iteration"])
    name = str(manifest["name"])
    candidates = manifest.get("candidates", [])
    if not candidates:
        raise ValueError("manifest candidates must not be empty")
    candidate_ids = [str(candidate["id"]) for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("manifest candidate ids must be unique")
    if manifest.get("smoke"):
        presentations = int(manifest.get("presentations", 0))
        if presentations <= 0 or presentations >= 120_000:
            raise ValueError("smoke presentations must be between 1 and 119999")
    elif "presentations" in manifest:
        raise ValueError("custom presentation budgets require smoke=true")
    else:
        presentations = resolve_presentation_budget(iteration=iteration, force_long=force_long)
    slug = f"iteration-{iteration:03d}-{name}"
    return {
        "iteration": iteration,
        "name": name,
        "slug": slug,
        "presentations": presentations,
        "long_run": presentations > 120_000,
        "smoke": bool(manifest.get("smoke", False)),
        "candidate_ids": candidate_ids,
        "prune_due": iteration > 0 and iteration % 10 == 0,
    }


def run_iteration(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    force_long: bool,
    diagnostics_only: bool,
) -> dict[str, Any]:
    plan = resolve_plan(manifest, force_long=force_long)
    training = dict(manifest.get("training", {}))
    diagnostics = dict(manifest.get("diagnostics", {}))
    dataset_root = _resolve_path(manifest.get("dataset_root", DEFAULT_DATASET_ROOT), manifest_path)
    run_root = _resolve_path(manifest.get("run_root", DEFAULT_RUN_ROOT), manifest_path) / plan["slug"]
    diagnostics_root = (
        _resolve_path(manifest.get("diagnostics_root", DEFAULT_DIAGNOSTICS_ROOT), manifest_path)
        / plan["slug"]
    )
    run_root.mkdir(parents=True, exist_ok=True)
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    seed = int(training.get("seed", 20260711))
    schedule_batch_size = int(training.get("schedule_batch_size", 64))
    shard_reuse_batches = int(training.get("shard_reuse_batches", 8))
    schedule_morphology_ids_value = training.get("schedule_morphology_ids")
    if schedule_morphology_ids_value is None:
        schedule_morphology_ids = None
        morphology_schedule_suffix = ""
    else:
        if not isinstance(schedule_morphology_ids_value, list):
            raise ValueError("schedule_morphology_ids must be a list")
        schedule_morphology_ids = [
            str(value) for value in schedule_morphology_ids_value
        ]
        if not schedule_morphology_ids:
            raise ValueError("schedule_morphology_ids must not be empty")
        morphology_schedule_hash = hashlib.sha256(
            json.dumps(sorted(schedule_morphology_ids)).encode("utf-8")
        ).hexdigest()[:10]
        morphology_schedule_suffix = f"-m{morphology_schedule_hash}"
    schedule_root = DEFAULT_RUN_ROOT / "schedules"
    schedule_path = schedule_root / (
        f"balanced-{plan['presentations']}-seed{seed}-b{schedule_batch_size}"
        f"-reuse{shard_reuse_batches}{morphology_schedule_suffix}.npz"
    )
    base_dataset = ShardedNeuronIODataset(dataset_root / "train", cache_shards=1)
    if not schedule_path.exists():
        create_frozen_window_schedule(
            base_dataset,
            schedule_path,
            presentations=plan["presentations"],
            schedule_batch_size=schedule_batch_size,
            shard_reuse_batches=shard_reuse_batches,
            window_size=int(training.get("window_size", 500)),
            start_offset=int(training.get("start_offset", 500)),
            sequence_length=int(training.get("sequence_length", 6000)),
            seed=seed,
            morphology_ids=schedule_morphology_ids,
        )
    base_schedule = FrozenWindowDataset(base_dataset, schedule_path)
    schedule = base_schedule

    candidate_records = _build_candidates(
        manifest["candidates"],
        training=training,
        run_root=run_root,
        manifest_path=manifest_path,
        seed=seed,
    )
    distillation_teacher = None
    distillation_teacher_record = None
    teacher_checkpoint_value = training.get("distillation_teacher_checkpoint")
    if teacher_checkpoint_value is not None:
        teacher_checkpoint = _resolve_path(teacher_checkpoint_value, manifest_path)
        distillation_teacher, teacher_metadata = load_checkpoint(
            teacher_checkpoint,
            map_location="cpu",
        )
        if not isinstance(distillation_teacher, AxoMamba):
            raise ValueError("distillation teacher must contain an AxoMamba model")
        distillation_teacher_record = {
            "path": str(teacher_checkpoint),
            "sha256": hashlib.sha256(teacher_checkpoint.read_bytes()).hexdigest(),
            "metadata": teacher_metadata,
        }
    training_results: dict[str, dict[str, Any]] = {}
    if diagnostics_only:
        missing = [
            str(record.checkpoint_path)
            for record in candidate_records
            if not Path(record.checkpoint_path).exists()
        ]
        if missing:
            raise FileNotFoundError(f"diagnostics-only checkpoints are missing: {missing}")
        for record in candidate_records:
            metrics_path = Path(record.metrics_path)
            training_results[record.candidate_id] = (
                json.loads(metrics_path.read_text(encoding="utf-8"))
                if metrics_path.exists()
                else {"candidate_id": record.candidate_id}
            )
    else:
        training_results = train_candidate_panel(
            candidate_records,
            schedule,
            objective=_objective_from_dict(training.get("objective", {})),
            device=str(training.get("device", "cuda")),
            burn_in=int(training.get("burn_in", AXOMAMBA_DEFAULT_BURN_IN)),
            seed=seed,
            prefetch_batches=int(
                training.get("prefetch_batches", min(shard_reuse_batches, 16))
            ),
            pin_memory=bool(training.get("pin_memory", False)),
            non_blocking=bool(training.get("non_blocking", True)),
            warmup_cuda=bool(training.get("warmup_cuda", True)),
            distillation_teacher=distillation_teacher,
        )
        for metrics in training_results.values():
            metrics["schedule_path"] = str(schedule_path.resolve())
        for record in candidate_records:
            training_results[record.candidate_id]["initial_parameter_values"] = getattr(
                record, "initial_parameter_values", {}
            )
            training_results[record.candidate_id]["initial_parameter_offsets"] = getattr(
                record, "initial_parameter_offsets", {}
            )
            training_results[record.candidate_id]["mean_initialized_morphology_rows"] = getattr(
                record, "mean_initialized_morphology_rows", None
            )
            training_results[record.candidate_id]["initialization"] = getattr(
                record, "initialization", "pretrained"
            )
            training_results[record.candidate_id]["frozen_parameter_names"] = getattr(
                record, "frozen_parameter_names", []
            )
            training_results[record.candidate_id]["trainable_parameter_names"] = getattr(
                record, "trainable_parameter_names", None
            )
            training_results[record.candidate_id]["buffer_artifacts"] = getattr(
                record, "buffer_artifacts", []
            )
            training_results[record.candidate_id]["initial_checkpoint"] = getattr(
                record, "initial_checkpoint", None
            )
            training_results[record.candidate_id]["factorized_source_weights"] = getattr(
                record, "factorized_source_weights", []
            )
            training_results[record.candidate_id]["state_transfer"] = getattr(
                record, "state_transfer", None
            )
            training_results[record.candidate_id]["retained_state_indices"] = getattr(
                record, "retained_state_indices", {}
            )
            training_results[record.candidate_id]["shared_module_prefix_groups"] = getattr(
                record, "shared_module_prefix_groups", []
            )
            training_results[record.candidate_id]["initialized_tcn_adapters"] = getattr(
                record, "initialized_tcn_adapters", []
            )
            training_results[record.candidate_id]["distillation_teacher"] = (
                distillation_teacher_record
            )
            Path(record.metrics_path).write_text(
                json.dumps(training_results[record.candidate_id], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    diagnostic_results = _run_diagnostics(
        manifest,
        candidate_records=candidate_records,
        training_results=training_results,
        dataset_root=dataset_root,
        diagnostics_root=diagnostics_root,
        diagnostics=diagnostics,
    )
    if bool(diagnostics.get("benchmark_inference", True)) or bool(
        diagnostics.get("benchmark_training_step", True)
    ):
        for record in candidate_records:
            benchmark_model, _ = load_checkpoint(record.checkpoint_path, map_location="cpu")
            if bool(diagnostics.get("benchmark_training_step", True)):
                training_results[record.candidate_id].update(
                    benchmark_model_training_step(
                        benchmark_model,
                        device=str(diagnostics.get("device", "cuda")),
                        batch_size=int(diagnostics.get("training_benchmark_batch_size", 32)),
                        window_size=int(diagnostics.get("training_benchmark_window_size", 500)),
                        warmup_repeats=int(
                            diagnostics.get("training_benchmark_warmup_repeats", 20)
                        ),
                        repeats=int(diagnostics.get("training_benchmark_repeats", 100)),
                    )
                )
            if bool(diagnostics.get("benchmark_inference", True)):
                training_results[record.candidate_id].update(
                    benchmark_model_inference(
                        benchmark_model,
                        device=str(diagnostics.get("device", "cuda")),
                        batch_size=int(diagnostics.get("inference_batch_size", 32)),
                        window_size=int(diagnostics.get("inference_window_size", 500)),
                        warmup_repeats=int(
                            diagnostics.get("inference_warmup_repeats", 20)
                        ),
                        repeats=int(diagnostics.get("inference_repeats", 100)),
                    )
                )
            del benchmark_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            Path(record.metrics_path).write_text(
                json.dumps(training_results[record.candidate_id], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    baseline_id = diagnostic_results["baseline_id"]
    baseline_training = training_results.get(
        baseline_id,
        diagnostic_results.get("baseline_training_metrics"),
    )
    if baseline_training is not None:
        for candidate_id, candidate_diagnostics in diagnostic_results["candidates"].items():
            if candidate_id == baseline_id or "deltas" not in candidate_diagnostics:
                continue
            candidate_diagnostics["practical_gate"] = practical_candidate_gate(
                candidate_diagnostics["deltas"],
                assessments=candidate_diagnostics["assessments"],
                baseline_training=baseline_training,
                candidate_training=training_results[candidate_id],
                metric_families=getattr(
                    next(
                        record
                        for record in candidate_records
                        if record.candidate_id == candidate_id
                    ),
                    "selection_metric_families",
                    ("sera", "f1"),
                ),
            )
    result = {
        **plan,
        "manifest": str(manifest_path),
        "dataset_root": str(dataset_root),
        "schedule_path": str(schedule_path.resolve()),
        "schedule_metadata": base_schedule.metadata,
        "run_root": str(run_root.resolve()),
        "diagnostics_root": str(diagnostics_root.resolve()),
        "training": training_results,
        "diagnostics": diagnostic_results,
    }
    (run_root / "iteration-summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (diagnostics_root / "iteration-summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _build_candidates(
    specifications: list[dict[str, Any]],
    *,
    training: dict[str, Any],
    run_root: Path,
    manifest_path: Path,
    seed: int,
) -> list[PanelCandidate]:
    base_config_path = _resolve_path(training.get("model_config", DEFAULT_CONFIG_PATH), manifest_path)
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    initial_checkpoint = None
    initial_checkpoint_value = training.get("initial_checkpoint")
    if initial_checkpoint_value is not None:
        initial_checkpoint_path = _resolve_path(initial_checkpoint_value, manifest_path)
        reference, initial_metadata = load_checkpoint(initial_checkpoint_path, map_location="cpu")
        if not isinstance(reference, AxoMamba):
            raise ValueError("initial_checkpoint must contain an AxoMamba model")
        if asdict(reference.config) != asdict(AxoMambaConfig(**base_config)):
            raise ValueError("initial_checkpoint config does not match training model_config")
        initial_checkpoint = {
            "path": str(initial_checkpoint_path),
            "sha256": hashlib.sha256(initial_checkpoint_path.read_bytes()).hexdigest(),
            "metadata": initial_metadata,
        }
    else:
        torch.manual_seed(seed)
        reference = AxoMamba(AxoMambaConfig(**base_config))
    reference_state = reference.state_dict()
    del reference

    records = []
    for specification in specifications:
        candidate_id = str(specification["id"])
        config = dict(base_config)
        config.update(specification.get("config_overrides", {}))
        torch.manual_seed(seed)
        model = AxoMamba(AxoMambaConfig(**config))
        initialization = str(specification.get("initialization", "pretrained"))
        if initialization not in {"pretrained", "scratch"}:
            raise ValueError("candidate initialization must be 'pretrained' or 'scratch'")
        state_transfer = specification.get("state_transfer")
        retained_state_indices: dict[str, list[int]] = {}
        candidate_reference_state = reference_state
        if state_transfer is not None:
            if state_transfer not in {"balanced_proxy", "random"}:
                raise ValueError("state_transfer must be 'balanced_proxy' or 'random'")
            if initialization != "pretrained":
                raise ValueError("state transfer requires pretrained initialization")
            if state_transfer == "balanced_proxy":
                candidate_reference_state, retained_state_indices = (
                    balanced_proxy_compress_mamba_state(
                        reference_state,
                        source_state_dim=int(base_config["state_dim"]),
                        target_state_dim=int(config["state_dim"]),
                    )
                )
        untied_candidate_reference_state = candidate_reference_state
        shared_module_prefix_groups: list[list[str]] = []
        if bool(config.get("share_local_tcn_pointwise", False)):
            shared_module_prefix_groups.append(
                [f"local_tcn_blocks.{index}.pointwise" for index in range(3)]
            )
        if shared_module_prefix_groups:
            candidate_reference_state = average_repeated_module_state(
                candidate_reference_state,
                prefix_groups=shared_module_prefix_groups,
            )
        if bool(config.get("share_morphology_synapse_gain", False)) and not bool(
            base_config.get("share_morphology_synapse_gain", False)
        ):
            candidate_reference_state = share_morphology_synapse_gain_state(
                candidate_reference_state
            )
        if int(config.get("morphology_synapse_gain_rank", 0)) and not int(
            base_config.get("morphology_synapse_gain_rank", 0)
        ):
            candidate_reference_state = factorize_morphology_synapse_gain_state(
                candidate_reference_state,
                rank=int(config["morphology_synapse_gain_rank"]),
                morphology_count=len(config["morphology_ids"]),
            )
        candidate_state = model.state_dict()
        if initialization == "pretrained":
            shared = {
                name: tensor
                for name, tensor in candidate_reference_state.items()
                if name in candidate_state and candidate_state[name].shape == tensor.shape
            }
            candidate_state.update(shared)
            model.load_state_dict(candidate_state, strict=True)
            factorized_source_weights = initialize_factorized_linears_from_state(
                model,
                candidate_reference_state,
                compatible_only=(
                    state_transfer == "random"
                    or int(base_config.get("mamba_projection_rank", 0)) > 0
                ),
            )
            initialized_tcn_adapters = (
                initialize_shared_tcn_adapters_from_pretrained_state(
                    model,
                    untied_candidate_reference_state,
                )
                if int(config.get("local_tcn_pointwise_adapter_rank", 0)) > 0
                else []
            )
        else:
            shared = {}
            factorized_source_weights = []
            initialized_tcn_adapters = []
        buffer_artifact = specification.get("buffer_artifact")
        buffer_artifacts = specification.get("buffer_artifacts")
        if buffer_artifact is not None and buffer_artifacts is not None:
            raise ValueError("candidate may use buffer_artifact or buffer_artifacts, not both")
        if buffer_artifacts is None:
            artifact_specs = [] if buffer_artifact is None else [buffer_artifact]
        elif isinstance(buffer_artifacts, list):
            artifact_specs = buffer_artifacts
        else:
            raise ValueError("candidate buffer_artifacts must be a list")
        loaded_buffer_artifacts = []
        loaded_buffer_names: set[str] = set()
        for artifact_spec in artifact_specs:
            if not isinstance(artifact_spec, dict):
                raise ValueError("candidate buffer artifact must be an object")
            required = {"buffer_name", "path", "key"}
            missing = sorted(required - set(artifact_spec))
            if missing:
                raise ValueError(f"candidate buffer_artifact is missing fields: {missing}")
            buffer_name = str(artifact_spec["buffer_name"])
            if buffer_name in loaded_buffer_names:
                raise ValueError(f"candidate buffer artifact repeats {buffer_name!r}")
            loaded_buffer_names.add(buffer_name)
            loaded_buffer_artifacts.append(load_named_buffer_artifact(
                model,
                buffer_name=buffer_name,
                artifact_path=_resolve_path(artifact_spec["path"], manifest_path),
                key=str(artifact_spec["key"]),
                expected_morphology_ids=config.get("morphology_ids"),
            ))
        mean_initialization_spec = specification.get(
            "mean_initialize_morphology_rows"
        )
        if mean_initialization_spec is None:
            mean_initialized_morphology_rows = None
        else:
            if not isinstance(mean_initialization_spec, dict):
                raise ValueError("mean_initialize_morphology_rows must be an object")
            required = {
                "target_morphology_id",
                "source_morphology_ids",
                "parameter_names",
            }
            missing = sorted(required - set(mean_initialization_spec))
            if missing:
                raise ValueError(
                    f"mean_initialize_morphology_rows is missing fields: {missing}"
                )
            mean_initialized_morphology_rows = (
                initialize_morphology_parameter_rows_from_mean(
                    model,
                    target_morphology_id=str(
                        mean_initialization_spec["target_morphology_id"]
                    ),
                    source_morphology_ids=mean_initialization_spec[
                        "source_morphology_ids"
                    ],
                    parameter_names=mean_initialization_spec["parameter_names"],
                )
            )
        initial_parameter_values = {
            str(name): float(value)
            for name, value in specification.get("initial_parameter_values", {}).items()
        }
        initialize_named_parameters(model, initial_parameter_values)
        initial_parameter_offsets = {
            str(name): float(value)
            for name, value in specification.get("initial_parameter_offsets", {}).items()
        }
        offset_named_parameters(model, initial_parameter_offsets)
        frozen_parameter_names = [
            str(name) for name in specification.get("frozen_parameter_names", [])
        ]
        trainable_parameter_names_value = specification.get("trainable_parameter_names")
        if trainable_parameter_names_value is not None and frozen_parameter_names:
            raise ValueError(
                "candidate may use frozen_parameter_names or trainable_parameter_names, not both"
            )
        if trainable_parameter_names_value is None:
            trainable_parameter_names = None
            freeze_named_parameters(model, frozen_parameter_names)
        else:
            if not isinstance(trainable_parameter_names_value, list):
                raise ValueError("trainable_parameter_names must be a list")
            trainable_parameter_names = [
                str(name) for name in trainable_parameter_names_value
            ]
            frozen_parameter_names = set_trainable_parameter_names(
                model, trainable_parameter_names
            )
        objective_values = dict(training.get("objective", {}))
        objective_values.update(specification.get("objective_overrides", {}))
        record = PanelCandidate(
            candidate_id=candidate_id,
            model=model,
            batch_size=int(specification.get("batch_size", training.get("batch_size", 64))),
            checkpoint_path=run_root / candidate_id / "checkpoint.pt",
            metrics_path=run_root / candidate_id / "training-metrics.json",
            learning_rate=float(specification.get("learning_rate", training.get("learning_rate", 3e-4))),
            optimizer_name=str(specification.get("optimizer", training.get("optimizer", "adamw"))),
            weight_decay=float(specification.get("weight_decay", training.get("weight_decay", 1e-4))),
            lr_schedule=str(specification.get("lr_schedule", training.get("lr_schedule", "constant"))),
            grad_clip_norm=float(specification.get("grad_clip_norm", training.get("grad_clip_norm", 0.0))),
            objective=_objective_from_dict(objective_values),
            burn_in=int(
                specification.get(
                    "burn_in", training.get("burn_in", AXOMAMBA_DEFAULT_BURN_IN)
                )
            ),
            output_distillation_weight=float(
                specification.get("output_distillation_weight", 0.0)
            ),
        )
        setattr(record, "model_config", config)
        setattr(record, "shared_initial_tensors", len(shared))
        setattr(record, "initial_parameter_values", initial_parameter_values)
        setattr(record, "initial_parameter_offsets", initial_parameter_offsets)
        setattr(record, "frozen_parameter_names", sorted(frozen_parameter_names))
        setattr(
            record,
            "trainable_parameter_names",
            None if trainable_parameter_names is None else sorted(trainable_parameter_names),
        )
        setattr(record, "buffer_artifacts", loaded_buffer_artifacts)
        setattr(
            record,
            "mean_initialized_morphology_rows",
            mean_initialized_morphology_rows,
        )
        setattr(
            record,
            "initial_checkpoint",
            initial_checkpoint if initialization == "pretrained" else None,
        )
        setattr(record, "initialization", initialization)
        setattr(record, "factorized_source_weights", factorized_source_weights)
        setattr(record, "state_transfer", state_transfer)
        setattr(record, "retained_state_indices", retained_state_indices)
        setattr(record, "shared_module_prefix_groups", shared_module_prefix_groups)
        setattr(record, "initialized_tcn_adapters", initialized_tcn_adapters)
        setattr(
            record,
            "selection_metric_families",
            tuple(
                str(family)
                for family in specification.get(
                    "selection_metric_families",
                    training.get("selection_metric_families", ("sera", "f1")),
                )
            ),
        )
        records.append(record)
    return records


def _run_diagnostics(
    manifest: dict[str, Any],
    *,
    candidate_records: list[PanelCandidate],
    training_results: dict[str, dict[str, Any]],
    dataset_root: Path,
    diagnostics_root: Path,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    eval_config = EvaluationConfig(
        ignore_start=int(diagnostics.get("ignore_start", 500)),
        calibration_fraction=float(diagnostics.get("calibration_fraction", 0.25)),
        default_spike_threshold=float(diagnostics.get("default_spike_threshold", 0.0)),
    )
    common = {
        "dataset_root": dataset_root,
        "max_samples": int(diagnostics.get("max_samples", 120)),
        "batch_size": int(diagnostics.get("batch_size", 128)),
        "max_intervention_samples": int(diagnostics.get("max_intervention_samples", 40)),
        "max_interventions": int(diagnostics.get("max_interventions", 8)),
        "intervention_workers": int(diagnostics.get("intervention_workers", 4)),
        "intervention_names": diagnostics.get("intervention_names"),
        "multiple_morphologies": bool(diagnostics.get("multiple_morphologies", True)),
        "morphology_id": diagnostics.get("morphology_id"),
        "config": eval_config,
    }
    record_by_id = {record.candidate_id: record for record in candidate_records}
    reuse_cached = bool(diagnostics.get("reuse_cached_comparisons", False))
    baseline_cache_value = manifest.get("baseline_cache")
    baseline_id = None
    if baseline_cache_value:
        baseline_cache_path = _resolve_path(baseline_cache_value, Path.cwd() / "manifest.json")
        baseline_cache = load_model_evaluation_cache(baseline_cache_path)
        baseline_id = baseline_cache.model_id
        if baseline_id in record_by_id:
            cached_checkpoint = baseline_cache.report.get("checkpoint")
            candidate_checkpoint = str(
                Path(record_by_id[baseline_id].checkpoint_path).resolve()
            )
            if cached_checkpoint != candidate_checkpoint:
                raise ValueError(
                    "external baseline cache cannot replace a different "
                    f"checkpoint with the same id {baseline_id!r}"
                )
    elif reuse_cached and (diagnostics_root / "baseline-cache.npz").exists():
        baseline_cache_path = diagnostics_root / "baseline-cache.npz"
        baseline_cache = load_model_evaluation_cache(baseline_cache_path)
        baseline_id = baseline_cache.model_id
    else:
        baseline_specs = [item for item in manifest["candidates"] if item.get("baseline")]
        if len(baseline_specs) != 1:
            raise ValueError("manifest needs one baseline candidate or baseline_cache")
        baseline_id = str(baseline_specs[0]["id"])
        record = record_by_id[baseline_id]
        predictor, checkpoint_metadata = make_checkpoint_predictor(
            record.checkpoint_path,
            device=str(diagnostics.get("device", "cuda")),
        )
        baseline_cache = create_model_evaluation_cache(
            predictor,
            model_id=baseline_id,
            **common,
        )
        baseline_cache.report["checkpoint"] = str(Path(record.checkpoint_path).resolve())
        baseline_cache.report["checkpoint_metadata"] = checkpoint_metadata
        baseline_cache.report["training_metrics"] = training_results[baseline_id]
        baseline_cache_path = diagnostics_root / "baseline-cache.npz"
        write_model_evaluation_cache(baseline_cache, baseline_cache_path)
        write_evaluation_report(baseline_cache.report, diagnostics_root / f"{baseline_id}.json")
        baseline_cache = load_model_evaluation_cache(baseline_cache_path)

    summaries: dict[str, Any] = {
        "baseline_id": baseline_id,
        "baseline_cache": str(Path(baseline_cache_path).resolve()),
        "baseline_training_metrics": baseline_cache.report.get("training_metrics"),
        "candidates": {},
    }
    for record in candidate_records:
        if record.candidate_id == baseline_id:
            summaries["candidates"][record.candidate_id] = {
                "role": "baseline",
                "report": str(Path(baseline_cache_path).resolve()),
                "metrics": _metric_values(baseline_cache.report),
            }
            continue
        output = diagnostics_root / f"{record.candidate_id}-vs-{baseline_id}.json"
        expected_checkpoint = str(Path(record.checkpoint_path).resolve())
        if reuse_cached and output.exists():
            comparison = json.loads(output.read_text(encoding="utf-8"))
            if comparison["candidate"].get("checkpoint") != expected_checkpoint:
                raise ValueError(
                    f"cached comparison checkpoint mismatch for {record.candidate_id}"
                )
        else:
            predictor, checkpoint_metadata = make_checkpoint_predictor(
                record.checkpoint_path,
                device=str(diagnostics.get("device", "cuda")),
            )
            comparison = evaluate_model_against_cache(
                predictor,
                baseline_cache=baseline_cache,
                candidate_id=record.candidate_id,
                bootstrap_replicates=int(diagnostics.get("bootstrap_replicates", 250)),
                bootstrap_seed=int(diagnostics.get("bootstrap_seed", 20260711)),
                **common,
            )
            comparison["candidate"]["checkpoint"] = expected_checkpoint
            comparison["candidate"]["checkpoint_metadata"] = checkpoint_metadata
            comparison["candidate"]["training_metrics"] = training_results[record.candidate_id]
            comparison["candidate"]["model_config"] = getattr(record, "model_config")
            write_model_comparison(comparison, output)
        summaries["candidates"][record.candidate_id] = {
            "role": "candidate",
            "comparison": str(output.resolve()),
            "metrics": _metric_values(comparison["candidate"]),
            "deltas": {row["id"]: row["delta"] for row in comparison["comparison"]},
            "assessments": {row["id"]: row["assessment"] for row in comparison["comparison"]},
            "attention_yes": [
                row["id"] for row in comparison["candidate"]["diagnostics"] if row["attention"] == "YES"
            ],
        }
    return summaries


def _objective_from_dict(values: dict[str, Any]) -> TrainingObjective:
    resolved = dict(values)
    if "f1_tolerances_steps" in resolved:
        resolved["f1_tolerances_steps"] = tuple(int(value) for value in resolved["f1_tolerances_steps"])
    return TrainingObjective(**resolved)


def _metric_values(report: dict[str, Any]) -> dict[str, float | None]:
    return {row["id"]: row["value"] for row in report["metrics"]}


def _resolve_path(value: str | Path, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    project_candidate = (PROJECT_ROOT / path).resolve()
    if project_candidate.exists() or str(path).startswith(("data/", "runs/", "docs/", "configs/")):
        return project_candidate
    return (manifest_path.parent / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
