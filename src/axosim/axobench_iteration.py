"""Thin AxoSim checkpoint adapter for AxoBench model-iteration reports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from axosim.checkpoint import load_checkpoint


def make_checkpoint_predictor(
    checkpoint: str | Path,
    *,
    device: str = "auto",
    dtype: str | torch.dtype = "float32",
    streaming: bool = False,
    streaming_chunk_size: int | None = None,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Load a standard AxoSim checkpoint as an AxoBench prediction callable."""
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype)
    model, metadata = load_checkpoint(checkpoint, map_location=resolved_device)
    model = model.to(device=resolved_device, dtype=resolved_dtype).eval()
    use_no_grad = bool(getattr(model, "learn_memory_tau", False) or hasattr(model, "branch_gain_enabled"))
    configured_morphologies = getattr(getattr(model, "config", None), "morphology_ids", None)
    morphology_mapping = (
        {str(name): index for index, name in enumerate(configured_morphologies)}
        if configured_morphologies is not None
        else None
    )

    def predict(
        inputs: np.ndarray,
        *,
        morphology_ids: list[str] | np.ndarray | None = None,
    ) -> np.ndarray:
        array = np.ascontiguousarray(inputs, dtype=np.float32)
        tensor = torch.from_numpy(array).to(
            device=resolved_device,
            dtype=resolved_dtype,
            non_blocking=resolved_device.startswith("cuda"),
        )
        morphology_indices = None
        if morphology_mapping is not None:
            if morphology_ids is None or len(morphology_ids) != tensor.shape[0]:
                raise ValueError("conditioned checkpoint requires one morphology ID per input sample")
            names = [_morphology_id(value) for value in morphology_ids]
            unknown = sorted(set(names) - set(morphology_mapping))
            if unknown:
                raise ValueError(f"unknown morphology IDs for checkpoint: {unknown}")
            morphology_indices = torch.tensor(
                [morphology_mapping[name] for name in names],
                dtype=torch.long,
                device=resolved_device,
            )
        context = torch.no_grad() if use_no_grad else torch.inference_mode()
        with context:
            if streaming:
                if not hasattr(model, "allocate_streaming_state") or not hasattr(
                    model,
                    "streaming_step",
                ):
                    raise ValueError("checkpoint model does not support streaming inference")
                state = model.allocate_streaming_state(
                    tensor.shape[0],
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                output = torch.empty(
                    tensor.shape[0],
                    tensor.shape[1],
                    int(getattr(model, "num_output", 2)),
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                for step in range(tensor.shape[1]):
                    output[:, step : step + 1] = model.streaming_step(
                        tensor[:, step],
                        state,
                        morphology_indices=morphology_indices,
                        chunk_size=streaming_chunk_size,
                    )
            else:
                output = (
                    model(tensor, morphology_indices=morphology_indices)
                    if morphology_indices is not None
                    else model(tensor)
                )
        return output.detach().to(dtype=torch.float32).cpu().numpy()

    predict.supports_morphology_ids = morphology_mapping is not None
    return predict, {
        **metadata,
        "inference_dtype": str(resolved_dtype).removeprefix("torch."),
        "streaming_inference": streaming,
        "streaming_chunk_size": streaming_chunk_size,
    }


def _morphology_id(value: object) -> str:
    name = str(value)
    for separator in (":", "/", "@"):
        if separator in name:
            return name.split(separator, 1)[0]
    return name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print bounded AxoBench metrics and attention diagnostics for an AxoSim checkpoint.",
        allow_abbrev=False,
    )
    parser.add_argument("checkpoint")
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--baseline-checkpoint")
    baseline.add_argument("--baseline-cache")
    parser.add_argument("--save-cache")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-intervention-samples", type=int)
    parser.add_argument("--max-interventions", type=int, default=8)
    parser.add_argument("--intervention-workers", type=int, default=4)
    parser.add_argument("--interventions", help="comma-separated intervention names; default discovers all")
    parser.add_argument(
        "--multiple-morphologies",
        action="store_true",
        help="balance the 120/40 evaluation profile across available morphologies",
    )
    parser.add_argument(
        "--morphology-id",
        help="specific morphology for the default single-morphology evaluation",
    )
    parser.add_argument("--ignore-start", type=int, default=500)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--default-spike-threshold", type=float, default=0.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=250)
    parser.add_argument("--bootstrap-seed", type=int, default=20260711)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output", help="optional .json or .csv report path")
    output.add_argument("--output-json", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        from axobench.benchmark.iteration_evaluator import (
            EvaluationConfig,
            create_model_evaluation_cache,
            evaluate_model,
            evaluate_model_against_cache,
            evaluate_model_comparison,
            format_evaluation_report,
            format_model_comparison,
            write_evaluation_report,
            write_model_evaluation_cache,
            write_model_comparison,
        )
    except ImportError as error:
        raise RuntimeError(
            "AxoBench with the model-iteration evaluator is required; install the current axobench package"
        ) from error

    predictor, metadata = make_checkpoint_predictor(args.checkpoint, device=args.device)
    model_id = args.model_id or Path(args.checkpoint).stem
    interventions = None
    if args.interventions:
        interventions = tuple(name.strip() for name in args.interventions.split(",") if name.strip())
    config = EvaluationConfig(
        ignore_start=args.ignore_start,
        calibration_fraction=args.calibration_fraction,
        default_spike_threshold=args.default_spike_threshold,
    )
    output_path = args.output or args.output_json
    if args.save_cache and (args.baseline_checkpoint or args.baseline_cache):
        parser.error("--save-cache cannot be combined with a baseline comparison")
    if args.baseline_checkpoint:
        baseline_predictor, baseline_metadata = make_checkpoint_predictor(
            args.baseline_checkpoint,
            device=args.device,
        )
        comparison = evaluate_model_comparison(
            predictor,
            baseline_predictor,
            dataset_root=args.dataset_root,
            candidate_id=model_id,
            baseline_id=Path(args.baseline_checkpoint).stem,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            max_intervention_samples=args.max_intervention_samples,
            max_interventions=args.max_interventions,
            intervention_workers=args.intervention_workers,
            intervention_names=interventions,
            multiple_morphologies=args.multiple_morphologies,
            morphology_id=args.morphology_id,
            config=config,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        comparison["candidate"]["checkpoint"] = str(Path(args.checkpoint).resolve())
        comparison["candidate"]["checkpoint_metadata"] = metadata
        comparison["baseline"]["checkpoint"] = str(Path(args.baseline_checkpoint).resolve())
        comparison["baseline"]["checkpoint_metadata"] = baseline_metadata
        print(format_evaluation_report(comparison["candidate"]))
        print()
        print(format_model_comparison(comparison))
        if output_path:
            write_model_comparison(comparison, output_path)
    elif args.baseline_cache:
        comparison = evaluate_model_against_cache(
            predictor,
            baseline_cache=args.baseline_cache,
            dataset_root=args.dataset_root,
            candidate_id=model_id,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            max_intervention_samples=args.max_intervention_samples,
            max_interventions=args.max_interventions,
            intervention_workers=args.intervention_workers,
            intervention_names=interventions,
            multiple_morphologies=args.multiple_morphologies,
            morphology_id=args.morphology_id,
            config=config,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        comparison["candidate"]["checkpoint"] = str(Path(args.checkpoint).resolve())
        comparison["candidate"]["checkpoint_metadata"] = metadata
        print(format_evaluation_report(comparison["candidate"]))
        print()
        print(format_model_comparison(comparison))
        if output_path:
            write_model_comparison(comparison, output_path)
    elif args.save_cache:
        cache = create_model_evaluation_cache(
            predictor,
            dataset_root=args.dataset_root,
            model_id=model_id,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            max_intervention_samples=args.max_intervention_samples,
            max_interventions=args.max_interventions,
            intervention_workers=args.intervention_workers,
            intervention_names=interventions,
            multiple_morphologies=args.multiple_morphologies,
            morphology_id=args.morphology_id,
            config=config,
        )
        cache.report["checkpoint"] = str(Path(args.checkpoint).resolve())
        cache.report["checkpoint_metadata"] = metadata
        write_model_evaluation_cache(cache, args.save_cache)
        print(format_evaluation_report(cache.report))
        if output_path:
            write_evaluation_report(cache.report, output_path)
    else:
        report = evaluate_model(
            predictor,
            dataset_root=args.dataset_root,
            model_id=model_id,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            max_intervention_samples=args.max_intervention_samples,
            max_interventions=args.max_interventions,
            intervention_workers=args.intervention_workers,
            intervention_names=interventions,
            multiple_morphologies=args.multiple_morphologies,
            morphology_id=args.morphology_id,
            config=config,
        )
        report["checkpoint"] = str(Path(args.checkpoint).resolve())
        report["checkpoint_metadata"] = metadata
        print(format_evaluation_report(report))
        if output_path:
            write_evaluation_report(report, output_path)
    return 0


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} requested but CUDA is unavailable")
    return device


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        resolved = dtype
    else:
        names = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in names:
            raise ValueError(
                "inference dtype must be 'float32', 'float16', or 'bfloat16'"
            )
        resolved = names[dtype]
    if resolved not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError("inference dtype must be float32, float16, or bfloat16")
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
