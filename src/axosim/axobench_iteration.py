"""Thin AxoSim checkpoint adapter for AxoBench model-iteration reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from axosim.checkpoint import load_checkpoint


def make_checkpoint_predictor(
    checkpoint: str | Path,
    *,
    device: str = "auto",
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Load a standard AxoSim checkpoint as an AxoBench prediction callable."""
    resolved_device = _resolve_device(device)
    model, metadata = load_checkpoint(checkpoint, map_location=resolved_device)
    model = model.to(resolved_device).eval()
    use_no_grad = bool(getattr(model, "learn_memory_tau", False) or hasattr(model, "branch_gain_enabled"))

    def predict(inputs: np.ndarray) -> np.ndarray:
        array = np.ascontiguousarray(inputs, dtype=np.float32)
        tensor = torch.from_numpy(array).to(resolved_device, non_blocking=resolved_device.startswith("cuda"))
        context = torch.no_grad() if use_no_grad else torch.inference_mode()
        with context:
            output = model(tensor)
        return output.detach().cpu().numpy().astype(np.float32, copy=False)

    return predict, metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print bounded AxoBench metrics and attention diagnostics for an AxoSim checkpoint."
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-intervention-samples", type=int, default=8)
    parser.add_argument("--max-interventions", type=int, default=8)
    parser.add_argument("--interventions", help="comma-separated intervention names; default discovers all")
    parser.add_argument("--ignore-start", type=int, default=500)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--default-spike-threshold", type=float, default=0.0)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    try:
        from axobench.benchmark.iteration_evaluator import (
            EvaluationConfig,
            evaluate_model,
            format_evaluation_report,
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
    report = evaluate_model(
        predictor,
        dataset_root=args.dataset_root,
        model_id=model_id,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_intervention_samples=args.max_intervention_samples,
        max_interventions=args.max_interventions,
        intervention_names=interventions,
        config=EvaluationConfig(
            ignore_start=args.ignore_start,
            calibration_fraction=args.calibration_fraction,
            default_spike_threshold=args.default_spike_threshold,
        ),
    )
    report["checkpoint"] = str(Path(args.checkpoint).resolve())
    report["checkpoint_metadata"] = metadata
    print(format_evaluation_report(report))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} requested but CUDA is unavailable")
    return device


if __name__ == "__main__":
    raise SystemExit(main())

