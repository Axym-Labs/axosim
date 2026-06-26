from __future__ import annotations

import json
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

import torch


def benchmark_inference_matrix(
    model: torch.nn.Module,
    *,
    batch_sizes: Iterable[int],
    time_steps: Iterable[int],
    input_dim: int,
    device: str = "cpu",
    precision: str = "float32",
    warmup_runs: int = 5,
    runs: int = 20,
    compile_model: bool = False,
    accuracy_metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    """Benchmark dense full-window inference for deployment-oriented comparisons."""
    batch_size_values = [int(value) for value in batch_sizes]
    time_step_values = [int(value) for value in time_steps]
    if not batch_size_values:
        raise ValueError("batch_sizes must not be empty")
    if not time_step_values:
        raise ValueError("time_steps must not be empty")
    if any(value <= 0 for value in batch_size_values):
        raise ValueError("batch_sizes must be positive")
    if any(value <= 0 for value in time_step_values):
        raise ValueError("time_steps must be positive")
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if runs <= 0:
        raise ValueError("runs must be positive")
    if precision not in {"float32", "float16", "bfloat16"}:
        raise ValueError("precision must be one of: float32, float16, bfloat16")

    model = model.to(device).eval()
    if compile_model:
        model = torch.compile(model)

    is_cuda = str(device).startswith("cuda")
    dtype = _precision_dtype(precision)
    input_dtype = dtype if is_cuda else torch.float32
    if precision == "float16" and not is_cuda:
        raise ValueError("float16 inference benchmarking is only supported on CUDA devices")
    if precision != "float32" and is_cuda:
        model = model.to(dtype=dtype)

    measurements = []
    with torch.inference_mode():
        for batch_size in batch_size_values:
            for steps in time_step_values:
                try:
                    measurement = _benchmark_one_shape(
                        model,
                        batch_size=batch_size,
                        time_steps=steps,
                        input_dim=input_dim,
                        device=device,
                        dtype=input_dtype,
                        precision=precision,
                        warmup_runs=warmup_runs,
                        runs=runs,
                    )
                except RuntimeError as error:
                    if not _is_out_of_memory(error):
                        raise
                    _clear_device_cache(device)
                    measurement = {
                        "batch_size": batch_size,
                        "time_steps": steps,
                        "error": "out_of_memory",
                        "error_message": str(error).splitlines()[0],
                    }
                measurements.append(measurement)

    accuracy_metrics = _load_accuracy_metrics(accuracy_metrics_path)
    report = {
        "accuracy_metrics": accuracy_metrics,
        "compiled": bool(compile_model),
        "device": device,
        "input_dim": input_dim,
        "measurements": measurements,
        "model_parameter_count": count_parameters(model),
        "precision": precision,
        "runs": runs,
        "warmup_runs": warmup_runs,
    }
    successful = [row for row in measurements if "neuron_steps_per_second" in row]
    if successful:
        best = max(successful, key=lambda row: row["neuron_steps_per_second"])
        report["best_neuron_steps_per_second"] = best["neuron_steps_per_second"]
        report["best_batch_size"] = best["batch_size"]
        report["best_time_steps"] = best["time_steps"]
    return report


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def write_inference_benchmark(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_int_list(value: str) -> list[int]:
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("expected a comma-separated list of integers")
    return parsed


def _benchmark_one_shape(
    model: torch.nn.Module,
    *,
    batch_size: int,
    time_steps: int,
    input_dim: int,
    device: str,
    dtype: torch.dtype,
    precision: str,
    warmup_runs: int,
    runs: int,
) -> dict[str, Any]:
    x = torch.randn(batch_size, time_steps, input_dim, device=device, dtype=dtype)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup_runs):
        _run_model(model, x, device=device, precision=precision)
    _synchronize(device)

    elapsed_values = []
    for _ in range(runs):
        _synchronize(device)
        start = time.perf_counter()
        _run_model(model, x, device=device, precision=precision)
        _synchronize(device)
        elapsed_values.append(time.perf_counter() - start)

    mean_seconds = statistics.fmean(elapsed_values)
    median_seconds = statistics.median(elapsed_values)
    p95_seconds = sorted(elapsed_values)[max(0, int(0.95 * len(elapsed_values)) - 1)]
    peak_memory_bytes = torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None
    return {
        "batch_size": batch_size,
        "time_steps": time_steps,
        "mean_latency_seconds": mean_seconds,
        "median_latency_seconds": median_seconds,
        "p95_latency_seconds": p95_seconds,
        "presentations_per_second": batch_size / mean_seconds,
        "neuron_steps_per_second": (batch_size * time_steps) / mean_seconds,
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_mib": None if peak_memory_bytes is None else peak_memory_bytes / (1024 * 1024),
    }


def _run_model(model: torch.nn.Module, x: torch.Tensor, *, device: str, precision: str) -> torch.Tensor:
    if precision == "float32":
        return model(x)
    if str(device).startswith("cuda"):
        dtype = _precision_dtype(precision)
        with torch.autocast(device_type="cuda", dtype=dtype):
            return model(x)
    if precision == "bfloat16":
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            return model(x)
    return model(x)


def _precision_dtype(precision: str) -> torch.dtype:
    if precision == "float16":
        return torch.float16
    if precision == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _load_accuracy_metrics(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    metrics = json.loads(Path(path).read_text(encoding="utf-8"))
    keys = (
        "best_validation_soma_rmse",
        "best_validation_spike_auc",
        "validation_soma_rmse",
        "validation_spike_auc",
        "soma_rmse",
        "spike_auc",
        "train_presentations",
    )
    summary = {key: metrics[key] for key in keys if key in metrics}
    history = metrics.get("history")
    if isinstance(history, list) and history:
        validation_rows = [row for row in history if isinstance(row, dict)]
        if validation_rows:
            rmse_rows = [row for row in validation_rows if "validation_soma_rmse" in row]
            auc_rows = [row for row in validation_rows if "validation_spike_auc" in row]
            if "best_validation_soma_rmse" not in summary and rmse_rows:
                summary["best_validation_soma_rmse"] = min(float(row["validation_soma_rmse"]) for row in rmse_rows)
            if auc_rows:
                summary["best_validation_spike_auc"] = max(float(row["validation_spike_auc"]) for row in auc_rows)
                summary["last_validation_spike_auc"] = float(auc_rows[-1]["validation_spike_auc"])
            if rmse_rows:
                summary["last_validation_soma_rmse"] = float(rmse_rows[-1]["validation_soma_rmse"])
    return summary


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _clear_device_cache(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def _is_out_of_memory(error: RuntimeError) -> bool:
    return "out of memory" in str(error).lower()
