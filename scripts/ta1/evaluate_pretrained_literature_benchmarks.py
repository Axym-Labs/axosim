#!/usr/bin/env python3
"""Evaluate selected literature pretrained models on the v1 benchmark bundle.

The script is intentionally environment-split:
- run `--family official` in the main project env with PyTorch;
- run `--family keras-tcn` in the TensorFlow env used for `.h5` checkpoints.

Both routes call the same dataframe-first `run_v1_benchmark_bundle` path and
can disable Markdown summaries so numeric rows/manifests remain the primary
stored artifacts.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from bnn_sim.benchmark.bundle import run_v1_benchmark_bundle
from bnn_sim.benchmark.diagnostic_rows import DiagnosticContext


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data/v1-neuronio-like-gpu-20260611"
DEFAULT_OUTPUT = ROOT / "docs/ta1/artifacts/analysis_sources/existing_models/pretrained_benchmark_eval_20260617"


OFFICIAL_MODELS = {
    "branch_elm_best": {
        "config": ".external/elmneuron/models/best_elm_neuron/model_config.json",
        "checkpoint": ".external/elmneuron/models/best_elm_neuron/neuronio_best_model_state.pt",
        "family": "official",
    },
    "branch_elm_m1": {
        "config": ".external/elmneuron/models/num_memory_1/model_config.json",
        "checkpoint": ".external/elmneuron/models/num_memory_1/neuronio_best_model_state.pt",
        "family": "official",
    },
    "branch_elm_m30": {
        "config": ".external/elmneuron/models/num_memory_30/model_config.json",
        "checkpoint": ".external/elmneuron/models/num_memory_30/neuronio_best_model_state.pt",
        "family": "official",
    },
    "branch_elm_m100": {
        "config": ".external/elmneuron/models/num_memory_100/model_config.json",
        "checkpoint": ".external/elmneuron/models/num_memory_100/neuronio_best_model_state.pt",
        "family": "official",
    },
}

KERAS_TCN_MODELS = {
    "tcn_dwt7_128_153": {
        "checkpoint": "data/raw/nmda-test/Models/NMDA_TCN__DWT_7_128_153__model.h5",
        "family": "keras-tcn",
    },
    "tcn_dwt8_224_217": {
        "checkpoint": "data/raw/nmda-test/Models/NMDA_TCN__DWT_8_224_217__model.h5",
        "family": "keras-tcn",
    },
    "tcn_dwt9_256_241": {
        "checkpoint": "data/raw/nmda-test/Models/NMDA_TCN__DWT_9_256_241__model.h5",
        "family": "keras-tcn",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--family", choices=["official", "keras-tcn", "all"], default="all")
    parser.add_argument("--suite", choices=["teacher-qc", "paper-summary", "full"], default="paper-summary")
    parser.add_argument("--max-val-samples", type=int, default=128)
    parser.add_argument("--max-intervention-samples", type=int, default=16)
    parser.add_argument("--cache-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-summaries", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = _selected_models(args.family)
    summary_by_model = {
        row["model_id"]: row
        for row in _load_existing_summaries(output_dir)
        if row.get("model_id")
    }
    for model_id, spec in models.items():
        model_output = output_dir / model_id
        if args.skip_existing and (model_output / "manifest.json").exists():
            summary_by_model[model_id] = _summarize_existing(model_id, spec, model_output)
            summaries = [summary_by_model[key] for key in sorted(summary_by_model)]
            _write_summary(output_dir, summaries, args)
            continue
        model_fn, adapter = _build_model_fn(spec, batch_size=args.batch_size, device=args.device)
        manifest = run_v1_benchmark_bundle(
            dataset_root=dataset_root,
            output_dir=model_output,
            base_context=DiagnosticContext(model_id=model_id),
            model_fn=model_fn,
            model_id=model_id,
            suite=args.suite,
            max_val_samples=args.max_val_samples,
            max_intervention_samples=args.max_intervention_samples,
            cache_shards=args.cache_shards,
            write_summaries=args.write_summaries,
        )
        model_summary = _summarize_model_rows(model_id, spec, model_output)
        model_summary["adapter"] = adapter
        model_summary["manifest"] = manifest
        summary_by_model[model_id] = model_summary
        summaries = [summary_by_model[key] for key in sorted(summary_by_model)]
        _write_summary(output_dir, summaries, args)

    summaries = [summary_by_model[key] for key in sorted(summary_by_model)]
    _write_summary(output_dir, summaries, args)
    print(json.dumps({"models": len(summaries), "output_dir": str(output_dir)}, indent=2, sort_keys=True))
    return 0


def _selected_models(family: str) -> dict[str, dict[str, str]]:
    if family == "official":
        return dict(OFFICIAL_MODELS)
    if family == "keras-tcn":
        return dict(KERAS_TCN_MODELS)
    return {**OFFICIAL_MODELS, **KERAS_TCN_MODELS}


def _build_model_fn(spec: dict[str, str], *, batch_size: int, device: str):
    if spec["family"] == "official":
        return _build_official_model_fn(spec, batch_size=batch_size, device=device)
    if spec["family"] == "keras-tcn":
        if device != "cpu":
            raise ValueError("keras-tcn evaluation should use --device cpu in the TensorFlow CPU env")
        return _build_keras_tcn_model_fn(spec, batch_size=batch_size)
    raise ValueError(f"unknown model family: {spec['family']}")


def _build_official_model_fn(spec: dict[str, str], *, batch_size: int, device: str):
    import torch

    from bnn_sim.official import create_official_elm, load_official_elm_config, load_official_state_dict

    config_path = ROOT / spec["config"]
    checkpoint_path = ROOT / spec["checkpoint"]
    model = create_official_elm(load_official_elm_config(config_path))
    model.load_state_dict(load_official_state_dict(checkpoint_path, map_location=device), strict=True)
    model = model.to(device).eval()

    def predict(inputs: np.ndarray) -> np.ndarray:
        outputs = []
        with torch.inference_mode():
            for start in range(0, inputs.shape[0], batch_size):
                batch = torch.as_tensor(inputs[start : start + batch_size], dtype=torch.float32, device=device)
                outputs.append(model(batch).detach().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)

    return predict, {
        "family": "official",
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "batch_size": batch_size,
        "device": device,
    }


def _build_keras_tcn_model_fn(spec: dict[str, str], *, batch_size: int):
    import tensorflow as tf

    checkpoint_path = ROOT / spec["checkpoint"]
    metadata_path = checkpoint_path.with_name(checkpoint_path.name.replace("__model.h5", "__training.pickle"))
    model = tf.keras.models.load_model(checkpoint_path, compile=False)
    input_shape = model.input_shape
    input_window_size = int(input_shape[1])
    input_dim = int(input_shape[2])
    if input_window_size <= 0:
        raise ValueError(f"TCN checkpoint must have fixed time dimension, got {input_shape}")
    overlap_size, time_window_t = _load_tcn_overlap(metadata_path, input_window_size=input_window_size)

    def predict(inputs: np.ndarray) -> np.ndarray:
        inputs_float = np.asarray(inputs, dtype=np.float32)
        if inputs_float.shape[-1] != input_dim:
            raise ValueError(f"TCN expected input_dim={input_dim}, got {inputs_float.shape[-1]}")
        return _predict_tcn_overlapping_windows(
            inputs_float,
            input_window_size=input_window_size,
            overlap_size=overlap_size,
            batch_size=batch_size,
            predict_chunk_fn=lambda chunk: model.predict(chunk, verbose=0),
        )

    return predict, {
        "family": "keras-tcn",
        "checkpoint": str(checkpoint_path),
        "metadata": str(metadata_path),
        "input_window_size": input_window_size,
        "overlap_size": overlap_size,
        "time_window_t": time_window_t,
        "input_dim": input_dim,
        "batch_size": batch_size,
        "adapter": (
            "overlapping fixed-length TCN windows using the original receptive-field-derived "
            "overlap rule; discard leading overlap on non-initial windows; zero-pad only "
            "the final short window and trim padded timesteps; no clipping or test-target "
            "mean/std alignment"
        ),
        "outputs": ["spikes", "somatic"],
    }


def _load_tcn_overlap(metadata_path: Path, *, input_window_size: int) -> tuple[int, int]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing TCN training metadata: {metadata_path}")
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle, encoding="latin1")
    architecture = metadata.get("architecture_dict", {})
    filter_sizes = architecture.get("filter_sizes_per_layer")
    if not filter_sizes:
        raise ValueError(f"TCN metadata missing architecture_dict.filter_sizes_per_layer: {metadata_path}")
    time_window_t = int((np.asarray(filter_sizes, dtype=int) - 1).sum() + 1)
    overlap_size = int(min(max(time_window_t + 1, min(150, input_window_size - 50)), 250))
    if overlap_size <= 0 or overlap_size >= input_window_size:
        raise ValueError(
            f"invalid TCN overlap_size={overlap_size} for input_window_size={input_window_size}"
        )
    return overlap_size, time_window_t


def _predict_tcn_overlapping_windows(
    inputs: np.ndarray,
    *,
    input_window_size: int,
    overlap_size: int,
    batch_size: int,
    predict_chunk_fn,
) -> np.ndarray:
    """Predict full traces by stitching overlapping TCN windows fairly.

    This mirrors the original TCN window stitching convention without the
    original evaluation-time soma distribution alignment. Only the final short
    input window is zero-padded, and predictions for padded timesteps are never
    returned.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch_count, time_steps, channels = inputs.shape
    stride = input_window_size - overlap_size
    if stride <= 0:
        raise ValueError("input_window_size must be larger than overlap_size")
    predictions = np.zeros((batch_count, time_steps, 2), dtype=np.float32)
    filled_until = 0
    if time_steps <= input_window_size:
        starts = [0]
    else:
        num_splits = int(2 + (time_steps - input_window_size) / stride)
        starts = [index * stride for index in range(num_splits)]
    for window_index, start in enumerate(starts):
        end = start + input_window_size
        chunk = inputs[:, start:min(end, time_steps), :]
        if chunk.shape[1] < input_window_size:
            padded = np.zeros((batch_count, input_window_size, channels), dtype=np.float32)
            padded[:, : chunk.shape[1], :] = chunk
            chunk = padded
        chunk_pred = _predict_tcn_chunks(chunk, batch_size=batch_size, predict_chunk_fn=predict_chunk_fn)
        if window_index == 0:
            write_start = start
            read_start = 0
        else:
            write_start = max(start + overlap_size, filled_until)
            read_start = write_start - start
        write_end = min(end, time_steps)
        if write_start >= write_end:
            continue
        predictions[:, write_start:write_end, :] = chunk_pred[:, read_start : read_start + (write_end - write_start), :]
        filled_until = max(filled_until, write_end)
    if filled_until < time_steps:
        raise RuntimeError(f"TCN window stitching left unfilled tail: filled_until={filled_until}, time_steps={time_steps}")
    return predictions


def _predict_tcn_chunks(chunk: np.ndarray, *, batch_size: int, predict_chunk_fn) -> np.ndarray:
    pred_chunks: list[np.ndarray] = []
    for start in range(0, chunk.shape[0], batch_size):
        outputs = predict_chunk_fn(chunk[start : start + batch_size])
        spikes, soma = outputs[0], outputs[1]
        pred_chunks.append(np.concatenate([spikes, soma], axis=-1))
    return np.concatenate(pred_chunks, axis=0).astype(np.float32, copy=False)


def _summarize_existing(model_id: str, spec: dict[str, str], model_output: Path) -> dict[str, Any]:
    summary = _summarize_model_rows(model_id, spec, model_output)
    summary["skipped_existing"] = True
    return summary


def _summarize_model_rows(model_id: str, spec: dict[str, str], model_output: Path) -> dict[str, Any]:
    model_rows_path = model_output / f"model_val_{_slug(model_id)}.jsonl"
    rows = _read_jsonl(model_rows_path) if model_rows_path.exists() else []
    metrics = {
        (row.get("diagnostic"), row.get("metric"), row.get("state")): row.get("value")
        for row in rows
        if row.get("trace_id") is None
    }
    return {
        "model_id": model_id,
        "family": spec["family"],
        "checkpoint": spec["checkpoint"],
        "output_dir": str(model_output),
        "row_count": len(rows),
        "soma_rmse_mv": metrics.get(("trace_core", "soma_rmse_mv", None)),
        "state_weighted_trace_distance_mv": metrics.get(
            ("trace_core", "state_weighted_trace_distance_mv", None)
        ),
        "dynamic_distance": metrics.get(("trace_core", "dynamic_distance", None)),
        "lag_tolerant_voltage_rmse_mv": metrics.get(
            ("trace_core", "lag_tolerant_voltage_rmse_mv", None)
        ),
        "spike_auc": metrics.get(("trace_core", "spike_auc", None)),
        "spike_brier": metrics.get(("trace_core", "spike_brier", None)),
        "spike_count_mae_per_trace": metrics.get(("trace_core", "spike_count_mae_per_trace", None)),
    }


def _write_summary(output_dir: Path, summaries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    payload = {
        "dataset_root": args.dataset_root,
        "suite": args.suite,
        "max_val_samples": args.max_val_samples,
        "max_intervention_samples": args.max_intervention_samples,
        "write_summaries": args.write_summaries,
        "models": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary_csv(output_dir / "summary.csv", summaries)


def _load_existing_summaries(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "summary.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    models = payload.get("models", [])
    return models if isinstance(models, list) else []


def _write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    import csv

    fields = [
        "model_id",
        "family",
        "checkpoint",
        "row_count",
        "soma_rmse_mv",
        "state_weighted_trace_distance_mv",
        "dynamic_distance",
        "lag_tolerant_voltage_rmse_mv",
        "spike_auc",
        "spike_brier",
        "spike_count_mae_per_trace",
        "output_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field) for field in fields})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _slug(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
