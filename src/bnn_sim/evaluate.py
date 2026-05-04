from __future__ import annotations

import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from bnn_sim.data import ShardedNeuronIODataset
from bnn_sim.metrics import binary_auc
from bnn_sim.model import BranchELM
from bnn_sim.official.constants import DEFAULT_Y_TRAIN_SOMA_SCALE


def evaluate_dataset(
    model: BranchELM,
    dataset: ShardedNeuronIODataset,
    *,
    batch_size: int = 8,
    device: str = "cpu",
    soma_units: str = "millivolts",
    y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE,
    ignore_start: int = 0,
    mask_mode: str = "ignore-start",
    stitch_burn_in: int = 150,
    soma_affine_calibration: bool = False,
    pin_memory: bool = False,
    non_blocking: bool = True,
) -> dict[str, float | int]:
    if soma_units not in {"millivolts", "normalized"}:
        raise ValueError("soma_units must be 'millivolts' or 'normalized'")
    if ignore_start < 0:
        raise ValueError("ignore_start must be non-negative")
    if mask_mode not in {"ignore-start", "official-overlap"}:
        raise ValueError("mask_mode must be 'ignore-start' or 'official-overlap'")
    if stitch_burn_in < 0:
        raise ValueError("stitch_burn_in must be non-negative")
    first_window_starts = _first_window_starts(dataset) if mask_mode == "official-overlap" else {}
    model = model.to(device)
    model.eval()
    soma_squared_error = 0.0
    soma_count = 0
    soma_predictions: list[np.ndarray] = []
    soma_targets: list[np.ndarray] = []
    spike_scores: list[np.ndarray] = []
    spike_labels: list[np.ndarray] = []
    _synchronize_device(device)
    start_time = time.perf_counter()

    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            batch_indices = range(start, min(start + batch_size, len(dataset)))
            if hasattr(dataset, "get_batch"):
                x_np, y = dataset.get_batch(batch_indices)
            else:
                samples = [dataset[i] for i in batch_indices]
                x_np = np.stack([sample.inputs for sample in samples]).astype(np.float32, copy=False)
                y = np.stack([sample.targets for sample in samples]).astype(np.float32, copy=False)
            sample_ids = (
                dataset.get_sample_ids(batch_indices)
                if hasattr(dataset, "get_sample_ids")
                else [dataset[i].sample_id for i in batch_indices]
            )
            x = _tensor_from_numpy(x_np, device, pin_memory=pin_memory, non_blocking=non_blocking)
            pred = model(x).cpu().numpy()
            mask = _metric_mask(
                sample_ids,
                sequence_length=pred.shape[1],
                ignore_start=ignore_start,
                mask_mode=mask_mode,
                stitch_burn_in=stitch_burn_in,
                first_window_starts=first_window_starts,
            )

            pred_soma = pred[..., 1][mask].astype(np.float64)
            target_soma = y[..., 1][mask].astype(np.float64)
            if soma_affine_calibration:
                soma_predictions.append(pred_soma.copy())
                soma_targets.append(target_soma.copy())
            else:
                soma_error = pred_soma - target_soma
                if soma_units == "millivolts":
                    soma_error = soma_error / y_train_soma_scale
                soma_squared_error += float(np.square(soma_error).sum())
                soma_count += int(soma_error.size)
            spike_scores.append(pred[..., 0][mask].copy())
            spike_labels.append(y[..., 0][mask].copy())

    if soma_affine_calibration:
        pred_soma_all = np.concatenate(soma_predictions)
        target_soma_all = np.concatenate(soma_targets)
        pred_std = float(pred_soma_all.std())
        if pred_std == 0.0:
            raise ValueError("cannot affine-calibrate soma predictions with zero standard deviation")
        pred_soma_all = (pred_soma_all - pred_soma_all.mean()) / pred_std
        pred_soma_all = target_soma_all.std() * pred_soma_all + target_soma_all.mean()
        soma_error = pred_soma_all - target_soma_all
        if soma_units == "millivolts":
            soma_error = soma_error / y_train_soma_scale
        soma_squared_error = float(np.square(soma_error).sum())
        soma_count = int(soma_error.size)

    _synchronize_device(device)
    total_seconds = time.perf_counter() - start_time
    return {
        "samples": len(dataset),
        "metric_ignore_start": ignore_start,
        "metric_mask_mode": mask_mode,
        "metric_stitch_burn_in": stitch_burn_in if mask_mode == "official-overlap" else None,
        "soma_affine_calibration": soma_affine_calibration,
        "pin_memory": pin_memory,
        "non_blocking": non_blocking,
        "total_seconds": total_seconds,
        "samples_per_second": len(dataset) / total_seconds if total_seconds > 0 else float("inf"),
        "soma_rmse": float(np.sqrt(soma_squared_error / soma_count)),
        "soma_rmse_units": soma_units,
        "spike_auc": binary_auc(np.concatenate(spike_scores), np.concatenate(spike_labels)),
    }


def write_metrics(metrics: dict[str, float | int], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tensor_from_numpy(
    array: np.ndarray,
    device: str,
    *,
    pin_memory: bool,
    non_blocking: bool,
) -> torch.Tensor:
    tensor = torch.from_numpy(array)
    if pin_memory and device.startswith("cuda") and torch.cuda.is_available():
        tensor = tensor.pin_memory()
    return tensor.to(device, non_blocking=non_blocking)


def _synchronize_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _metric_mask(
    sample_ids: list[str],
    *,
    sequence_length: int,
    ignore_start: int,
    mask_mode: str,
    stitch_burn_in: int,
    first_window_starts: dict[str, int],
) -> np.ndarray:
    mask = np.ones((len(sample_ids), sequence_length), dtype=bool)
    if ignore_start:
        if ignore_start >= sequence_length:
            raise ValueError(f"ignore_start={ignore_start} leaves no timesteps for sequence length {sequence_length}")
        mask[:, :ignore_start] = False
    if mask_mode == "official-overlap":
        if stitch_burn_in >= sequence_length:
            raise ValueError(
                f"stitch_burn_in={stitch_burn_in} leaves no timesteps for sequence length {sequence_length}"
            )
        for row, sample_id in enumerate(sample_ids):
            simulation_id, window_start = _parse_window_sample_id(sample_id)
            if window_start > first_window_starts[simulation_id]:
                mask[row, :stitch_burn_in] = False
    if not mask.any():
        raise ValueError("metric mask removed all timesteps")
    return mask


def _first_window_starts(dataset: ShardedNeuronIODataset) -> dict[str, int]:
    result: dict[str, int] = {}
    for index in range(len(dataset)):
        sample_id = dataset.get_sample_ids([index])[0] if hasattr(dataset, "get_sample_ids") else dataset[index].sample_id
        simulation_id, window_start = _parse_window_sample_id(sample_id)
        if simulation_id not in result or window_start < result[simulation_id]:
            result[simulation_id] = window_start
    return result


_WINDOW_SAMPLE_ID = re.compile(r"^(?P<simulation>.+:\d+)@(?P<start>\d+)$")


def _parse_window_sample_id(sample_id: str) -> tuple[str, int]:
    match = _WINDOW_SAMPLE_ID.match(sample_id)
    if match is None:
        raise ValueError(f"sample_id does not contain a window start: {sample_id!r}")
    return match.group("simulation"), int(match.group("start"))
