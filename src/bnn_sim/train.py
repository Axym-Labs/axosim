from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from bnn_sim.checkpoint import save_checkpoint
from bnn_sim.data import ShardedNeuronIODataset
from bnn_sim.evaluate import evaluate_dataset, _tensor_from_numpy
from bnn_sim.model import BranchELM


def train_dataset(
    model: BranchELM,
    dataset: ShardedNeuronIODataset,
    *,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    burn_in: int = 0,
    device: str = "cpu",
    seed: int = 0,
    validation_dataset: ShardedNeuronIODataset | None = None,
    validation_batch_size: int | None = None,
    best_checkpoint_path: str | Path | None = None,
    validation_soma_units: str = "millivolts",
    validation_metric_ignore_start: int = 0,
    validation_metric_mask_mode: str = "ignore-start",
    validation_metric_stitch_burn_in: int = 150,
    validation_soma_affine_calibration: bool = False,
    max_train_batches: int | None = None,
    lr_schedule: str = "constant",
    lr_schedule_steps: int | None = None,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    l1_lambda: float = 0.0,
    spike_loss_weight: float = 0.5,
    soma_loss_weight: float = 0.5,
    prefetch_batches: int = 0,
    pin_memory: bool = False,
    non_blocking: bool = True,
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if prefetch_batches < 0:
        raise ValueError("prefetch_batches must be non-negative")

    torch.manual_seed(seed)
    model = model.to(device)
    model.train()
    optimizer = _build_optimizer(model, optimizer_name, learning_rate, weight_decay)
    batches_per_epoch = _batches_per_epoch(len(dataset), batch_size, max_train_batches)
    total_train_steps = epochs * batches_per_epoch
    scheduler = _build_scheduler(optimizer, lr_schedule, lr_schedule_steps or total_train_steps)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    last_loss = 0.0
    history: list[dict[str, float | int]] = []
    best_validation_soma_rmse = float("inf")
    best_epoch = 0
    _synchronize_device(device)
    total_start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        if hasattr(dataset, "reshuffle"):
            dataset.reshuffle(seed + epoch - 1)
        _synchronize_device(device)
        epoch_start_time = time.perf_counter()
        epoch_batches = 0
        for batch_index, x_np, y_np in _iter_epoch_batches(
            dataset,
            batch_size,
            max_train_batches,
            prefetch_batches=prefetch_batches,
        ):
            x = _tensor_from_numpy(x_np, device, pin_memory=pin_memory, non_blocking=non_blocking)
            y = _tensor_from_numpy(y_np, device, pin_memory=pin_memory, non_blocking=non_blocking)

            pred = model(x)
            pred_trimmed = pred[:, burn_in:, :]
            target_trimmed = y[:, burn_in:, :]
            loss = spike_loss_weight * bce(
                pred_trimmed[..., 0], target_trimmed[..., 0]
            ) + soma_loss_weight * mse(pred_trimmed[..., 1], target_trimmed[..., 1])
            if l1_lambda:
                loss = loss + l1_lambda * sum(parameter.abs().sum() for parameter in model.parameters())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            last_loss = float(loss.detach().cpu())
            epoch_batches += 1

        _synchronize_device(device)
        epoch_seconds = time.perf_counter() - epoch_start_time
        train_samples = epoch_batches * batch_size
        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train_last_loss": last_loss,
            "train_samples": train_samples,
            "train_batches": epoch_batches,
            "epoch_seconds": epoch_seconds,
            "train_samples_per_second": train_samples / epoch_seconds if epoch_seconds > 0 else float("inf"),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if validation_dataset is not None:
            validation_metrics = evaluate_dataset(
                model,
                validation_dataset,
                batch_size=validation_batch_size or batch_size,
                device=device,
                soma_units=validation_soma_units,
                ignore_start=validation_metric_ignore_start,
                mask_mode=validation_metric_mask_mode,
                stitch_burn_in=validation_metric_stitch_burn_in,
                soma_affine_calibration=validation_soma_affine_calibration,
                pin_memory=pin_memory,
                non_blocking=non_blocking,
            )
            epoch_metrics.update(
                {
                    "validation_samples": int(validation_metrics["samples"]),
                    "validation_metric_ignore_start": int(validation_metrics["metric_ignore_start"]),
                    "validation_metric_mask_mode": str(validation_metrics["metric_mask_mode"]),
                    "validation_metric_stitch_burn_in": validation_metrics["metric_stitch_burn_in"],
                    "validation_soma_affine_calibration": bool(validation_metrics["soma_affine_calibration"]),
                    "validation_soma_rmse": float(validation_metrics["soma_rmse"]),
                    "validation_soma_rmse_units": str(validation_metrics["soma_rmse_units"]),
                    "validation_spike_auc": float(validation_metrics["spike_auc"]),
                }
            )
            if float(validation_metrics["soma_rmse"]) < best_validation_soma_rmse:
                best_validation_soma_rmse = float(validation_metrics["soma_rmse"])
                best_epoch = epoch
                if best_checkpoint_path is not None:
                    save_checkpoint(model, best_checkpoint_path, metadata=epoch_metrics)
        history.append(epoch_metrics)

    _synchronize_device(device)
    total_seconds = time.perf_counter() - total_start_time
    return {
        "epoch": epochs,
        "last_loss": last_loss,
        "samples": len(dataset),
        "train_presentations": sum(int(row["train_samples"]) for row in history),
        "burn_in": burn_in,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lr_schedule": lr_schedule,
        "lr_schedule_steps": lr_schedule_steps,
        "optimizer": optimizer_name,
        "weight_decay": weight_decay,
        "l1_lambda": l1_lambda,
        "spike_loss_weight": spike_loss_weight,
        "soma_loss_weight": soma_loss_weight,
        "prefetch_batches": prefetch_batches,
        "pin_memory": pin_memory,
        "non_blocking": non_blocking,
        "max_train_batches": max_train_batches,
        "device": device,
        "total_seconds": total_seconds,
        "train_samples_per_second": sum(int(row["train_samples"]) for row in history) / total_seconds if total_seconds > 0 else float("inf"),
        "best_epoch": best_epoch,
        "best_validation_soma_rmse": best_validation_soma_rmse if validation_dataset is not None else None,
        "validation_soma_units": validation_soma_units if validation_dataset is not None else None,
        "validation_metric_ignore_start": validation_metric_ignore_start if validation_dataset is not None else None,
        "validation_metric_mask_mode": validation_metric_mask_mode if validation_dataset is not None else None,
        "validation_metric_stitch_burn_in": validation_metric_stitch_burn_in if validation_dataset is not None else None,
        "validation_soma_affine_calibration": validation_soma_affine_calibration if validation_dataset is not None else None,
        "history": history,
    }


def train_and_save(
    model: BranchELM,
    dataset: ShardedNeuronIODataset,
    *,
    checkpoint_path: str | Path,
    metrics_path: str | Path,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    burn_in: int = 0,
    device: str = "cpu",
    seed: int = 0,
    validation_dataset: ShardedNeuronIODataset | None = None,
    validation_batch_size: int | None = None,
    best_checkpoint_path: str | Path | None = None,
    validation_soma_units: str = "millivolts",
    validation_metric_ignore_start: int = 0,
    validation_metric_mask_mode: str = "ignore-start",
    validation_metric_stitch_burn_in: int = 150,
    validation_soma_affine_calibration: bool = False,
    max_train_batches: int | None = None,
    lr_schedule: str = "constant",
    lr_schedule_steps: int | None = None,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    l1_lambda: float = 0.0,
    spike_loss_weight: float = 0.5,
    soma_loss_weight: float = 0.5,
    prefetch_batches: int = 0,
    pin_memory: bool = False,
    non_blocking: bool = True,
) -> dict[str, Any]:
    metrics = train_dataset(
        model,
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        burn_in=burn_in,
        device=device,
        seed=seed,
        validation_dataset=validation_dataset,
        validation_batch_size=validation_batch_size,
        best_checkpoint_path=best_checkpoint_path,
        validation_soma_units=validation_soma_units,
        validation_metric_ignore_start=validation_metric_ignore_start,
        validation_metric_mask_mode=validation_metric_mask_mode,
        validation_metric_stitch_burn_in=validation_metric_stitch_burn_in,
        validation_soma_affine_calibration=validation_soma_affine_calibration,
        max_train_batches=max_train_batches,
        lr_schedule=lr_schedule,
        lr_schedule_steps=lr_schedule_steps,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        l1_lambda=l1_lambda,
        spike_loss_weight=spike_loss_weight,
        soma_loss_weight=soma_loss_weight,
        prefetch_batches=prefetch_batches,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
    )
    save_checkpoint(model, checkpoint_path, metadata=metrics)
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _synchronize_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _stack_batch(dataset: ShardedNeuronIODataset, indices: range) -> tuple[np.ndarray, np.ndarray]:
    samples = [dataset[i] for i in indices]
    return (
        np.stack([sample.inputs for sample in samples]).astype(np.float32, copy=False),
        np.stack([sample.targets for sample in samples]).astype(np.float32, copy=False),
    )


def _load_batch(dataset: ShardedNeuronIODataset, indices: range) -> tuple[np.ndarray, np.ndarray]:
    return dataset.get_batch(indices) if hasattr(dataset, "get_batch") else _stack_batch(dataset, indices)


def _iter_epoch_batches(
    dataset: ShardedNeuronIODataset,
    batch_size: int,
    max_train_batches: int | None,
    *,
    prefetch_batches: int = 0,
):
    ranges = [
        range(start, min(start + batch_size, len(dataset)))
        for batch_index, start in enumerate(range(0, len(dataset), batch_size))
        if max_train_batches is None or batch_index < max_train_batches
    ]
    if prefetch_batches <= 0:
        for batch_index, batch_indices in enumerate(ranges):
            x_np, y_np = _load_batch(dataset, batch_indices)
            yield batch_index, x_np, y_np
        return

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_load_batch, dataset, ranges[0])
        for batch_index in range(len(ranges)):
            x_np, y_np = future.result()
            next_index = batch_index + 1
            future = executor.submit(_load_batch, dataset, ranges[next_index]) if next_index < len(ranges) else None
            yield batch_index, x_np, y_np


def _batches_per_epoch(dataset_size: int, batch_size: int, max_train_batches: int | None) -> int:
    available = (dataset_size + batch_size - 1) // batch_size
    return min(available, max_train_batches) if max_train_batches is not None else available


def _build_scheduler(optimizer: torch.optim.Optimizer, lr_schedule: str, total_train_steps: int):
    if lr_schedule == "constant":
        return None
    if lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_train_steps, 1))
    raise ValueError("lr_schedule must be 'constant' or 'cosine'")


def _build_optimizer(
    model: torch.nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError("optimizer_name must be 'adam' or 'adamw'")
