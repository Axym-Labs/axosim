from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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
    sparse_soma_loss_weight: float = 0.0,
    sparse_soma_high_voltage_quantile: float = 0.9,
    sparse_soma_high_dvdt_quantile: float = 0.9,
    sparse_soma_input_event_quantile: float = 0.95,
    sparse_soma_spike_window: int = 5,
    sparse_soma_post_event_window: int = 5,
    soma_slope_loss_weight: float = 0.0,
    grad_clip_norm: float = 0.0,
    update_log_interval: int = 0,
    update_log_path: str | Path | None = None,
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
    if sparse_soma_loss_weight < 0:
        raise ValueError("sparse_soma_loss_weight must be non-negative")
    if not 0 <= sparse_soma_high_voltage_quantile <= 1:
        raise ValueError("sparse_soma_high_voltage_quantile must be in [0, 1]")
    if not 0 <= sparse_soma_high_dvdt_quantile <= 1:
        raise ValueError("sparse_soma_high_dvdt_quantile must be in [0, 1]")
    if not 0 <= sparse_soma_input_event_quantile <= 1:
        raise ValueError("sparse_soma_input_event_quantile must be in [0, 1]")
    if sparse_soma_spike_window < 0:
        raise ValueError("sparse_soma_spike_window must be non-negative")
    if sparse_soma_post_event_window < 0:
        raise ValueError("sparse_soma_post_event_window must be non-negative")
    if soma_slope_loss_weight < 0:
        raise ValueError("soma_slope_loss_weight must be non-negative")
    if update_log_interval < 0:
        raise ValueError("update_log_interval must be non-negative")

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
    last_sparse_soma_loss: float | None = None
    last_sparse_soma_fraction: float | None = None
    last_soma_slope_loss: float | None = None
    history: list[dict[str, float | int]] = []
    best_validation_soma_rmse = float("inf")
    best_epoch = 0
    global_step = 0
    update_log_file = None
    if update_log_interval and update_log_path is not None:
        update_log_path = Path(update_log_path)
        update_log_path.parent.mkdir(parents=True, exist_ok=True)
        update_log_file = update_log_path.open("w", encoding="utf-8")
    _synchronize_device(device)
    total_start_time = time.perf_counter()

    try:
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
                pred_soma = pred_trimmed[..., 1]
                target_soma = target_trimmed[..., 1]
                spike_loss = bce(pred_trimmed[..., 0], target_trimmed[..., 0])
                soma_loss = mse(pred_soma, target_soma)
                loss = spike_loss_weight * spike_loss + soma_loss_weight * soma_loss
                sparse_soma_loss_value: float | None = None
                sparse_soma_fraction_value: float | None = None
                soma_slope_loss_value: float | None = None
                if sparse_soma_loss_weight:
                    sparse_mask = _sparse_soma_mask(
                        x[:, burn_in:, :],
                        target_trimmed,
                        high_voltage_quantile=sparse_soma_high_voltage_quantile,
                        high_dvdt_quantile=sparse_soma_high_dvdt_quantile,
                        input_event_quantile=sparse_soma_input_event_quantile,
                        spike_window=sparse_soma_spike_window,
                        post_event_window=sparse_soma_post_event_window,
                    )
                    sparse_weight = sparse_mask.to(dtype=pred_soma.dtype)
                    sparse_soma_loss = ((pred_soma - target_soma).square() * sparse_weight).sum()
                    sparse_soma_loss = sparse_soma_loss / sparse_weight.sum().clamp_min(1.0)
                    loss = loss + sparse_soma_loss_weight * sparse_soma_loss
                    sparse_soma_loss_value = float(sparse_soma_loss.detach().cpu())
                    sparse_soma_fraction_value = float(sparse_weight.mean().detach().cpu())
                    last_sparse_soma_loss = sparse_soma_loss_value
                    last_sparse_soma_fraction = sparse_soma_fraction_value
                if soma_slope_loss_weight and pred_soma.shape[1] > 1:
                    pred_slope = pred_soma[:, 1:] - pred_soma[:, :-1]
                    target_slope = target_soma[:, 1:] - target_soma[:, :-1]
                    soma_slope_loss = mse(pred_slope, target_slope)
                    loss = loss + soma_slope_loss_weight * soma_slope_loss
                    soma_slope_loss_value = float(soma_slope_loss.detach().cpu())
                    last_soma_slope_loss = soma_slope_loss_value
                l1_loss_value: float | None = None
                if l1_lambda:
                    l1_loss = sum(parameter.abs().sum() for parameter in model.parameters())
                    loss = loss + l1_lambda * l1_loss
                    l1_loss_value = float(l1_loss.detach().cpu())

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                last_loss = float(loss.detach().cpu())
                global_step += 1
                epoch_batches += 1

                if update_log_file is not None and global_step % update_log_interval == 0:
                    elapsed_seconds = time.perf_counter() - total_start_time
                    update_row: dict[str, float | int] = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "epoch_batch": batch_index + 1,
                        "presentations_seen": global_step * batch_size,
                        "loss": last_loss,
                        "spike_loss": float(spike_loss.detach().cpu()),
                        "soma_loss": float(soma_loss.detach().cpu()),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "elapsed_seconds": elapsed_seconds,
                        "train_samples_per_second": (global_step * batch_size) / elapsed_seconds if elapsed_seconds > 0 else float("inf"),
                    }
                    if sparse_soma_loss_value is not None:
                        update_row["sparse_soma_loss"] = sparse_soma_loss_value
                        update_row["sparse_soma_fraction"] = sparse_soma_fraction_value or 0.0
                    if soma_slope_loss_value is not None:
                        update_row["soma_slope_loss"] = soma_slope_loss_value
                    if l1_loss_value is not None:
                        update_row["l1_loss"] = l1_loss_value
                    update_log_file.write(json.dumps(update_row, sort_keys=True) + "\n")
                    update_log_file.flush()

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
            if last_sparse_soma_loss is not None:
                epoch_metrics["train_last_sparse_soma_loss"] = last_sparse_soma_loss
                epoch_metrics["train_last_sparse_soma_fraction"] = last_sparse_soma_fraction or 0.0
            if last_soma_slope_loss is not None:
                epoch_metrics["train_last_soma_slope_loss"] = last_soma_slope_loss
            if validation_dataset is not None:
                effective_validation_ignore_start = _effective_validation_ignore_start(
                    validation_dataset,
                    validation_metric_ignore_start,
                )
                validation_metrics = evaluate_dataset(
                    model,
                    validation_dataset,
                    batch_size=validation_batch_size or batch_size,
                    device=device,
                    soma_units=validation_soma_units,
                    ignore_start=effective_validation_ignore_start,
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
    finally:
        if update_log_file is not None:
            update_log_file.close()

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
        "sparse_soma_loss_weight": sparse_soma_loss_weight,
        "sparse_soma_high_voltage_quantile": sparse_soma_high_voltage_quantile,
        "sparse_soma_high_dvdt_quantile": sparse_soma_high_dvdt_quantile,
        "sparse_soma_input_event_quantile": sparse_soma_input_event_quantile,
        "sparse_soma_spike_window": sparse_soma_spike_window,
        "sparse_soma_post_event_window": sparse_soma_post_event_window,
        "soma_slope_loss_weight": soma_slope_loss_weight,
        "grad_clip_norm": grad_clip_norm,
        "update_log_interval": update_log_interval,
        "update_log_path": str(update_log_path) if update_log_path is not None else None,
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
    sparse_soma_loss_weight: float = 0.0,
    sparse_soma_high_voltage_quantile: float = 0.9,
    sparse_soma_high_dvdt_quantile: float = 0.9,
    sparse_soma_input_event_quantile: float = 0.95,
    sparse_soma_spike_window: int = 5,
    sparse_soma_post_event_window: int = 5,
    soma_slope_loss_weight: float = 0.0,
    grad_clip_norm: float = 0.0,
    update_log_interval: int = 0,
    update_log_path: str | Path | None = None,
    prefetch_batches: int = 0,
    pin_memory: bool = False,
    non_blocking: bool = True,
) -> dict[str, Any]:
    if update_log_interval and update_log_path is None:
        metrics_path_for_log = Path(metrics_path)
        update_log_path = metrics_path_for_log.with_name(f"{metrics_path_for_log.stem}-updates.jsonl")
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
        sparse_soma_loss_weight=sparse_soma_loss_weight,
        sparse_soma_high_voltage_quantile=sparse_soma_high_voltage_quantile,
        sparse_soma_high_dvdt_quantile=sparse_soma_high_dvdt_quantile,
        sparse_soma_input_event_quantile=sparse_soma_input_event_quantile,
        sparse_soma_spike_window=sparse_soma_spike_window,
        sparse_soma_post_event_window=sparse_soma_post_event_window,
        soma_slope_loss_weight=soma_slope_loss_weight,
        grad_clip_norm=grad_clip_norm,
        update_log_interval=update_log_interval,
        update_log_path=update_log_path,
        prefetch_batches=prefetch_batches,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
    )
    save_checkpoint(model, checkpoint_path, metadata=metrics)
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _sparse_soma_mask(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    high_voltage_quantile: float,
    high_dvdt_quantile: float,
    input_event_quantile: float,
    spike_window: int,
    post_event_window: int,
) -> torch.Tensor:
    soma = targets[..., 1].detach()
    spikes = targets[..., 0].detach() > 0.5
    soma_dvdt = torch.zeros_like(soma)
    soma_dvdt[:, 1:] = soma[:, 1:] - soma[:, :-1]
    high_voltage_threshold = torch.quantile(soma.flatten(), high_voltage_quantile)
    high_dvdt_threshold = torch.quantile(soma_dvdt.abs().flatten(), high_dvdt_quantile)
    high_voltage = soma >= high_voltage_threshold
    high_dvdt = soma_dvdt.abs() >= high_dvdt_threshold
    spike_near = _dilate_time_mask(spikes, spike_window, causal=False)
    input_activity = inputs.detach().abs().sum(dim=-1)
    positive_activity = input_activity[input_activity > 0]
    if positive_activity.numel():
        input_event_threshold = torch.quantile(positive_activity, input_event_quantile)
        input_event = input_activity >= input_event_threshold
    else:
        input_event = input_activity > 0
    post_event = _dilate_time_mask(input_event, post_event_window, causal=True)
    return high_voltage | high_dvdt | spike_near | post_event


def _dilate_time_mask(mask: torch.Tensor, window: int, *, causal: bool) -> torch.Tensor:
    if window <= 0:
        return mask
    values = mask.to(dtype=torch.float32).unsqueeze(1)
    if causal:
        dilated = F.max_pool1d(F.pad(values, (window, 0)), kernel_size=window + 1, stride=1)
    else:
        dilated = F.max_pool1d(values, kernel_size=2 * window + 1, stride=1, padding=window)
    return dilated.squeeze(1) > 0


def _synchronize_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _stack_batch(dataset: ShardedNeuronIODataset, indices: range) -> tuple[np.ndarray, np.ndarray]:
    samples = [dataset[i] for i in indices]
    return (
        np.stack([sample.inputs for sample in samples]).astype(np.float32, copy=False),
        np.stack([sample.targets for sample in samples]).astype(np.float32, copy=False),
    )


def _effective_validation_ignore_start(dataset: ShardedNeuronIODataset, ignore_start: int) -> int:
    if ignore_start <= 0:
        return ignore_start
    if len(dataset) == 0:
        return ignore_start
    sequence_length = int(dataset[0].inputs.shape[0])
    return 0 if ignore_start >= sequence_length else ignore_start


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
