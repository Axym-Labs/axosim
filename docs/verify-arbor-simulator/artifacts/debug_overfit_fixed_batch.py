#!/usr/bin/env python3
"""Overfit one fixed real-data batch to test model/training correctness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bnn_sim.cli import DEFAULT_MAMBA_OFFICIAL_MODEL_CONFIG
from bnn_sim.data import RandomFullTraceWindowDataset, ShardedNeuronIODataset
from bnn_sim.metrics import binary_auc
from bnn_sim.mamba_official import BranchOfficialMamba, BranchOfficialMambaConfig, BranchPyTorchMamba
from bnn_sim.mamba_pytorch import SimpleMambaModel
from bnn_sim.official import load_official_elm_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/arbor-allen-nmda-source-medium-20260510/train")
    parser.add_argument("--model-kind", choices=["mamba-official", "branch-mamba-pytorch", "mamba-pytorch"], default="mamba-official")
    parser.add_argument("--model-config", default=DEFAULT_MAMBA_OFFICIAL_MODEL_CONFIG)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--ignore-start", type=int, default=500)
    parser.add_argument("--full-trace-length", type=int, default=6000)
    parser.add_argument("--burn-in", type=int, default=150)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--output")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    base = ShardedNeuronIODataset(args.data, shuffle=False, cache_shards=1)
    dataset = RandomFullTraceWindowDataset(
        base,
        window_size=args.window_size,
        start_offset=args.ignore_start,
        samples_per_epoch=args.batch_size,
        batch_size=args.batch_size,
        shard_reuse_batches=1,
        sequence_length=args.full_trace_length,
        seed=args.seed,
    )
    x_np, y_np = dataset.get_batch(range(args.batch_size))
    x = torch.from_numpy(x_np).float().to(args.device)
    y = torch.from_numpy(y_np).float().to(args.device)

    model = build_model(args.model_kind, args.model_config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rows = []
    for step in range(1, args.steps + 1):
        model.train()
        pred = model(x)
        pred_trimmed = pred[:, args.burn_in :, :]
        target_trimmed = y[:, args.burn_in :, :]
        soma_loss = F.mse_loss(pred_trimmed[..., 1], target_trimmed[..., 1])
        spike_loss = F.binary_cross_entropy_with_logits(pred_trimmed[..., 0], target_trimmed[..., 0])
        loss = 0.5 * soma_loss + 0.5 * spike_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            row = metrics_row(step, loss, soma_loss, spike_loss, model, x, y, args.burn_in)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def build_model(model_kind: str, config_path: str) -> torch.nn.Module:
    if model_kind == "mamba-pytorch":
        config = load_official_elm_config(config_path)
        if "input_dim" in config:
            return SimpleMambaModel(**config)
        return SimpleMambaModel(input_dim=int(config.get("num_input", 1278)))
    config = BranchOfficialMambaConfig(**load_official_elm_config(config_path))
    if model_kind == "mamba-official":
        return BranchOfficialMamba(config)
    return BranchPyTorchMamba(config)


def metrics_row(
    step: int,
    loss: torch.Tensor,
    soma_loss: torch.Tensor,
    spike_loss: torch.Tensor,
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    burn_in: int,
) -> dict[str, float | int]:
    model.eval()
    with torch.inference_mode():
        pred = model(x)[:, burn_in:, :]
        target = y[:, burn_in:, :]
        residual = (pred[..., 1] - target[..., 1]).detach().cpu().numpy()
        spike_scores = pred[..., 0].detach().cpu().numpy()
        spike_labels = target[..., 0].detach().cpu().numpy()
    return {
        "step": step,
        "loss": float(loss.detach().cpu()),
        "soma_loss": float(soma_loss.detach().cpu()),
        "spike_loss": float(spike_loss.detach().cpu()),
        "rmse_mv": float(np.sqrt(np.mean(residual**2)) / 0.1),
        "auc": binary_auc(spike_scores.reshape(-1), spike_labels.reshape(-1)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
