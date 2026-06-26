#!/usr/bin/env python3
"""Train Mamba (our primary model) for benchmark evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from axosim.mamba_pytorch import SimpleMambaModel
from axosim.data import ShardedNeuronIODataset


def train_epoch(model, dataset, batch_size, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for i in range(0, len(dataset), batch_size):
        batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        inputs = torch.stack([torch.from_numpy(b.inputs).float() for b in batch]).to(device)
        targets = torch.stack([torch.from_numpy(b.targets).float() for b in batch]).to(device)

        optimizer.zero_grad()
        preds = model(inputs)

        # Loss with burn-in ignore
        burn_in = 150
        soma_loss = nn.functional.mse_loss(preds[:, burn_in:, 1], targets[:, burn_in:, 1])
        spike_loss = nn.functional.binary_cross_entropy_with_logits(
            preds[:, burn_in:, 0], targets[:, burn_in:, 0]
        )
        loss = 0.5 * soma_loss + 0.5 * spike_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if n_batches % 10 == 0:
            print(f"  Batch {n_batches}, loss={loss.item():.4f}")

    return total_loss / n_batches


def evaluate(model, dataset, batch_size, device):
    """Evaluate on dataset."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            inputs = torch.stack([torch.from_numpy(b.inputs).float() for b in batch]).to(device)
            targets = torch.stack([torch.from_numpy(b.targets).float() for b in batch]).to(device)
            preds = model(inputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Compute metrics
    burn_in = 500
    rmse_mv = np.sqrt(np.mean((preds[:, burn_in:, 1] - targets[:, burn_in:, 1]) ** 2)) / 0.1

    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(targets[:, burn_in:, 0].flatten(), preds[:, burn_in:, 0].flatten())
    except ValueError:
        auc = 0.5

    return {"rmse_mv": float(rmse_mv), "auc": float(auc)}


def main():
    data_dir = "data/arbor-allen-nmda-source-medium-20260510"
    train_dir = f"{data_dir}/train"
    val_dir = f"{data_dir}/val"

    checkpoint_path = "runs/mamba_best_trained.pt"
    metrics_path = "runs/mamba_best_metrics.json"

    device = "cuda"
    epochs = 30
    batch_size = 16
    lr = 3e-4
    weight_decay = 1e-4
    seed = 42

    print(f"Training Mamba (primary model)")
    print(f"  Architecture: Pure PyTorch selective scan")
    print(f"  Train: {train_dir}")
    print(f"  Val:   {val_dir}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")

    # Create model
    model = SimpleMambaModel(
        input_dim=1278,
        d_model=64,
        n_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = ShardedNeuronIODataset(
        train_dir,
        shuffle=True,
        seed=seed,
        shuffle_mode="sample",
    )
    val_dataset = ShardedNeuronIODataset(
        val_dir,
        shuffle=False,
        seed=seed,
    )
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train
    print(f"\nStarting training...")
    start_time = time.time()
    best_val_rmse = float('inf')
    history = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss = train_epoch(model, train_dataset, batch_size, optimizer, device)
        val_metrics = evaluate(model, val_dataset, batch_size, device)
        scheduler.step()

        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val RMSE: {val_metrics['rmse_mv']:.2f} mV")
        print(f"  Val AUC: {val_metrics['auc']:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_rmse_mv": val_metrics["rmse_mv"],
            "val_auc": val_metrics["auc"],
        })

        # Save best
        if val_metrics['rmse_mv'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse_mv']
            torch.save({
                "model_kind": "mamba-pytorch",
                "config": {
                    "input_dim": 1278,
                    "d_model": 64,
                    "n_layers": 2,
                    "d_state": 16,
                    "d_conv": 4,
                    "expand": 2,
                },
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_rmse_mv": best_val_rmse,
            }, checkpoint_path)
            print(f"  >> Saved best checkpoint (RMSE: {best_val_rmse:.2f} mV)")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save final
    torch.save({
        "model_kind": "mamba-pytorch",
        "config": {
            "input_dim": 1278,
            "d_model": 64,
            "n_layers": 2,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
        },
        "state_dict": model.state_dict(),
        "history": history,
    }, "runs/mamba_best_final.pt")

    # Save metrics
    with open(metrics_path, "w") as f:
        json.dump({
            "best_val_rmse_mv": best_val_rmse,
            "history": history,
            "training_time_sec": elapsed,
        }, f, indent=2)

    print(f"\nBest validation RMSE: {best_val_rmse:.2f} mV")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
