from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bnn_sim.checkpoint import save_checkpoint
from bnn_sim.data import ShardedNeuronIODataset
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
) -> dict[str, float | int]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    torch.manual_seed(seed)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    last_loss = 0.0

    for epoch in range(1, epochs + 1):
        for start in range(0, len(dataset), batch_size):
            samples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            x = torch.from_numpy(np.stack([sample.inputs for sample in samples])).to(device)
            y = torch.from_numpy(np.stack([sample.targets for sample in samples])).to(device)

            pred = model(x)
            pred_trimmed = pred[:, burn_in:, :]
            target_trimmed = y[:, burn_in:, :]
            loss = bce(pred_trimmed[..., 0], target_trimmed[..., 0]) + mse(
                pred_trimmed[..., 1], target_trimmed[..., 1]
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

    return {
        "epoch": epochs,
        "last_loss": last_loss,
        "samples": len(dataset),
        "burn_in": burn_in,
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
) -> dict[str, float | int]:
    metrics = train_dataset(
        model,
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        burn_in=burn_in,
        device=device,
        seed=seed,
    )
    save_checkpoint(model, checkpoint_path, metadata=metrics)
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics
