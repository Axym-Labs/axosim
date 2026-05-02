from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from bnn_sim.data import ShardedNeuronIODataset
from bnn_sim.metrics import soma_rmse, spike_auc
from bnn_sim.model import BranchELM


def evaluate_dataset(
    model: BranchELM,
    dataset: ShardedNeuronIODataset,
    *,
    batch_size: int = 8,
    device: str = "cpu",
) -> dict[str, float | int]:
    model = model.to(device)
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            samples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            x = torch.from_numpy(np.stack([sample.inputs for sample in samples])).to(device)
            y = np.stack([sample.targets for sample in samples])
            pred = model(x).cpu().numpy()
            predictions.append(pred)
            targets.append(y)

    prediction_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    return {
        "samples": len(dataset),
        "soma_rmse": soma_rmse(prediction_array, target_array),
        "spike_auc": spike_auc(prediction_array, target_array),
    }


def write_metrics(metrics: dict[str, float | int], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
