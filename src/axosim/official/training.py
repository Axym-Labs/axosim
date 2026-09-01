"""Training utilities for the published Branch-ELM NeuronIO contract."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def official_neuronio_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    burn_in: int = 0,
) -> torch.Tensor:
    """Return the published equal-weight spike-logit and soma regression loss."""

    if (
        prediction.shape != target.shape
        or prediction.ndim != 3
        or prediction.shape[-1] != 2
    ):
        raise ValueError(
            "prediction and target must have matching (batch, time, 2) shapes"
        )
    if burn_in < 0 or burn_in >= prediction.shape[1]:
        raise ValueError("burn_in must leave at least one training timestep")
    predicted = prediction[:, burn_in:]
    expected = target[:, burn_in:]
    spike_loss = F.binary_cross_entropy_with_logits(
        predicted[..., 0],
        expected[..., 0],
    )
    soma_loss = F.mse_loss(
        predicted[..., 1],
        expected[..., 1],
    )
    return 0.5 * spike_loss + 0.5 * soma_loss
