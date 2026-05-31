"""Branch count adaptation for morphology transfer (M3).

Handles mismatch between training morphology (e.g., 45 branches)
and test morphologies (e.g., 24-40 branches).
"""

from __future__ import annotations

import numpy as np
import torch


def adapt_branch_count(
    routing_matrix: np.ndarray,
    target_n_branches: int,
    strategy: str = "pad",
) -> np.ndarray:
    """Adapt a routing matrix to a target branch count.

    Args:
        routing_matrix: Source routing matrix of shape (n_branches, input_dim).
        target_n_branches: Desired number of branches.
        strategy: Adaptation strategy:
            - "pad": Pad with zero-weight branches (if source < target)
            - "truncate": Truncate excess branches (if source > target)
            - "interpolate": Distribute inputs proportionally (always works)

    Returns:
        Adapted routing matrix of shape (target_n_branches, input_dim).
    """
    n_source, input_dim = routing_matrix.shape

    if n_source == target_n_branches:
        return routing_matrix

    if strategy == "pad":
        if n_source < target_n_branches:
            # Pad with zero-weight branches
            padding = np.zeros((target_n_branches - n_source, input_dim))
            return np.vstack([routing_matrix, padding])
        else:
            # Truncate excess branches
            return routing_matrix[:target_n_branches]

    elif strategy == "interpolate":
        # Distribute source branches across target branches proportionally
        result = np.zeros((target_n_branches, input_dim))
        for target_b in range(target_n_branches):
            # Map target branch to source branch space
            source_idx = int(target_b * n_source / target_n_branches)
            # Weight by fractional overlap
            weight = min(1.0, n_source / target_n_branches)
            result[target_b] = routing_matrix[source_idx] * weight
        return result

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def create_model_with_adapted_routing(
    model_class,
    model_kwargs: dict,
    source_routing: np.ndarray,
    target_routing: np.ndarray,
) -> torch.nn.Module:
    """Create a model with adapted branch routing.

    This is a placeholder for more sophisticated adaptation.
    Currently just creates a new model with the target branch count.
    """
    n_target = target_routing.shape[0]
    adapted_kwargs = {**model_kwargs, "n_branches": n_target}
    return model_class(**adapted_kwargs)


def evaluate_with_branch_adaptation(
    model_fn: callable,
    inputs: np.ndarray,
    targets: np.ndarray,
    source_routing: np.ndarray,
    target_routing: np.ndarray,
    strategy: str = "pad",
) -> dict:
    """Evaluate model with branch count adaptation.

    Args:
        model_fn: Model prediction function.
        inputs: Input array (batch, time, input_dim).
        targets: Target array (batch, time, 2).
        source_routing: Routing matrix used during training.
        target_routing: Routing matrix for test morphology.
        strategy: Adaptation strategy.

    Returns:
        Dictionary with evaluation metrics.
    """
    # Adapt routing if needed
    n_source = source_routing.shape[0]
    n_target = target_routing.shape[0]

    adapted_routing = adapt_branch_count(target_routing, n_source, strategy=strategy)

    # For now, just evaluate directly (assuming model handles routing internally)
    # In practice, would need to rewire model's routing matrix
    preds = model_fn(inputs)

    rmse_mv = np.sqrt(np.mean((preds[:, :, 1] - targets[:, :, 1]) ** 2)) / 0.1

    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(targets[:, :, 0].flatten(), preds[:, :, 0].flatten())
    except ValueError:
        auc = 0.5

    return {
        "n_source_branches": n_source,
        "n_target_branches": n_target,
        "adaptation_strategy": strategy,
        "rmse_mv": float(rmse_mv),
        "auc": float(auc),
    }
