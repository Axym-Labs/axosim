from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BlockDisagreement:
    """Teacher disagreement for each causal macro block."""

    voltage_rmse_mv: torch.Tensor
    event_mismatch_fraction: torch.Tensor
    spike_score_rmse: torch.Tensor
    combined: torch.Tensor


def block_disagreement_scores(
    surrogate: torch.Tensor,
    teacher: torch.Tensor,
    *,
    patch_size: int,
    spike_threshold: float = 0.0,
    soma_scale: float = 0.1,
    voltage_budget_mv: float = 0.5,
    event_budget_fraction: float = 0.08,
) -> BlockDisagreement:
    """Measure block-local disagreement without biological targets."""

    _validate_prediction_pair(
        surrogate,
        teacher,
        patch_size=patch_size,
    )
    if soma_scale <= 0 or voltage_budget_mv <= 0:
        raise ValueError("voltage scales must be positive")
    if event_budget_fraction <= 0:
        raise ValueError("event_budget_fraction must be positive")
    batch, steps, outputs = surrogate.shape
    blocks = steps // patch_size
    surrogate_blocks = surrogate.view(
        batch,
        blocks,
        patch_size,
        outputs,
    )
    teacher_blocks = teacher.view(
        batch,
        blocks,
        patch_size,
        outputs,
    )
    voltage_error_mv = (
        surrogate_blocks[..., 1] - teacher_blocks[..., 1]
    ) / soma_scale
    voltage_rmse_mv = voltage_error_mv.square().mean(dim=-1).sqrt()
    surrogate_events = surrogate_blocks[..., 0] >= spike_threshold
    teacher_events = teacher_blocks[..., 0] >= spike_threshold
    event_mismatch = (
        surrogate_events != teacher_events
    ).to(surrogate.dtype).mean(dim=-1)
    spike_score_rmse = (
        surrogate_blocks[..., 0] - teacher_blocks[..., 0]
    ).square().mean(dim=-1).sqrt()
    combined = torch.maximum(
        voltage_rmse_mv / voltage_budget_mv,
        event_mismatch / event_budget_fraction,
    )
    return BlockDisagreement(
        voltage_rmse_mv=voltage_rmse_mv,
        event_mismatch_fraction=event_mismatch,
        spike_score_rmse=spike_score_rmse,
        combined=combined,
    )


def top_fraction_correction_mask(
    scores: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    """Select the globally highest-scoring blocks with exact cardinality."""

    selected = top_fraction_correction_indices(scores, fraction)
    mask = torch.zeros(
        scores.numel(),
        dtype=torch.bool,
        device=scores.device,
    )
    mask[selected] = True
    return mask.view_as(scores)


def top_fraction_correction_indices(
    scores: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    """Return flat indices for a fixed correction-compute budget."""

    if scores.ndim != 2 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty two-dimensional tensor")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    count = (
        0
        if fraction == 0.0
        else min(scores.numel(), math.ceil(fraction * scores.numel()))
    )
    if count == 0:
        return torch.empty(
            0, dtype=torch.long, device=scores.device
        )
    return torch.topk(
        scores.flatten(),
        count,
        sorted=False,
    ).indices


def apply_block_corrections(
    surrogate: torch.Tensor,
    teacher: torch.Tensor,
    correction_mask: torch.Tensor,
    *,
    patch_size: int,
) -> torch.Tensor:
    """Replace selected surrogate blocks with mature-model outputs."""

    _validate_prediction_pair(
        surrogate,
        teacher,
        patch_size=patch_size,
    )
    expected = (surrogate.shape[0], surrogate.shape[1] // patch_size)
    if correction_mask.shape != expected:
        raise ValueError(
            f"correction_mask must have shape {expected}"
        )
    blocks = surrogate.view(
        surrogate.shape[0],
        expected[1],
        patch_size,
        surrogate.shape[2],
    )
    teacher_blocks = teacher.view_as(blocks)
    corrected = torch.where(
        correction_mask[:, :, None, None],
        teacher_blocks,
        blocks,
    )
    return corrected.reshape_as(surrogate)


def _validate_prediction_pair(
    surrogate: torch.Tensor,
    teacher: torch.Tensor,
    *,
    patch_size: int,
) -> None:
    if surrogate.shape != teacher.shape:
        raise ValueError("surrogate and teacher shapes must match")
    if surrogate.ndim != 3 or surrogate.shape[-1] < 2:
        raise ValueError(
            "predictions must have shape (batch, time, outputs>=2)"
        )
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if surrogate.shape[1] % patch_size:
        raise ValueError(
            "prediction time dimension must be divisible by patch_size"
        )
