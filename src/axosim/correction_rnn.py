from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CorrectionGRUConfig:
    """Compact GRU that replaces one selected native P4 output block."""

    input_dim: int = 58
    hidden_dim: int = 32
    history_blocks: int = 4
    patch_size: int = 4
    output_dim: int = 2

    def __post_init__(self) -> None:
        if min(
            self.input_dim,
            self.hidden_dim,
            self.history_blocks,
        ) <= 0:
            raise ValueError("correction GRU dimensions must be positive")
        if self.patch_size != 4:
            raise ValueError("the correction model replaces exact P4 blocks")
        if self.output_dim != 2:
            raise ValueError("the correction model predicts two outputs")


class CompactCorrectionGRU(nn.Module):
    """Small selected-history GRU trained against the mature P4 model."""

    def __init__(self, config: CorrectionGRUConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(
            config.input_dim,
            config.hidden_dim,
            bias=False,
        )
        self.gru = nn.GRU(
            config.hidden_dim,
            config.hidden_dim,
            batch_first=True,
        )
        self.output = nn.Linear(
            config.hidden_dim,
            config.patch_size * config.output_dim,
        )

    def forward(
        self,
        history: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            history.ndim != 3
            or history.shape[-1] != self.config.input_dim
            or history.shape[1] != self.config.history_blocks
        ):
            raise ValueError(
                "history must have shape "
                f"(batch, {self.config.history_blocks}, "
                f"{self.config.input_dim})"
            )
        hidden, _ = self.gru(self.input_projection(history), state)
        return self.output(hidden[:, -1]).view(
            history.shape[0],
            self.config.patch_size,
            self.config.output_dim,
        )
