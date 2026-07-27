from __future__ import annotations

from dataclasses import dataclass

import torch
from mamba_ssm import Mamba
from torch import nn


@dataclass(frozen=True)
class CorrectionMambaConfig:
    """Compact Mamba that replaces one selected native P4 output block."""

    input_dim: int = 58
    model_dim: int = 32
    state_dim: int = 8
    conv_kernel: int = 2
    expansion: int = 1
    history_blocks: int = 4
    patch_size: int = 4
    output_dim: int = 2

    def __post_init__(self) -> None:
        if min(
            self.input_dim,
            self.model_dim,
            self.state_dim,
            self.conv_kernel,
            self.expansion,
            self.history_blocks,
        ) <= 0:
            raise ValueError("correction Mamba dimensions must be positive")
        if self.patch_size != 4:
            raise ValueError("the correction model replaces exact P4 blocks")
        if self.output_dim != 2:
            raise ValueError("the correction model predicts two outputs")


class CompactCorrectionMamba(nn.Module):
    """Small selected-history Mamba distilled from the mature P4 model."""

    def __init__(self, config: CorrectionMambaConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(
            config.input_dim,
            config.model_dim,
            bias=False,
        )
        self.norm = nn.LayerNorm(config.model_dim)
        self.mamba = Mamba(
            d_model=config.model_dim,
            d_state=config.state_dim,
            d_conv=config.conv_kernel,
            expand=config.expansion,
            use_fast_path=True,
        )
        self.residual_scale = nn.Parameter(
            torch.full((config.model_dim,), 0.1)
        )
        self.output = nn.Linear(
            config.model_dim,
            config.patch_size * config.output_dim,
        )

    @property
    def output_width(self) -> int:
        return self.config.patch_size * self.config.output_dim

    def forward(self, history: torch.Tensor) -> torch.Tensor:
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
        hidden = self.input_projection(history)
        hidden = hidden + self.residual_scale.view(
            1, 1, -1
        ) * self.mamba(self.norm(hidden))
        return self.output(hidden[:, -1]).view(
            history.shape[0],
            self.config.patch_size,
            self.config.output_dim,
        )

    def allocate_replay_state(
        self,
        batch_size: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mamba.allocate_inference_cache(
            batch_size,
            1,
            dtype=dtype,
        )

    def recurrent_replay(
        self,
        history: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Replay a short selected history with persistent scratch buffers."""

        if (
            history.ndim != 3
            or history.shape[-1] != self.config.input_dim
            or history.shape[1] != self.config.history_blocks
        ):
            raise ValueError("history has incompatible shape")
        conv_state, ssm_state = state
        conv_state.zero_()
        ssm_state.zero_()
        projected = self.input_projection(history)
        hidden = None
        for step in range(self.config.history_blocks):
            hidden = projected[:, step : step + 1]
            update, _, _ = self.mamba.step(
                self.norm(hidden),
                conv_state,
                ssm_state,
            )
            hidden = (
                hidden
                + self.residual_scale.view(1, 1, -1) * update
            )
        assert hidden is not None
        return self.output(hidden[:, 0]).view(
            history.shape[0],
            self.config.patch_size,
            self.config.output_dim,
        )
