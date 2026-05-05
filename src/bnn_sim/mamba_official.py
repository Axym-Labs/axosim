from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from bnn_sim.official.modeling_utils import (
    create_interlocking_indices,
    create_overlapping_window_indices,
)


@dataclass(frozen=True)
class BranchOfficialMambaConfig:
    num_input: int = 1278
    num_output: int = 2
    num_branch: int = 45
    num_synapse_per_branch: int = 100
    input_to_synapse_routing: str | None = "neuronio_routing"
    model_dim: int = 64
    num_layers: int = 2
    state_dim: int = 16
    conv_kernel: int = 4
    expansion: int = 2
    dropout: float = 0.0
    residual_scale_init: float = 0.1


class BranchOfficialMamba(nn.Module):
    """Branch-routed wrapper around the official mamba-ssm Mamba module."""

    def __init__(self, config: BranchOfficialMambaConfig) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as error:
            raise ImportError(
                "mamba-ssm is required for BranchOfficialMamba. "
                "Use the mamba-official venv or install a compatible mamba-ssm wheel."
            ) from error

        if config.num_input <= 0:
            raise ValueError("num_input must be positive")
        if config.num_output <= 0:
            raise ValueError("num_output must be positive")
        if config.num_branch <= 0:
            raise ValueError("num_branch must be positive")
        if config.num_synapse_per_branch <= 0:
            raise ValueError("num_synapse_per_branch must be positive")
        if config.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if config.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if config.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if config.conv_kernel <= 0:
            raise ValueError("conv_kernel must be positive")
        if config.expansion <= 0:
            raise ValueError("expansion must be positive")

        self.config = config
        self.num_input = config.num_input
        self.num_output = config.num_output
        self.num_branch = config.num_branch
        self.num_synapse_per_branch = config.num_synapse_per_branch
        self.input_to_synapse_routing = config.input_to_synapse_routing

        routing_indices, valid_mask = self._create_input_to_synapse_indices()
        self.register_buffer("input_to_synapse_indices", routing_indices if routing_indices is not None else torch.empty(0, dtype=torch.long))
        self.register_buffer("valid_indices_mask", valid_mask if valid_mask is not None else torch.empty(0, dtype=torch.float32))

        self.branch_projection = nn.Linear(config.num_branch, config.model_dim)
        self.blocks = nn.ModuleList(
            [
                _OfficialMambaResidualBlock(
                    Mamba(
                        d_model=config.model_dim,
                        d_state=config.state_dim,
                        d_conv=config.conv_kernel,
                        expand=config.expansion,
                    ),
                    model_dim=config.model_dim,
                    dropout=config.dropout,
                    residual_scale_init=config.residual_scale_init,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.model_dim)
        self.readout = nn.Linear(config.model_dim, config.num_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (batch, time, channels), got {tuple(x.shape)}")
        if x.shape[-1] != self.num_input:
            raise ValueError(f"expected num_input={self.num_input}, got {x.shape[-1]}")

        hidden = self.branch_projection(self._branch_reduce(x))
        for block in self.blocks:
            hidden = block(hidden)
        return self.readout(self.norm(hidden))

    def _create_input_to_synapse_indices(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.input_to_synapse_routing is None:
            return None, None
        if self.input_to_synapse_routing != "neuronio_routing":
            raise ValueError(f"unsupported input_to_synapse_routing: {self.input_to_synapse_routing}")
        assert math.ceil(self.num_input / self.num_branch) <= self.num_synapse_per_branch
        interlocking_indices = create_interlocking_indices(self.num_input)
        overlapping_indices, valid_indices_mask = create_overlapping_window_indices(
            self.num_input,
            self.num_branch,
            self.num_synapse_per_branch,
        )
        input_to_synapse_indices = interlocking_indices[overlapping_indices]
        return input_to_synapse_indices.to(dtype=torch.long), valid_indices_mask.to(dtype=torch.float32)

    def _route_input_to_synapses(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_to_synapse_routing is None:
            return x
        routed = torch.index_select(x, 2, self.input_to_synapse_indices)
        return routed * self.valid_indices_mask.to(dtype=x.dtype, device=x.device)

    def _branch_reduce(self, x: torch.Tensor) -> torch.Tensor:
        routed = self._route_input_to_synapses(x)
        batch_size, time_steps, _ = routed.shape
        return routed.view(batch_size, time_steps, self.num_branch, -1).sum(dim=-1)


class _OfficialMambaResidualBlock(nn.Module):
    def __init__(
        self,
        mamba: nn.Module,
        *,
        model_dim: int,
        dropout: float,
        residual_scale_init: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.mamba = mamba
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.full((model_dim,), float(residual_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.dropout(self.mamba(self.norm(x)))
        return x + self.residual_scale.view(1, 1, -1) * update
