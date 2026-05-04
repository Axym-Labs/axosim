from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bnn_sim.official.modeling_utils import (
    create_interlocking_indices,
    create_overlapping_window_indices,
)


@dataclass(frozen=True)
class BranchMambaLiteConfig:
    num_input: int = 1278
    num_output: int = 2
    num_branch: int = 45
    num_synapse_per_branch: int = 100
    input_to_synapse_routing: str | None = "neuronio_routing"
    model_dim: int = 64
    num_layers: int = 4
    conv_kernel: int = 9
    long_conv_kernel: int = 65
    expansion: int = 2
    dropout: float = 0.0
    residual_scale_init: float = 0.1
    long_path_bias: float = 0.5


class BranchMambaLite(nn.Module):
    """Mamba-inspired experimental model with parallel temporal mixing."""

    def __init__(self, config: BranchMambaLiteConfig) -> None:
        super().__init__()
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
        if config.conv_kernel <= 0:
            raise ValueError("conv_kernel must be positive")
        if config.expansion <= 0:
            raise ValueError("expansion must be positive")
        if config.long_conv_kernel <= 0:
            raise ValueError("long_conv_kernel must be positive")

        self.config = config
        self.num_input = config.num_input
        self.num_output = config.num_output
        self.num_branch = config.num_branch
        self.num_synapse_per_branch = config.num_synapse_per_branch
        self.input_to_synapse_routing = config.input_to_synapse_routing
        self.model_dim = config.model_dim
        self.num_layers = config.num_layers
        self.conv_kernel = config.conv_kernel
        self.expansion = config.expansion
        self.dropout = config.dropout

        routing_indices, valid_mask = self._create_input_to_synapse_indices()
        self.register_buffer("input_to_synapse_indices", routing_indices if routing_indices is not None else torch.empty(0, dtype=torch.long))
        self.register_buffer("valid_indices_mask", valid_mask if valid_mask is not None else torch.empty(0, dtype=torch.float32))

        self.branch_projection = nn.Linear(config.num_branch, config.model_dim)
        self.blocks = nn.ModuleList(
            [
                _ParallelMambaBlock(
                    model_dim=config.model_dim,
                    conv_kernel=config.conv_kernel,
                    long_conv_kernel=config.long_conv_kernel,
                    expansion=config.expansion,
                    dropout=config.dropout,
                    residual_scale_init=config.residual_scale_init,
                    long_path_bias=config.long_path_bias,
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

        branch_inputs = self._branch_reduce(x)
        hidden = self.branch_projection(branch_inputs)
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


class _ParallelMambaBlock(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        conv_kernel: int,
        long_conv_kernel: int,
        expansion: int,
        dropout: float,
        residual_scale_init: float,
        long_path_bias: float,
    ) -> None:
        super().__init__()
        expanded_dim = expansion * model_dim
        self.norm = nn.LayerNorm(model_dim)
        self.in_proj = nn.Linear(model_dim, expanded_dim)
        self.gate_proj = nn.Linear(model_dim, expanded_dim)
        self.short_conv = nn.Conv1d(
            expanded_dim,
            expanded_dim,
            kernel_size=conv_kernel,
            groups=expanded_dim,
            padding=conv_kernel - 1,
        )
        self.long_conv = nn.Conv1d(
            expanded_dim,
            expanded_dim,
            kernel_size=long_conv_kernel,
            groups=expanded_dim,
            padding=long_conv_kernel - 1,
            bias=False,
        )
        nn.init.constant_(self.long_conv.weight, 1.0 / long_conv_kernel)
        self.out_proj = nn.Linear(expanded_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.path_mix_logit = nn.Parameter(torch.full((expanded_dim,), float(long_path_bias)))
        self.residual_scale = nn.Parameter(torch.full((model_dim,), float(residual_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        value = self.in_proj(x)
        gate = torch.sigmoid(self.gate_proj(x))
        value_t = value.transpose(1, 2)
        short = self.short_conv(value_t)[..., : value.shape[1]].transpose(1, 2)
        long = self.long_conv(value_t)[..., : value.shape[1]].transpose(1, 2)
        path_mix = torch.sigmoid(self.path_mix_logit).view(1, 1, -1)
        mixed = (1.0 - path_mix) * F.silu(short) + path_mix * F.silu(long)
        mixed = mixed * gate
        update = self.dropout(self.out_proj(mixed))
        return residual + self.residual_scale.view(1, 1, -1) * update
