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
class BranchSelectiveMambaConfig:
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
    dt_rank: int = 8
    dt_min: float = 0.001
    dt_max: float = 0.1
    dropout: float = 0.0
    residual_scale_init: float = 0.1


class BranchSelectiveMamba(nn.Module):
    """Branch-routed Mamba-v1 style selective SSM probe.

    This is a dependency-free reference implementation of the Mamba mechanism:
    causal depthwise conv, input-dependent delta/B/C terms, diagonal SSM state,
    D skip, and gated output projection. The scan is intentionally plain
    PyTorch, so it tests the mechanism rather than optimized deployment speed.
    """

    def __init__(self, config: BranchSelectiveMambaConfig) -> None:
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
        if config.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if config.conv_kernel <= 0:
            raise ValueError("conv_kernel must be positive")
        if config.expansion <= 0:
            raise ValueError("expansion must be positive")
        if config.dt_rank <= 0:
            raise ValueError("dt_rank must be positive")
        if config.dt_min <= 0 or config.dt_max <= config.dt_min:
            raise ValueError("dt_min/dt_max must satisfy 0 < dt_min < dt_max")

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
                _SelectiveMambaBlock(
                    model_dim=config.model_dim,
                    state_dim=config.state_dim,
                    conv_kernel=config.conv_kernel,
                    expansion=config.expansion,
                    dt_rank=config.dt_rank,
                    dt_min=config.dt_min,
                    dt_max=config.dt_max,
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


class _SelectiveMambaBlock(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        state_dim: int,
        conv_kernel: int,
        expansion: int,
        dt_rank: int,
        dt_min: float,
        dt_max: float,
        dropout: float,
        residual_scale_init: float,
    ) -> None:
        super().__init__()
        inner_dim = expansion * model_dim
        self.inner_dim = inner_dim
        self.state_dim = state_dim
        self.conv_kernel = conv_kernel

        self.norm = nn.LayerNorm(model_dim)
        self.in_proj = nn.Linear(model_dim, 2 * inner_dim)
        self.conv1d = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=conv_kernel,
            groups=inner_dim,
            padding=conv_kernel - 1,
        )
        self.x_proj = nn.Linear(inner_dim, dt_rank + 2 * state_dim, bias=False)
        self.dt_proj = nn.Linear(dt_rank, inner_dim)
        self.out_proj = nn.Linear(inner_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.full((model_dim,), float(residual_scale_init)))

        a = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(inner_dim, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(inner_dim))
        self._init_dt(dt_min, dt_max)

    def _init_dt(self, dt_min: float, dt_max: float) -> None:
        dt = torch.exp(torch.empty(self.inner_dim).uniform_(math.log(dt_min), math.log(dt_max)))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x_and_z = self.in_proj(x)
        x_part, z = x_and_z.chunk(2, dim=-1)
        x_conv = self.conv1d(x_part.transpose(1, 2))[..., : x_part.shape[1]].transpose(1, 2)
        x_conv = F.silu(x_conv)
        scan_input = self.x_proj(x_conv)
        dt_latent, b_t, c_t = torch.split(scan_input, [self.dt_proj.in_features, self.state_dim, self.state_dim], dim=-1)
        dt = F.softplus(self.dt_proj(dt_latent))
        y = self._selective_scan(x_conv, dt, b_t, c_t)
        y = y * F.silu(z)
        update = self.dropout(self.out_proj(y))
        return residual + self.residual_scale.view(1, 1, -1) * update

    def _selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        b_t: torch.Tensor,
        c_t: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, time_steps, _ = x.shape
        state = torch.zeros(batch_size, self.inner_dim, self.state_dim, dtype=x.dtype, device=x.device)
        a = -torch.exp(self.A_log).to(dtype=x.dtype, device=x.device)
        d = self.D.to(dtype=x.dtype, device=x.device)
        outputs = []
        for t in range(time_steps):
            dt_t = dt[:, t, :]
            x_t = x[:, t, :]
            d_a = torch.exp(dt_t.unsqueeze(-1) * a.unsqueeze(0))
            d_b_x = dt_t.unsqueeze(-1) * b_t[:, t, :].unsqueeze(1) * x_t.unsqueeze(-1)
            state = d_a * state + d_b_x
            y_t = (state * c_t[:, t, :].unsqueeze(1)).sum(dim=-1) + d * x_t
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)
