from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BranchELMConfig:
    input_dim: int = 1278
    memory_units: int = 30
    num_branches: int = 32
    hidden_units: int = 64
    synapse_decay: float = 0.85
    memory_decay: float = 0.90

    @classmethod
    def branch_elm_30(cls, *, input_dim: int = 1278, num_branches: int = 32) -> "BranchELMConfig":
        return cls(input_dim=input_dim, memory_units=30, num_branches=num_branches)


class BranchELM(nn.Module):
    """Branch-factorized recurrent surrogate with spike and soma outputs."""

    def __init__(self, config: BranchELMConfig) -> None:
        super().__init__()
        if config.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if config.memory_units <= 0:
            raise ValueError("memory_units must be positive")
        if config.num_branches <= 0:
            raise ValueError("num_branches must be positive")

        self.config = config
        self.routing = _build_routing(config.input_dim, config.num_branches)
        self.register_buffer("routing_matrix", self.routing)

        self.input_gain = nn.Parameter(torch.ones(config.input_dim))
        self.integration = nn.Sequential(
            nn.Linear(config.num_branches + config.memory_units, config.hidden_units),
            nn.Tanh(),
            nn.Linear(config.hidden_units, config.memory_units),
            nn.Tanh(),
        )
        self.readout = nn.Linear(config.memory_units, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (batch, time, channels), got {tuple(x.shape)}")
        if x.shape[-1] != self.config.input_dim:
            raise ValueError(f"expected input_dim={self.config.input_dim}, got {x.shape[-1]}")

        batch_size, time_steps, _ = x.shape
        syn_state = torch.zeros(batch_size, self.config.input_dim, dtype=x.dtype, device=x.device)
        mem_state = torch.zeros(batch_size, self.config.memory_units, dtype=x.dtype, device=x.device)
        outputs = []

        routing = self.routing_matrix.to(dtype=x.dtype, device=x.device)
        input_gain = self.input_gain.to(dtype=x.dtype, device=x.device)

        for t in range(time_steps):
            syn_state = self.config.synapse_decay * syn_state + x[:, t, :] * input_gain
            branch_state = syn_state @ routing
            branch_state = branch_state / routing.sum(dim=0).clamp_min(1.0).to(dtype=x.dtype, device=x.device)
            recurrent_input = torch.cat([branch_state, self.config.memory_decay * mem_state], dim=-1)
            update = self.integration(recurrent_input)
            mem_state = self.config.memory_decay * mem_state + (1.0 - self.config.memory_decay) * update
            outputs.append(self.readout(mem_state))

        return torch.stack(outputs, dim=1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def _build_routing(input_dim: int, num_branches: int) -> torch.Tensor:
    routing = torch.zeros(input_dim, num_branches, dtype=torch.float32)
    for channel in range(input_dim):
        routing[channel, channel % num_branches] = 1.0
    return routing
