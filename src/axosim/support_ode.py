from __future__ import annotations

import math

import torch
from torch import nn


SUPPORT_ODE_MODES = frozenset(
    {"synaptic", "generic", "polarity_swapped", "instantaneous"}
)


class SynapticSupportODE(nn.Module):
    """Fixed causal receptor states derived from the source teacher ODEs."""

    def __init__(self, *, num_branches: int, mode: str = "synaptic") -> None:
        super().__init__()
        if num_branches <= 0:
            raise ValueError("num_branches must be positive")
        if mode not in SUPPORT_ODE_MODES:
            raise ValueError(f"unsupported support ODE mode: {mode}")
        self.num_branches = int(num_branches)
        self.mode = mode

        if mode == "generic":
            receptors = ((1.0, 12.0, 1.0), (4.0, 24.0, 1.0), (0.8, 18.0, 2.5))
        else:
            receptors = ((0.3, 3.0, 1.0), (2.0, 70.0, 1.0), (0.2, 8.0, 2.5))
        state_matrix, readout_matrix = _support_matrices(
            num_branches=self.num_branches,
            receptors=receptors,
        )
        self.register_buffer("state_matrix", state_matrix, persistent=True)
        self.register_buffer("readout_matrix", readout_matrix, persistent=True)

    @property
    def output_dim(self) -> int:
        return 3 * self.num_branches

    def forward(
        self,
        excitatory: torch.Tensor,
        inhibitory: torch.Tensor,
    ) -> torch.Tensor:
        if excitatory.shape != inhibitory.shape:
            raise ValueError("excitatory and inhibitory support inputs must have equal shape")
        if excitatory.ndim != 3 or excitatory.shape[-1] != self.num_branches:
            raise ValueError(
                "support inputs must have shape (batch, time, num_branches)"
            )
        if self.mode == "polarity_swapped":
            drives = (inhibitory, inhibitory, excitatory)
        else:
            drives = (excitatory, excitatory, inhibitory)
        forcing = torch.cat(drives, dim=-1)
        if self.mode == "instantaneous":
            return forcing

        forcing_channels = forcing.transpose(1, 2).contiguous()
        state_matrix = self.state_matrix.to(
            dtype=torch.float32,
            device=forcing_channels.device,
        )
        readout_matrix = self.readout_matrix.to(
            dtype=torch.float32,
            device=forcing_channels.device,
        )
        if forcing_channels.is_cuda:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

            output = selective_scan_fn(
                forcing_channels,
                torch.ones_like(forcing_channels),
                state_matrix,
                torch.ones_like(state_matrix),
                readout_matrix,
            )
        else:
            output = _reference_scan(
                forcing_channels,
                state_matrix=state_matrix,
                readout_matrix=readout_matrix,
            )
        return output.transpose(1, 2).to(dtype=forcing.dtype)


def _support_matrices(
    *,
    num_branches: int,
    receptors: tuple[tuple[float, float, float], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    state_rows = []
    readout_rows = []
    for rise_ms, decay_ms, relative_gmax in receptors:
        if not 0 < rise_ms < decay_ms:
            raise ValueError("support ODE requires 0 < rise_ms < decay_ms")
        peak_time = (
            rise_ms
            * decay_ms
            / (decay_ms - rise_ms)
            * math.log(decay_ms / rise_ms)
        )
        factor = 1.0 / (
            -math.exp(-peak_time / rise_ms) + math.exp(-peak_time / decay_ms)
        )
        state_rows.extend([[-1.0 / rise_ms, -1.0 / decay_ms]] * num_branches)
        readout_rows.extend(
            [[-relative_gmax * factor, relative_gmax * factor]] * num_branches
        )
    return (
        torch.tensor(state_rows, dtype=torch.float32),
        torch.tensor(readout_rows, dtype=torch.float32),
    )


def _reference_scan(
    forcing: torch.Tensor,
    *,
    state_matrix: torch.Tensor,
    readout_matrix: torch.Tensor,
) -> torch.Tensor:
    forcing_float = forcing.float()
    decay = state_matrix.exp().unsqueeze(0)
    readout = readout_matrix.unsqueeze(0)
    state = forcing_float.new_zeros(
        forcing_float.shape[0], forcing_float.shape[1], state_matrix.shape[1]
    )
    outputs = []
    for value in forcing_float.unbind(dim=-1):
        state = decay * state + value.unsqueeze(-1)
        outputs.append((state * readout).sum(dim=-1))
    return torch.stack(outputs, dim=-1).to(dtype=forcing.dtype)
