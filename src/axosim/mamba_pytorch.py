"""Pure PyTorch Mamba implementation (fallback when mamba_ssm unavailable)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan_fn(u, delta, A, B, C, D=None, delta_bias=None,
                      delta_softplus=True, n_rows=1):
    """Differentiable pure-PyTorch selective scan.

    This fallback is intentionally simple and slow. It must remain ordinary
    PyTorch code so autograd can train the Mamba trunk when fused kernels are
    unavailable.
    """
    batch, dim, length = u.shape
    dstate = A.shape[1]

    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(-1)
    if delta_softplus:
        delta = F.softplus(delta)

    delta_a = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2))
    b_t = B.transpose(1, 2).unsqueeze(1)
    delta_b_u = delta.unsqueeze(-1) * b_t * u.unsqueeze(-1)

    state = torch.zeros(batch, dim, dstate, device=u.device, dtype=u.dtype)
    outputs = []
    for index in range(length):
        state = delta_a[:, :, index, :] * state + delta_b_u[:, :, index, :]
        output = torch.einsum("bnd,bd->bn", state, C[:, :, index])
        outputs.append(output)

    y = torch.stack(outputs, dim=-1)
    if D is not None:
        y = y + u * D.unsqueeze(-1)
    return y


class MambaBlock(nn.Module):
    """Single Mamba SSM block (pure PyTorch)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Initialize A (S4D real initialization)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (batch, length, d_model)
        Returns: (batch, length, d_model)
        """
        batch, length, dim = x.shape

        # Input projection (split into x and residual)
        xz = self.in_proj(x)  # (batch, length, d_inner * 2)
        x_ssm, z = xz.chunk(2, dim=-1)  # each (batch, length, d_inner)

        # Convolution
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :length].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # SSM
        x_proj_out = self.x_proj(x_conv)  # (batch, length, d_state * 2 + 1)
        delta, B, C = torch.split(
            x_proj_out,
            [1, self.d_state, self.d_state],
            dim=-1
        )

        delta = F.softplus(self.dt_proj(delta))  # (batch, length, d_inner)

        # Transpose for selective scan
        x_conv_t = x_conv.transpose(1, 2)  # (batch, d_inner, length)
        delta_t = delta.transpose(1, 2)
        B_t = B.transpose(1, 2)
        C_t = C.transpose(1, 2)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Selective scan
        y = selective_scan_fn(
            x_conv_t, delta_t, A, B_t, C_t,
            self.D, delta_softplus=False,
        )

        y = y.transpose(1, 2)  # (batch, length, d_inner)

        # Gating
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        return output

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del max_seqlen
        device = kwargs.get("device", self.out_proj.weight.device)
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        state_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        return (
            torch.zeros(
                batch_size,
                self.d_inner,
                self.d_conv,
                device=device,
                dtype=conv_dtype,
            ),
            torch.zeros(
                batch_size,
                self.d_inner,
                self.d_state,
                device=device,
                dtype=state_dtype,
            ),
        )

    def step(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError("MambaBlock.step expects one timestep")
        x, z = self.in_proj(hidden_states[:, 0]).chunk(2, dim=-1)
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = x
        conv_weight = self.conv1d.weight[:, 0]
        x = (conv_state * conv_weight.unsqueeze(0)).sum(dim=-1)
        if self.conv1d.bias is not None:
            x = x + self.conv1d.bias
        x = F.silu(x)

        delta, state_input, state_readout = torch.split(
            self.x_proj(x),
            [1, self.d_state, self.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(delta))
        transition = -torch.exp(self.A_log.float())
        discrete_transition = torch.exp(
            torch.einsum("bd,dn->bdn", delta, transition)
        )
        discrete_input = torch.einsum("bd,bn->bdn", delta, state_input)
        ssm_state.copy_(
            ssm_state * discrete_transition
            + x.unsqueeze(-1) * discrete_input
        )
        output = torch.einsum(
            "bdn,bn->bd",
            ssm_state.to(dtype=x.dtype),
            state_readout,
        )
        output = (output + self.D.to(dtype=x.dtype) * x) * F.silu(z)
        return self.out_proj(output).unsqueeze(1), conv_state, ssm_state


class SimpleMambaModel(nn.Module):
    """Simple Mamba-based sequence model."""

    def __init__(self, input_dim, d_model=64, n_layers=2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.config = {
            "input_dim": int(input_dim),
            "d_model": int(d_model),
            "n_layers": int(n_layers),
            "d_state": int(d_state),
            "d_conv": int(d_conv),
            "expand": int(expand),
        }
        self.input_proj = nn.Linear(input_dim, d_model)

        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 2)  # [spike, soma]

    def forward(self, x):
        """
        x: (batch, length, input_dim)
        Returns: (batch, length, 2)
        """
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x) + x  # residual

        x = self.norm(x)
        return self.output_proj(x)
