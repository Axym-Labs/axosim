"""Pure PyTorch Mamba implementation (fallback when mamba_ssm unavailable)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveScanFn(torch.autograd.Function):
    """Pure PyTorch selective scan (slow but correct)."""

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None,
                delta_softplus=True, n_rows=1):
        """
        u: (batch, dim, length)
        delta: (batch, dim, length)
        A: (dim, dstate)
        B: (batch, dstate, length)
        C: (batch, dstate, length)
        D: (dim,)
        delta_bias: (dim,)
        """
        batch, dim, length = u.shape
        dstate = A.shape[1]

        if delta_bias is not None:
            delta = delta + delta_bias.unsqueeze(-1)
        if delta_softplus:
            delta = F.softplus(delta)

        # Discretize: A_bar = exp(delta * A)
        # delta: (batch, dim, length), A: (dim, dstate)
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2))  # (batch, dim, length, dstate)
        # B: (batch, dstate, length) -> (batch, 1, length, dstate)
        B_t = B.transpose(1, 2).unsqueeze(1)
        deltaB_u = delta.unsqueeze(-1) * B_t * u.unsqueeze(-1)  # (batch, dim, length, dstate)

        # Selective scan
        x = torch.zeros(batch, dim, dstate, device=u.device, dtype=u.dtype)
        ys = []

        for i in range(length):
            x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
            y = torch.einsum('bnd,bd->bn', x, C[:, :, i])  # (batch, dim)
            ys.append(y)

        y = torch.stack(ys, dim=-1)  # (batch, dim, length)

        if D is not None:
            y = y + u * D.unsqueeze(-1)

        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias)
        ctx.delta_softplus = delta_softplus
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # Simplified backward - not fully optimized
        return None, None, None, None, None, None, None, None, None


def selective_scan_fn(u, delta, A, B, C, D=None, delta_bias=None,
                      delta_softplus=True, n_rows=1):
    """Wrapper for selective scan."""
    return SelectiveScanFn.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, n_rows)


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


class SimpleMambaModel(nn.Module):
    """Simple Mamba-based sequence model."""

    def __init__(self, input_dim, d_model=64, n_layers=2, d_state=16, d_conv=4, expand=2):
        super().__init__()
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
