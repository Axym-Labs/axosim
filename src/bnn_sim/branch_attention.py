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
class BranchAttentionConfig:
    num_input: int = 1278
    num_output: int = 2
    num_branch: int = 45
    num_synapse_per_branch: int = 100
    input_to_synapse_routing: str | None = "neuronio_routing"
    model_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ff_multiplier: int = 2
    dropout: float = 0.0
    residual_scale_init: float = 0.1
    position_encoding: str = "none"
    branch_gain: bool = False
    branch_nonlinearity: str = "none"
    branch_nonlinearity_scale_init: float = 0.1
    integrative_mixer: bool = False
    integrative_mixer_short_kernel: int = 9
    integrative_mixer_long_kernel: int = 65
    integrative_mixer_long_path_bias: float = 0.5
    soma_multiplicative_gate: bool = False
    soma_multiplicative_scale_init: float = 0.1


class BranchAttention(nn.Module):
    """Branch-routed causal self-attention surrogate."""

    def __init__(self, config: BranchAttentionConfig) -> None:
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
        if config.num_heads <= 0 or config.model_dim % config.num_heads != 0:
            raise ValueError("num_heads must divide model_dim")
        if config.ff_multiplier <= 0:
            raise ValueError("ff_multiplier must be positive")
        if config.position_encoding not in {"none", "sinusoidal"}:
            raise ValueError("position_encoding must be 'none' or 'sinusoidal'")
        if config.branch_nonlinearity not in {"none", "tanh"}:
            raise ValueError("branch_nonlinearity must be 'none' or 'tanh'")
        if config.branch_nonlinearity_scale_init <= 0:
            raise ValueError("branch_nonlinearity_scale_init must be positive")
        if config.integrative_mixer_short_kernel <= 0:
            raise ValueError("integrative_mixer_short_kernel must be positive")
        if config.integrative_mixer_long_kernel <= 0:
            raise ValueError("integrative_mixer_long_kernel must be positive")
        if config.soma_multiplicative_scale_init < 0:
            raise ValueError("soma_multiplicative_scale_init must be non-negative")

        self.config = config
        self.num_input = config.num_input
        self.num_output = config.num_output
        self.num_branch = config.num_branch
        self.num_synapse_per_branch = config.num_synapse_per_branch
        self.input_to_synapse_routing = config.input_to_synapse_routing

        routing_indices, valid_mask = self._create_input_to_synapse_indices()
        self.register_buffer("input_to_synapse_indices", routing_indices if routing_indices is not None else torch.empty(0, dtype=torch.long))
        self.register_buffer("valid_indices_mask", valid_mask if valid_mask is not None else torch.empty(0, dtype=torch.float32))

        if config.branch_gain:
            self.branch_gain = nn.Parameter(torch.ones(config.num_branch))
        else:
            self.branch_gain = None
        if config.branch_nonlinearity == "tanh":
            self.branch_nonlinearity_log_scale = nn.Parameter(
                torch.full((config.num_branch,), math.log(config.branch_nonlinearity_scale_init))
            )
            self.branch_nonlinearity_bias = nn.Parameter(torch.zeros(config.num_branch))
        else:
            self.branch_nonlinearity_log_scale = None
            self.branch_nonlinearity_bias = None

        self.branch_projection = nn.Linear(config.num_branch, config.model_dim)
        self.integrative_pre = (
            _ShortLongTemporalMixer(
                config.model_dim,
                short_kernel=config.integrative_mixer_short_kernel,
                long_kernel=config.integrative_mixer_long_kernel,
                residual_scale_init=config.residual_scale_init,
                long_path_bias=config.integrative_mixer_long_path_bias,
            )
            if config.integrative_mixer
            else None
        )
        self.blocks = nn.ModuleList(
            [
                _CausalAttentionBlock(
                    model_dim=config.model_dim,
                    num_heads=config.num_heads,
                    ff_multiplier=config.ff_multiplier,
                    dropout=config.dropout,
                    residual_scale_init=config.residual_scale_init,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.model_dim)
        if config.soma_multiplicative_gate:
            self.spike_readout = nn.Linear(config.model_dim, 1)
            self.soma_readout = nn.Linear(config.model_dim, 1)
            self.soma_gate_readout = nn.Linear(config.model_dim, 1)
            self.soma_mod_readout = nn.Linear(config.model_dim, 1)
            nn.init.zeros_(self.soma_mod_readout.weight)
            nn.init.zeros_(self.soma_mod_readout.bias)
            self.soma_multiplicative_scale = nn.Parameter(torch.tensor(float(config.soma_multiplicative_scale_init)))
            self.readout = None
        else:
            self.readout = nn.Linear(config.model_dim, config.num_output)
            self.spike_readout = None
            self.soma_readout = None
            self.soma_gate_readout = None
            self.soma_mod_readout = None
            self.soma_multiplicative_scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (batch, time, channels), got {tuple(x.shape)}")
        if x.shape[-1] != self.num_input:
            raise ValueError(f"expected num_input={self.num_input}, got {x.shape[-1]}")

        branch_inputs = self._branch_reduce(x)
        if self.branch_gain is not None:
            branch_inputs = branch_inputs * self.branch_gain.to(dtype=branch_inputs.dtype, device=branch_inputs.device).view(1, 1, -1)
        if self.branch_nonlinearity_log_scale is not None:
            scale = self.branch_nonlinearity_log_scale.exp().to(dtype=branch_inputs.dtype, device=branch_inputs.device)
            bias = self.branch_nonlinearity_bias.to(dtype=branch_inputs.dtype, device=branch_inputs.device)
            branch_inputs = torch.tanh(scale.view(1, 1, -1) * (branch_inputs + bias.view(1, 1, -1)))
            branch_inputs = branch_inputs / scale.view(1, 1, -1).clamp_min(torch.finfo(branch_inputs.dtype).eps)

        hidden = self.branch_projection(branch_inputs)
        if self.config.position_encoding == "sinusoidal":
            hidden = hidden + _sinusoidal_position_encoding(
                hidden.shape[1],
                hidden.shape[2],
                dtype=hidden.dtype,
                device=hidden.device,
            ).unsqueeze(0)
        if self.integrative_pre is not None:
            hidden = self.integrative_pre(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        features = self.norm(hidden)
        if self.readout is not None:
            return self.readout(features)

        assert self.spike_readout is not None and self.soma_readout is not None
        spike = self.spike_readout(features)
        soma = self.soma_readout(features)
        assert self.soma_gate_readout is not None and self.soma_mod_readout is not None
        gate = torch.sigmoid(self.soma_gate_readout(features))
        modulation = torch.tanh(self.soma_mod_readout(features))
        scale = self.soma_multiplicative_scale.to(dtype=soma.dtype, device=soma.device)
        soma = soma + scale * gate * soma * modulation
        return torch.cat([spike, soma], dim=-1)

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


class _CausalAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        ff_multiplier: int,
        dropout: float,
        residual_scale_init: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.attn_norm = nn.LayerNorm(model_dim)
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.attn_out = nn.Linear(model_dim, model_dim)
        self.attn_dropout = dropout
        self.ff_norm = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, ff_multiplier * model_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_multiplier * model_dim, model_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.attn_residual_scale = nn.Parameter(torch.full((model_dim,), float(residual_scale_init)))
        self.ff_residual_scale = nn.Parameter(torch.full((model_dim,), float(residual_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, model_dim = x.shape
        qkv = self.qkv(self.attn_norm(x))
        q, k, v = qkv.view(batch_size, time_steps, 3, self.num_heads, self.head_dim).unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch_size, time_steps, model_dim)
        x = x + self.attn_residual_scale.view(1, 1, -1) * self.dropout(self.attn_out(attn))
        ff_update = self.ff(self.ff_norm(x))
        return x + self.ff_residual_scale.view(1, 1, -1) * self.dropout(ff_update)


def _sinusoidal_position_encoding(
    length: int,
    dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / dim)
    )
    encoding = torch.zeros(length, dim, dtype=torch.float32, device=device)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    if dim > 1:
        encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


class _ShortLongTemporalMixer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        *,
        short_kernel: int,
        long_kernel: int,
        residual_scale_init: float,
        long_path_bias: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.short_conv = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=short_kernel,
            groups=model_dim,
            padding=short_kernel - 1,
        )
        self.long_conv = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=long_kernel,
            groups=model_dim,
            padding=long_kernel - 1,
            bias=False,
        )
        nn.init.constant_(self.long_conv.weight, 1.0 / long_kernel)
        self.path_mix_logit = nn.Parameter(torch.full((model_dim,), float(long_path_bias)))
        self.residual_scale = nn.Parameter(torch.full((model_dim,), float(residual_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.norm(x).transpose(1, 2)
        short = self.short_conv(value)[..., : x.shape[1]].transpose(1, 2)
        long = self.long_conv(value)[..., : x.shape[1]].transpose(1, 2)
        path_mix = torch.sigmoid(self.path_mix_logit).view(1, 1, -1)
        update = (1.0 - path_mix) * F.silu(short) + path_mix * F.silu(long)
        return x + self.residual_scale.view(1, 1, -1) * update
