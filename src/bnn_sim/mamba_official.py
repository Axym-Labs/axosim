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
    block_type: str = "mamba1"
    head_dim: int = 64
    num_groups: int = 1
    chunk_size: int = 256
    dropout: float = 0.0
    residual_scale_init: float = 0.1
    separate_heads: bool = False
    soma_filter: bool = False
    soma_filter_tau: float = 25.0
    soma_filter_kernel: int = 129
    learn_soma_filter_decay: bool = True
    soma_multiplicative_gate: bool = False
    soma_multiplicative_scale_init: float = 0.1
    branch_gain: bool = False
    branch_nonlinearity: str = "none"
    branch_nonlinearity_scale_init: float = 0.1
    branch_trace_taus: list[float] | None = None
    branch_trace_kernel: int = 65
    hidden_trace_taus: list[float] | None = None
    hidden_trace_kernel: int = 129
    temporal_refine_kernel: int = 0
    integrative_mixer: bool = False
    integrative_mixer_short_kernel: int = 9
    integrative_mixer_long_kernel: int = 65
    integrative_mixer_position: str = "pre"
    integrative_mixer_long_path_bias: float = 0.5


class BranchOfficialMamba(nn.Module):
    """Branch-routed wrapper around the official mamba-ssm Mamba module."""

    def __init__(self, config: BranchOfficialMambaConfig) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba, Mamba2
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
        if config.block_type not in {"mamba1", "mamba2"}:
            raise ValueError("block_type must be 'mamba1' or 'mamba2'")
        if config.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if config.num_groups <= 0:
            raise ValueError("num_groups must be positive")
        if config.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if config.soma_filter_tau <= 0:
            raise ValueError("soma_filter_tau must be positive")
        if config.soma_filter_kernel <= 0:
            raise ValueError("soma_filter_kernel must be positive")
        if config.soma_multiplicative_scale_init < 0:
            raise ValueError("soma_multiplicative_scale_init must be non-negative")
        if config.branch_nonlinearity not in {"none", "tanh"}:
            raise ValueError("branch_nonlinearity must be 'none' or 'tanh'")
        if config.branch_nonlinearity_scale_init <= 0:
            raise ValueError("branch_nonlinearity_scale_init must be positive")
        if config.branch_trace_kernel <= 0:
            raise ValueError("branch_trace_kernel must be positive")
        if config.hidden_trace_kernel <= 0:
            raise ValueError("hidden_trace_kernel must be positive")
        if config.temporal_refine_kernel < 0:
            raise ValueError("temporal_refine_kernel must be non-negative")
        if config.temporal_refine_kernel == 1:
            raise ValueError("temporal_refine_kernel must be 0 or >= 2")
        if config.integrative_mixer_short_kernel <= 0:
            raise ValueError("integrative_mixer_short_kernel must be positive")
        if config.integrative_mixer_long_kernel <= 0:
            raise ValueError("integrative_mixer_long_kernel must be positive")
        if config.integrative_mixer_position not in {"pre", "post", "both"}:
            raise ValueError("integrative_mixer_position must be 'pre', 'post', or 'both'")

        self.config = config
        self.num_input = config.num_input
        self.num_output = config.num_output
        self.num_branch = config.num_branch
        self.num_synapse_per_branch = config.num_synapse_per_branch
        self.input_to_synapse_routing = config.input_to_synapse_routing
        self.branch_trace_taus = tuple(float(tau) for tau in (config.branch_trace_taus or ()))
        self.hidden_trace_taus = tuple(float(tau) for tau in (config.hidden_trace_taus or ()))
        if any(tau <= 0 for tau in self.branch_trace_taus):
            raise ValueError("branch_trace_taus must be positive")
        if any(tau <= 0 for tau in self.hidden_trace_taus):
            raise ValueError("hidden_trace_taus must be positive")

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
        self.branch_projection = nn.Linear(config.num_branch * (1 + len(self.branch_trace_taus)), config.model_dim)
        self.integrative_pre = (
            _ShortLongTemporalMixer(
                config.model_dim,
                short_kernel=config.integrative_mixer_short_kernel,
                long_kernel=config.integrative_mixer_long_kernel,
                residual_scale_init=config.residual_scale_init,
                long_path_bias=config.integrative_mixer_long_path_bias,
            )
            if config.integrative_mixer and config.integrative_mixer_position in {"pre", "both"}
            else None
        )
        self.blocks = nn.ModuleList(
            [
                _OfficialMambaResidualBlock(
                    self._build_mamba_block(Mamba, Mamba2),
                    model_dim=config.model_dim,
                    dropout=config.dropout,
                    residual_scale_init=config.residual_scale_init,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.integrative_post = (
            _ShortLongTemporalMixer(
                config.model_dim,
                short_kernel=config.integrative_mixer_short_kernel,
                long_kernel=config.integrative_mixer_long_kernel,
                residual_scale_init=config.residual_scale_init,
                long_path_bias=config.integrative_mixer_long_path_bias,
            )
            if config.integrative_mixer and config.integrative_mixer_position in {"post", "both"}
            else None
        )
        self.temporal_refine = (
            _DepthwiseTemporalRefine(config.model_dim, config.temporal_refine_kernel, config.residual_scale_init)
            if config.temporal_refine_kernel > 0
            else None
        )
        self.norm = nn.LayerNorm(config.model_dim)
        readout_dim = config.model_dim * (1 + len(self.hidden_trace_taus))
        if config.separate_heads or config.soma_filter or config.soma_multiplicative_gate:
            self.spike_readout = nn.Linear(readout_dim, 1)
            self.soma_readout = nn.Linear(readout_dim, 1)
            self.readout = None
        else:
            self.readout = nn.Linear(readout_dim, config.num_output)
            self.spike_readout = None
            self.soma_readout = None
        if config.soma_multiplicative_gate:
            self.soma_gate_readout = nn.Linear(readout_dim, 1)
            self.soma_mod_readout = nn.Linear(readout_dim, 1)
            nn.init.zeros_(self.soma_mod_readout.weight)
            nn.init.zeros_(self.soma_mod_readout.bias)
            self.soma_multiplicative_scale = nn.Parameter(torch.tensor(float(config.soma_multiplicative_scale_init)))
        else:
            self.soma_gate_readout = None
            self.soma_mod_readout = None
            self.soma_multiplicative_scale = None
        if config.soma_filter:
            initial_decay = math.exp(-1.0 / config.soma_filter_tau)
            initial_logit = math.log(initial_decay / (1.0 - initial_decay))
            self.soma_filter_logit = nn.Parameter(
                torch.tensor(initial_logit, dtype=torch.float32),
                requires_grad=config.learn_soma_filter_decay,
            )
        else:
            self.soma_filter_logit = None

    def _build_mamba_block(self, mamba_cls: type[nn.Module], mamba2_cls: type[nn.Module]) -> nn.Module:
        if self.config.block_type == "mamba1":
            return mamba_cls(
                d_model=self.config.model_dim,
                d_state=self.config.state_dim,
                d_conv=self.config.conv_kernel,
                expand=self.config.expansion,
            )
        return mamba2_cls(
            d_model=self.config.model_dim,
            d_state=self.config.state_dim,
            d_conv=self.config.conv_kernel,
            expand=self.config.expansion,
            headdim=self.config.head_dim,
            ngroups=self.config.num_groups,
            chunk_size=self.config.chunk_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (batch, time, channels), got {tuple(x.shape)}")
        if x.shape[-1] != self.num_input:
            raise ValueError(f"expected num_input={self.num_input}, got {x.shape[-1]}")

        branch_inputs = self._branch_reduce(x)
        if self.branch_gain is not None:
            gain = self.branch_gain.to(dtype=branch_inputs.dtype, device=branch_inputs.device)
            branch_inputs = branch_inputs * gain.view(1, 1, -1)
        if self.branch_nonlinearity_log_scale is not None:
            scale = self.branch_nonlinearity_log_scale.exp().to(dtype=branch_inputs.dtype, device=branch_inputs.device)
            bias = self.branch_nonlinearity_bias.to(dtype=branch_inputs.dtype, device=branch_inputs.device)
            branch_inputs = torch.tanh(scale.view(1, 1, -1) * (branch_inputs + bias.view(1, 1, -1)))
            branch_inputs = branch_inputs / scale.view(1, 1, -1).clamp_min(torch.finfo(branch_inputs.dtype).eps)
        if self.branch_trace_taus:
            branch_inputs = torch.cat(
                [branch_inputs]
                + [
                    _causal_exponential_filter(branch_inputs, tau=tau, kernel_size=self.config.branch_trace_kernel)
                    for tau in self.branch_trace_taus
                ],
                dim=-1,
            )
        hidden = self.branch_projection(branch_inputs)
        if self.integrative_pre is not None:
            hidden = self.integrative_pre(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        if self.integrative_post is not None:
            hidden = self.integrative_post(hidden)
        if self.temporal_refine is not None:
            hidden = self.temporal_refine(hidden)
        features = self.norm(hidden)
        if self.hidden_trace_taus:
            features = torch.cat(
                [features]
                + [
                    _causal_exponential_filter(features, tau=tau, kernel_size=self.config.hidden_trace_kernel)
                    for tau in self.hidden_trace_taus
                ],
                dim=-1,
            )
        if self.readout is not None:
            return self.readout(features)
        assert self.spike_readout is not None and self.soma_readout is not None
        spike = self.spike_readout(features)
        soma = self.soma_readout(features)
        if self.soma_multiplicative_scale is not None:
            assert self.soma_gate_readout is not None and self.soma_mod_readout is not None
            gate = torch.sigmoid(self.soma_gate_readout(features))
            modulation = torch.tanh(self.soma_mod_readout(features))
            scale = self.soma_multiplicative_scale.to(dtype=soma.dtype, device=soma.device)
            soma = soma + scale * gate * soma * modulation
        if self.soma_filter_logit is not None:
            soma = self._filter_soma(soma)
        return torch.cat([spike, soma], dim=-1)

    def _filter_soma(self, soma: torch.Tensor) -> torch.Tensor:
        if self.soma_filter_logit is None:
            raise RuntimeError("soma filter is disabled")
        decay = torch.sigmoid(self.soma_filter_logit).to(dtype=soma.dtype, device=soma.device)
        powers = torch.arange(self.config.soma_filter_kernel, dtype=soma.dtype, device=soma.device)
        kernel_values = decay.pow(powers)
        kernel_values = kernel_values / kernel_values.sum().clamp_min(torch.finfo(kernel_values.dtype).eps)
        kernel = torch.flip(kernel_values, dims=[0]).view(1, 1, -1)
        filtered = F.conv1d(
            soma.transpose(1, 2),
            kernel,
            padding=self.config.soma_filter_kernel - 1,
        )[..., : soma.shape[1]]
        return filtered.transpose(1, 2)

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


class _DepthwiseTemporalRefine(nn.Module):
    def __init__(self, model_dim: int, kernel_size: int, residual_scale_init: float) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.norm = nn.LayerNorm(model_dim)
        self.conv = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=kernel_size,
            groups=model_dim,
            padding=kernel_size - 1,
        )
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        self.residual_scale = nn.Parameter(torch.full((model_dim,), float(residual_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.conv(self.norm(x).transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        return x + self.residual_scale.view(1, 1, -1) * update


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
        self.short_kernel = short_kernel
        self.long_kernel = long_kernel
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


def _causal_exponential_filter(x: torch.Tensor, *, tau: float, kernel_size: int) -> torch.Tensor:
    channels = x.shape[-1]
    decay = math.exp(-1.0 / tau)
    powers = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
    kernel_values = decay ** powers
    kernel_values = kernel_values / kernel_values.sum().clamp_min(torch.finfo(kernel_values.dtype).eps)
    kernel = torch.flip(kernel_values, dims=[0]).view(1, 1, kernel_size).repeat(channels, 1, 1)
    filtered = F.conv1d(
        x.transpose(1, 2),
        kernel,
        padding=kernel_size - 1,
        groups=channels,
    )[..., : x.shape[1]]
    return filtered.transpose(1, 2)
