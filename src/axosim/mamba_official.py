from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from axosim.official.modeling_utils import (
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
    block_repeats: int = 1
    state_dim: int = 16
    conv_kernel: int = 4
    expansion: int = 2
    block_type: str = "mamba1"
    head_dim: int = 64
    num_groups: int = 1
    chunk_size: int = 256
    block_norm: bool = True
    final_norm: bool = True
    dropout: float = 0.0
    residual_scale_init: float = 0.1
    separate_heads: bool = False
    soma_filter: bool = False
    soma_filter_tau: float = 25.0
    soma_filter_kernel: int = 129
    learn_soma_filter_decay: bool = True
    soma_multiplicative_gate: bool = False
    soma_multiplicative_scale_init: float = 0.1
    soma_peak_correction: bool = False
    soma_peak_correction_scale_init: float = 0.1
    soma_highpass_correction: bool = False
    soma_highpass_correction_scale_init: float = 0.1
    spike_voltage_coupling_scale: float = 0.0
    morphology_ids: list[str] | None = None
    morphology_embedding_scale: float | None = None
    synapse_gain_scale: float | None = None
    morphology_synapse_gain_scale: float | None = None
    morphology_synapse_feature_dim: int = 0
    morphology_synapse_feature_hidden: int = 0
    morphology_synapse_feature_scale: float | None = None
    local_tcn_scale: float | None = None
    local_tcn_mode: str = "dilated"
    local_tcn_position: str = "parallel"
    soma_tcn_scale: float | None = None
    soma_tcn_coupling_source: str = "corrected"
    branch_gain: bool = False
    branch_nonlinearity: str = "none"
    branch_nonlinearity_scale_init: float = 0.1
    branch_subunits: int = 0
    branch_subunit_routing: str = "contiguous"
    branch_subunit_scale: float = 1.0
    branch_bilinear_rank: int = 0
    branch_bilinear_scale: float | None = None
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

    def __init__(
        self,
        config: BranchOfficialMambaConfig,
        *,
        mamba_classes: tuple[type[nn.Module], type[nn.Module] | None] | None = None,
    ) -> None:
        super().__init__()
        if mamba_classes is None:
            try:
                from mamba_ssm import Mamba, Mamba2
            except ImportError as error:
                raise ImportError(
                    "mamba-ssm is required for BranchOfficialMamba. "
                    "Use the mamba-official venv or install a compatible mamba-ssm wheel."
                ) from error
        else:
            Mamba, Mamba2 = mamba_classes

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
        if config.block_repeats <= 0:
            raise ValueError("block_repeats must be positive")
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
        if config.morphology_ids is None and config.morphology_embedding_scale is not None:
            raise ValueError("morphology_embedding_scale requires morphology_ids")
        if config.morphology_ids is not None:
            if not config.morphology_ids:
                raise ValueError("morphology_ids must not be empty")
            if len(set(config.morphology_ids)) != len(config.morphology_ids):
                raise ValueError("morphology_ids must be unique")
            if config.morphology_embedding_scale is not None and config.morphology_embedding_scale < 0:
                raise ValueError("morphology_embedding_scale must be non-negative")
        if config.soma_filter_tau <= 0:
            raise ValueError("soma_filter_tau must be positive")
        if config.soma_filter_kernel <= 0:
            raise ValueError("soma_filter_kernel must be positive")
        if config.soma_multiplicative_scale_init < 0:
            raise ValueError("soma_multiplicative_scale_init must be non-negative")
        if config.soma_peak_correction_scale_init < 0:
            raise ValueError("soma_peak_correction_scale_init must be non-negative")
        if config.soma_highpass_correction_scale_init < 0:
            raise ValueError("soma_highpass_correction_scale_init must be non-negative")
        if config.synapse_gain_scale is not None and config.synapse_gain_scale < 0:
            raise ValueError("synapse_gain_scale must be non-negative")
        if config.morphology_synapse_gain_scale is not None:
            if config.morphology_ids is None:
                raise ValueError("morphology_synapse_gain_scale requires morphology_ids")
            if config.morphology_synapse_gain_scale < 0:
                raise ValueError("morphology_synapse_gain_scale must be non-negative")
        feature_operator_enabled = (
            config.morphology_synapse_feature_dim > 0
            or config.morphology_synapse_feature_hidden > 0
            or config.morphology_synapse_feature_scale is not None
        )
        if feature_operator_enabled:
            if config.morphology_ids is None:
                raise ValueError("morphology_synapse_feature operator requires morphology_ids")
            if config.morphology_synapse_feature_dim <= 0:
                raise ValueError("morphology_synapse_feature_dim must be positive")
            if config.morphology_synapse_feature_hidden <= 0:
                raise ValueError("morphology_synapse_feature_hidden must be positive")
            if config.morphology_synapse_feature_scale is None or config.morphology_synapse_feature_scale < 0:
                raise ValueError("morphology_synapse_feature_scale must be non-negative")
        local_tcn_enabled = config.local_tcn_scale is not None
        if local_tcn_enabled and config.local_tcn_scale is not None and config.local_tcn_scale < 0:
            raise ValueError("local_tcn_scale must be non-negative")
        if config.local_tcn_mode not in {"dilated", "undilated"}:
            raise ValueError("local_tcn_mode must be 'dilated' or 'undilated'")
        if config.local_tcn_position not in {"parallel", "interlayer", "post"}:
            raise ValueError(
                "local_tcn_position must be 'parallel', 'interlayer', or 'post'"
            )
        if config.soma_tcn_scale is not None and config.soma_tcn_scale < 0:
            raise ValueError("soma_tcn_scale must be non-negative")
        if config.soma_tcn_coupling_source not in {"corrected", "base"}:
            raise ValueError("soma_tcn_coupling_source must be 'corrected' or 'base'")
        if local_tcn_enabled and config.local_tcn_position == "interlayer" and config.num_layers < 2:
            raise ValueError("interlayer local TCN requires at least two Mamba layers")
        if config.branch_nonlinearity not in {"none", "tanh"}:
            raise ValueError("branch_nonlinearity must be 'none' or 'tanh'")
        if config.branch_nonlinearity_scale_init <= 0:
            raise ValueError("branch_nonlinearity_scale_init must be positive")
        if config.branch_subunits < 0:
            raise ValueError("branch_subunits must be non-negative")
        if config.branch_subunit_routing not in {"dense", "contiguous"}:
            raise ValueError("branch_subunit_routing must be 'dense' or 'contiguous'")
        if config.branch_subunit_scale < 0:
            raise ValueError("branch_subunit_scale must be non-negative")
        if config.branch_bilinear_rank < 0:
            raise ValueError("branch_bilinear_rank must be non-negative")
        if (config.branch_bilinear_rank > 0) != (config.branch_bilinear_scale is not None):
            raise ValueError("positive branch_bilinear_rank and branch_bilinear_scale are required together")
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
        self.synapse_log_gain = (
            nn.Parameter(torch.zeros(config.num_branch, config.num_synapse_per_branch))
            if config.synapse_gain_scale is not None
            else None
        )
        self.morphology_synapse_log_gain = (
            nn.Parameter(
                torch.zeros(
                    len(config.morphology_ids),
                    config.num_branch,
                    config.num_synapse_per_branch,
                )
            )
            if config.morphology_synapse_gain_scale is not None
            else None
        )
        feature_values = (
            torch.zeros(
                len(config.morphology_ids),
                config.num_input,
                config.morphology_synapse_feature_dim,
            )
            if feature_operator_enabled
            else torch.empty(0)
        )
        self.register_buffer(
            "morphology_synapse_features",
            feature_values,
            persistent=feature_operator_enabled,
        )
        self.branch_subunit_log_gain = (
            nn.Parameter(
                torch.zeros(
                    config.num_branch,
                    config.branch_subunits,
                    config.num_synapse_per_branch,
                )
            )
            if config.branch_subunits
            else None
        )
        self.register_buffer("branch_subunit_mask", self._create_branch_subunit_mask())

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
        if config.branch_bilinear_rank:
            self.branch_bilinear_u = nn.Linear(config.num_branch, config.branch_bilinear_rank, bias=False)
            self.branch_bilinear_v = nn.Linear(config.num_branch, config.branch_bilinear_rank, bias=False)
            self.branch_bilinear_out = nn.Linear(config.branch_bilinear_rank, config.model_dim, bias=False)
            nn.init.zeros_(self.branch_bilinear_out.weight)
        else:
            self.branch_bilinear_u = None
            self.branch_bilinear_v = None
            self.branch_bilinear_out = None
        branch_projection_features = config.num_branch * (1 + len(self.branch_trace_taus))
        self.branch_projection = nn.Linear(branch_projection_features, config.model_dim)
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
                self._build_residual_block(Mamba, Mamba2)
                for _ in range(config.num_layers)
            ]
        )
        local_dilations = (1, 4, 16) if config.local_tcn_mode == "dilated" else (1, 1, 1)
        self.local_tcn_blocks = nn.ModuleList(
            [
                _CausalDepthwiseTCNBlock(config.model_dim, dilation=dilation)
                for dilation in local_dilations
            ]
            if local_tcn_enabled
            else []
        )
        self.local_tcn_out = (
            nn.Linear(config.model_dim, config.model_dim, bias=False)
            if local_tcn_enabled
            else None
        )
        if self.local_tcn_out is not None:
            nn.init.zeros_(self.local_tcn_out.weight)
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
        self.norm = nn.LayerNorm(config.model_dim) if config.final_norm else nn.Identity()
        readout_dim = config.model_dim * (1 + len(self.hidden_trace_taus))
        if config.separate_heads or config.soma_filter or config.soma_multiplicative_gate or config.soma_peak_correction:
            self.spike_readout = nn.Linear(readout_dim, 1)
            self.soma_readout = nn.Linear(readout_dim, 1)
            self.readout = None
        else:
            self.readout = nn.Linear(readout_dim, config.num_output)
            self.spike_readout = None
            self.soma_readout = None
        self.base_soma_prediction: torch.Tensor | None = None
        self.soma_tcn_blocks = nn.ModuleList(
            [
                _CausalDepthwiseTCNBlock(readout_dim, dilation=dilation)
                for dilation in (1, 4, 16)
            ]
            if config.soma_tcn_scale is not None
            else []
        )
        self.soma_tcn_out = (
            nn.Linear(readout_dim, 1, bias=False)
            if config.soma_tcn_scale is not None
            else None
        )
        if self.soma_tcn_out is not None:
            nn.init.zeros_(self.soma_tcn_out.weight)
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
        if config.soma_peak_correction:
            self.soma_peak_gate_readout = nn.Linear(readout_dim, 1)
            self.soma_peak_delta_readout = nn.Linear(readout_dim, 1)
            nn.init.zeros_(self.soma_peak_delta_readout.weight)
            nn.init.zeros_(self.soma_peak_delta_readout.bias)
            self.soma_peak_correction_scale = nn.Parameter(torch.tensor(float(config.soma_peak_correction_scale_init)))
        else:
            self.soma_peak_gate_readout = None
            self.soma_peak_delta_readout = None
            self.soma_peak_correction_scale = None
        if config.soma_highpass_correction:
            self.soma_highpass_readout = nn.Linear(config.num_branch, 2)
            nn.init.zeros_(self.soma_highpass_readout.weight[1])
            nn.init.zeros_(self.soma_highpass_readout.bias[1])
            self.soma_highpass_correction_scale = nn.Parameter(
                torch.tensor(float(config.soma_highpass_correction_scale_init))
            )
        else:
            self.soma_highpass_readout = None
            self.soma_highpass_correction_scale = None
        if config.soma_filter:
            initial_decay = math.exp(-1.0 / config.soma_filter_tau)
            initial_logit = math.log(initial_decay / (1.0 - initial_decay))
            self.soma_filter_logit = nn.Parameter(
                torch.tensor(initial_logit, dtype=torch.float32),
                requires_grad=config.learn_soma_filter_decay,
            )
        else:
            self.soma_filter_logit = None
        self.morphology_embedding = (
            nn.Embedding(len(config.morphology_ids), config.model_dim)
            if config.morphology_embedding_scale is not None
            else None
        )
        self.branch_subunit_projection = (
            nn.Linear(config.num_branch * config.branch_subunits, config.model_dim)
            if config.branch_subunits
            else None
        )
        if self.branch_subunit_projection is not None:
            nn.init.zeros_(self.branch_subunit_projection.weight)
            nn.init.zeros_(self.branch_subunit_projection.bias)
        self.morphology_synapse_feature_in = (
            nn.Linear(
                config.morphology_synapse_feature_dim,
                config.morphology_synapse_feature_hidden,
            )
            if feature_operator_enabled
            else None
        )
        self.morphology_synapse_feature_out = (
            nn.Linear(config.morphology_synapse_feature_hidden, 1, bias=False)
            if feature_operator_enabled
            else None
        )
        if self.morphology_synapse_feature_out is not None:
            nn.init.zeros_(self.morphology_synapse_feature_out.weight)

    def _build_mamba_block(self, mamba_cls: type[nn.Module], mamba2_cls: type[nn.Module]) -> nn.Module:
        return self._build_mamba_block_for_dim(mamba_cls, mamba2_cls, self.config.model_dim)

    def _build_mamba_block_for_dim(
        self,
        mamba_cls: type[nn.Module],
        mamba2_cls: type[nn.Module],
        model_dim: int,
    ) -> nn.Module:
        if self.config.block_type == "mamba1":
            return mamba_cls(
                d_model=model_dim,
                d_state=self.config.state_dim,
                d_conv=self.config.conv_kernel,
                expand=self.config.expansion,
            )
        if mamba2_cls is None:
            raise ValueError("block_type='mamba2' is not supported by this Mamba backend")
        return mamba2_cls(
            d_model=model_dim,
            d_state=self.config.state_dim,
            d_conv=self.config.conv_kernel,
            expand=self.config.expansion,
            headdim=self.config.head_dim,
            ngroups=self.config.num_groups,
            chunk_size=self.config.chunk_size,
        )

    def _build_residual_block(self, mamba_cls: type[nn.Module], mamba2_cls: type[nn.Module]) -> nn.Module:
        return _OfficialMambaResidualBlock(
            self._build_mamba_block_for_dim(mamba_cls, mamba2_cls, self.config.model_dim),
            model_dim=self.config.model_dim,
            dropout=self.config.dropout,
            residual_scale_init=self.config.residual_scale_init,
            use_norm=self.config.block_norm,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        morphology_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (batch, time, channels), got {tuple(x.shape)}")
        if x.shape[-1] != self.num_input:
            raise ValueError(f"expected num_input={self.num_input}, got {x.shape[-1]}")

        routed_synapses = self._weighted_routed_synapses(x, morphology_indices=morphology_indices)
        branch_inputs = routed_synapses.sum(dim=-1)
        if self.branch_gain is not None:
            gain = self.branch_gain.to(dtype=branch_inputs.dtype, device=branch_inputs.device)
            branch_inputs = branch_inputs * gain.view(1, 1, -1)
        if self.config.branch_nonlinearity == "tanh":
            assert self.branch_nonlinearity_log_scale is not None
            assert self.branch_nonlinearity_bias is not None
            scale = self.branch_nonlinearity_log_scale.exp().to(dtype=branch_inputs.dtype, device=branch_inputs.device)
            bias = self.branch_nonlinearity_bias.to(dtype=branch_inputs.dtype, device=branch_inputs.device)
            branch_inputs = torch.tanh(scale.view(1, 1, -1) * (branch_inputs + bias.view(1, 1, -1)))
            branch_inputs = branch_inputs / scale.view(1, 1, -1).clamp_min(torch.finfo(branch_inputs.dtype).eps)
        bilinear_inputs = branch_inputs
        highpass_features = None
        if self.soma_highpass_correction_scale is not None:
            branch_delta = torch.zeros_like(branch_inputs)
            branch_delta[:, 1:] = branch_inputs[:, 1:] - branch_inputs[:, :-1]
            highpass_features = branch_delta
        projection_inputs = branch_inputs
        subunit_projection = None
        if self.branch_trace_taus:
            projection_inputs = torch.cat(
                [branch_inputs]
                + [
                    _causal_exponential_filter(branch_inputs, tau=tau, kernel_size=self.config.branch_trace_kernel)
                    for tau in self.branch_trace_taus
                ],
                dim=-1,
            )
        if self.branch_subunit_projection is not None:
            subunit_inputs = self._branch_subunit_features(routed_synapses)
            if self.branch_gain is not None:
                gain = self.branch_gain.to(dtype=subunit_inputs.dtype, device=subunit_inputs.device)
                subunit_inputs = subunit_inputs * gain.view(1, 1, -1, 1)
            if self.config.branch_nonlinearity == "tanh":
                assert self.branch_nonlinearity_log_scale is not None
                assert self.branch_nonlinearity_bias is not None
                scale = self.branch_nonlinearity_log_scale.exp().to(
                    dtype=subunit_inputs.dtype,
                    device=subunit_inputs.device,
                )
                bias = self.branch_nonlinearity_bias.to(
                    dtype=subunit_inputs.dtype,
                    device=subunit_inputs.device,
                )
                subunit_inputs = torch.tanh(
                    scale.view(1, 1, -1, 1)
                    * (subunit_inputs + bias.view(1, 1, -1, 1))
                )
                subunit_inputs = subunit_inputs / scale.view(1, 1, -1, 1).clamp_min(
                    torch.finfo(subunit_inputs.dtype).eps
                )
            subunit_projection = self.branch_subunit_projection(
                subunit_inputs.flatten(start_dim=-2)
            )
        hidden = self.branch_projection(projection_inputs)
        if subunit_projection is not None:
            hidden = hidden + self.config.branch_subunit_scale * subunit_projection
        bilinear_projection = self._branch_bilinear_projection(bilinear_inputs)
        hidden = self._apply_projected_branch_bilinear(hidden, bilinear_projection)
        if self.morphology_embedding is not None:
            if morphology_indices is None:
                raise ValueError("morphology_indices are required for morphology-conditioned AxoMamba")
            if morphology_indices.ndim != 1 or morphology_indices.shape[0] != x.shape[0]:
                raise ValueError("morphology_indices must have shape (batch,)")
            morphology = self.morphology_embedding(
                morphology_indices.to(dtype=torch.long, device=hidden.device)
            ).to(dtype=hidden.dtype)
            assert self.config.morphology_embedding_scale is not None
            hidden = hidden + self.config.morphology_embedding_scale * morphology.unsqueeze(1)
        elif morphology_indices is not None and self.config.morphology_ids is None:
            raise ValueError("morphology_indices were provided to an unconditioned AxoMamba")
        local_source = hidden
        if self.integrative_pre is not None:
            hidden = self.integrative_pre(hidden)
        for _ in range(self.config.block_repeats):
            for layer_index, block in enumerate(self.blocks):
                hidden = block(hidden)
                if layer_index == 0:
                    if self.config.local_tcn_position == "interlayer":
                        hidden = hidden + self._local_tcn_correction(hidden)
        if self.integrative_post is not None:
            hidden = self.integrative_post(hidden)
        if self.temporal_refine is not None:
            hidden = self.temporal_refine(hidden)
        if self.config.local_tcn_position == "parallel":
            hidden = hidden + self._local_tcn_correction(local_source)
        elif self.config.local_tcn_position == "post":
            hidden = hidden + self._local_tcn_correction(hidden)
        features = self.norm(hidden)
        if self.hidden_trace_taus:
            features = torch.cat(
                [features]
                + [
                    _causal_exponential_filter(
                        features,
                        tau=tau,
                        kernel_size=self.config.hidden_trace_kernel,
                    )
                    for tau in self.hidden_trace_taus
                ],
                dim=-1,
            )
        if self.readout is not None:
            output = self.readout(features)
            spike = output[..., 0:1]
            base_soma = output[..., 1:2]
            self.base_soma_prediction = base_soma
            soma = self._apply_soma_tcn(features, base_soma)
            if self.config.spike_voltage_coupling_scale:
                coupling_soma = (
                    base_soma
                    if self.config.soma_tcn_coupling_source == "base"
                    else soma
                )
                spike = self._couple_spike_voltage(spike, coupling_soma)
            return torch.cat([spike, soma], dim=-1)
        assert self.spike_readout is not None and self.soma_readout is not None
        spike = self.spike_readout(features)
        base_soma = self._readout_soma(features, highpass_features)
        self.base_soma_prediction = base_soma
        soma = self._apply_soma_tcn(features, base_soma)
        if self.config.spike_voltage_coupling_scale:
            coupling_soma = (
                base_soma
                if self.config.soma_tcn_coupling_source == "base"
                else soma
            )
            spike = self._couple_spike_voltage(spike, coupling_soma)
        return torch.cat([spike, soma], dim=-1)

    def _local_tcn_correction(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.local_tcn_out is None:
            return torch.zeros_like(hidden)
        assert self.config.local_tcn_scale is not None
        local = hidden
        for block in self.local_tcn_blocks:
            local = block(local)
        return self.config.local_tcn_scale * self.local_tcn_out(local)

    def _apply_branch_bilinear(
        self,
        hidden: torch.Tensor,
        branch_inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self._apply_projected_branch_bilinear(
            hidden,
            self._branch_bilinear_projection(branch_inputs),
        )

    def _branch_bilinear_projection(
        self,
        branch_inputs: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.branch_bilinear_out is None:
            return None
        assert self.branch_bilinear_u is not None and self.branch_bilinear_v is not None
        first_factor = F.silu(self.branch_bilinear_u(branch_inputs))
        second_factor = F.silu(self.branch_bilinear_v(branch_inputs))
        interaction = first_factor * second_factor
        return self.branch_bilinear_out(interaction)

    def _apply_projected_branch_bilinear(
        self,
        hidden: torch.Tensor,
        projected: torch.Tensor | None,
    ) -> torch.Tensor:
        if projected is None:
            return hidden
        assert self.config.branch_bilinear_scale is not None
        return hidden + self.config.branch_bilinear_scale * projected

    def _readout_soma(
        self,
        soma_features: torch.Tensor,
        highpass_features: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.soma_readout is not None
        soma = self.soma_readout(soma_features)
        if self.soma_multiplicative_scale is not None:
            assert self.soma_gate_readout is not None and self.soma_mod_readout is not None
            gate = torch.sigmoid(self.soma_gate_readout(soma_features))
            modulation = torch.tanh(self.soma_mod_readout(soma_features))
            scale = self.soma_multiplicative_scale.to(dtype=soma.dtype, device=soma.device)
            soma = soma + scale * gate * soma * modulation
        if self.soma_peak_correction_scale is not None:
            assert self.soma_peak_gate_readout is not None and self.soma_peak_delta_readout is not None
            gate = torch.sigmoid(self.soma_peak_gate_readout(soma_features))
            delta = torch.tanh(self.soma_peak_delta_readout(soma_features))
            scale = self.soma_peak_correction_scale.to(dtype=soma.dtype, device=soma.device)
            soma = soma + scale * gate * delta
        if self.soma_highpass_correction_scale is not None:
            assert highpass_features is not None
            assert self.soma_highpass_readout is not None
            gate_delta = self.soma_highpass_readout(highpass_features)
            gate = torch.sigmoid(gate_delta[..., 0:1])
            delta = torch.tanh(gate_delta[..., 1:2])
            scale = self.soma_highpass_correction_scale.to(dtype=soma.dtype, device=soma.device)
            soma = soma + scale * gate * delta
        if self.soma_filter_logit is not None:
            soma = self._filter_soma(soma)
        return soma

    def _couple_spike_voltage(self, spike: torch.Tensor, soma: torch.Tensor) -> torch.Tensor:
        return spike + self.config.spike_voltage_coupling_scale * soma

    def _apply_soma_tcn(
        self, features: torch.Tensor, soma: torch.Tensor
    ) -> torch.Tensor:
        if self.soma_tcn_out is None:
            return soma
        local = features
        for block in self.soma_tcn_blocks:
            local = block(local)
        assert self.config.soma_tcn_scale is not None
        return soma + self.config.soma_tcn_scale * self.soma_tcn_out(local)

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

    def _create_branch_subunit_mask(self) -> torch.Tensor:
        if not self.config.branch_subunits:
            return torch.empty(0, dtype=torch.float32)
        if self.config.branch_subunit_routing == "dense":
            return torch.ones(
                self.config.branch_subunits,
                self.num_synapse_per_branch,
                dtype=torch.float32,
            )
        slot_indices = torch.arange(self.num_synapse_per_branch)
        assignments = torch.div(
            slot_indices * self.config.branch_subunits,
            self.num_synapse_per_branch,
            rounding_mode="floor",
        ).clamp_max(self.config.branch_subunits - 1)
        mask = F.one_hot(assignments, num_classes=self.config.branch_subunits).T.to(torch.float32)
        counts = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return mask * torch.sqrt(torch.tensor(float(self.num_synapse_per_branch)) / counts)

    def _route_input_to_synapses(
        self,
        x: torch.Tensor,
        *,
        morphology_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.input_to_synapse_routing is None:
            return x
        routed = torch.index_select(x, 2, self.input_to_synapse_indices)
        return routed * self.valid_indices_mask.to(dtype=x.dtype, device=x.device)

    def _branch_reduce(
        self,
        x: torch.Tensor,
        morphology_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._weighted_routed_synapses(
            x,
            morphology_indices=morphology_indices,
        ).sum(dim=-1)

    def _weighted_routed_synapses(
        self,
        x: torch.Tensor,
        *,
        morphology_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        routed = self._route_input_to_synapses(
            x,
            morphology_indices=morphology_indices,
        )
        batch_size, time_steps, _ = routed.shape
        routed = routed.view(batch_size, time_steps, self.num_branch, -1)
        if self.synapse_log_gain is not None:
            assert self.config.synapse_gain_scale is not None
            gain = torch.exp(self.config.synapse_gain_scale * self.synapse_log_gain).to(
                dtype=routed.dtype,
                device=routed.device,
            )
            routed = routed * gain.view(1, 1, self.num_branch, self.num_synapse_per_branch)
        if self.morphology_synapse_log_gain is not None:
            if morphology_indices is None:
                raise ValueError("morphology_indices are required for morphology-conditioned synaptic gains")
            if morphology_indices.ndim != 1 or morphology_indices.shape[0] != batch_size:
                raise ValueError("morphology_indices must have shape (batch,)")
            assert self.config.morphology_synapse_gain_scale is not None
            selected_log_gain = self.morphology_synapse_log_gain.index_select(
                0,
                morphology_indices.to(
                    dtype=torch.long,
                    device=self.morphology_synapse_log_gain.device,
                ),
            )
            morphology_gain = torch.exp(
                self.config.morphology_synapse_gain_scale * selected_log_gain
            ).to(dtype=routed.dtype, device=routed.device)
            routed = routed * morphology_gain.unsqueeze(1)
        if self.morphology_synapse_feature_out is not None:
            if morphology_indices is None:
                raise ValueError("morphology_indices are required for morphology synapse features")
            assert self.morphology_synapse_feature_in is not None
            assert self.config.morphology_synapse_feature_scale is not None
            selected_features = self.morphology_synapse_features.index_select(
                0,
                morphology_indices.to(
                    dtype=torch.long,
                    device=self.morphology_synapse_features.device,
                ),
            )
            if self.input_to_synapse_routing is not None:
                selected_features = torch.index_select(
                    selected_features,
                    1,
                    self.input_to_synapse_indices,
                )
            selected_features = selected_features.view(
                batch_size,
                self.num_branch,
                self.num_synapse_per_branch,
                self.config.morphology_synapse_feature_dim,
            ).to(dtype=routed.dtype, device=routed.device)
            feature_hidden = F.silu(self.morphology_synapse_feature_in(selected_features))
            feature_delta = self.morphology_synapse_feature_out(feature_hidden).squeeze(-1)
            feature_gain = torch.exp(
                self.config.morphology_synapse_feature_scale * feature_delta
            )
            routed = routed * feature_gain.unsqueeze(1)
        return routed

    def _branch_subunit_features(self, routed_synapses: torch.Tensor) -> torch.Tensor:
        if self.branch_subunit_log_gain is None:
            raise RuntimeError("branch subunits are disabled")
        weights = torch.exp(self.branch_subunit_log_gain).to(
            dtype=routed_synapses.dtype,
            device=routed_synapses.device,
        )
        weights = weights * self.branch_subunit_mask.to(
            dtype=routed_synapses.dtype,
            device=routed_synapses.device,
        ).view(1, self.config.branch_subunits, self.num_synapse_per_branch)
        return torch.einsum("btns,nhs->btnh", routed_synapses, weights)

class BranchPyTorchMamba(BranchOfficialMamba):
    """Branch-routed Mamba wrapper using the differentiable PyTorch fallback block."""

    def __init__(self, config: BranchOfficialMambaConfig) -> None:
        if config.block_type != "mamba1":
            raise ValueError("BranchPyTorchMamba only supports block_type='mamba1'")
        from axosim.mamba_pytorch import MambaBlock

        super().__init__(config, mamba_classes=(MambaBlock, None))


class _OfficialMambaResidualBlock(nn.Module):
    def __init__(
        self,
        mamba: nn.Module,
        *,
        model_dim: int,
        dropout: float,
        residual_scale_init: float,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(model_dim) if use_norm else nn.Identity()
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


class _CausalDepthwiseTCNBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int) -> None:
        super().__init__()
        self.right_padding = 2 * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            padding=self.right_padding,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.depthwise(x.transpose(1, 2))[..., : x.shape[1]]
        local = self.pointwise(F.silu(local)).transpose(1, 2)
        return x + local


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
