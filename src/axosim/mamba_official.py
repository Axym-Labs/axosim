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


DEFAULT_STREAMING_CHUNK_SIZE = 32_768


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
    population_adapter_morphology_id: str | None = None
    synapse_gain_scale: float | None = None
    morphology_synapse_gain_scale: float | None = None
    share_morphology_synapse_gain: bool = False
    morphology_synapse_gain_rank: int = 0
    morphology_synapse_feature_dim: int = 0
    morphology_synapse_feature_hidden: int = 0
    morphology_synapse_feature_scale: float | None = None
    morphology_synapse_feature_storage_dtype: str = "float32"
    local_tcn_scale: float | None = None
    local_tcn_mode: str = "dilated"
    local_tcn_position: str = "parallel"
    share_local_tcn_pointwise: bool = False
    local_tcn_pointwise_adapter_rank: int = 0
    local_tcn_pointwise_rank: int = 0
    local_tcn_output_rank: int = 0
    mamba_projection_rank: int = 0
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


@dataclass
class BranchMambaStreamingState:
    """Mutable recurrent state for exact one-timestep AxoMamba inference."""

    mamba_states: list[tuple[torch.Tensor, torch.Tensor]]
    local_tcn_histories: list[torch.Tensor]
    morphology_feature_gains: torch.Tensor | None = None


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
        if config.population_adapter_morphology_id is not None:
            if (
                config.morphology_ids is None
                or config.population_adapter_morphology_id not in config.morphology_ids
            ):
                raise ValueError(
                    "population_adapter_morphology_id must name a configured morphology"
                )
            if (
                config.morphology_embedding_scale is None
            ):
                raise ValueError(
                    "population_adapter_morphology_id requires a morphology embedding"
                )
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
        if (
            config.share_morphology_synapse_gain
            and config.morphology_synapse_gain_scale is None
        ):
            raise ValueError(
                "share_morphology_synapse_gain requires morphology_synapse_gain_scale"
            )
        if config.morphology_synapse_gain_rank < 0:
            raise ValueError("morphology_synapse_gain_rank must be non-negative")
        if config.morphology_synapse_gain_rank:
            if config.morphology_synapse_gain_scale is None:
                raise ValueError(
                    "morphology_synapse_gain_rank requires morphology_synapse_gain_scale"
                )
            if config.share_morphology_synapse_gain:
                raise ValueError(
                    "morphology_synapse_gain_rank is incompatible with "
                    "share_morphology_synapse_gain"
                )
            assert config.morphology_ids is not None
            if config.morphology_synapse_gain_rank > len(config.morphology_ids):
                raise ValueError(
                    "morphology_synapse_gain_rank exceeds morphology count"
                )
        feature_operator_enabled = (
            config.morphology_synapse_feature_dim > 0
            or config.morphology_synapse_feature_hidden > 0
            or config.morphology_synapse_feature_scale is not None
        )
        if config.morphology_synapse_feature_storage_dtype not in {
            "float32",
            "float16",
        }:
            raise ValueError(
                "morphology_synapse_feature_storage_dtype must be "
                "'float32' or 'float16'"
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
        if config.mamba_projection_rank < 0:
            raise ValueError("mamba_projection_rank must be non-negative")
        if config.soma_tcn_scale is not None and config.soma_tcn_scale < 0:
            raise ValueError("soma_tcn_scale must be non-negative")
        if config.soma_tcn_coupling_source not in {"corrected", "base"}:
            raise ValueError("soma_tcn_coupling_source must be 'corrected' or 'base'")
        if local_tcn_enabled and config.local_tcn_position == "interlayer" and config.num_layers < 2:
            raise ValueError("interlayer local TCN requires at least two Mamba layers")
        if config.share_local_tcn_pointwise and not local_tcn_enabled:
            raise ValueError("share_local_tcn_pointwise requires local TCN")
        if config.local_tcn_pointwise_adapter_rank < 0:
            raise ValueError("local_tcn_pointwise_adapter_rank must be non-negative")
        if config.local_tcn_pointwise_adapter_rank > config.model_dim:
            raise ValueError("local TCN pointwise adapter rank exceeds model dimension")
        if config.local_tcn_pointwise_adapter_rank and not config.share_local_tcn_pointwise:
            raise ValueError("local TCN pointwise adapters require shared pointwise maps")
        if config.local_tcn_pointwise_rank < 0:
            raise ValueError("local TCN pointwise rank must be non-negative")
        if config.local_tcn_pointwise_rank > config.model_dim:
            raise ValueError("local TCN pointwise rank exceeds model dimension")
        if config.local_tcn_pointwise_rank and not config.share_local_tcn_pointwise:
            raise ValueError("factorized local TCN pointwise map requires sharing")
        if config.local_tcn_output_rank < 0:
            raise ValueError("local TCN output rank must be non-negative")
        if config.local_tcn_output_rank > config.model_dim:
            raise ValueError("local TCN output rank exceeds model dimension")
        if (config.local_tcn_pointwise_rank or config.local_tcn_output_rank) and not local_tcn_enabled:
            raise ValueError("factorized local TCN maps require local TCN")
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
        self.register_buffer(
            "sparse_input_slot_indices",
            self._create_sparse_input_slot_indices(routing_indices, valid_mask),
            persistent=False,
        )
        self.synapse_log_gain = (
            nn.Parameter(torch.zeros(config.num_branch, config.num_synapse_per_branch))
            if config.synapse_gain_scale is not None
            else None
        )
        self.morphology_synapse_log_gain = (
            nn.Parameter(
                torch.zeros(
                    config.morphology_synapse_gain_rank
                    or (
                        1
                        if config.share_morphology_synapse_gain
                        else len(config.morphology_ids)
                    ),
                    config.num_branch,
                    config.num_synapse_per_branch,
                )
            )
            if config.morphology_synapse_gain_scale is not None
            else None
        )
        self.morphology_synapse_gain_context = (
            nn.Parameter(
                torch.ones(
                    len(config.morphology_ids),
                    config.morphology_synapse_gain_rank,
                )
            )
            if config.morphology_synapse_gain_rank
            else None
        )
        feature_values = (
            torch.zeros(
                len(config.morphology_ids),
                config.num_input,
                config.morphology_synapse_feature_dim,
                dtype=(
                    torch.float16
                    if config.morphology_synapse_feature_storage_dtype == "float16"
                    else torch.float32
                ),
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
        if config.mamba_projection_rank:
            self._factorize_mamba_projections(config.mamba_projection_rank)
        local_dilations = (1, 4, 16) if config.local_tcn_mode == "dilated" else (1, 1, 1)
        self.local_tcn_blocks = nn.ModuleList(
            [
                _CausalDepthwiseTCNBlock(
                    config.model_dim,
                    dilation=dilation,
                    pointwise_adapter_rank=config.local_tcn_pointwise_adapter_rank,
                )
                for dilation in local_dilations
            ]
            if local_tcn_enabled
            else []
        )
        if config.share_local_tcn_pointwise and len(self.local_tcn_blocks) > 1:
            shared_pointwise = self.local_tcn_blocks[0].pointwise
            for block in self.local_tcn_blocks[1:]:
                block.pointwise = shared_pointwise
        if config.local_tcn_pointwise_rank:
            from axosim.structured import FactorizedPointwiseConv1d

            dense_pointwise = self.local_tcn_blocks[0].pointwise
            if not isinstance(dense_pointwise, nn.Conv1d):
                raise ValueError("local TCN pointwise factorization requires a dense map")
            shared_pointwise = FactorizedPointwiseConv1d.from_conv(
                dense_pointwise,
                rank=config.local_tcn_pointwise_rank,
                source_weight_name="local_tcn_blocks.0.pointwise.weight",
            )
            for block in self.local_tcn_blocks:
                block.pointwise = shared_pointwise
        if not local_tcn_enabled:
            self.local_tcn_out = None
        elif config.local_tcn_output_rank:
            from axosim.structured import FactorizedLinear

            self.local_tcn_out = FactorizedLinear(
                config.model_dim,
                config.model_dim,
                config.local_tcn_output_rank,
                bias=False,
                source_weight_name="local_tcn_out.weight",
            )
            nn.init.zeros_(self.local_tcn_out.left)
        else:
            self.local_tcn_out = nn.Linear(config.model_dim, config.model_dim, bias=False)
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
        self._population_adapter_index = (
            config.morphology_ids.index(config.population_adapter_morphology_id)
            if config.population_adapter_morphology_id is not None
            else None
        )
        self.population_embedding_adapter = (
            nn.Parameter(torch.zeros(config.model_dim))
            if self._population_adapter_index is not None
            and self.morphology_embedding is not None
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

    def _factorize_mamba_projections(self, rank: int) -> None:
        from axosim.structured import FactorizedLinear

        for block_index, block in enumerate(self.blocks):
            for projection_name in ("in_proj", "x_proj", "out_proj"):
                projection = getattr(block.mamba, projection_name, None)
                if not isinstance(projection, nn.Linear):
                    raise ValueError(
                        "mamba_projection_rank requires linear in_proj/x_proj/out_proj"
                    )
                factor_parameters = rank * (
                    projection.in_features + projection.out_features
                )
                dense_parameters = projection.in_features * projection.out_features
                if (
                    rank > min(projection.in_features, projection.out_features)
                    or factor_parameters >= dense_parameters
                ):
                    continue
                source_name = (
                    f"blocks.{block_index}.mamba.{projection_name}.weight"
                )
                setattr(
                    block.mamba,
                    projection_name,
                    FactorizedLinear.from_linear(
                        projection,
                        rank=rank,
                        source_weight_name=source_name,
                    ),
                )

    def _morphology_embedding_values(
        self,
        morphology_indices: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        assert self.morphology_embedding is not None
        indices = morphology_indices.to(dtype=torch.long, device=device)
        values = self.morphology_embedding(indices).to(dtype=dtype)
        if self.population_embedding_adapter is not None:
            assert self._population_adapter_index is not None
            mask = (indices == self._population_adapter_index).to(dtype=dtype)
            values = values + mask.unsqueeze(1) * self.population_embedding_adapter.to(
                dtype=dtype, device=device
            ).unsqueeze(0)
        return values

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

        routed_events = self._route_input_to_synapses(
            x,
            morphology_indices=morphology_indices,
        )
        routed_synapses = self._apply_synapse_weights(
            routed_events,
            morphology_indices=morphology_indices,
        )
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
            morphology = self._morphology_embedding_values(
                morphology_indices, dtype=hidden.dtype, device=hidden.device
            )
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

    def allocate_streaming_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> BranchMambaStreamingState:
        """Allocate state that advances a population without replaying history."""
        self._validate_streaming_configuration()
        if batch_size <= 0:
            raise ValueError("streaming batch_size must be positive")
        parameter = next(self.parameters())
        resolved_device = parameter.device if device is None else torch.device(device)
        resolved_dtype = parameter.dtype if dtype is None else dtype
        mamba_states = [
            block.mamba.allocate_inference_cache(
                batch_size,
                max_seqlen=1,
                dtype=resolved_dtype,
                device=resolved_device,
            )
            for block in self.blocks
        ]
        local_tcn_histories = [
            torch.zeros(
                batch_size,
                block.depthwise.in_channels,
                block.right_padding,
                device=resolved_device,
                dtype=resolved_dtype,
            )
            for block in self.local_tcn_blocks
        ]
        return BranchMambaStreamingState(
            mamba_states=mamba_states,
            local_tcn_histories=local_tcn_histories,
            morphology_feature_gains=self._all_morphology_feature_gains(
                dtype=resolved_dtype,
                device=resolved_device,
            ),
        )

    def streaming_step(
        self,
        x: torch.Tensor,
        state: BranchMambaStreamingState,
        *,
        morphology_indices: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Advance one timestep and return `(batch, 1, output)` predictions."""
        self._validate_streaming_configuration()
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[2] != self.num_input:
            raise ValueError(
                "streaming input must have shape (batch, channels) or "
                "(batch, 1, channels)"
            )
        if chunk_size is None and x.shape[0] > DEFAULT_STREAMING_CHUNK_SIZE:
            chunk_size = DEFAULT_STREAMING_CHUNK_SIZE
        if chunk_size is not None:
            if chunk_size <= 0:
                raise ValueError("streaming chunk_size must be positive")
            if x.shape[0] > chunk_size:
                outputs = []
                base_soma_predictions = []
                for start in range(0, x.shape[0], chunk_size):
                    stop = min(start + chunk_size, x.shape[0])
                    chunk_state = BranchMambaStreamingState(
                        mamba_states=[
                            (conv_state[start:stop], ssm_state[start:stop])
                            for conv_state, ssm_state in state.mamba_states
                        ],
                        local_tcn_histories=[
                            history[start:stop]
                            for history in state.local_tcn_histories
                        ],
                        morphology_feature_gains=state.morphology_feature_gains,
                    )
                    outputs.append(
                        self.streaming_step(
                            x[start:stop],
                            chunk_state,
                            morphology_indices=(
                                None
                                if morphology_indices is None
                                else morphology_indices[start:stop]
                            ),
                        )
                    )
                    if self.base_soma_prediction is not None:
                        base_soma_predictions.append(self.base_soma_prediction)
                    for parent, updated in zip(
                        state.mamba_states,
                        chunk_state.mamba_states,
                        strict=True,
                    ):
                        parent[0][start:stop].copy_(updated[0])
                        parent[1][start:stop].copy_(updated[1])
                    for parent, updated in zip(
                        state.local_tcn_histories,
                        chunk_state.local_tcn_histories,
                        strict=True,
                    ):
                        parent[start:stop].copy_(updated)
                if base_soma_predictions:
                    self.base_soma_prediction = torch.cat(
                        base_soma_predictions,
                        dim=0,
                    )
                return torch.cat(outputs, dim=0)
        if len(state.mamba_states) != len(self.blocks):
            raise ValueError("streaming Mamba state does not match model layers")
        if len(state.local_tcn_histories) != len(self.local_tcn_blocks):
            raise ValueError("streaming TCN state does not match model blocks")

        routed_events = self._route_input_to_synapses(
            x,
            morphology_indices=morphology_indices,
        )
        routed_synapses = self._apply_synapse_weights(
            routed_events,
            morphology_indices=morphology_indices,
            morphology_feature_gains=state.morphology_feature_gains,
        )
        branch_inputs = routed_synapses.sum(dim=-1)
        branch_inputs = self._apply_streaming_branch_transforms(branch_inputs)
        return self._streaming_step_from_branch_inputs(
            branch_inputs,
            state,
            morphology_indices=morphology_indices,
            routed_synapses=routed_synapses,
        )

    def _apply_streaming_branch_transforms(
        self,
        branch_inputs: torch.Tensor,
    ) -> torch.Tensor:
        if self.branch_gain is not None:
            gain = self.branch_gain.to(
                dtype=branch_inputs.dtype,
                device=branch_inputs.device,
            )
            branch_inputs = branch_inputs * gain.view(1, 1, -1)
        if self.config.branch_nonlinearity == "tanh":
            assert self.branch_nonlinearity_log_scale is not None
            assert self.branch_nonlinearity_bias is not None
            scale = self.branch_nonlinearity_log_scale.exp().to(
                dtype=branch_inputs.dtype,
                device=branch_inputs.device,
            )
            bias = self.branch_nonlinearity_bias.to(
                dtype=branch_inputs.dtype,
                device=branch_inputs.device,
            )
            branch_inputs = torch.tanh(
                scale.view(1, 1, -1)
                * (branch_inputs + bias.view(1, 1, -1))
            )
            branch_inputs = branch_inputs / scale.view(1, 1, -1).clamp_min(
                torch.finfo(branch_inputs.dtype).eps
            )
        return branch_inputs

    def _streaming_step_from_branch_inputs(
        self,
        branch_inputs: torch.Tensor,
        state: BranchMambaStreamingState,
        *,
        morphology_indices: torch.Tensor | None,
        routed_synapses: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if branch_inputs.ndim != 3 or branch_inputs.shape[1:] != (1, self.num_branch):
            raise ValueError("streaming branch inputs must have shape (batch, 1, branches)")
        hidden = self.branch_projection(branch_inputs)
        if self.branch_subunit_projection is not None:
            if routed_synapses is None:
                raise ValueError("branch subunits require routed synapse inputs")
            subunit_inputs = self._branch_subunit_features(routed_synapses)
            if self.branch_gain is not None:
                assert self.branch_gain is not None
                subunit_inputs = subunit_inputs * self.branch_gain.to(
                    dtype=subunit_inputs.dtype,
                    device=subunit_inputs.device,
                ).view(1, 1, -1, 1)
            hidden = hidden + self.config.branch_subunit_scale * self.branch_subunit_projection(
                subunit_inputs.flatten(start_dim=-2)
            )
        hidden = self._apply_projected_branch_bilinear(
            hidden,
            self._branch_bilinear_projection(branch_inputs),
        )
        if self.morphology_embedding is not None:
            if morphology_indices is None:
                raise ValueError(
                    "morphology_indices are required for morphology-conditioned AxoMamba"
                )
            if morphology_indices.ndim != 1 or morphology_indices.shape[0] != branch_inputs.shape[0]:
                raise ValueError("morphology_indices must have shape (batch,)")
            morphology = self._morphology_embedding_values(
                morphology_indices, dtype=hidden.dtype, device=hidden.device
            )
            assert self.config.morphology_embedding_scale is not None
            hidden = hidden + self.config.morphology_embedding_scale * morphology.unsqueeze(1)
        elif morphology_indices is not None and self.config.morphology_ids is None:
            raise ValueError("morphology_indices were provided to an unconditioned AxoMamba")

        for block_index, block in enumerate(self.blocks):
            conv_state, ssm_state = state.mamba_states[block_index]
            update, conv_state, ssm_state = block.mamba.step(
                block.norm(hidden),
                conv_state,
                ssm_state,
            )
            state.mamba_states[block_index] = (conv_state, ssm_state)
            hidden = hidden + block.residual_scale.view(1, 1, -1) * block.dropout(update)

        hidden = hidden + self._stream_local_tcn_correction(
            hidden,
            state.local_tcn_histories,
        )
        features = self.norm(hidden)
        if self.readout is not None:
            output = self.readout(features)
            spike = output[..., 0:1]
            base_soma = output[..., 1:2]
            soma = base_soma
        else:
            assert self.spike_readout is not None and self.soma_readout is not None
            spike = self.spike_readout(features)
            base_soma = self._readout_soma(features, None)
            soma = base_soma
        self.base_soma_prediction = base_soma
        if self.config.spike_voltage_coupling_scale:
            spike = self._couple_spike_voltage(spike, soma)
        return torch.cat([spike, soma], dim=-1)

    def _sparse_streaming_branch_inputs(
        self,
        event_indices: torch.Tensor,
        event_values: torch.Tensor,
        *,
        morphology_indices: torch.Tensor | None,
        morphology_feature_gains: torch.Tensor | None,
    ) -> torch.Tensor:
        slots = self.sparse_input_slot_indices.index_select(
            0,
            event_indices.clamp_min(0).reshape(-1),
        ).view_as(event_indices)
        valid = event_indices >= 0
        weighted = event_values * valid.to(dtype=event_values.dtype)
        if self.synapse_log_gain is not None:
            assert self.config.synapse_gain_scale is not None
            gain = torch.exp(
                self.config.synapse_gain_scale * self.synapse_log_gain
            ).to(dtype=weighted.dtype, device=weighted.device)
            weighted = weighted * gain.flatten().index_select(0, slots.flatten()).view_as(slots)
        if self.morphology_synapse_log_gain is not None:
            if morphology_indices is None and not self.config.share_morphology_synapse_gain:
                raise ValueError("morphology_indices are required for morphology-conditioned synaptic gains")
            assert self.config.morphology_synapse_gain_scale is not None
            if self.config.morphology_synapse_gain_rank:
                assert morphology_indices is not None
                assert self.morphology_synapse_gain_context is not None
                morphology_indices = morphology_indices.to(
                    dtype=torch.long,
                    device=self.morphology_synapse_gain_context.device,
                )
                context = self.morphology_synapse_gain_context.index_select(
                    0, morphology_indices
                )
                basis = self.morphology_synapse_log_gain.flatten(1).transpose(0, 1)
                selected_basis = basis.index_select(0, slots.flatten()).view(
                    *slots.shape,
                    self.config.morphology_synapse_gain_rank,
                )
                selected_log_gain = (
                    selected_basis * context.unsqueeze(1)
                ).sum(dim=-1)
                selected_gain = torch.exp(
                    self.config.morphology_synapse_gain_scale * selected_log_gain
                ).to(dtype=weighted.dtype, device=weighted.device)
            else:
                gain = torch.exp(
                    self.config.morphology_synapse_gain_scale
                    * self.morphology_synapse_log_gain
                ).to(dtype=weighted.dtype, device=weighted.device)
            if self.config.share_morphology_synapse_gain:
                selected_gain = gain.flatten().index_select(0, slots.flatten())
            elif not self.config.morphology_synapse_gain_rank:
                assert morphology_indices is not None
                slot_count = self.num_branch * self.num_synapse_per_branch
                morphology_slots = morphology_indices.unsqueeze(1) * slot_count + slots
                selected_gain = gain.flatten().index_select(
                    0,
                    morphology_slots.flatten(),
                )
            weighted = weighted * selected_gain.view_as(slots)
        if self.morphology_synapse_feature_out is not None:
            if morphology_indices is None:
                raise ValueError("morphology_indices are required for morphology synapse features")
            feature_gains = morphology_feature_gains
            if feature_gains is None:
                feature_gains = self._all_morphology_feature_gains(
                    dtype=weighted.dtype,
                    device=weighted.device,
                )
            assert feature_gains is not None
            slot_count = self.num_branch * self.num_synapse_per_branch
            morphology_slots = morphology_indices.unsqueeze(1) * slot_count + slots
            weighted = weighted * feature_gains.flatten().index_select(
                0,
                morphology_slots.flatten(),
            ).view_as(slots)
        branch_indices = torch.div(
            slots,
            self.num_synapse_per_branch,
            rounding_mode="floor",
        )
        branch_inputs = torch.zeros(
            event_indices.shape[0],
            self.num_branch,
            device=weighted.device,
            dtype=weighted.dtype,
        )
        branch_inputs.scatter_add_(1, branch_indices, weighted)
        return self._apply_streaming_branch_transforms(branch_inputs.unsqueeze(1))

    def streaming_step_events(
        self,
        event_indices: torch.Tensor,
        event_values: torch.Tensor,
        state: BranchMambaStreamingState,
        *,
        morphology_indices: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Advance one timestep from padded sparse channel/value event rows.

        ``event_indices`` and ``event_values`` have shape ``(batch, events)``.
        Negative indices are padding; repeated channels are summed. Dense input
        exists only for the current population chunk.
        """
        self._validate_streaming_configuration()
        if event_indices.ndim != 2 or event_values.shape != event_indices.shape:
            raise ValueError(
                "streaming events must have matching (batch, events) indices and values"
            )
        if event_indices.shape[0] <= 0:
            raise ValueError("streaming event batch must be positive")
        if event_indices.is_floating_point() or event_indices.is_complex():
            raise ValueError("streaming event indices must be integers")
        resolved_chunk_size = (
            DEFAULT_STREAMING_CHUNK_SIZE if chunk_size is None else chunk_size
        )
        if resolved_chunk_size <= 0:
            raise ValueError("streaming chunk_size must be positive")
        if len(state.mamba_states) != len(self.blocks):
            raise ValueError("streaming Mamba state does not match model layers")
        if len(state.local_tcn_histories) != len(self.local_tcn_blocks):
            raise ValueError("streaming TCN state does not match model blocks")

        parameter = next(self.parameters())
        outputs = []
        base_soma_predictions = []
        for start in range(0, event_indices.shape[0], resolved_chunk_size):
            stop = min(start + resolved_chunk_size, event_indices.shape[0])
            chunk_indices = event_indices[start:stop].to(
                device=parameter.device,
                dtype=torch.long,
            )
            valid = chunk_indices >= 0
            chunk_values = event_values[start:stop].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            chunk_state = BranchMambaStreamingState(
                mamba_states=[
                    (conv_state[start:stop], ssm_state[start:stop])
                    for conv_state, ssm_state in state.mamba_states
                ],
                local_tcn_histories=[
                    history[start:stop]
                    for history in state.local_tcn_histories
                ],
                morphology_feature_gains=state.morphology_feature_gains,
            )
            chunk_morphology_indices = (
                None
                if morphology_indices is None
                else morphology_indices[start:stop].to(
                    device=parameter.device,
                    dtype=torch.long,
                )
            )
            if (
                self.sparse_input_slot_indices.numel() == self.num_input
                and self.branch_subunit_projection is None
            ):
                branch_inputs = self._sparse_streaming_branch_inputs(
                    chunk_indices,
                    chunk_values,
                    morphology_indices=chunk_morphology_indices,
                    morphology_feature_gains=state.morphology_feature_gains,
                )
                outputs.append(
                    self._streaming_step_from_branch_inputs(
                        branch_inputs,
                        chunk_state,
                        morphology_indices=chunk_morphology_indices,
                    )
                )
            else:
                dense = torch.zeros(
                    stop - start,
                    self.num_input,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                dense.scatter_add_(
                    1,
                    chunk_indices.clamp_min(0),
                    chunk_values * valid.to(dtype=parameter.dtype),
                )
                outputs.append(
                    self.streaming_step(
                        dense,
                        chunk_state,
                        morphology_indices=chunk_morphology_indices,
                    )
                )
            if self.base_soma_prediction is not None:
                base_soma_predictions.append(self.base_soma_prediction)
            for parent, updated in zip(
                state.mamba_states,
                chunk_state.mamba_states,
                strict=True,
            ):
                parent[0][start:stop].copy_(updated[0])
                parent[1][start:stop].copy_(updated[1])
            for parent, updated in zip(
                state.local_tcn_histories,
                chunk_state.local_tcn_histories,
                strict=True,
            ):
                parent[start:stop].copy_(updated)
        if base_soma_predictions:
            self.base_soma_prediction = torch.cat(base_soma_predictions, dim=0)
        return torch.cat(outputs, dim=0)

    def _stream_local_tcn_correction(
        self,
        hidden: torch.Tensor,
        histories: list[torch.Tensor],
    ) -> torch.Tensor:
        if self.local_tcn_out is None:
            return torch.zeros_like(hidden)
        local = hidden
        for block, history in zip(self.local_tcn_blocks, histories, strict=True):
            value = local.transpose(1, 2)
            if history.shape[:2] != value.shape[:2] or history.shape[2] != block.right_padding:
                raise ValueError("streaming TCN history has the wrong shape")
            window = torch.cat([history, value], dim=-1)
            filtered = F.conv1d(
                window,
                block.depthwise.weight,
                block.depthwise.bias,
                dilation=block.depthwise.dilation,
                groups=block.depthwise.groups,
            )
            history.copy_(window[..., -block.right_padding :])
            activated = F.silu(filtered)
            update = block.pointwise(activated)
            if block.pointwise_adapter_in is not None:
                assert block.pointwise_adapter_out is not None
                update = update + block.pointwise_adapter_out(
                    block.pointwise_adapter_in(activated)
                )
            local = local + update.transpose(1, 2)
        assert self.config.local_tcn_scale is not None
        return self.config.local_tcn_scale * self.local_tcn_out(local)

    def _validate_streaming_configuration(self) -> None:
        unsupported = []
        if self.training:
            unsupported.append("training mode")
        if self.config.block_type != "mamba1":
            unsupported.append("non-Mamba-1 block")
        if self.config.block_repeats != 1:
            unsupported.append("block repeats")
        if self.branch_trace_taus:
            unsupported.append("branch traces")
        if self.hidden_trace_taus:
            unsupported.append("hidden traces")
        if self.integrative_pre is not None or self.integrative_post is not None:
            unsupported.append("integrative mixer")
        if self.temporal_refine is not None:
            unsupported.append("temporal refine")
        if self.local_tcn_blocks and self.config.local_tcn_position != "post":
            unsupported.append("non-post local TCN")
        if self.soma_tcn_out is not None:
            unsupported.append("soma TCN")
        if self.soma_filter_logit is not None:
            unsupported.append("soma filter")
        if self.soma_highpass_correction_scale is not None:
            unsupported.append("soma highpass correction")
        if any(
            not hasattr(block.mamba, "allocate_inference_cache")
            or not hasattr(block.mamba, "step")
            for block in self.blocks
        ):
            unsupported.append("Mamba backend without streaming state")
        if unsupported:
            raise ValueError(
                "streaming AxoMamba does not support " + ", ".join(unsupported)
            )

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

    def _create_sparse_input_slot_indices(
        self,
        routing_indices: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        slot_count = self.num_branch * self.num_synapse_per_branch
        if routing_indices is None or valid_mask is None:
            if self.num_input == slot_count:
                return torch.arange(self.num_input, dtype=torch.long)
            return torch.empty(0, dtype=torch.long)
        valid_slots = torch.nonzero(valid_mask > 0, as_tuple=False).flatten()
        valid_channels = routing_indices.index_select(0, valid_slots)
        channel_counts = torch.bincount(valid_channels, minlength=self.num_input)
        if valid_channels.numel() != self.num_input or not torch.all(channel_counts == 1):
            return torch.empty(0, dtype=torch.long)
        sparse_slots = torch.empty(self.num_input, dtype=torch.long)
        sparse_slots[valid_channels] = valid_slots
        return sparse_slots

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
        return self._apply_synapse_weights(
            routed,
            morphology_indices=morphology_indices,
        )

    def _apply_synapse_weights(
        self,
        routed: torch.Tensor,
        *,
        morphology_indices: torch.Tensor | None = None,
        morphology_feature_gains: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
            assert self.config.morphology_synapse_gain_scale is not None
            if self.config.share_morphology_synapse_gain:
                morphology_gain = torch.exp(
                    self.config.morphology_synapse_gain_scale
                    * self.morphology_synapse_log_gain[0]
                ).to(dtype=routed.dtype, device=routed.device)
                routed = routed * morphology_gain.view(
                    1,
                    1,
                    self.num_branch,
                    self.num_synapse_per_branch,
                )
            else:
                if morphology_indices is None:
                    raise ValueError("morphology_indices are required for morphology-conditioned synaptic gains")
                if morphology_indices.ndim != 1 or morphology_indices.shape[0] != batch_size:
                    raise ValueError("morphology_indices must have shape (batch,)")
                morphology_indices = morphology_indices.to(
                    dtype=torch.long,
                    device=self.morphology_synapse_log_gain.device,
                )
                if self.config.morphology_synapse_gain_rank:
                    assert self.morphology_synapse_gain_context is not None
                    context = self.morphology_synapse_gain_context.index_select(
                        0, morphology_indices
                    )
                    selected_log_gain = torch.einsum(
                        "br,rns->bns",
                        context,
                        self.morphology_synapse_log_gain,
                    )
                else:
                    selected_log_gain = self.morphology_synapse_log_gain.index_select(
                        0, morphology_indices
                    )
                morphology_gain = torch.exp(
                    self.config.morphology_synapse_gain_scale * selected_log_gain
                ).to(dtype=routed.dtype, device=routed.device)
                routed = routed * morphology_gain.unsqueeze(1)
        if self.morphology_synapse_feature_out is not None:
            if morphology_indices is None:
                raise ValueError("morphology_indices are required for morphology synapse features")
            feature_gains = morphology_feature_gains
            if feature_gains is None:
                feature_gains = self._all_morphology_feature_gains(
                    dtype=routed.dtype,
                    device=routed.device,
                )
            assert feature_gains is not None
            feature_gain = feature_gains.index_select(
                0,
                morphology_indices.to(
                    dtype=torch.long,
                    device=feature_gains.device,
                ),
            )
            routed = routed * feature_gain.to(
                dtype=routed.dtype,
                device=routed.device,
            ).unsqueeze(1)
        return routed

    def _all_morphology_feature_gains(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor | None:
        if self.morphology_synapse_feature_out is None:
            return None
        assert self.morphology_synapse_feature_in is not None
        assert self.config.morphology_synapse_feature_scale is not None
        assert self.config.morphology_ids is not None
        features = self.morphology_synapse_features
        if self.input_to_synapse_routing is not None:
            features = torch.index_select(
                features,
                1,
                self.input_to_synapse_indices,
            )
        features = features.view(
            len(self.config.morphology_ids),
            self.num_branch,
            self.num_synapse_per_branch,
            self.config.morphology_synapse_feature_dim,
        ).to(dtype=dtype, device=device)
        feature_hidden = F.silu(self.morphology_synapse_feature_in(features))
        feature_delta = self.morphology_synapse_feature_out(feature_hidden).squeeze(-1)
        return torch.exp(
            self.config.morphology_synapse_feature_scale * feature_delta
        )

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
    def __init__(
        self,
        channels: int,
        *,
        dilation: int,
        pointwise_adapter_rank: int = 0,
    ) -> None:
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
        self.pointwise_adapter_in = (
            nn.Conv1d(channels, pointwise_adapter_rank, kernel_size=1, bias=False)
            if pointwise_adapter_rank
            else None
        )
        self.pointwise_adapter_out = (
            nn.Conv1d(pointwise_adapter_rank, channels, kernel_size=1)
            if pointwise_adapter_rank
            else None
        )
        if self.pointwise_adapter_out is not None:
            nn.init.zeros_(self.pointwise_adapter_out.weight)
            nn.init.zeros_(self.pointwise_adapter_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.depthwise(x.transpose(1, 2))[..., : x.shape[1]]
        activated = F.silu(local)
        local = self.pointwise(activated)
        if self.pointwise_adapter_in is not None:
            assert self.pointwise_adapter_out is not None
            local = local + self.pointwise_adapter_out(
                self.pointwise_adapter_in(activated)
            )
        local = local.transpose(1, 2)
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
