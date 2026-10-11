from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn

from axosim.mamba_official import BranchOfficialMamba

TemporalCoreKind = Literal["mamba", "gru", "branch_rnn"]


@dataclass
class AxoTemporalStreamingState:
    """Mutable online state for a replacement temporal core."""

    core_state: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    local_tcn_histories: list[torch.Tensor]
    morphology_feature_gains: torch.Tensor | None
    patch_sum: torch.Tensor | None = None
    patch_correction: torch.Tensor | None = None
    patch_position: int = 0


@dataclass(frozen=True)
class AxoTemporalCoreConfig:
    """Configuration for a temporal core behind the shared AxoMamba encoder."""

    kind: TemporalCoreKind
    gru_hidden_units: int | None = None
    branch_memory_units: int = 30
    branch_hidden_units: int = 64
    branch_synapse_decay: float = 0.85
    branch_memory_decay: float = 0.90
    residual_scale_init: float = 0.1
    keep_local_tcn: bool = False
    gru_layers: int = 1
    patch_size: int = 1
    behavior_adapter_morphology_id: str | None = None
    behavior_adapter_rank: int = 0
    behavior_adapter_branch_token_offset: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"mamba", "gru", "branch_rnn"}:
            raise ValueError(f"unknown temporal core: {self.kind}")
        if self.branch_memory_units <= 0:
            raise ValueError("branch_memory_units must be positive")
        if self.gru_hidden_units is not None and self.gru_hidden_units <= 0:
            raise ValueError("gru_hidden_units must be positive")
        if self.branch_hidden_units <= 0:
            raise ValueError("branch_hidden_units must be positive")
        if not 0.0 <= self.branch_synapse_decay < 1.0:
            raise ValueError("branch_synapse_decay must be in [0, 1)")
        if not 0.0 <= self.branch_memory_decay < 1.0:
            raise ValueError("branch_memory_decay must be in [0, 1)")
        if self.residual_scale_init < 0:
            raise ValueError("residual_scale_init must be non-negative")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.kind == "mamba" and self.patch_size != 1:
            raise ValueError("patching is supported by replacement cores")
        if (
            self.behavior_adapter_morphology_id is not None
            and self.kind != "gru"
        ):
            raise ValueError(
                "behavior adaptation currently requires a GRU core"
            )
        if self.behavior_adapter_rank < 0:
            raise ValueError("behavior_adapter_rank must be non-negative")
        if (
            self.behavior_adapter_rank
            and self.behavior_adapter_morphology_id is None
        ):
            raise ValueError(
                "low-rank behavior adaptation requires a target morphology"
            )
        if (
            self.behavior_adapter_branch_token_offset
            and self.behavior_adapter_morphology_id is None
        ):
            raise ValueError(
                "branch token adaptation requires a target morphology"
            )


class NeuronBehaviorAdapter(nn.Module):
    """Neuron-specific response parameters around shared GRU dynamics."""

    def __init__(
        self,
        *,
        branches: int,
        width: int,
        outputs: int,
        morphology_index: int,
        rank: int = 0,
        branch_token_offset: bool = False,
    ) -> None:
        super().__init__()
        self.branches = branches
        self.width = width
        self.outputs = outputs
        self.morphology_index = morphology_index
        self.rank = rank
        if branch_token_offset:
            self.register_parameter("branch_log_gain", None)
            self.register_parameter("branch_bias", None)
        else:
            self.branch_log_gain = nn.Parameter(torch.zeros(branches))
            self.branch_bias = nn.Parameter(torch.zeros(branches))
        self.latent_gain = nn.Parameter(torch.zeros(width))
        self.latent_offset = nn.Parameter(torch.zeros(width))
        self.recurrent_output_gain = nn.Parameter(torch.zeros(width))
        self.readout_weight = nn.Parameter(torch.zeros(outputs, width))
        self.readout_bias = nn.Parameter(torch.zeros(outputs))
        if branch_token_offset:
            self.branch_token_offset = nn.Parameter(torch.zeros(width))
        else:
            self.register_parameter("branch_token_offset", None)
        if rank:
            if branch_token_offset:
                self.register_parameter("branch_projection_down", None)
                self.register_parameter("branch_projection_up", None)
            else:
                self.branch_projection_down = nn.Parameter(
                    torch.empty(rank, branches)
                )
                self.branch_projection_up = nn.Parameter(
                    torch.zeros(width, rank)
                )
            self.gru_input_down = nn.Parameter(
                torch.empty(rank, width)
            )
            self.gru_input_up = nn.Parameter(
                torch.zeros(3 * width, rank)
            )
            self.gru_recurrent_down = nn.Parameter(
                torch.empty(rank, width)
            )
            self.gru_recurrent_up = nn.Parameter(
                torch.zeros(3 * width, rank)
            )
            down_parameters = [
                self.gru_input_down,
                self.gru_recurrent_down,
            ]
            if self.branch_projection_down is not None:
                down_parameters.append(self.branch_projection_down)
            for parameter in down_parameters:
                nn.init.kaiming_uniform_(parameter, a=5**0.5)
        else:
            self.branch_projection_down = None
            self.branch_projection_up = None
            self.gru_input_down = None
            self.gru_input_up = None
            self.gru_recurrent_down = None
            self.gru_recurrent_up = None

    @property
    def parameter_counts(self) -> dict[str, int]:
        counts = {
            "latent_gain": self.latent_gain.numel(),
            "latent_offset": self.latent_offset.numel(),
            "recurrent_output_gain": (
                self.recurrent_output_gain.numel()
            ),
            "readout": (
                self.readout_weight.numel()
                + self.readout_bias.numel()
            ),
        }
        if self.branch_log_gain is not None:
            assert self.branch_bias is not None
            counts["branch_gain"] = self.branch_log_gain.numel()
            counts["branch_bias"] = self.branch_bias.numel()
        if self.branch_token_offset is not None:
            counts["branch_token_offset"] = (
                self.branch_token_offset.numel()
            )
        if self.rank:
            assert self.gru_input_down is not None
            assert self.gru_input_up is not None
            if self.branch_projection_down is not None:
                assert self.branch_projection_up is not None
                counts["branch_projection_low_rank"] = (
                    self.branch_projection_down.numel()
                    + self.branch_projection_up.numel()
                )
            counts["gru_input_low_rank"] = (
                self.gru_input_down.numel()
                + self.gru_input_up.numel()
            )
            assert self.gru_recurrent_down is not None
            assert self.gru_recurrent_up is not None
            counts["gru_recurrent_low_rank"] = (
                self.gru_recurrent_down.numel()
                + self.gru_recurrent_up.numel()
            )
        return counts

    @property
    def parameter_count(self) -> int:
        return sum(self.parameter_counts.values())


class StackedResidualGRUTemporalCore(nn.Module):
    """Several residual GRU layers applied in sequence (full-sequence use).

    Adapter deltas, if any, apply to the first layer only. Streaming
    inference is not supported for stacked cores.
    """

    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.width = layers[0].width

    def forward(self, hidden: torch.Tensor, **kwargs) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, **(kwargs if index == 0 else {}))
        return hidden


class ResidualGRUTemporalCore(nn.Module):
    """Vendor-fused GRU with the same residual contract as AxoMamba blocks."""

    def __init__(
        self,
        width: int,
        *,
        hidden_units: int | None = None,
        residual_scale_init: float,
    ) -> None:
        super().__init__()
        self.width = width
        self.hidden_units = hidden_units or width
        self.pre_norm = nn.LayerNorm(width)
        self.gru = nn.GRU(
            width,
            self.hidden_units,
            batch_first=True,
        )
        self.output_projection: nn.Module
        if self.hidden_units == width:
            self.output_projection = nn.Identity()
        else:
            self.output_projection = nn.Linear(
                self.hidden_units,
                width,
                bias=False,
            )
        self.residual_scale = nn.Parameter(
            torch.full((width,), residual_scale_init)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        weight_ih_delta: torch.Tensor | None = None,
        weight_hh_delta: torch.Tensor | None = None,
        adapter_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return hidden + self.temporal_correction(
            hidden,
            weight_ih_delta=weight_ih_delta,
            weight_hh_delta=weight_hh_delta,
            adapter_mask=adapter_mask,
        )

    def temporal_correction(
        self,
        hidden: torch.Tensor,
        *,
        weight_ih_delta: torch.Tensor | None = None,
        weight_hh_delta: torch.Tensor | None = None,
        adapter_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.pre_norm(hidden)
        recurrent, _ = self.gru(normalized)
        if (
            weight_ih_delta is not None
            or weight_hh_delta is not None
        ):
            if adapter_mask is None:
                raise ValueError(
                    "GRU adaptation requires an adapter mask"
                )
            adapted = self._adapted_gru(
                normalized,
                weight_ih_delta=weight_ih_delta,
                weight_hh_delta=weight_hh_delta,
            )
            mask = adapter_mask.to(
                dtype=recurrent.dtype,
                device=recurrent.device,
            ).view(-1, 1, 1)
            recurrent = recurrent + mask * (adapted - recurrent)
        return (
            self.residual_scale.view(1, 1, -1)
            * self.output_projection(recurrent)
        )

    def _adapted_gru(
        self,
        hidden: torch.Tensor,
        *,
        weight_ih_delta: torch.Tensor | None,
        weight_hh_delta: torch.Tensor | None,
    ) -> torch.Tensor:
        state = hidden.new_zeros(
            1,
            hidden.shape[0],
            self.hidden_units,
        )
        output, _ = torch._VF.gru(
            hidden,
            state,
            [
                (
                    self.gru.weight_ih_l0
                    if weight_ih_delta is None
                    else self.gru.weight_ih_l0 + weight_ih_delta
                ),
                (
                    self.gru.weight_hh_l0
                    if weight_hh_delta is None
                    else self.gru.weight_hh_l0 + weight_hh_delta
                ),
                self.gru.bias_ih_l0,
                self.gru.bias_hh_l0,
            ],
            True,
            1,
            0.0,
            self.training,
            False,
            True,
        )
        return output

    @property
    def recurrent_state_elements(self) -> int:
        return self.hidden_units


class BranchELMTemporalCore(nn.Module):
    """Leaky branch/memory recurrence behind the shared biological encoder."""

    def __init__(
        self,
        width: int,
        *,
        memory_units: int,
        hidden_units: int,
        synapse_decay: float,
        memory_decay: float,
        residual_scale_init: float,
    ) -> None:
        super().__init__()
        self.width = width
        self.memory_units = memory_units
        self.synapse_decay = synapse_decay
        self.memory_decay = memory_decay
        self.pre_norm = nn.LayerNorm(width)
        self.integration = nn.Sequential(
            nn.Linear(width + memory_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, memory_units),
            nn.Tanh(),
        )
        self.output_projection = nn.Linear(memory_units, width)
        self.residual_scale = nn.Parameter(
            torch.full((width,), residual_scale_init)
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.temporal_correction(hidden)

    def temporal_correction(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.pre_norm(hidden)
        branch_state = hidden.new_zeros(hidden.shape[0], self.width)
        memory_state = hidden.new_zeros(
            hidden.shape[0],
            self.memory_units,
        )
        outputs = []
        for time_index in range(hidden.shape[1]):
            branch_state = (
                self.synapse_decay * branch_state
                + normalized[:, time_index]
            )
            decayed_memory = self.memory_decay * memory_state
            update = self.integration(
                torch.cat((branch_state, decayed_memory), dim=-1)
            )
            memory_state = (
                decayed_memory
                + (1.0 - self.memory_decay) * update
            )
            recurrent = self.output_projection(memory_state)
            outputs.append(
                self.residual_scale * recurrent
            )
        return torch.stack(outputs, dim=1)

    @property
    def recurrent_state_elements(self) -> int:
        return self.width + self.memory_units


class CausalPatchedTemporalCore(nn.Module):
    """Run a temporal core on causal patch means and delay its correction."""

    def __init__(
        self,
        core: nn.Module,
        *,
        width: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        if patch_size <= 1:
            raise ValueError("patch_size must be greater than one")
        self.core = core
        self.width = width
        self.patch_size = patch_size

    def forward(
        self,
        hidden: torch.Tensor,
        **temporal_kwargs,
    ) -> torch.Tensor:
        if hidden.shape[-1] != self.width:
            raise ValueError("hidden width does not match patch wrapper")
        batch, steps, width = hidden.shape
        full_steps = steps - steps % self.patch_size
        if full_steps == 0:
            return hidden
        coarse = hidden[:, :full_steps].reshape(
            batch,
            full_steps // self.patch_size,
            self.patch_size,
            width,
        ).mean(dim=2)
        correction = getattr(self.core, "temporal_correction", None)
        if correction is None:
            coarse_correction = self.core(coarse) - coarse
        else:
            coarse_correction = correction(
                coarse,
                **temporal_kwargs,
            )
        native_correction = torch.zeros_like(hidden)
        if coarse_correction.shape[1] > 1:
            native_correction[:, self.patch_size:full_steps] = (
                coarse_correction[:, :-1].repeat_interleave(
                    self.patch_size,
                    dim=1,
                )
            )
        if full_steps < steps:
            native_correction[:, full_steps:] = coarse_correction[:, -1:]
        return hidden + native_correction

    @property
    def recurrent_state_elements(self) -> int:
        core_elements = int(self.core.recurrent_state_elements)
        # Online execution retains an in-progress patch sum and the previous
        # patch correction while producing the current native-rate outputs.
        return core_elements + 2 * self.width


class AxoTemporalModel(nn.Module):
    """Shared AxoMamba encoder/heads with an interchangeable temporal core."""

    def __init__(
        self,
        source: BranchOfficialMamba,
        temporal_config: AxoTemporalCoreConfig,
    ) -> None:
        super().__init__()
        if temporal_config.kind == "mamba":
            raise ValueError(
                "AxoTemporalModel is for replacement cores; keep the "
                "source model for kind='mamba'"
            )
        self.temporal_config = temporal_config
        self.backbone = deepcopy(source)
        self.backbone.blocks = nn.ModuleList()
        if not temporal_config.keep_local_tcn:
            self.backbone.local_tcn_blocks = nn.ModuleList()
            self.backbone.local_tcn_out = None
        width = int(self.backbone.config.model_dim)
        if temporal_config.kind == "gru" and temporal_config.gru_layers > 1:
            temporal_core: nn.Module = StackedResidualGRUTemporalCore(
                [
                    ResidualGRUTemporalCore(
                        width,
                        hidden_units=temporal_config.gru_hidden_units,
                        residual_scale_init=temporal_config.residual_scale_init,
                    )
                    for _ in range(temporal_config.gru_layers)
                ]
            )
        elif temporal_config.kind == "gru":
            temporal_core = ResidualGRUTemporalCore(
                width,
                hidden_units=temporal_config.gru_hidden_units,
                residual_scale_init=temporal_config.residual_scale_init,
            )
        else:
            temporal_core = BranchELMTemporalCore(
                width,
                memory_units=temporal_config.branch_memory_units,
                hidden_units=temporal_config.branch_hidden_units,
                synapse_decay=temporal_config.branch_synapse_decay,
                memory_decay=temporal_config.branch_memory_decay,
                residual_scale_init=temporal_config.residual_scale_init,
            )
        if temporal_config.patch_size > 1:
            temporal_core = CausalPatchedTemporalCore(
                temporal_core,
                width=width,
                patch_size=temporal_config.patch_size,
            )
        self.temporal_core = temporal_core
        behavior_id = temporal_config.behavior_adapter_morphology_id
        if behavior_id is None:
            self.behavior_adapter = None
        else:
            morphology_ids = tuple(
                self.backbone.config.morphology_ids or ()
            )
            if behavior_id not in morphology_ids:
                raise ValueError(
                    "behavior adapter morphology is not configured in "
                    "the backbone"
                )
            self.behavior_adapter = NeuronBehaviorAdapter(
                branches=self.num_branch,
                width=width,
                outputs=self.num_output,
                morphology_index=morphology_ids.index(behavior_id),
                rank=temporal_config.behavior_adapter_rank,
                branch_token_offset=(
                    temporal_config.behavior_adapter_branch_token_offset
                ),
            )

    @property
    def config(self):
        return self.backbone.config

    @property
    def num_input(self) -> int:
        return int(self.backbone.num_input)

    @property
    def num_output(self) -> int:
        return int(self.backbone.num_output)

    @property
    def num_branch(self) -> int:
        return int(self.backbone.num_branch)

    @property
    def base_soma_prediction(self) -> torch.Tensor | None:
        return self.backbone.base_soma_prediction

    def forward(
        self,
        x: torch.Tensor,
        *,
        morphology_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        adapter = self.behavior_adapter
        adapter_mask = None
        branch_log_gain = None
        branch_bias = None
        branch_projection_down = None
        branch_projection_up = None
        if adapter is not None:
            if morphology_indices is None:
                raise ValueError(
                    "morphology indices are required for behavior "
                    "adaptation"
                )
            adapter_mask = (
                morphology_indices == adapter.morphology_index
            ).to(dtype=x.dtype, device=x.device)
            if adapter.branch_log_gain is not None:
                assert adapter.branch_bias is not None
                branch_log_gain = (
                    adapter_mask.unsqueeze(1)
                    * adapter.branch_log_gain
                )
                branch_bias = (
                    adapter_mask.unsqueeze(1) * adapter.branch_bias
                )
                branch_projection_down = (
                    adapter.branch_projection_down
                )
                branch_projection_up = adapter.branch_projection_up
        hidden, local_source, highpass_features = (
            self.backbone._encode_sequence(
                x,
                morphology_indices=morphology_indices,
                branch_log_gain_delta=branch_log_gain,
                branch_bias_delta=branch_bias,
                branch_projection_down=branch_projection_down,
                branch_projection_up=branch_projection_up,
                branch_projection_adapter_mask=adapter_mask,
            )
        )
        temporal_input = hidden
        if adapter is not None:
            assert adapter_mask is not None
            latent_mask = adapter_mask.view(-1, 1, 1)
            hidden = (
                hidden
                * (
                    1.0
                    + latent_mask
                    * adapter.latent_gain.view(1, 1, -1)
                )
                + latent_mask
                * adapter.latent_offset.view(1, 1, -1)
            )
            if adapter.branch_token_offset is not None:
                hidden = (
                    hidden
                    + latent_mask
                    * adapter.branch_token_offset.view(1, 1, -1)
                )
            local_source = hidden
            temporal_input = hidden
        temporal_kwargs = {}
        if adapter is not None and adapter.rank:
            assert adapter.gru_input_up is not None
            assert adapter.gru_input_down is not None
            assert adapter.gru_recurrent_up is not None
            assert adapter.gru_recurrent_down is not None
            temporal_kwargs = {
                "weight_ih_delta": (
                    adapter.gru_input_up @ adapter.gru_input_down
                ),
                "weight_hh_delta": (
                    adapter.gru_recurrent_up
                    @ adapter.gru_recurrent_down
                ),
                "adapter_mask": adapter_mask,
            }
        hidden = self.backbone._run_temporal_sequence_with_core(
            hidden,
            local_source,
            temporal_core=self.temporal_core,
            temporal_kwargs=temporal_kwargs,
        )
        if adapter is not None:
            assert adapter_mask is not None
            recurrent_mask = adapter_mask.view(-1, 1, 1)
            correction = hidden - temporal_input
            hidden = temporal_input + correction * (
                1.0
                + recurrent_mask
                * adapter.recurrent_output_gain.view(1, 1, -1)
            )
        return self.backbone._decode_sequence(
            hidden,
            highpass_features,
            readout_weight_delta=(
                None if adapter is None else adapter.readout_weight
            ),
            readout_bias_delta=(
                None if adapter is None else adapter.readout_bias
            ),
            readout_adapter_mask=adapter_mask,
        )

    def allocate_streaming_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> AxoTemporalStreamingState:
        """Allocate exact online state without retaining native-rate history."""

        self._validate_streaming_configuration()
        if batch_size <= 0:
            raise ValueError("streaming batch_size must be positive")
        parameter = next(self.parameters())
        resolved_device = (
            parameter.device if device is None else torch.device(device)
        )
        resolved_dtype = parameter.dtype if dtype is None else dtype
        core = self._unwrapped_temporal_core()
        if isinstance(core, ResidualGRUTemporalCore):
            core_state: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            core_state = torch.zeros(
                batch_size,
                core.hidden_units,
                device=resolved_device,
                dtype=resolved_dtype,
            )
        elif isinstance(core, BranchELMTemporalCore):
            core_state = (
                torch.zeros(
                    batch_size,
                    core.width,
                    device=resolved_device,
                    dtype=resolved_dtype,
                ),
                torch.zeros(
                    batch_size,
                    core.memory_units,
                    device=resolved_device,
                    dtype=resolved_dtype,
                ),
            )
        else:
            raise TypeError(f"unsupported temporal core: {type(core)}")
        local_tcn_histories = [
            torch.zeros(
                batch_size,
                block.depthwise.in_channels,
                block.right_padding,
                device=resolved_device,
                dtype=resolved_dtype,
            )
            for block in self.backbone.local_tcn_blocks
        ]
        patch_sum = None
        patch_correction = None
        if isinstance(self.temporal_core, CausalPatchedTemporalCore):
            width = self.temporal_core.width
            patch_sum = torch.zeros(
                batch_size,
                width,
                device=resolved_device,
                dtype=resolved_dtype,
            )
            patch_correction = torch.zeros_like(patch_sum)
        return AxoTemporalStreamingState(
            core_state=core_state,
            local_tcn_histories=local_tcn_histories,
            morphology_feature_gains=(
                self.backbone._all_morphology_feature_gains(
                    dtype=resolved_dtype,
                    device=resolved_device,
                )
            ),
            patch_sum=patch_sum,
            patch_correction=patch_correction,
        )

    def streaming_step(
        self,
        x: torch.Tensor,
        state: AxoTemporalStreamingState,
        *,
        morphology_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance one dense native-rate timestep."""

        self._validate_streaming_configuration()
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if (
            x.ndim != 3
            or x.shape[1] != 1
            or x.shape[2] != self.num_input
        ):
            raise ValueError(
                "streaming input must have shape (batch, channels) or "
                "(batch, 1, channels)"
            )
        routed_events = self.backbone._route_input_to_synapses(
            x,
            morphology_indices=morphology_indices,
        )
        routed_synapses = self.backbone._apply_synapse_weights(
            routed_events,
            morphology_indices=morphology_indices,
            morphology_feature_gains=state.morphology_feature_gains,
        )
        branch_inputs = self.backbone._apply_streaming_branch_transforms(
            routed_synapses.sum(dim=-1)
        )
        return self._streaming_step_from_branch_inputs(
            branch_inputs,
            state,
            morphology_indices=morphology_indices,
            routed_synapses=routed_synapses,
        )

    def streaming_step_events(
        self,
        event_indices: torch.Tensor,
        event_values: torch.Tensor,
        state: AxoTemporalStreamingState,
        *,
        morphology_indices: torch.Tensor | None = None,
        chunk_size: int | None = None,
        output_buffer: torch.Tensor | None = None,
        retain_base_soma_prediction: bool = True,
    ) -> torch.Tensor:
        """Advance one timestep from padded sparse channel/value rows."""

        self._validate_streaming_configuration()
        if (
            event_indices.ndim != 2
            or event_values.shape != event_indices.shape
        ):
            raise ValueError(
                "streaming events must have matching "
                "(batch, events) indices and values"
            )
        if event_indices.shape[0] <= 0:
            raise ValueError("streaming event batch must be positive")
        if event_indices.is_floating_point() or event_indices.is_complex():
            raise ValueError("streaming event indices must be integers")
        resolved_chunk_size = (
            event_indices.shape[0]
            if chunk_size is None
            else chunk_size
        )
        if resolved_chunk_size <= 0:
            raise ValueError("streaming chunk_size must be positive")
        parameter = next(self.parameters())
        expected_output_shape = (
            event_indices.shape[0],
            1,
            self.num_output,
        )
        if output_buffer is not None:
            if output_buffer.shape != expected_output_shape:
                raise ValueError(
                    "streaming output_buffer must have shape "
                    f"{expected_output_shape}"
                )
            if output_buffer.device != parameter.device:
                raise ValueError(
                    "streaming output_buffer must be on the model device"
                )
            if output_buffer.dtype != parameter.dtype:
                raise ValueError(
                    "streaming output_buffer must match the model dtype"
                )
        outputs = []
        base_soma_predictions = []
        next_patch_position = None
        for start in range(
            0,
            event_indices.shape[0],
            resolved_chunk_size,
        ):
            stop = min(
                start + resolved_chunk_size,
                event_indices.shape[0],
            )
            chunk_indices = event_indices[start:stop].to(
                device=parameter.device,
                dtype=torch.long,
            )
            chunk_values = event_values[start:stop].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            chunk_morphology = (
                None
                if morphology_indices is None
                else morphology_indices[start:stop].to(
                    device=parameter.device,
                    dtype=torch.long,
                )
            )
            chunk_state = self._slice_streaming_state(
                state,
                start,
                stop,
            )
            if self.backbone.branch_subunit_projection is None:
                branch_inputs = (
                    self.backbone._sparse_streaming_branch_inputs(
                        chunk_indices,
                        chunk_values,
                        morphology_indices=chunk_morphology,
                        morphology_feature_gains=(
                            state.morphology_feature_gains
                        ),
                    )
                )
                chunk_output = self._streaming_step_from_branch_inputs(
                    branch_inputs,
                    chunk_state,
                    morphology_indices=chunk_morphology,
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
                    chunk_values
                    * (chunk_indices >= 0).to(dtype=parameter.dtype),
                )
                chunk_output = self.streaming_step(
                    dense,
                    chunk_state,
                    morphology_indices=chunk_morphology,
                )
            if output_buffer is None:
                outputs.append(chunk_output)
            else:
                output_buffer[start:stop].copy_(chunk_output)
            if (
                retain_base_soma_prediction
                and self.base_soma_prediction is not None
            ):
                base_soma_predictions.append(
                    self.base_soma_prediction
                )
            elif not retain_base_soma_prediction:
                self.backbone.base_soma_prediction = None
            if next_patch_position is None:
                next_patch_position = chunk_state.patch_position
            elif next_patch_position != chunk_state.patch_position:
                raise RuntimeError(
                    "streaming chunks advanced to different patch phases"
                )
        if next_patch_position is not None:
            state.patch_position = next_patch_position
        if base_soma_predictions:
            self.backbone.base_soma_prediction = torch.cat(
                base_soma_predictions,
                dim=0,
            )
        elif not retain_base_soma_prediction:
            self.backbone.base_soma_prediction = None
        if output_buffer is not None:
            return output_buffer
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _slice_streaming_state(
        state: AxoTemporalStreamingState,
        start: int,
        stop: int,
    ) -> AxoTemporalStreamingState:
        core_state = state.core_state
        if isinstance(core_state, torch.Tensor):
            chunk_core_state: (
                torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            ) = core_state[start:stop]
        else:
            chunk_core_state = (
                core_state[0][start:stop],
                core_state[1][start:stop],
            )
        return AxoTemporalStreamingState(
            core_state=chunk_core_state,
            local_tcn_histories=[
                history[start:stop]
                for history in state.local_tcn_histories
            ],
            morphology_feature_gains=state.morphology_feature_gains,
            patch_sum=(
                None
                if state.patch_sum is None
                else state.patch_sum[start:stop]
            ),
            patch_correction=(
                None
                if state.patch_correction is None
                else state.patch_correction[start:stop]
            ),
            patch_position=state.patch_position,
        )

    def _streaming_step_from_branch_inputs(
        self,
        branch_inputs: torch.Tensor,
        state: AxoTemporalStreamingState,
        *,
        morphology_indices: torch.Tensor | None,
        routed_synapses: torch.Tensor | None = None,
    ) -> torch.Tensor:
        backbone = self.backbone
        hidden = backbone.branch_projection(branch_inputs)
        if backbone.branch_subunit_projection is not None:
            if routed_synapses is None:
                raise ValueError(
                    "sparse streaming does not support branch subunits"
                )
            subunit_inputs = backbone._branch_subunit_features(
                routed_synapses
            )
            if backbone.branch_gain is not None:
                subunit_inputs = subunit_inputs * backbone.branch_gain.to(
                    dtype=subunit_inputs.dtype,
                    device=subunit_inputs.device,
                ).view(1, 1, -1, 1)
            hidden = hidden + (
                backbone.config.branch_subunit_scale
                * backbone.branch_subunit_projection(
                    subunit_inputs.flatten(start_dim=-2)
                )
            )
        hidden = backbone._apply_projected_branch_bilinear(
            hidden,
            backbone._branch_bilinear_projection(branch_inputs),
        )
        if backbone.morphology_embedding is not None:
            if morphology_indices is None:
                raise ValueError(
                    "morphology_indices are required for "
                    "morphology-conditioned AxoTemporalModel"
                )
            morphology = backbone._morphology_embedding_values(
                morphology_indices,
                dtype=hidden.dtype,
                device=hidden.device,
            )
            assert backbone.config.morphology_embedding_scale is not None
            hidden = hidden + (
                backbone.config.morphology_embedding_scale
                * morphology.unsqueeze(1)
            )
        hidden = self._advance_temporal_core(hidden, state)
        hidden = hidden + backbone._stream_local_tcn_correction(
            hidden,
            state.local_tcn_histories,
        )
        return backbone._decode_sequence(hidden, None)

    def _advance_temporal_core(
        self,
        hidden: torch.Tensor,
        state: AxoTemporalStreamingState,
    ) -> torch.Tensor:
        if not isinstance(
            self.temporal_core,
            CausalPatchedTemporalCore,
        ):
            return hidden + self._advance_unwrapped_core(hidden, state)
        assert state.patch_sum is not None
        assert state.patch_correction is not None
        corrected = hidden + state.patch_correction.unsqueeze(1)
        state.patch_sum.add_(hidden.squeeze(1))
        state.patch_position += 1
        if state.patch_position == self.temporal_core.patch_size:
            patch_mean = (
                state.patch_sum / self.temporal_core.patch_size
            ).unsqueeze(1)
            next_correction = self._advance_unwrapped_core(
                patch_mean,
                state,
            )
            state.patch_correction.copy_(next_correction.squeeze(1))
            state.patch_sum.zero_()
            state.patch_position = 0
        return corrected

    def _advance_unwrapped_core(
        self,
        hidden: torch.Tensor,
        state: AxoTemporalStreamingState,
    ) -> torch.Tensor:
        core = self._unwrapped_temporal_core()
        if isinstance(core, ResidualGRUTemporalCore):
            if not isinstance(state.core_state, torch.Tensor):
                raise TypeError("GRU streaming state has the wrong type")
            recurrent, next_state = core.gru(
                core.pre_norm(hidden),
                state.core_state.unsqueeze(0),
            )
            state.core_state.copy_(next_state.squeeze(0))
            return (
                core.residual_scale.view(1, 1, -1)
                * core.output_projection(recurrent)
            )
        if not isinstance(state.core_state, tuple):
            raise TypeError("Branch-RNN streaming state has the wrong type")
        branch_state, memory_state = state.core_state
        normalized = core.pre_norm(hidden).squeeze(1)
        branch_state = (
            core.synapse_decay * branch_state + normalized
        )
        decayed_memory = core.memory_decay * memory_state
        update = core.integration(
            torch.cat((branch_state, decayed_memory), dim=-1)
        )
        memory_state = (
            decayed_memory
            + (1.0 - core.memory_decay) * update
        )
        state.core_state[0].copy_(branch_state)
        state.core_state[1].copy_(memory_state)
        return (
            core.residual_scale.view(1, 1, -1)
            * core.output_projection(memory_state).unsqueeze(1)
        )

    def _unwrapped_temporal_core(
        self,
    ) -> ResidualGRUTemporalCore | BranchELMTemporalCore:
        core = self.temporal_core
        if isinstance(core, CausalPatchedTemporalCore):
            core = core.core
        if not isinstance(
            core,
            (ResidualGRUTemporalCore, BranchELMTemporalCore),
        ):
            raise TypeError(f"unsupported temporal core: {type(core)}")
        return core

    def _validate_streaming_configuration(self) -> None:
        unsupported = []
        backbone = self.backbone
        if self.training:
            unsupported.append("training mode")
        if backbone.branch_trace_taus:
            unsupported.append("branch traces")
        if backbone.hidden_trace_taus:
            unsupported.append("hidden traces")
        if (
            backbone.integrative_pre is not None
            or backbone.integrative_post is not None
        ):
            unsupported.append("integrative mixer")
        if backbone.temporal_refine is not None:
            unsupported.append("temporal refine")
        if (
            backbone.local_tcn_blocks
            and backbone.config.local_tcn_position != "post"
        ):
            unsupported.append("non-post local TCN")
        if backbone.soma_filter_logit is not None:
            unsupported.append("soma filter")
        if backbone.soma_highpass_correction_scale is not None:
            unsupported.append("soma highpass correction")
        if unsupported:
            raise ValueError(
                "streaming AxoTemporalModel does not support "
                + ", ".join(unsupported)
            )

    def recurrent_state_bytes(
        self,
        *,
        dtype: torch.dtype,
    ) -> int:
        return (
            self.temporal_core.recurrent_state_elements
            * torch.empty((), dtype=dtype).element_size()
        )

    def component_manifest(self) -> dict[str, object]:
        config = self.backbone.config
        return {
            "temporal_core": self.temporal_config.kind,
            "shared_axomamba_front_end": True,
            "shared_axomamba_heads": True,
            "morphology_conditioning": (
                config.morphology_ids is not None
            ),
            "morphology_embedding": (
                config.morphology_embedding_scale is not None
            ),
            "morphology_synapse_gain": (
                config.morphology_synapse_gain_scale is not None
            ),
            "branch_nonlinearity": config.branch_nonlinearity,
            "local_tcn": config.local_tcn_scale is not None,
            "local_tcn_retained": self.backbone.local_tcn_out is not None,
            "local_tcn_mode": config.local_tcn_mode,
            "local_tcn_position": config.local_tcn_position,
            "temporal_patch_size": self.temporal_config.patch_size,
            "separate_heads": config.separate_heads,
            "spike_voltage_coupling_scale": (
                config.spike_voltage_coupling_scale
            ),
            "temporal_config": asdict(self.temporal_config),
        }


def create_temporal_model(
    source: BranchOfficialMamba,
    temporal_config: AxoTemporalCoreConfig,
) -> BranchOfficialMamba | AxoTemporalModel:
    """Compose a selected AxoMamba front end with one temporal core."""

    if temporal_config.kind == "mamba":
        return source
    return AxoTemporalModel(source, temporal_config)


def migrate_legacy_gru_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map the internal I108 GRU wrapper into the unified checkpoint schema."""

    migrated = {}
    for name, value in state_dict.items():
        if name == "residual_scale":
            target = "temporal_core.residual_scale"
        elif name.startswith("pre_gru_norm."):
            target = "temporal_core.pre_norm." + name.removeprefix(
                "pre_gru_norm."
            )
        elif name.startswith("gru."):
            target = "temporal_core.gru." + name.removeprefix("gru.")
        else:
            target = name
        migrated[target] = value
    return migrated
