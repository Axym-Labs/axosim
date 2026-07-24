from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn

from axosim.mamba_official import BranchOfficialMamba

TemporalCoreKind = Literal["mamba", "gru", "branch_rnn"]


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
    patch_size: int = 1

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

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.temporal_correction(hidden)

    def temporal_correction(self, hidden: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.gru(self.pre_norm(hidden))
        return (
            self.residual_scale.view(1, 1, -1)
            * self.output_projection(recurrent)
        )

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

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
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
            coarse_correction = correction(coarse)
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
        if temporal_config.kind == "gru":
            temporal_core: nn.Module = ResidualGRUTemporalCore(
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
        hidden, local_source, highpass_features = (
            self.backbone._encode_sequence(
                x,
                morphology_indices=morphology_indices,
            )
        )
        hidden = self.backbone._run_temporal_sequence_with_core(
            hidden,
            local_source,
            temporal_core=self.temporal_core,
        )
        return self.backbone._decode_sequence(
            hidden,
            highpass_features,
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
