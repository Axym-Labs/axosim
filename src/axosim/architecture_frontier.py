from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from axosim.autoresearch import (
    model_population_memory_metrics,
    model_resident_state_metrics,
)
from axosim.block_forecast import (
    BlockForecastConfig,
    CausalBlockForecastModel,
)
from axosim.mamba_official import BranchOfficialMamba
from axosim.temporal_core import (
    AxoTemporalCoreConfig,
    AxoTemporalModel,
    create_temporal_model,
    migrate_legacy_gru_state_dict,
)

TaskKind = Literal["regression", "spike"]


@dataclass(frozen=True)
class ArchitectureSpec:
    """One task/core composition in the unified architecture frontier."""

    candidate_id: str
    task: TaskKind
    temporal_core: Literal["mamba", "gru", "branch_rnn"]
    keep_local_tcn: bool = False
    gru_hidden_units: int | None = None
    branch_memory_units: int = 30
    branch_hidden_units: int = 64
    patch_size: int = 1
    block_forecast: bool = False
    max_patch_rank: int = 8
    max_trajectory_rank: int = 8
    voltage_anchor_count: int | None = None
    event_template: tuple[float, ...] | None = None
    event_template_center_init: float = 0.9208316802978516
    event_template_temperature: float = 0.25
    event_template_hard: bool = False

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if self.task not in {"regression", "spike"}:
            raise ValueError(f"unknown architecture task: {self.task}")
        if self.block_forecast:
            if self.temporal_core not in {"mamba", "gru"}:
                raise ValueError("block forecasting supports Mamba or GRU")
            if self.patch_size <= 1:
                raise ValueError(
                    "block forecasting requires patch_size greater than one"
                )

    def temporal_config(self) -> AxoTemporalCoreConfig:
        return AxoTemporalCoreConfig(
            kind=self.temporal_core,
            keep_local_tcn=self.keep_local_tcn,
            gru_hidden_units=self.gru_hidden_units,
            branch_memory_units=self.branch_memory_units,
            branch_hidden_units=self.branch_hidden_units,
            patch_size=self.patch_size,
        )


def build_architecture_candidate(
    source: BranchOfficialMamba,
    spec: ArchitectureSpec,
) -> BranchOfficialMamba | AxoTemporalModel | CausalBlockForecastModel:
    """Compose one task checkpoint with the requested temporal core."""

    if spec.block_forecast:
        return CausalBlockForecastModel(
            source,
            BlockForecastConfig(
                core_kind=spec.temporal_core,
                patch_size=spec.patch_size,
                max_patch_rank=spec.max_patch_rank,
                max_trajectory_rank=spec.max_trajectory_rank,
                keep_local_tcn=spec.keep_local_tcn,
                gru_hidden_units=spec.gru_hidden_units,
                voltage_anchor_count=spec.voltage_anchor_count,
                event_template=spec.event_template,
                event_template_center_init=(spec.event_template_center_init),
                event_template_temperature=(spec.event_template_temperature),
                event_template_hard=spec.event_template_hard,
            ),
        )
    return create_temporal_model(source, spec.temporal_config())


def load_legacy_gru_candidate(
    source: BranchOfficialMamba,
    checkpoint_path: str | Path,
    *,
    candidate_id: str,
    task: TaskKind,
) -> tuple[AxoTemporalModel, dict]:
    """Load an I108/I166/I170 GRU artifact into the unified model."""

    spec = ArchitectureSpec(
        candidate_id=candidate_id,
        task=task,
        temporal_core="gru",
        keep_local_tcn=False,
    )
    model = build_architecture_candidate(source, spec)
    if not isinstance(model, AxoTemporalModel):
        raise TypeError("legacy GRU migration did not build a replacement core")
    payload = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError("legacy GRU checkpoint has no state_dict")
    model.load_state_dict(
        migrate_legacy_gru_state_dict(state_dict),
        strict=True,
    )
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    return model, metadata


def architecture_evidence(
    model: nn.Module,
    spec: ArchitectureSpec,
) -> dict[str, object]:
    """Record composition and memory evidence in one schema."""

    manifest = (
        model.component_manifest()
        if hasattr(model, "component_manifest")
        else {
            "temporal_core": "mamba",
            "shared_axomamba_front_end": True,
            "shared_axomamba_heads": True,
        }
    )
    return {
        "spec": asdict(spec),
        "component_manifest": manifest,
        "resident": model_resident_state_metrics(model),
        "population_memory": model_population_memory_metrics(model),
    }


def transfer_compatible_state(
    target: nn.Module,
    source_state: dict[str, torch.Tensor],
) -> dict[str, object]:
    """Load equal-shaped tensors while preserving target-only components."""

    target_state = target.state_dict()
    copied: dict[str, torch.Tensor] = {}
    copied_sources = set()
    aliases = {}
    for source_name, value in source_state.items():
        candidates = [source_name]
        if source_name.startswith("temporal_core."):
            suffix = source_name.removeprefix("temporal_core.")
            if suffix.startswith("core."):
                candidates.append("temporal_core." + suffix.removeprefix("core."))
            else:
                candidates.append("temporal_core.core." + suffix)
        for target_name in candidates:
            if (
                target_name in target_state
                and target_state[target_name].shape == value.shape
            ):
                copied[target_name] = value
                copied_sources.add(source_name)
                if target_name != source_name:
                    aliases[source_name] = target_name
                break
    target_state.update(copied)
    target.load_state_dict(target_state, strict=True)
    return {
        "copied": sorted(copied),
        "aliases": aliases,
        "target_only": sorted(set(target_state) - set(copied)),
        "source_only_or_incompatible": sorted(set(source_state) - copied_sources),
    }
