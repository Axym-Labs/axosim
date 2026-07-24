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
    branch_memory_units: int = 30
    branch_hidden_units: int = 64

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if self.task not in {"regression", "spike"}:
            raise ValueError(f"unknown architecture task: {self.task}")

    def temporal_config(self) -> AxoTemporalCoreConfig:
        return AxoTemporalCoreConfig(
            kind=self.temporal_core,
            keep_local_tcn=self.keep_local_tcn,
            branch_memory_units=self.branch_memory_units,
            branch_hidden_units=self.branch_hidden_units,
        )


def build_architecture_candidate(
    source: BranchOfficialMamba,
    spec: ArchitectureSpec,
) -> BranchOfficialMamba | AxoTemporalModel:
    """Compose one task checkpoint with the requested temporal core."""

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
        raise ValueError("legacy GRU checkpoint has no state_dict")
    model.load_state_dict(
        migrate_legacy_gru_state_dict(state_dict),
        strict=True,
    )
    metadata = {
        key: value
        for key, value in payload.items()
        if key != "state_dict"
    }
    return model, metadata


def architecture_evidence(
    model: nn.Module,
    spec: ArchitectureSpec,
) -> dict[str, object]:
    """Record composition and memory evidence in one schema."""

    manifest = (
        model.component_manifest()
        if isinstance(model, AxoTemporalModel)
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
