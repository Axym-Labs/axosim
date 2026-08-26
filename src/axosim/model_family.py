from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Literal, Mapping

from torch import nn

from axosim.axomamba import (
    AxoMambaConfig,
    create_axomamba,
    default_axomamba_config,
)
from axosim.temporal_core import (
    AxoTemporalCoreConfig,
    AxoTemporalModel,
    create_temporal_model,
)

ModelFamily = Literal["gru", "mamba"]


@dataclass(frozen=True)
class AxoSimModelProfile:
    """Stable recipe for one trained-model capacity point."""

    profile_id: str
    public_name: str
    family: ModelFamily
    model_dim: int
    state_dim: int
    num_layers: int
    head_dim: int


_PROFILES = (
    AxoSimModelProfile(
        "axosim-gru-small",
        "AxoSim-GRU Small",
        "gru",
        model_dim=32,
        state_dim=8,
        num_layers=2,
        head_dim=32,
    ),
    AxoSimModelProfile(
        "axosim-gru-medium",
        "AxoSim-GRU Medium",
        "gru",
        model_dim=58,
        state_dim=23,
        num_layers=2,
        head_dim=64,
    ),
    AxoSimModelProfile(
        "axosim-mamba-small",
        "AxoSim-Mamba Small",
        "mamba",
        model_dim=32,
        state_dim=8,
        num_layers=2,
        head_dim=32,
    ),
    AxoSimModelProfile(
        "axosim-mamba-medium",
        "AxoSim-Mamba Medium",
        "mamba",
        model_dim=58,
        state_dim=23,
        num_layers=2,
        head_dim=64,
    ),
    AxoSimModelProfile(
        "axosim-mamba-large",
        "AxoSim-Mamba Large",
        "mamba",
        model_dim=256,
        state_dim=64,
        num_layers=6,
        head_dim=64,
    ),
)

AXOSIM_MODEL_PROFILES: Mapping[str, AxoSimModelProfile] = MappingProxyType(
    {profile.profile_id: profile for profile in _PROFILES}
)


def create_axosim_profile(
    profile_id: str,
    *,
    base_config: AxoMambaConfig | None = None,
    use_pytorch_fallback: bool = False,
) -> nn.Module:
    """Build one named GRU or Mamba profile from the common AxoSim shell."""

    try:
        profile = AXOSIM_MODEL_PROFILES[profile_id]
    except KeyError as error:
        raise ValueError(f"unknown AxoSim model profile: {profile_id}") from error
    values = asdict(base_config or default_axomamba_config())
    values.update(
        model_dim=profile.model_dim,
        state_dim=profile.state_dim,
        num_layers=profile.num_layers,
        head_dim=profile.head_dim,
    )
    backbone = create_axomamba(
        AxoMambaConfig(**values),
        use_pytorch_fallback=use_pytorch_fallback,
    )
    if profile.family == "mamba":
        return backbone
    model = create_temporal_model(
        backbone,
        AxoTemporalCoreConfig(
            kind="gru",
            gru_hidden_units=profile.model_dim,
            keep_local_tcn=False,
        ),
    )
    assert isinstance(model, AxoTemporalModel)
    return model
