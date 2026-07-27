from __future__ import annotations

from axosim.model_family import (
    AXOSIM_MODEL_PROFILES,
    create_axosim_profile,
)
from axosim.temporal_core import AxoTemporalModel


def test_named_model_family_has_two_gru_and_three_mamba_profiles() -> None:
    assert tuple(AXOSIM_MODEL_PROFILES) == (
        "axosim-gru-small",
        "axosim-gru-medium",
        "axosim-mamba-small",
        "axosim-mamba-medium",
        "axosim-mamba-large",
    )
    assert [profile.public_name for profile in AXOSIM_MODEL_PROFILES.values()] == [
        "AxoSim-GRU Small",
        "AxoSim-GRU Medium",
        "AxoSim-Mamba Small",
        "AxoSim-Mamba Medium",
        "AxoSim-Mamba Large",
    ]


def test_profile_factory_builds_the_declared_temporal_family() -> None:
    gru = create_axosim_profile(
        "axosim-gru-small",
        use_pytorch_fallback=True,
    )
    mamba = create_axosim_profile(
        "axosim-mamba-small",
        use_pytorch_fallback=True,
    )

    assert isinstance(gru, AxoTemporalModel)
    assert gru.temporal_config.kind == "gru"
    assert gru.config.model_dim == 32
    assert mamba.config.model_dim == 32
    assert mamba.config.state_dim == 8
    assert mamba.config.num_layers == 2


def test_large_mamba_is_a_material_capacity_step() -> None:
    medium = create_axosim_profile(
        "axosim-mamba-medium",
        use_pytorch_fallback=True,
    )
    large = create_axosim_profile(
        "axosim-mamba-large",
        use_pytorch_fallback=True,
    )

    medium_parameters = sum(parameter.numel() for parameter in medium.parameters())
    large_parameters = sum(parameter.numel() for parameter in large.parameters())

    assert large.config.model_dim == 256
    assert large.config.state_dim == 64
    assert large.config.num_layers == 6
    assert large_parameters >= 20 * medium_parameters
