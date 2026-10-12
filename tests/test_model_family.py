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
    assert {
        profile_id: profile.mamba_update_normalization
        for profile_id, profile in AXOSIM_MODEL_PROFILES.items()
        if profile.mamba_update_normalization != "none"
    } == {"axosim-mamba-large": "fixed_rms"}


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
    assert (
        mamba.config.mamba_update_normalization
        == AXOSIM_MODEL_PROFILES[
            "axosim-mamba-small"
        ].mamba_update_normalization
    )


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

    assert medium.config.model_dim == 176
    assert medium.config.state_dim == 44
    assert medium.config.num_layers == 4
    assert 1_000_000 <= medium_parameters <= 1_250_000
    assert large_parameters >= 15_000_000
    assert large_parameters >= 12 * medium_parameters


def test_gru_medium_is_a_real_capacity_step_above_one_hundred_thousand() -> None:
    small = create_axosim_profile(
        "axosim-gru-small",
        use_pytorch_fallback=True,
    )
    medium = create_axosim_profile(
        "axosim-gru-medium",
        use_pytorch_fallback=True,
    )

    small_parameters = sum(parameter.numel() for parameter in small.parameters())
    medium_parameters = sum(parameter.numel() for parameter in medium.parameters())

    assert medium.config.model_dim == 128
    assert medium.temporal_config.gru_hidden_units == 128
    assert medium_parameters >= 100_000
    assert medium_parameters >= 5 * small_parameters
