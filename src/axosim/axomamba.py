from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from axosim.mamba_official import BranchOfficialMamba, BranchOfficialMambaConfig


AXOMAMBA_MODEL_KIND = "axomamba"
AXOMAMBA_DEFAULT_CONFIG = "configs/axomamba_45x64.json"


@dataclass(frozen=True)
class AxoMambaConfig(BranchOfficialMambaConfig):
    """Named config for the current best compact AxoSim Mamba surrogate."""


def default_axomamba_config(**overrides: Any) -> AxoMambaConfig:
    """Return the promoted compact Mamba config.

    This is the best retained Mamba mechanism stack from the AxoSim notes:
    branch gain, fixed branch-current saturation, pre-Mamba short/long
    temporal mixing, and a multiplicative soma gate.
    """

    values: dict[str, Any] = {
        "num_input": 1278,
        "num_output": 2,
        "num_branch": 45,
        "num_synapse_per_branch": 100,
        "input_to_synapse_routing": "neuronio_routing",
        "model_dim": 64,
        "num_layers": 2,
        "block_repeats": 1,
        "state_dim": 16,
        "conv_kernel": 4,
        "expansion": 2,
        "block_type": "mamba1",
        "dropout": 0.0,
        "residual_scale_init": 0.1,
        "branch_gain": True,
        "branch_nonlinearity": "tanh",
        "branch_nonlinearity_scale_init": 0.1,
        "soma_multiplicative_gate": True,
        "soma_multiplicative_scale_init": 0.1,
        "integrative_mixer": True,
        "integrative_mixer_short_kernel": 9,
        "integrative_mixer_long_kernel": 65,
        "integrative_mixer_position": "pre",
        "integrative_mixer_long_path_bias": 0.5,
    }
    values.update(overrides)
    return AxoMambaConfig(**values)


def load_axomamba_config(path: str | Path | None = None, **overrides: Any) -> AxoMambaConfig:
    if path is None:
        return default_axomamba_config(**overrides)
    with Path(path).open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    values.update(overrides)
    return AxoMambaConfig(**values)


def coerce_axomamba_config(config: AxoMambaConfig | BranchOfficialMambaConfig | Mapping[str, Any] | None) -> AxoMambaConfig:
    if config is None:
        return default_axomamba_config()
    if isinstance(config, AxoMambaConfig):
        return config
    if isinstance(config, BranchOfficialMambaConfig):
        return AxoMambaConfig(**config.__dict__)
    return AxoMambaConfig(**dict(config))


class AxoMamba(BranchOfficialMamba):
    """Reusable named AxoSim Mamba surrogate built on the official Mamba backend."""

    def __init__(
        self,
        config: AxoMambaConfig | BranchOfficialMambaConfig | Mapping[str, Any] | None = None,
        *,
        mamba_classes: tuple[type, type | None] | None = None,
    ) -> None:
        super().__init__(coerce_axomamba_config(config), mamba_classes=mamba_classes)


class AxoPyTorchMamba(AxoMamba):
    """AxoMamba with the dependency-free PyTorch Mamba block, for tests/smokes."""

    def __init__(self, config: AxoMambaConfig | BranchOfficialMambaConfig | Mapping[str, Any] | None = None) -> None:
        resolved = coerce_axomamba_config(config)
        if resolved.block_type != "mamba1":
            raise ValueError("AxoPyTorchMamba only supports block_type='mamba1'")
        from axosim.mamba_pytorch import MambaBlock

        super().__init__(resolved, mamba_classes=(MambaBlock, None))


def create_axomamba(
    config: AxoMambaConfig | BranchOfficialMambaConfig | Mapping[str, Any] | None = None,
    *,
    use_pytorch_fallback: bool = False,
) -> AxoMamba:
    if use_pytorch_fallback:
        return AxoPyTorchMamba(config)
    return AxoMamba(config)
