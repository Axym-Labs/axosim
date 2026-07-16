from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from axosim.mamba_official import BranchOfficialMamba, BranchOfficialMambaConfig


AXOMAMBA_MODEL_KIND = "axomamba"
AXOMAMBA_DEFAULT_CONFIG = "configs/axomamba.json"


@dataclass(frozen=True)
class AxoMambaConfig(BranchOfficialMambaConfig):
    """Named config for the current best compact AxoSim Mamba surrogate."""


def default_axomamba_config(**overrides: Any) -> AxoMambaConfig:
    """Return the promoted compact Mamba config.

    This is the ablation-retained compact stack: 118 routed branches,
    shared and morphology-conditioned synaptic efficacy, a learned route-
    occupancy operator, branch-current saturation, separate task readouts,
    and a post-Mamba causal TCN correction at dilations 1, 4, and 16. The local
    correction passed the iteration-160 300k SERA-first gate; iteration 180
    promoted raw-soma adaptation, and the post-I200 larger ablation removed
    the recoverable soma gate and bilinear interaction after a matched 300k
    continuation.
    """

    values: dict[str, Any] = {
        "num_input": 1278,
        "num_output": 2,
        "num_branch": 118,
        "num_synapse_per_branch": 11,
        "input_to_synapse_routing": "neuronio_routing",
        "model_dim": 58,
        "num_layers": 2,
        "block_repeats": 1,
        "state_dim": 23,
        "conv_kernel": 4,
        "expansion": 2,
        "block_type": "mamba1",
        "dropout": 0.0,
        "morphology_ids": [
            "specimen_479770916",
            "specimen_487664663",
            "specimen_488679042",
            "specimen_488683425",
            "specimen_589427435",
        ],
        "morphology_embedding_scale": 0.15,
        "residual_scale_init": 0.1,
        "synapse_gain_scale": 1.0,
        "morphology_synapse_gain_scale": 1.0,
        "morphology_synapse_feature_dim": 9,
        "morphology_synapse_feature_hidden": 16,
        "morphology_synapse_feature_scale": 1.0,
        "branch_gain": False,
        "branch_nonlinearity": "tanh",
        "branch_nonlinearity_scale_init": 0.1,
        "branch_bilinear_rank": 0,
        "branch_bilinear_scale": None,
        "separate_heads": True,
        "soma_multiplicative_gate": False,
        "soma_multiplicative_scale_init": 0.1,
        "spike_voltage_coupling_scale": 4.0,
        "integrative_mixer": False,
        "integrative_mixer_short_kernel": 9,
        "integrative_mixer_long_kernel": 65,
        "integrative_mixer_position": "pre",
        "integrative_mixer_long_path_bias": 0.5,
        "local_tcn_scale": 1.0,
        "local_tcn_mode": "dilated",
        "local_tcn_position": "post",
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
