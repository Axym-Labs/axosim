from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_official_elm_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return json.loads(config_path.read_text(encoding="utf-8"))


def create_official_elm(config: Mapping[str, Any]):
    from bnn_sim.official.expressive_leaky_memory_neuron import ELM

    return ELM(**dict(config))


def load_official_state_dict(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    import torch

    checkpoint_path = Path(path)
    try:
        state = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError(f"expected a state dict mapping in {checkpoint_path}")
    return dict(state)


__all__ = [
    "ELM",
    "create_official_elm",
    "load_official_elm_config",
    "load_official_state_dict",
]


def __getattr__(name: str) -> Any:
    if name == "ELM":
        from bnn_sim.official.expressive_leaky_memory_neuron import ELM

        return ELM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
