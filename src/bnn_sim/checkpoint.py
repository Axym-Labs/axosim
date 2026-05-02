from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from bnn_sim.model import BranchELM, BranchELMConfig
from bnn_sim.official import ELM, create_official_elm


def save_checkpoint(model: torch.nn.Module, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_kind, config = _serialize_model_config(model)
    metadata_payload = {"model_kind": model_kind, **(metadata or {})}
    payload = {
        "model_kind": model_kind,
        "config": config,
        "state_dict": model.state_dict(),
        "metadata": metadata_payload,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model_kind = payload.get("model_kind", payload.get("metadata", {}).get("model_kind", "baseline"))
    if model_kind in {"baseline", "branch_elm"}:
        config = BranchELMConfig(**payload["config"])
        model = BranchELM(config)
    elif model_kind == "official":
        model = create_official_elm(payload["config"])
    else:
        raise ValueError(f"unknown model_kind in checkpoint: {model_kind}")
    model.load_state_dict(payload["state_dict"])
    return model, dict(payload.get("metadata", {}))


def _serialize_model_config(model: torch.nn.Module) -> tuple[str, dict[str, Any]]:
    if isinstance(model, BranchELM):
        return "baseline", asdict(model.config)
    if isinstance(model, ELM):
        return "official", {
            "num_input": int(model.num_input),
            "num_output": int(model.num_output),
            "num_memory": int(model.num_memory),
            "lambda_value": float(model.lambda_value),
            "mlp_num_layers": int(model.mlp_num_layers),
            "mlp_hidden_size": int(model.mlp_hidden_size),
            "mlp_activation": str(model.mlp_activation),
            "tau_s_value": float(model.tau_s_value),
            "memory_tau_min": float(model.memory_tau_min),
            "memory_tau_max": float(model.memory_tau_max),
            "learn_memory_tau": bool(model.learn_memory_tau),
            "w_s_value": float(model.w_s_value),
            "num_branch": int(model.num_branch),
            "num_synapse_per_branch": int(model.num_synapse_per_branch),
            "input_to_synapse_routing": model.input_to_synapse_routing,
            "delta_t": float(model.delta_t),
        }
    raise TypeError(f"unsupported checkpoint model type: {type(model).__name__}")
