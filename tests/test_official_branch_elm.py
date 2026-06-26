from __future__ import annotations

from pathlib import Path

import pytest
import torch

from axosim.official import create_official_elm, load_official_elm_config, load_official_state_dict


def test_official_num_memory_30_config_builds_paper_model() -> None:
    config = load_official_elm_config(Path("configs/branch_elm_30_official.json"))

    assert config["input_to_synapse_routing"] == "neuronio_routing"
    assert config["num_input"] == 1278
    assert config["num_branch"] == 45
    assert config["num_synapse_per_branch"] == 100
    assert config["num_memory"] == 30

    model = create_official_elm(config)

    assert model.num_input == 1278
    assert model.num_branch == 45
    assert model.num_synapse == 4500
    assert model.num_memory == 30


def test_official_neuronio_eval_forward_preserves_upstream_postprocessing() -> None:
    model = create_official_elm(
        {
            "input_to_synapse_routing": "neuronio_routing",
            "learn_memory_tau": False,
            "memory_tau_max": 150.0,
            "memory_tau_min": 1.0,
            "num_branch": 3,
            "num_input": 6,
            "num_memory": 4,
            "num_output": 2,
            "num_synapse_per_branch": 2,
        }
    )
    x = torch.zeros(2, 5, 6)

    raw = model(x)
    postprocessed = model.neuronio_eval_forward(x)

    assert raw.shape == (2, 5, 2)
    assert postprocessed.shape == (2, 5, 2)
    assert torch.all((0 <= postprocessed[..., 0]) & (postprocessed[..., 0] <= 1))
    assert torch.allclose(postprocessed[..., 1], raw[..., 1] / 0.1)


def test_official_num_memory_30_checkpoint_loads_strictly_when_available() -> None:
    checkpoint_path = Path(".external/elmneuron/models/num_memory_30/neuronio_best_model_state.pt")
    if not checkpoint_path.exists():
        pytest.skip("upstream elmneuron checkout is not present")

    config = load_official_elm_config(Path("configs/branch_elm_30_official.json"))
    model = create_official_elm(config)
    state_dict = load_official_state_dict(checkpoint_path, map_location="cpu")

    model.load_state_dict(state_dict, strict=True)
