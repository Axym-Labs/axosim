from __future__ import annotations

from pathlib import Path

import pytest
import torch

from axosim.axobench_iteration import make_official_elm_predictor
from axosim.official import (
    create_official_elm,
    load_official_elm_config,
    load_official_state_dict,
)
from axosim.official.training import official_neuronio_loss


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
    checkpoint_path = Path(
        ".external/elmneuron/models/num_memory_30/neuronio_best_model_state.pt"
    )
    if not checkpoint_path.exists():
        pytest.skip("upstream elmneuron checkout is not present")

    config = load_official_elm_config(Path("configs/branch_elm_30_official.json"))
    model = create_official_elm(config)
    state_dict = load_official_state_dict(checkpoint_path, map_location="cpu")

    model.load_state_dict(state_dict, strict=True)


def test_official_predictor_uses_shared_axobench_output_semantics(
    tmp_path: Path,
) -> None:
    config = {
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
    config_path = tmp_path / "model_config.json"
    checkpoint_path = tmp_path / "model.pt"
    config_path.write_text(__import__("json").dumps(config), encoding="utf-8")
    reference = create_official_elm(config).eval()
    torch.save(reference.state_dict(), checkpoint_path)

    predictor, metadata = make_official_elm_predictor(
        config_path,
        checkpoint_path,
        device="cpu",
    )
    inputs = torch.zeros(2, 5, 6)

    expected = reference(inputs).detach().numpy()
    actual = predictor(inputs.numpy())

    assert actual == pytest.approx(expected)
    assert metadata["output_semantics"] == {
        "spike": "logit on the shared AxoBench target contract",
        "voltage": "centered normalized voltage on the shared AxoBench target contract",
    }
    assert metadata["parameter_count"] == sum(
        parameter.numel()
        for parameter in reference.parameters()
        if parameter.requires_grad
    )
    assert metadata["resident_parameter_count"] == sum(
        parameter.numel() for parameter in reference.parameters()
    )


def test_official_training_loss_matches_published_neuronio_objective() -> None:
    prediction = torch.tensor(
        [
            [
                [0.2, -0.4],
                [-0.3, 0.7],
                [1.1, -0.2],
            ]
        ]
    )
    target = torch.tensor(
        [
            [
                [0.0, -0.1],
                [1.0, 0.5],
                [1.0, -0.4],
            ]
        ]
    )

    actual = official_neuronio_loss(
        prediction,
        target,
        burn_in=1,
    )
    expected = 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
        prediction[:, 1:, 0],
        target[:, 1:, 0],
    ) + 0.5 * torch.nn.functional.mse_loss(
        prediction[:, 1:, 1],
        target[:, 1:, 1],
    )

    assert actual == pytest.approx(expected)


def test_official_training_loss_rejects_invalid_contract() -> None:
    prediction = torch.zeros(2, 3, 2)

    with pytest.raises(ValueError, match="matching"):
        official_neuronio_loss(prediction, torch.zeros(2, 4, 2))
    with pytest.raises(ValueError, match="burn_in"):
        official_neuronio_loss(prediction, prediction, burn_in=3)
