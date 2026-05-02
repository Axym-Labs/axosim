import torch

from bnn_sim.model import BranchELM, BranchELMConfig


def test_branch_elm_30_default_has_expected_shape_and_parameter_budget():
    config = BranchELMConfig.branch_elm_30(input_dim=12, num_branches=4)

    model = BranchELM(config)
    x = torch.zeros(3, 7, 12)
    output = model(x)

    assert output.shape == (3, 7, 2)
    assert config.memory_units == 30
    assert model.parameter_count() > 0


def test_branch_elm_forward_is_deterministic_for_same_seed_and_input():
    torch.manual_seed(123)
    model_a = BranchELM(BranchELMConfig(input_dim=8, memory_units=6, num_branches=2))
    torch.manual_seed(123)
    model_b = BranchELM(BranchELMConfig(input_dim=8, memory_units=6, num_branches=2))

    x = torch.randn(2, 5, 8)

    assert torch.allclose(model_a(x), model_b(x))
