import torch

from axosim.model import BranchELM, BranchELMConfig
from axosim.rnn_variants import BranchTraceRNN, BranchTraceRNNConfig


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


def test_branch_trace_rnn_forward_routed_matches_forward():
    torch.manual_seed(0)
    model = BranchTraceRNN(
        BranchTraceRNNConfig(
            num_input=6,
            num_branch=3,
            num_synapse_per_branch=2,
            num_memory=4,
            trace_banks=2,
            readout_hidden_size=8,
        )
    ).eval()
    x = torch.randn(2, 5, 6)

    with torch.inference_mode():
        routed = model.route_sequence(x)
        direct = model(x)
        routed_out = model.forward_routed(routed)

    assert routed.shape == (2, 5, 6)
    assert torch.allclose(direct, routed_out)


def test_branch_trace_rnn_forward_routed_accepts_inference_tensor():
    model = BranchTraceRNN(
        BranchTraceRNNConfig(
            num_input=6,
            num_branch=3,
            num_synapse_per_branch=2,
            num_memory=4,
            trace_banks=2,
            readout_hidden_size=8,
        )
    ).eval()

    with torch.inference_mode():
        routed = model.route_sequence(torch.randn(2, 5, 6))
        output = model.forward_routed(routed)

    assert output.shape == (2, 5, 2)
