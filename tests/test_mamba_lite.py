import torch

from bnn_sim.mamba_lite import BranchMambaLite, BranchMambaLiteConfig


def test_mamba_lite_branch_trace_preserves_output_shape():
    model = BranchMambaLite(
        BranchMambaLiteConfig(
            num_input=6,
            num_output=2,
            num_branch=3,
            num_synapse_per_branch=2,
            model_dim=8,
            num_layers=1,
            conv_kernel=3,
            long_conv_kernel=5,
            use_branch_trace=True,
            branch_trace_kernel=5,
        )
    )

    output = model(torch.zeros(2, 7, 6))

    assert output.shape == (2, 7, 2)


def test_mamba_lite_branch_trace_matches_causal_exponential_filter():
    model = BranchMambaLite(
        BranchMambaLiteConfig(
            num_input=2,
            num_output=2,
            num_branch=2,
            num_synapse_per_branch=1,
            input_to_synapse_routing=None,
            model_dim=4,
            num_layers=1,
            conv_kernel=3,
            long_conv_kernel=3,
            use_branch_trace=True,
            branch_trace_tau=1.0,
            branch_trace_kernel=4,
            branch_trace_gain_init=1.0,
            learn_branch_trace_gain=False,
        )
    )
    branch_inputs = torch.zeros(1, 5, 2)
    branch_inputs[0, 0, 0] = 1.0

    traced = model._branch_trace(branch_inputs)
    decay = torch.exp(torch.tensor(-1.0))
    expected = torch.cat([decay ** torch.arange(4, dtype=traced.dtype), torch.zeros(1)])

    assert torch.allclose(traced[0, :, 0], expected, atol=1e-6)
    assert torch.allclose(traced[0, :, 1], torch.zeros(5), atol=1e-6)


def test_mamba_lite_state_trace_path_preserves_output_shape():
    model = BranchMambaLite(
        BranchMambaLiteConfig(
            num_input=6,
            num_output=2,
            num_branch=3,
            num_synapse_per_branch=2,
            model_dim=8,
            num_layers=1,
            conv_kernel=3,
            long_conv_kernel=5,
            use_state_trace=True,
            state_trace_kernel=7,
        )
    )

    output = model(torch.zeros(2, 7, 6))

    assert output.shape == (2, 7, 2)
