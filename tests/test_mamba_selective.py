import torch

from bnn_sim.mamba_selective import BranchSelectiveMamba, BranchSelectiveMambaConfig


def test_selective_mamba_preserves_output_shape():
    model = BranchSelectiveMamba(
        BranchSelectiveMambaConfig(
            num_input=6,
            num_output=2,
            num_branch=3,
            num_synapse_per_branch=2,
            model_dim=8,
            num_layers=1,
            state_dim=4,
            conv_kernel=3,
            expansion=2,
            dt_rank=2,
        )
    )

    output = model(torch.zeros(2, 7, 6))

    assert output.shape == (2, 7, 2)


def test_selective_mamba_is_deterministic_for_same_seed_and_input():
    torch.manual_seed(123)
    model_a = BranchSelectiveMamba(
        BranchSelectiveMambaConfig(
            num_input=6,
            num_output=2,
            num_branch=3,
            num_synapse_per_branch=2,
            model_dim=8,
            num_layers=1,
            state_dim=4,
            conv_kernel=3,
            expansion=2,
            dt_rank=2,
        )
    )
    torch.manual_seed(123)
    model_b = BranchSelectiveMamba(
        BranchSelectiveMambaConfig(
            num_input=6,
            num_output=2,
            num_branch=3,
            num_synapse_per_branch=2,
            model_dim=8,
            num_layers=1,
            state_dim=4,
            conv_kernel=3,
            expansion=2,
            dt_rank=2,
        )
    )
    x = torch.randn(2, 7, 6)

    assert torch.allclose(model_a(x), model_b(x))
