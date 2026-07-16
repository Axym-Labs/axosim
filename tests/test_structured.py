import torch
from torch import nn

from axosim.axomamba import AxoMambaConfig, AxoPyTorchMamba
from axosim.structured import FactorizedLinear, initialize_factorized_linears_from_state


def test_factorized_linear_reconstructs_full_rank_dense_map():
    torch.manual_seed(91)
    dense = nn.Linear(4, 6, bias=True)
    factorized = FactorizedLinear.from_linear(
        dense,
        rank=4,
        source_weight_name="dense.weight",
    )
    x = torch.randn(3, 4)

    torch.testing.assert_close(factorized(x), dense(x), atol=1e-5, rtol=1e-5)


def test_axomamba_factorized_mamba_maps_reduce_parameters_and_train():
    common = dict(
        num_input=8,
        num_output=2,
        num_branch=2,
        num_synapse_per_branch=4,
        model_dim=8,
        num_layers=2,
        state_dim=4,
        conv_kernel=2,
        expansion=2,
    )
    torch.manual_seed(92)
    dense = AxoPyTorchMamba(AxoMambaConfig(**common))
    torch.manual_seed(92)
    factorized = AxoPyTorchMamba(
        AxoMambaConfig(**common, mamba_projection_rank=4)
    )
    dense_state = dense.state_dict()
    candidate_state = factorized.state_dict()
    candidate_state.update(
        {
            name: value
            for name, value in dense_state.items()
            if name in candidate_state and candidate_state[name].shape == value.shape
        }
    )
    factorized.load_state_dict(candidate_state)
    initialized = initialize_factorized_linears_from_state(factorized, dense_state)

    assert len(initialized) == 6
    assert sum(parameter.numel() for parameter in factorized.parameters()) < sum(
        parameter.numel() for parameter in dense.parameters()
    )
    output = factorized(torch.randn(2, 10, 8))
    output.square().mean().backward()
    factor_gradients = [
        parameter.grad
        for name, parameter in factorized.named_parameters()
        if name.endswith(("left", "right"))
    ]
    assert factor_gradients
    assert all(gradient is not None for gradient in factor_gradients)
