import torch
from torch import nn

from axosim.axomamba import AxoMambaConfig, AxoPyTorchMamba
from axosim.structured import (
    FactorizedLinear,
    FactorizedPointwiseConv1d,
    average_repeated_module_state,
    balanced_proxy_compress_mamba_state,
    initialize_factorized_linears_from_state,
    initialize_shared_tcn_adapters_from_state,
)


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


def test_factorized_pointwise_conv_reconstructs_full_rank_dense_map():
    torch.manual_seed(93)
    dense = nn.Conv1d(4, 6, kernel_size=1, bias=True)
    factorized = FactorizedPointwiseConv1d.from_conv(
        dense,
        rank=4,
        source_weight_name="pointwise.weight",
    )
    x = torch.randn(3, 4, 7)

    torch.testing.assert_close(factorized(x), dense(x), atol=1e-5, rtol=1e-5)


def test_repeated_module_transfer_averages_every_shared_tensor():
    state = {
        "blocks.0.weight": torch.tensor([1.0, 3.0]),
        "blocks.1.weight": torch.tensor([5.0, 7.0]),
        "blocks.0.bias": torch.tensor([2.0]),
        "blocks.1.bias": torch.tensor([6.0]),
        "readout.weight": torch.tensor([11.0]),
    }

    averaged = average_repeated_module_state(
        state,
        prefix_groups=[["blocks.0", "blocks.1"]],
    )

    torch.testing.assert_close(averaged["blocks.0.weight"], torch.tensor([3.0, 5.0]))
    torch.testing.assert_close(averaged["blocks.1.weight"], torch.tensor([3.0, 5.0]))
    torch.testing.assert_close(averaged["blocks.0.bias"], torch.tensor([4.0]))
    torch.testing.assert_close(averaged["blocks.1.bias"], torch.tensor([4.0]))
    torch.testing.assert_close(averaged["readout.weight"], state["readout.weight"])


def test_full_rank_tcn_adapters_reconstruct_untied_pointwise_maps():
    common = dict(
        num_input=4,
        num_output=2,
        num_branch=2,
        num_synapse_per_branch=2,
        model_dim=4,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
        local_tcn_scale=1.0,
    )
    torch.manual_seed(104)
    source = AxoPyTorchMamba(AxoMambaConfig(**common))
    torch.manual_seed(104)
    candidate = AxoPyTorchMamba(
        AxoMambaConfig(
            **common,
            share_local_tcn_pointwise=True,
            local_tcn_pointwise_adapter_rank=4,
        )
    )
    source_state = source.state_dict()
    averaged_state = average_repeated_module_state(
        source_state,
        prefix_groups=[
            [f"local_tcn_blocks.{index}.pointwise" for index in range(3)]
        ],
    )
    candidate_state = candidate.state_dict()
    candidate_state.update(
        {
            name: value
            for name, value in averaged_state.items()
            if name in candidate_state and candidate_state[name].shape == value.shape
        }
    )
    candidate.load_state_dict(candidate_state)

    initialized = initialize_shared_tcn_adapters_from_state(candidate, source_state)

    assert len(initialized) == 3
    for index, block in enumerate(candidate.local_tcn_blocks):
        reconstructed_weight = block.pointwise.weight.flatten(1) + (
            block.pointwise_adapter_out.weight.flatten(1)
            @ block.pointwise_adapter_in.weight.flatten(1)
        )
        reconstructed_bias = (
            block.pointwise.bias + block.pointwise_adapter_out.bias
        )
        torch.testing.assert_close(
            reconstructed_weight,
            source_state[f"local_tcn_blocks.{index}.pointwise.weight"].flatten(1),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            reconstructed_bias,
            source_state[f"local_tcn_blocks.{index}.pointwise.bias"],
        )


def test_axomamba_factorized_tcn_width_maps_transfer_and_train():
    common = dict(
        num_input=8,
        num_output=2,
        num_branch=2,
        num_synapse_per_branch=4,
        model_dim=8,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
        local_tcn_scale=1.0,
        share_local_tcn_pointwise=True,
        local_tcn_pointwise_adapter_rank=2,
    )
    torch.manual_seed(94)
    dense = AxoPyTorchMamba(AxoMambaConfig(**common))
    torch.manual_seed(94)
    factorized = AxoPyTorchMamba(
        AxoMambaConfig(
            **common,
            local_tcn_pointwise_rank=3,
            local_tcn_output_rank=3,
        )
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

    initialized = initialize_factorized_linears_from_state(
        factorized,
        dense_state,
    )

    assert initialized == [
        "local_tcn_blocks.0.pointwise.weight",
        "local_tcn_out.weight",
    ]
    assert factorized.local_tcn_blocks[0].pointwise is factorized.local_tcn_blocks[2].pointwise
    assert sum(parameter.numel() for parameter in factorized.parameters()) < sum(
        parameter.numel() for parameter in dense.parameters()
    )
    x = torch.randn(2, 9, 8)
    factorized(x).square().mean().backward()
    assert factorized.local_tcn_out.left.grad is not None
    assert factorized.local_tcn_blocks[0].pointwise.left.weight.grad is not None


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


def test_balanced_proxy_compression_retains_influential_mamba_states():
    source = {
        "blocks.0.mamba.A_log": torch.zeros(2, 4),
        "blocks.0.mamba.x_proj.weight": torch.tensor(
            [
                [10.0, 10.0, 10.0],  # dt row
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        "unrelated": torch.tensor([7.0]),
    }

    compressed, selections = balanced_proxy_compress_mamba_state(
        source,
        source_state_dim=4,
        target_state_dim=2,
    )

    assert selections == {"blocks.0.mamba": [2, 3]}
    assert compressed["blocks.0.mamba.A_log"].shape == (2, 2)
    torch.testing.assert_close(
        compressed["blocks.0.mamba.x_proj.weight"],
        source["blocks.0.mamba.x_proj.weight"][[0, 3, 4, 7, 8]],
    )
    assert compressed["unrelated"] is source["unrelated"]


def test_balanced_proxy_favors_longer_memory_at_equal_io_strength():
    source = {
        "blocks.0.mamba.A_log": torch.tensor([[0.0, -2.0], [0.0, -2.0]]),
        "blocks.0.mamba.x_proj.weight": torch.ones(5, 3),
    }

    _, selections = balanced_proxy_compress_mamba_state(
        source,
        source_state_dim=2,
        target_state_dim=1,
    )

    assert selections == {"blocks.0.mamba": [1]}


def test_projection_factorization_skips_maps_where_rank_is_not_compressive():
    model = AxoPyTorchMamba(
        AxoMambaConfig(
            num_input=8,
            num_output=2,
            num_branch=2,
            num_synapse_per_branch=4,
            model_dim=8,
            num_layers=2,
            state_dim=1,
            conv_kernel=2,
            expansion=2,
            mamba_projection_rank=4,
        )
    )

    factorized_names = [
        name for name, module in model.named_modules() if isinstance(module, FactorizedLinear)
    ]
    assert factorized_names == [
        "blocks.0.mamba.in_proj",
        "blocks.0.mamba.out_proj",
        "blocks.1.mamba.in_proj",
        "blocks.1.mamba.out_proj",
    ]


def test_factorized_initialization_can_skip_incompatible_state_projection():
    model = FactorizedLinear(
        4,
        3,
        2,
        bias=False,
        source_weight_name="x_proj.weight",
    )
    before = model.weight.detach().clone()

    initialized = initialize_factorized_linears_from_state(
        model,
        {"x_proj.weight": torch.ones(5, 4)},
        compatible_only=True,
    )

    assert initialized == []
    torch.testing.assert_close(model.weight, before)
