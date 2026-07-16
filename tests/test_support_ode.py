import math

import torch

from axosim.axomamba import AxoMambaConfig, AxoPyTorchMamba
from axosim.support_ode import SynapticSupportODE


def _manual_dual_exponential(event: torch.Tensor, rise_ms: float, decay_ms: float) -> torch.Tensor:
    peak_time = rise_ms * decay_ms / (decay_ms - rise_ms) * math.log(decay_ms / rise_ms)
    factor = 1.0 / (-math.exp(-peak_time / rise_ms) + math.exp(-peak_time / decay_ms))
    rise_state = torch.tensor(0.0)
    decay_state = torch.tensor(0.0)
    values = []
    for value in event:
        rise_state = math.exp(-1.0 / rise_ms) * rise_state + value
        decay_state = math.exp(-1.0 / decay_ms) * decay_state + value
        values.append(factor * (decay_state - rise_state))
    return torch.stack(values)


def test_synaptic_support_ode_matches_teacher_dual_exponential_states():
    support = SynapticSupportODE(num_branches=1, mode="synaptic")
    excitatory = torch.zeros(1, 12, 1)
    inhibitory = torch.zeros(1, 12, 1)
    excitatory[0, 0, 0] = 1.0
    inhibitory[0, 0, 0] = 1.0

    output = support(excitatory, inhibitory)[0]

    torch.testing.assert_close(
        output[:, 0],
        _manual_dual_exponential(excitatory[0, :, 0], 0.3, 3.0),
    )
    torch.testing.assert_close(
        output[:, 1],
        _manual_dual_exponential(excitatory[0, :, 0], 2.0, 70.0),
    )
    torch.testing.assert_close(
        output[:, 2],
        2.5 * _manual_dual_exponential(inhibitory[0, :, 0], 0.2, 8.0),
    )


def test_support_ode_controls_preserve_shape_but_change_the_features():
    excitatory = torch.zeros(2, 16, 3)
    inhibitory = torch.zeros(2, 16, 3)
    excitatory[:, 0, :] = 1.0
    inhibitory[:, 4, :] = 1.0
    physical = SynapticSupportODE(num_branches=3, mode="synaptic")
    mismatched = SynapticSupportODE(num_branches=3, mode="generic")
    swapped = SynapticSupportODE(num_branches=3, mode="polarity_swapped")
    instantaneous = SynapticSupportODE(num_branches=3, mode="instantaneous")

    outputs = [module(excitatory, inhibitory) for module in (physical, mismatched, swapped, instantaneous)]

    assert all(output.shape == (2, 16, 9) for output in outputs)
    assert not torch.allclose(outputs[0], outputs[1])
    assert not torch.allclose(outputs[0], outputs[2])
    assert not torch.allclose(outputs[0], outputs[3])


def test_axomamba_support_ode_is_zero_initialized_and_receives_gradients():
    common = dict(
        num_input=8,
        num_output=2,
        num_branch=2,
        num_synapse_per_branch=4,
        model_dim=4,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
    )
    torch.manual_seed(81)
    control = AxoPyTorchMamba(AxoMambaConfig(**common))
    torch.manual_seed(81)
    supported = AxoPyTorchMamba(
        AxoMambaConfig(**common, support_ode_mode="synaptic", support_ode_scale=1.0)
    )
    supported_state = supported.state_dict()
    supported_state.update(
        {
            name: value
            for name, value in control.state_dict().items()
            if name in supported_state and supported_state[name].shape == value.shape
        }
    )
    supported.load_state_dict(supported_state)
    x = torch.randn(2, 9, 8)

    torch.testing.assert_close(supported(x), control(x))
    assert sum(parameter.numel() for parameter in supported.parameters()) == (
        sum(parameter.numel() for parameter in control.parameters()) + 3 * 2 * 4
    )

    supported(x).square().mean().backward()
    assert supported.support_ode_projection is not None
    gradient = supported.support_ode_projection.weight.grad
    assert gradient is not None
    assert gradient.abs().sum() > 0


def test_scale_zero_support_ode_is_an_exact_matched_control():
    common = dict(
        num_input=8,
        num_output=2,
        num_branch=2,
        num_synapse_per_branch=4,
        model_dim=4,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
        support_ode_mode="synaptic",
    )
    torch.manual_seed(82)
    active = AxoPyTorchMamba(AxoMambaConfig(**common, support_ode_scale=1.0))
    inactive = AxoPyTorchMamba(AxoMambaConfig(**common, support_ode_scale=0.0))
    control_values = dict(common)
    control_values.pop("support_ode_mode")
    control = AxoPyTorchMamba(AxoMambaConfig(**control_values))
    inactive.load_state_dict(active.state_dict())
    control_state = control.state_dict()
    control_state.update(
        {
            name: value
            for name, value in active.state_dict().items()
            if name in control_state and control_state[name].shape == value.shape
        }
    )
    control.load_state_dict(control_state)
    assert active.support_ode_projection is not None
    with torch.no_grad():
        weights = torch.linspace(
            -0.3,
            0.3,
            active.support_ode_projection.weight.numel(),
        ).view_as(active.support_ode_projection.weight)
        active.support_ode_projection.weight.copy_(weights)
        inactive.support_ode_projection.weight.copy_(weights)
    x = torch.randn(2, 9, 8)

    assert not torch.allclose(active(x), inactive(x))
    torch.testing.assert_close(inactive(x), control(x))


def test_support_ode_normalization_bounds_features_without_adding_parameters():
    common = dict(
        num_input=8,
        num_output=2,
        num_branch=2,
        num_synapse_per_branch=4,
        model_dim=4,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
        support_ode_mode="synaptic",
        support_ode_scale=1.0,
    )
    raw = AxoPyTorchMamba(AxoMambaConfig(**common, support_ode_normalization="none"))
    compressed = AxoPyTorchMamba(
        AxoMambaConfig(**common, support_ode_normalization="log1p")
    )
    routed_events = torch.ones(2, 32, 2, 4)

    raw_features = raw._support_ode_features(routed_events)
    compressed_features = compressed._support_ode_features(routed_events)

    assert raw_features is not None and compressed_features is not None
    torch.testing.assert_close(compressed_features, torch.log1p(raw_features))
    assert compressed_features.abs().max() < raw_features.abs().max()
    assert sum(parameter.numel() for parameter in raw.parameters()) == sum(
        parameter.numel() for parameter in compressed.parameters()
    )


def test_raw_support_input_is_independent_of_learned_synaptic_gains():
    config = AxoMambaConfig(
        num_input=8,
        num_output=2,
        num_branch=2,
        num_synapse_per_branch=4,
        model_dim=4,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
        synapse_gain_scale=1.0,
        support_ode_mode="synaptic",
        support_ode_scale=1.0,
        support_ode_input="raw",
    )
    model = AxoPyTorchMamba(config)
    x = torch.ones(2, 12, 8)
    routed = model._route_input_to_synapses(x)
    before = model._support_ode_features(routed)
    assert model.synapse_log_gain is not None
    with torch.no_grad():
        model.synapse_log_gain.fill_(2.0)
    weighted = model._apply_synapse_weights(routed)
    after = model._support_ode_features(routed)
    weighted_features = model._support_ode_features(weighted)

    assert before is not None and after is not None and weighted_features is not None
    torch.testing.assert_close(before, after)
    assert not torch.allclose(before, weighted_features)


def test_branch_local_support_readouts_are_exact_low_parameter_controls():
    common = dict(
        num_input=8,
        num_output=2,
        num_branch=2,
        num_synapse_per_branch=4,
        model_dim=4,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
        support_ode_mode="synaptic",
        support_ode_scale=1.0,
        support_ode_normalization="log1p",
    )
    torch.manual_seed(83)
    dense = AxoPyTorchMamba(
        AxoMambaConfig(**common, support_ode_projection="dense")
    )
    torch.manual_seed(83)
    branch = AxoPyTorchMamba(
        AxoMambaConfig(**common, support_ode_projection="branch")
    )
    torch.manual_seed(83)
    shared = AxoPyTorchMamba(
        AxoMambaConfig(**common, support_ode_projection="shared_branch")
    )
    dense_state = dense.state_dict()
    for model in (branch, shared):
        state = model.state_dict()
        state.update(
            {
                name: value
                for name, value in dense_state.items()
                if name in state and state[name].shape == value.shape
            }
        )
        model.load_state_dict(state)
    x = torch.randn(2, 12, 8)

    torch.testing.assert_close(branch(x), dense(x))
    torch.testing.assert_close(shared(x), dense(x))
    dense_parameters = sum(parameter.numel() for parameter in dense.parameters())
    branch_parameters = sum(parameter.numel() for parameter in branch.parameters())
    shared_parameters = sum(parameter.numel() for parameter in shared.parameters())
    assert dense_parameters - branch_parameters == 3 * 2 * 4 - 3 * 2
    assert branch_parameters - shared_parameters == 3 * 2 - 3

    assert branch.support_ode_branch_weights is not None
    assert shared.support_ode_branch_weights is not None
    with torch.no_grad():
        branch.support_ode_branch_weights.copy_(
            torch.tensor([[0.5, 0.25, -0.5], [0.2, 0.1, -0.1]])
        )
        shared.support_ode_branch_weights.copy_(torch.tensor([[0.5, 0.25, -0.5]]))
    assert not torch.allclose(branch(x), dense(x))
    assert not torch.allclose(shared(x), dense(x))


def test_support_route_multiplicity_is_morphology_specific_and_fixed():
    model = AxoPyTorchMamba(
        AxoMambaConfig(
            num_input=8,
            num_output=2,
            num_branch=2,
            num_synapse_per_branch=4,
            model_dim=4,
            num_layers=1,
            state_dim=2,
            conv_kernel=2,
            expansion=2,
            morphology_ids=["unused", "duplicated"],
            support_ode_mode="synaptic",
            support_ode_scale=1.0,
            support_ode_projection="shared_branch",
            support_ode_route_weighting="multiplicity",
        )
    )
    with torch.no_grad():
        model.support_ode_route_weights[0].zero_()
        model.support_ode_route_weights[1].fill_(2.0)
    x = torch.ones(2, 12, 8)
    routed = model._route_input_to_synapses(x)
    weighted = model._apply_synapse_weights(routed)
    support_input = model._support_ode_routed_input(
        x,
        routed_events=routed,
        routed_synapses=weighted,
        morphology_indices=torch.tensor([0, 1]),
    )

    assert torch.count_nonzero(support_input[0]) == 0
    torch.testing.assert_close(support_input[1], 2.0 * routed[1])
