import torch

from axosim.axomamba import AxoMamba
from axosim.block_forecast import CausalBlockForecastModel
from axosim.interfaces import (
    AxoSimGRU,
    AxoSimLite,
    AxoSimMamba,
    AxoSimPopulation,
)
from axosim.support_surrogate import (
    AdaptiveSupportP4Surrogate,
    SupportP4Config,
)


def _population() -> AxoSimPopulation:
    torch.manual_seed(7)
    config = SupportP4Config(
        input_dim=6,
        route_feature_dim=3,
        token_dim=4,
        state_dim=3,
        morphology_ids=("m0", "m1"),
        behavior_adaptation=False,
    )
    neuron = AdaptiveSupportP4Surrogate(
        config,
        torch.randn(2, 6, 3),
    )
    return AxoSimPopulation(
        neuron,
        morphology_indices=torch.tensor([0, 0, 1]),
        contact_branch_indices=torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [2, 3, 4, 5],
            ]
        ),
    )


def test_population_initializes_pretrained_morphology_adapters() -> None:
    torch.manual_seed(11)
    config = SupportP4Config(
        input_dim=6,
        route_feature_dim=3,
        token_dim=4,
        state_dim=3,
        morphology_ids=("m0", "m1"),
        behavior_adaptation=True,
    )
    neuron = AdaptiveSupportP4Surrogate(
        config,
        torch.randn(2, 6, 3),
    )
    assert neuron.behavior_adaptation_bank is not None
    expected = neuron.compile_adaptation(
        neuron.behavior_adaptation_bank
    ).detach()

    population = AxoSimPopulation(
        neuron,
        morphology_indices=torch.tensor([0, 0, 1]),
        contact_branch_indices=torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [2, 3, 4, 5],
            ]
        ),
    )

    torch.testing.assert_close(population.morphology_adapter, expected)
    torch.testing.assert_close(
        population.behavior_adapter,
        torch.zeros_like(population.behavior_adapter),
    )


def test_public_model_names_resolve_to_their_architectures() -> None:
    assert AxoSimMamba is AxoMamba
    assert AxoSimGRU is CausalBlockForecastModel
    assert AxoSimLite is AdaptiveSupportP4Surrogate


def test_population_adaptation_mode_freezes_shared_neuron_weights() -> None:
    population = _population()

    population.enable_adaptation_training()

    adapted = set(population.adaptation_parameters())
    assert adapted == {
        population.behavior_adapter,
        population.morphology_adapter,
        population.synaptic_log_efficacy,
    }
    assert all(parameter.requires_grad for parameter in adapted)
    assert all(
        not parameter.requires_grad
        for parameter in population.neuron.parameters()
    )


def test_population_can_train_one_adaptation_mechanism() -> None:
    population = _population()

    population.enable_adaptation_training(
        behavior=True,
        morphology=False,
        synaptic=False,
    )

    assert list(population.trainable_adaptation_parameters()) == [
        population.behavior_adapter
    ]
    assert population.behavior_adapter.requires_grad
    assert not population.morphology_adapter.requires_grad
    assert not population.synaptic_log_efficacy.requires_grad


def test_population_forward_preserves_full_temporal_gradients() -> None:
    population = _population()
    population.enable_adaptation_training()
    contacts = torch.randn(3, 12, 4, requires_grad=True)

    prediction = population(contacts)
    prediction[:, -1].sum().backward()

    assert prediction.shape == (3, 12, 2)
    assert contacts.grad is not None
    assert contacts.grad[:, :4].abs().sum() > 0
    assert population.behavior_adapter.grad is not None
    assert population.behavior_adapter.grad.abs().sum() > 0
    assert population.morphology_adapter.grad is not None
    assert population.morphology_adapter.grad.abs().sum() > 0
    assert population.synaptic_log_efficacy.grad is not None
    assert population.synaptic_log_efficacy.grad.abs().sum() > 0


def test_population_contact_path_matches_dense_branch_reference() -> None:
    population = _population()
    contacts = torch.randn(3, 12, 4)
    branch_inputs = population._branch_inputs(contacts)
    summaries = population.neuron.aggregate_inputs(
        branch_inputs,
        morphology_indices=population.morphology_indices,
    )
    expected = population.neuron.forward_runtime_coefficients(
        summaries,
        runtime_coefficients=population._runtime_coefficients(),
    )

    actual = population(contacts)

    torch.testing.assert_close(actual, expected)


def test_population_token_path_preserves_full_temporal_gradients() -> None:
    population = _population()
    population.enable_adaptation_training()
    tokens = torch.randn(3, 4, 4, requires_grad=True)

    prediction = population.forward_tokens(tokens)
    prediction[:, -1].sum().backward()

    assert prediction.shape == (3, 16, 2)
    assert tokens.grad is not None
    assert tokens.grad[:, 0].abs().sum() > 0
    assert population.behavior_adapter.grad is not None
    assert population.behavior_adapter.grad.abs().sum() > 0


def test_population_batch_path_matches_independent_examples() -> None:
    population = _population()
    contacts = torch.randn(2, 3, 12, 4)

    batched = population.forward_batch(contacts)
    independent = torch.stack(
        [population(example) for example in contacts],
        dim=0,
    )

    assert batched.shape == (2, 3, 12, 2)
    torch.testing.assert_close(batched, independent)


def test_population_batched_tokens_preserve_full_bptt() -> None:
    population = _population()
    population.enable_adaptation_training()
    tokens = torch.randn(2, 3, 4, 4, requires_grad=True)

    prediction = population.forward_token_batch(tokens)
    prediction[:, :, -1].sum().backward()

    assert prediction.shape == (2, 3, 16, 2)
    assert tokens.grad is not None
    assert tokens.grad[:, :, 0].abs().sum() > 0
    assert population.behavior_adapter.grad is not None
    assert population.behavior_adapter.grad.abs().sum() > 0


def test_behavior_adapter_changes_only_the_selected_neuron() -> None:
    population = _population()
    contacts = torch.randn(3, 12, 4)
    baseline = population(contacts)

    with torch.no_grad():
        population.behavior_adapter[0, 0] = 0.5
    changed = population(contacts)

    assert not torch.equal(changed[0], baseline[0])
    torch.testing.assert_close(changed[1:], baseline[1:])


def test_synaptic_efficacies_are_positive_and_contact_specific() -> None:
    population = _population()
    contacts = torch.zeros(3, 12, 4)
    contacts[:, :, 0] = 1.0
    baseline = population(contacts)

    with torch.no_grad():
        population.synaptic_log_efficacy[0, 0] = torch.log(
            torch.tensor(2.0)
        )
    changed = population(contacts)

    assert torch.all(population.synaptic_efficacies() > 0)
    assert not torch.equal(changed[0], baseline[0])
    torch.testing.assert_close(changed[1:], baseline[1:])


def test_population_accepts_differentiable_external_synaptic_values() -> None:
    population = _population()
    contacts = torch.randn(3, 12, 4)
    external = torch.zeros(3, 4, requires_grad=True)

    prediction = population(
        contacts,
        synaptic_log_efficacy=external,
    )
    prediction.square().mean().backward()

    assert external.grad is not None
    assert external.grad.abs().sum() > 0
    assert population.synaptic_log_efficacy.grad is None
