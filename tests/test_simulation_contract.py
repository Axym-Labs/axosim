from __future__ import annotations

from dataclasses import replace

import pytest

from axosim.simulation_contract import MILLION_NEURON_REALTIME_CONTRACT


def test_million_neuron_contract_preserves_defensive_adaptation() -> None:
    contract = MILLION_NEURON_REALTIME_CONTRACT

    assert contract.every_neuron_has_custom_adaptation
    assert contract.logical_adaptation_values_per_neuron == 1_808
    assert contract.adaptation_storage_bits == 8
    assert contract.adaptation_bank_bytes == 1_895_825_408


def test_million_neuron_contract_includes_network_work() -> None:
    contract = MILLION_NEURON_REALTIME_CONTRACT

    assert contract.active_sources_per_step == 10_486
    assert contract.events_per_step == 13_401_108
    assert contract.target_neuron_steps_per_second == 1_048_576_000
    assert contract.source_discovery_included
    assert contract.event_delivery_included
    assert contract.delay_queue_included
    assert contract.morphology_aware_branch_targeting
    assert contract.source_type_aware_branch_targeting
    assert contract.local_source_identity_required
    assert contract.long_range_source_identity_required
    assert contract.morphology_assignment_independent
    assert contract.activity_must_be_model_generated
    assert not contract.quota_activity_forcing_allowed
    assert contract.adaptation_reference_envelope_validation_required
    assert contract.delay_slots == 8
    assert contract.delay_queue_bytes == 1_979_711_488
    assert not contract.empirical_connectome


def test_million_neuron_contract_keeps_native_temporal_io() -> None:
    contract = MILLION_NEURON_REALTIME_CONTRACT

    assert contract.step_ms == 1.0
    assert contract.input_step_ms == 1.0
    assert contract.output_step_ms == 1.0
    assert contract.patch_size == 4
    assert contract.temporal_execution_mode == "causal_next_block_forecast"
    assert contract.lossless_ordered_temporal_io
    assert contract.causal_block_feedback_required
    assert contract.minimum_routing_delay_steps == 4


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("input_step_ms", 4.0, "inputs"),
        ("output_step_ms", 4.0, "outputs"),
        ("lossless_ordered_temporal_io", False, "every ordered"),
        ("causal_block_feedback_required", False, "causal block"),
        (
            "temporal_execution_mode",
            "mean",
            "causal next-block",
        ),
        ("minimum_routing_delay_steps", 3, "routing delays"),
        ("delay_slots", 4, "delay ring"),
    ],
)
def test_contract_rejects_coarse_temporal_io(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(MILLION_NEURON_REALTIME_CONTRACT, **{field: value})


def test_contract_rejects_shared_only_population_proxy() -> None:
    with pytest.raises(ValueError, match="custom adaptation"):
        replace(
            MILLION_NEURON_REALTIME_CONTRACT,
            every_neuron_has_custom_adaptation=False,
        )
