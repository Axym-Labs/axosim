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
    assert contract.delay_queue_bytes == 989_855_744
    assert not contract.empirical_connectome


def test_contract_rejects_shared_only_population_proxy() -> None:
    with pytest.raises(ValueError, match="custom adaptation"):
        replace(
            MILLION_NEURON_REALTIME_CONTRACT,
            every_neuron_has_custom_adaptation=False,
        )
