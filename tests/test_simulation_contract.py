from __future__ import annotations

from dataclasses import replace

import pytest

from axosim.simulation_contract import (
    AXOSIM_POPULATION_ACCURATE_PROFILE,
    AXOSIM_POPULATION_THROUGHPUT_PROFILE,
    MILLION_NEURON_BEST_QUALITY_REALTIME_PROFILE,
    MILLION_NEURON_MAX_THROUGHPUT_PROFILE,
    MILLION_NEURON_REALTIME_CONTRACT,
    PopulationInferenceProfile,
)


def test_million_neuron_contract_preserves_defensive_adaptation() -> None:
    contract = MILLION_NEURON_REALTIME_CONTRACT

    assert contract.every_neuron_has_custom_adaptation
    assert contract.synaptic_efficacy_values_per_neuron == 320
    assert contract.adaptation_storage_bits == 8
    assert contract.synaptic_efficacy_bank_bytes == 335_544_320


def test_million_neuron_contract_includes_network_work() -> None:
    contract = MILLION_NEURON_REALTIME_CONTRACT

    assert contract.active_sources_per_step == 4_057
    assert contract.recurrent_contact_fraction == 0.25
    assert contract.events_per_step == 1_296_212
    assert contract.target_neuron_steps_per_second == 1_048_576_000
    assert contract.source_discovery_included
    assert contract.event_delivery_included
    assert contract.delay_queue_included
    assert contract.morphology_aware_branch_targeting
    assert contract.source_type_aware_branch_targeting
    assert contract.local_source_identity_required
    assert contract.long_range_source_identity_required
    assert contract.local_branch_input_mode == (
        "binary_channels_then_exact_weighted_branch_currents"
    )
    assert (
        contract.input_channel_collision_mode
        == "binary_or_before_branch_weighting"
    )
    assert contract.local_source_summary_is_lossless_for_declared_router
    assert not contract.posthoc_source_projection_allowed
    assert contract.morphology_assignment_independent
    assert contract.activity_must_be_model_generated
    assert not contract.quota_activity_forcing_allowed
    assert contract.adaptation_reference_envelope_validation_required
    assert contract.delay_slots == 8
    assert contract.delay_queue_bytes == 1_979_711_488
    assert contract.external_drive_storage_bits == 8
    assert contract.external_drive_block_bytes == 494_927_872
    assert not contract.empirical_connectome


def test_million_neuron_contract_keeps_native_temporal_io() -> None:
    contract = MILLION_NEURON_REALTIME_CONTRACT

    assert contract.step_ms == 1.0
    assert contract.input_step_ms == 1.0
    assert contract.output_step_ms == 1.0
    assert contract.patch_size == 4
    assert contract.temporal_execution_mode == "causal_next_block_forecast"
    assert contract.patch_phase_mode == "staggered"
    assert (
        contract.patch_phase_assignment
        == "morphology_role_stratified_random"
    )
    assert contract.patch_phase_storage_layout == "phase_tile_major"
    assert contract.largest_patch_phase_population == 262_144
    assert contract.lossless_ordered_temporal_io
    assert contract.causal_block_feedback_required
    assert contract.minimum_routing_delay_steps == 4


def test_million_neuron_inference_profiles_are_explicit() -> None:
    throughput = AXOSIM_POPULATION_THROUGHPUT_PROFILE
    quality = AXOSIM_POPULATION_ACCURATE_PROFILE

    assert MILLION_NEURON_MAX_THROUGHPUT_PROFILE is throughput
    assert MILLION_NEURON_BEST_QUALITY_REALTIME_PROFILE is quality

    assert throughput.name == "axosim-population-throughput"
    assert throughput.support_model_id == "support_t24_s8_distilled_i600"
    assert throughput.support_token_width == 24
    assert throughput.support_state_width == 8
    assert throughput.support_block_rows == 32
    assert throughput.compiled_adaptation_values_per_neuron == 88
    assert throughput.morphology_adapter_values_per_class == 88
    assert throughput.output_export_fraction == 0.001
    assert not throughput.population_adaptation_individually_fitted
    assert throughput.individual_behavior_adaptation_supported
    assert throughput.individual_synaptic_adaptation_supported
    assert throughput.primary_metric == "neuron_steps_per_second"
    assert (
        throughput.measured_p90_neuron_steps_per_second
        == pytest.approx(1_361_378_268.2786045)
    )
    assert throughput.measured_spike_mean_f1 == pytest.approx(
        0.35383167978104685
    )
    assert quality.name == "axosim-population-accurate"
    assert quality.support_model_id == "support_t58_s32_distilled_i600"
    assert quality.support_token_width == 58
    assert quality.support_state_width == 32
    assert quality.support_block_rows == 32
    assert quality.compiled_adaptation_values_per_neuron == 228
    assert quality.morphology_adapter_values_per_class == 228
    assert quality.output_export_fraction == 0.001
    assert quality.primary_metric == "voltage_sera_mv2"
    assert (
        quality.measured_voltage_sera_mv2
        < throughput.measured_voltage_sera_mv2
    )
    assert (
        quality.measured_voltage_sera_mv2
        == pytest.approx(quality.measured_voltage_root_sera_mv**2)
    )
    assert (
        throughput.measured_voltage_sera_mv2
        == pytest.approx(throughput.measured_voltage_root_sera_mv**2)
    )
    assert (
        quality.measured_p90_neuron_steps_per_second
        == pytest.approx(1_145_868_452.1765864)
    )
    assert quality.measured_spike_mean_f1 == pytest.approx(
        0.3789423022478908
    )
    assert (
        quality.minimum_neuron_steps_per_second
        == throughput.minimum_neuron_steps_per_second
        == 1_000_000_000.0
    )

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
        ("patch_phase_mode", "aligned", "staggered"),
        (
            "patch_phase_assignment",
            "index_modulo",
            "randomized and stratified",
        ),
        (
            "patch_phase_storage_layout",
            "identity",
            "phase-tile-major",
        ),
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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "local_source_summary_is_lossless_for_declared_router",
            False,
            "lossless",
        ),
        (
            "posthoc_source_projection_allowed",
            True,
            "post-hoc",
        ),
        (
            "local_branch_input_mode",
            "rank16_source_projection",
            "exact physical branch",
        ),
        (
            "input_channel_collision_mode",
            "additive_events",
            "binary OR",
        ),
    ],
)
def test_contract_rejects_approximate_local_source_paths(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(MILLION_NEURON_REALTIME_CONTRACT, **{field: value})
