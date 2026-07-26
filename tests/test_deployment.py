from __future__ import annotations

from dataclasses import replace

import pytest

from axosim.deployment import (
    GRU_BEHAVIOR_ADAPTATION,
    GRU_DIRECT_BRANCH_ADAPTATION,
    ONE_MILLION_ADAPTED_TARGET,
    PopulationDeploymentMeasurement,
)


def _valid_measurement() -> PopulationDeploymentMeasurement:
    return PopulationDeploymentMeasurement(
        population=1_048_576,
        adapted_parameters_per_neuron=1_278,
        adapted_components=tuple(
            sorted(GRU_BEHAVIOR_ADAPTATION.adapted_components)
        ),
        shared_components=tuple(
            sorted(GRU_BEHAVIOR_ADAPTATION.shared_components)
        ),
        unique_adaptation_per_neuron=True,
        adaptation_quality_validated=True,
        shared_frozen_base=True,
        output_materialized=True,
        routing_feeds_model=True,
        model_latency_ms=0.62,
        routing_latency_ms=0.35,
        complete_stack_latency_ms=0.95,
        model_peak_bytes=1_000_000_000,
        routing_peak_bytes=900_000_000,
        complete_stack_peak_bytes=1_500_000_000,
        local_source_identity_retained=True,
        long_range_source_identity_retained=True,
        morphology_assignment_independent=True,
        activity_generation_mode="model_threshold",
        activity_dynamics_validated=True,
        adaptation_execution_mode="direct",
        adaptation_reference_envelope_validated=True,
        forecast_block_size=4,
        complete_block_latency_ms=3.8,
    )


def test_one_million_target_requires_adaptation_and_routing():
    target = ONE_MILLION_ADAPTED_TARGET

    assert target.headline_population == 1_000_000
    assert target.benchmark_population >= target.headline_population
    assert target.adaptation == GRU_BEHAVIOR_ADAPTATION
    assert target.adapted_parameters_per_neuron == 1_278
    assert target.adaptation.parameter_counts == {
        "branch_token_offset": 58,
        "latent_gain": 58,
        "latent_offset": 58,
        "recurrent_output_gain": 58,
        "readout": 118,
        "gru_input_low_rank": 464,
        "gru_recurrent_low_rank": 464,
    }
    assert target.adaptation.tensor_shapes == {
        "branch_token_offset": (58,),
        "latent_gain": (58,),
        "latent_offset": (58,),
        "recurrent_output_gain": (58,),
        "readout_weight": (2, 58),
        "readout_bias": (2,),
        "gru_input_down": (2, 58),
        "gru_input_up": (174, 2),
        "gru_recurrent_down": (2, 58),
        "gru_recurrent_up": (174, 2),
    }
    assert (
        target.adaptation.tensor_slices["gru_recurrent_up"].stop
        == 1_278
    )
    assert target.adaptation.shared_components == frozenset(
        {
            "branch_projection_base",
            "gru_base_transition",
            "normalization",
            "patch_dynamics",
        }
    )
    assert target.shared_frozen_base
    assert target.output_materialization_required
    assert target.network_routing_required
    assert target.local_source_identity_required
    assert target.long_range_source_identity_required
    assert target.independent_morphology_assignment_required
    assert target.activity_dynamics_validation_required
    assert not target.quota_activity_allowed
    assert target.forecast_block_size == 4
    assert target.firing_rate == 0.01
    assert target.fanout == 1_278


def test_direct_branch_adaptation_remains_an_explicit_reference():
    assert GRU_DIRECT_BRANCH_ADAPTATION.parameter_count == 1_808
    assert not GRU_DIRECT_BRANCH_ADAPTATION.branch_token_offset
    assert (
        "branch_projection_low_rank"
        in GRU_DIRECT_BRANCH_ADAPTATION.adapted_components
    )


def test_one_million_target_rejects_model_only_measurement():
    measurement = PopulationDeploymentMeasurement(
        population=1_048_576,
        adapted_parameters_per_neuron=1_278,
        adapted_components=(
            "branch_token_offset",
            "gru_input_low_rank",
            "gru_recurrent_low_rank",
            "latent_gain",
            "latent_offset",
            "recurrent_output_gain",
            "readout",
        ),
        shared_components=(
            "branch_projection_base",
            "gru_base_transition",
            "normalization",
            "patch_dynamics",
        ),
        unique_adaptation_per_neuron=True,
        adaptation_quality_validated=True,
        shared_frozen_base=True,
        output_materialized=True,
        routing_feeds_model=False,
        model_latency_ms=0.7,
        routing_latency_ms=None,
        complete_stack_latency_ms=None,
        model_peak_bytes=1_000_000_000,
        routing_peak_bytes=None,
        complete_stack_peak_bytes=None,
    )

    with pytest.raises(ValueError, match="routing"):
        ONE_MILLION_ADAPTED_TARGET.evaluate(measurement)


def test_one_million_target_uses_complete_step_latency():
    measurement = _valid_measurement()

    result = ONE_MILLION_ADAPTED_TARGET.evaluate(measurement)

    assert result["complete_step_latency_ms"] == pytest.approx(0.95)
    assert result["complete_block_latency_ms"] == pytest.approx(3.8)
    assert result["forecast_block_size"] == 4
    assert result["real_time_target_met"]
    assert result["adapted_parameter_count"] == 1_278 * 1_048_576


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"local_source_identity_retained": False}, "local source identity"),
        (
            {"long_range_source_identity_retained": False},
            "long-range source identity",
        ),
        (
            {"morphology_assignment_independent": False},
            "morphology assignment",
        ),
        (
            {
                "activity_generation_mode": "exact_tile_quota",
                "activity_dynamics_validated": False,
            },
            "activity",
        ),
        (
            {
                "adaptation_execution_mode": (
                    "compiled_reference_field"
                ),
                "adaptation_reference_envelope_validated": False,
            },
            "adaptation reference envelope",
        ),
        ({"forecast_block_size": 0}, "forecast block size"),
        ({"forecast_block_size": 2}, "forecast block size"),
        ({"complete_block_latency_ms": None}, "block latency"),
        ({"complete_block_latency_ms": 3.0}, "inconsistent"),
    ],
)
def test_one_million_target_rejects_incomplete_population_fidelity(
    changes,
    message,
):
    measurement = replace(_valid_measurement(), **changes)

    with pytest.raises(ValueError, match=message):
        ONE_MILLION_ADAPTED_TARGET.evaluate(measurement)


def test_target_reports_all_provisional_headline_violations():
    measurement = replace(
        _valid_measurement(),
        long_range_source_identity_retained=False,
        activity_generation_mode="exact_tile_quota",
        activity_dynamics_validated=False,
        adaptation_execution_mode="compiled_reference_field",
        adaptation_reference_envelope_validated=False,
    )

    errors = ONE_MILLION_ADAPTED_TARGET.validation_errors(measurement)

    assert len(errors) == 3
    assert any("long-range source identity" in error for error in errors)
    assert any("activity" in error for error in errors)
    assert any("adaptation reference envelope" in error for error in errors)


def test_one_million_target_rejects_inferred_complete_latency():
    measurement = PopulationDeploymentMeasurement(
        population=1_048_576,
        adapted_parameters_per_neuron=1_278,
        adapted_components=tuple(
            sorted(
                GRU_BEHAVIOR_ADAPTATION.adapted_components
            )
        ),
        shared_components=tuple(
            sorted(GRU_BEHAVIOR_ADAPTATION.shared_components)
        ),
        unique_adaptation_per_neuron=True,
        adaptation_quality_validated=True,
        shared_frozen_base=True,
        output_materialized=True,
        routing_feeds_model=True,
        model_latency_ms=0.6,
        routing_latency_ms=0.3,
        complete_stack_latency_ms=None,
        model_peak_bytes=1_000_000_000,
        routing_peak_bytes=900_000_000,
        complete_stack_peak_bytes=1_500_000_000,
    )

    with pytest.raises(ValueError, match="complete-stack latency"):
        ONE_MILLION_ADAPTED_TARGET.evaluate(measurement)


@pytest.mark.parametrize(
    ("parameter_count", "components", "unique"),
    [
        (0, (), False),
        (58, ("latent_offset",), True),
    ],
)
def test_one_million_target_rejects_underadapted_population(
    parameter_count,
    components,
    unique,
):
    measurement = PopulationDeploymentMeasurement(
        population=1_048_576,
        adapted_parameters_per_neuron=parameter_count,
        adapted_components=components,
        shared_components=(
            "branch_projection_base",
            "gru_base_transition",
            "normalization",
            "patch_dynamics",
        ),
        unique_adaptation_per_neuron=unique,
        adaptation_quality_validated=False,
        shared_frozen_base=True,
        output_materialized=True,
        routing_feeds_model=True,
        model_latency_ms=0.4,
        routing_latency_ms=0.4,
        complete_stack_latency_ms=0.8,
        model_peak_bytes=800_000_000,
        routing_peak_bytes=800_000_000,
        complete_stack_peak_bytes=1_200_000_000,
    )

    with pytest.raises(ValueError, match="adapted"):
        ONE_MILLION_ADAPTED_TARGET.evaluate(measurement)


def test_one_million_target_rejects_disconnected_routing_benchmark():
    measurement = PopulationDeploymentMeasurement(
        population=1_048_576,
        adapted_parameters_per_neuron=1_278,
        adapted_components=(
            "branch_token_offset",
            "gru_input_low_rank",
            "gru_recurrent_low_rank",
            "latent_gain",
            "latent_offset",
            "recurrent_output_gain",
            "readout",
        ),
        shared_components=(
            "branch_projection_base",
            "gru_base_transition",
            "normalization",
            "patch_dynamics",
        ),
        unique_adaptation_per_neuron=True,
        adaptation_quality_validated=True,
        shared_frozen_base=True,
        output_materialized=True,
        routing_feeds_model=False,
        model_latency_ms=0.4,
        routing_latency_ms=0.4,
        complete_stack_latency_ms=0.8,
        model_peak_bytes=800_000_000,
        routing_peak_bytes=800_000_000,
        complete_stack_peak_bytes=1_200_000_000,
    )

    with pytest.raises(ValueError, match="feed"):
        ONE_MILLION_ADAPTED_TARGET.evaluate(measurement)
