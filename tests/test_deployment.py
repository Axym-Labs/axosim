from __future__ import annotations

import pytest

from axosim.deployment import (
    GRU_BEHAVIOR_PERSONALIZATION,
    ONE_MILLION_PERSONALIZED_TARGET,
    PopulationDeploymentMeasurement,
)


def test_one_million_target_requires_personalization_and_routing():
    target = ONE_MILLION_PERSONALIZED_TARGET

    assert target.headline_population == 1_000_000
    assert target.benchmark_population >= target.headline_population
    assert target.personalization == GRU_BEHAVIOR_PERSONALIZATION
    assert target.personalized_parameters_per_neuron == 1_808
    assert target.personalization.parameter_counts == {
        "branch_gain": 118,
        "branch_bias": 118,
        "latent_gain": 58,
        "latent_offset": 58,
        "recurrent_output_gain": 58,
        "readout": 118,
        "branch_projection_low_rank": 352,
        "gru_input_low_rank": 464,
        "gru_recurrent_low_rank": 464,
    }
    assert target.personalization.tensor_shapes == {
        "branch_log_gain": (118,),
        "branch_bias": (118,),
        "latent_gain": (58,),
        "latent_offset": (58,),
        "recurrent_output_gain": (58,),
        "readout_weight": (2, 58),
        "readout_bias": (2,),
        "branch_projection_down": (2, 118),
        "branch_projection_up": (58, 2),
        "gru_input_down": (2, 58),
        "gru_input_up": (174, 2),
        "gru_recurrent_down": (2, 58),
        "gru_recurrent_up": (174, 2),
    }
    assert (
        target.personalization.tensor_slices["gru_recurrent_up"].stop
        == 1_808
    )
    assert target.personalization.shared_components == frozenset(
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
    assert target.firing_rate == 0.01
    assert target.fanout == 1_278


def test_one_million_target_rejects_model_only_measurement():
    measurement = PopulationDeploymentMeasurement(
        population=1_048_576,
        personalized_parameters_per_neuron=1_808,
        personalized_components=(
            "branch_gain",
            "branch_bias",
            "branch_projection_low_rank",
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
        unique_adapter_per_neuron=True,
        personalization_quality_validated=True,
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
        ONE_MILLION_PERSONALIZED_TARGET.evaluate(measurement)


def test_one_million_target_uses_complete_step_latency():
    measurement = PopulationDeploymentMeasurement(
        population=1_048_576,
        personalized_parameters_per_neuron=1_808,
        personalized_components=(
            "branch_gain",
            "branch_bias",
            "branch_projection_low_rank",
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
        unique_adapter_per_neuron=True,
        personalization_quality_validated=True,
        shared_frozen_base=True,
        output_materialized=True,
        routing_feeds_model=True,
        model_latency_ms=0.62,
        routing_latency_ms=0.35,
        complete_stack_latency_ms=0.95,
        model_peak_bytes=1_000_000_000,
        routing_peak_bytes=900_000_000,
        complete_stack_peak_bytes=1_500_000_000,
    )

    result = ONE_MILLION_PERSONALIZED_TARGET.evaluate(measurement)

    assert result["complete_step_latency_ms"] == pytest.approx(0.95)
    assert result["real_time_target_met"]
    assert result["personalized_parameter_count"] == 1_808 * 1_048_576


def test_one_million_target_rejects_inferred_complete_latency():
    measurement = PopulationDeploymentMeasurement(
        population=1_048_576,
        personalized_parameters_per_neuron=1_808,
        personalized_components=tuple(
            sorted(
                GRU_BEHAVIOR_PERSONALIZATION.personalized_components
            )
        ),
        shared_components=tuple(
            sorted(GRU_BEHAVIOR_PERSONALIZATION.shared_components)
        ),
        unique_adapter_per_neuron=True,
        personalization_quality_validated=True,
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
        ONE_MILLION_PERSONALIZED_TARGET.evaluate(measurement)


@pytest.mark.parametrize(
    ("parameter_count", "components", "unique"),
    [
        (0, (), False),
        (58, ("latent_offset",), True),
    ],
)
def test_one_million_target_rejects_underpersonalized_population(
    parameter_count,
    components,
    unique,
):
    measurement = PopulationDeploymentMeasurement(
        population=1_048_576,
        personalized_parameters_per_neuron=parameter_count,
        personalized_components=components,
        shared_components=(
            "branch_projection_base",
            "gru_base_transition",
            "normalization",
            "patch_dynamics",
        ),
        unique_adapter_per_neuron=unique,
        personalization_quality_validated=False,
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

    with pytest.raises(ValueError, match="personalized"):
        ONE_MILLION_PERSONALIZED_TARGET.evaluate(measurement)


def test_one_million_target_rejects_disconnected_routing_benchmark():
    measurement = PopulationDeploymentMeasurement(
        population=1_048_576,
        personalized_parameters_per_neuron=1_808,
        personalized_components=(
            "branch_gain",
            "branch_bias",
            "branch_projection_low_rank",
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
        unique_adapter_per_neuron=True,
        personalization_quality_validated=True,
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
        ONE_MILLION_PERSONALIZED_TARGET.evaluate(measurement)
