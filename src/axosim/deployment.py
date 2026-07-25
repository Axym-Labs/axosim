from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NeuronBehaviorAdaptation:
    """Mechanism-level parameters that remain unique to each neuron."""

    width: int
    branches: int
    outputs: int
    matrix_rank: int
    adapt_recurrent: bool

    @property
    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes = {
            "branch_log_gain": (self.branches,),
            "branch_bias": (self.branches,),
            "latent_gain": (self.width,),
            "latent_offset": (self.width,),
            "recurrent_output_gain": (self.width,),
            "readout_weight": (self.outputs, self.width),
            "readout_bias": (self.outputs,),
        }
        if self.matrix_rank:
            shapes.update(
                {
                    "branch_projection_down": (
                        self.matrix_rank,
                        self.branches,
                    ),
                    "branch_projection_up": (
                        self.width,
                        self.matrix_rank,
                    ),
                    "gru_input_down": (
                        self.matrix_rank,
                        self.width,
                    ),
                    "gru_input_up": (
                        3 * self.width,
                        self.matrix_rank,
                    ),
                }
            )
            if self.adapt_recurrent:
                shapes.update(
                    {
                        "gru_recurrent_down": (
                            self.matrix_rank,
                            self.width,
                        ),
                        "gru_recurrent_up": (
                            3 * self.width,
                            self.matrix_rank,
                        ),
                    }
                )
        return shapes

    @property
    def tensor_slices(self) -> dict[str, slice]:
        offset = 0
        slices = {}
        for name, shape in self.tensor_shapes.items():
            elements = 1
            for dimension in shape:
                elements *= dimension
            slices[name] = slice(offset, offset + elements)
            offset += elements
        return slices

    @property
    def parameter_counts(self) -> dict[str, int]:
        counts = {
            "branch_gain": self.branches,
            "branch_bias": self.branches,
            "latent_gain": self.width,
            "latent_offset": self.width,
            "recurrent_output_gain": self.width,
            "readout": self.outputs * (self.width + 1),
        }
        if self.matrix_rank:
            counts["branch_projection_low_rank"] = (
                self.matrix_rank * (self.branches + self.width)
            )
            counts["gru_input_low_rank"] = (
                self.matrix_rank * (4 * self.width)
            )
            if self.adapt_recurrent:
                counts["gru_recurrent_low_rank"] = (
                    self.matrix_rank * (4 * self.width)
                )
        return counts

    @property
    def parameter_count(self) -> int:
        return sum(self.parameter_counts.values())

    @property
    def adapted_components(self) -> frozenset[str]:
        return frozenset(self.parameter_counts)

    @property
    def shared_components(self) -> frozenset[str]:
        return frozenset(
            {
                "branch_projection_base",
                "gru_base_transition",
                "normalization",
                "patch_dynamics",
            }
        )


@dataclass(frozen=True)
class PopulationDeploymentMeasurement:
    """One complete population-simulation measurement."""

    population: int
    adapted_parameters_per_neuron: int
    adapted_components: tuple[str, ...]
    shared_components: tuple[str, ...]
    unique_adaptation_per_neuron: bool
    adaptation_quality_validated: bool
    shared_frozen_base: bool
    output_materialized: bool
    routing_feeds_model: bool
    model_latency_ms: float
    routing_latency_ms: float | None
    complete_stack_latency_ms: float | None
    model_peak_bytes: int
    routing_peak_bytes: int | None
    complete_stack_peak_bytes: int | None
    local_source_identity_retained: bool = False
    long_range_source_identity_retained: bool = False
    morphology_assignment_independent: bool = False
    activity_generation_mode: str = "unknown"
    activity_dynamics_validated: bool = False
    adaptation_execution_mode: str = "direct"
    adaptation_reference_envelope_validated: bool = False
    forecast_block_size: int = 1
    complete_block_latency_ms: float | None = None


@dataclass(frozen=True)
class PopulationDeploymentTarget:
    """Requirements for a defensible real-time population claim."""

    headline_population: int
    benchmark_population: int
    biological_step_ms: float
    adaptation: NeuronBehaviorAdaptation
    shared_frozen_base: bool
    output_materialization_required: bool
    network_routing_required: bool
    local_source_identity_required: bool
    long_range_source_identity_required: bool
    independent_morphology_assignment_required: bool
    activity_dynamics_validation_required: bool
    quota_activity_allowed: bool
    forecast_block_size: int
    firing_rate: float
    fanout: int

    @property
    def adapted_parameters_per_neuron(self) -> int:
        return self.adaptation.parameter_count

    def validation_errors(
        self,
        measurement: PopulationDeploymentMeasurement,
    ) -> tuple[str, ...]:
        errors: list[str] = []
        if measurement.population < self.headline_population:
            errors.append(
                "measurement population is below the headline target"
            )
        if (
            measurement.adapted_parameters_per_neuron
            != self.adapted_parameters_per_neuron
            or not measurement.unique_adaptation_per_neuron
            or frozenset(measurement.adapted_components)
            != self.adaptation.adapted_components
        ):
            errors.append(
                "measurement does not use the required adapted "
                "behavior parameters for every neuron"
            )
        if (
            frozenset(measurement.shared_components)
            != self.adaptation.shared_components
        ):
            errors.append(
                "measurement does not preserve the required shared "
                "population mechanisms"
            )
        if not measurement.adaptation_quality_validated:
            errors.append(
                "measurement adaptation has not passed the "
                "held-out quality gate"
            )
        if self.shared_frozen_base and not measurement.shared_frozen_base:
            errors.append(
                "measurement does not use a shared frozen base"
            )
        if (
            self.output_materialization_required
            and not measurement.output_materialized
        ):
            errors.append("measurement omits output materialization")
        if self.network_routing_required:
            if (
                measurement.routing_latency_ms is None
                or measurement.routing_peak_bytes is None
            ):
                errors.append(
                    "measurement omits required network routing"
                )
            if not measurement.routing_feeds_model:
                errors.append(
                    "routed events do not feed the measured model step"
                )
        if (
            self.local_source_identity_required
            and not measurement.local_source_identity_retained
        ):
            errors.append(
                "measurement does not retain local source identity"
            )
        if (
            self.long_range_source_identity_required
            and not measurement.long_range_source_identity_retained
        ):
            errors.append(
                "measurement does not retain long-range source identity"
            )
        if (
            self.independent_morphology_assignment_required
            and not measurement.morphology_assignment_independent
        ):
            errors.append(
                "measurement morphology assignment is coupled to "
                "source type or storage index"
            )
        activity_invalid = (
            self.activity_dynamics_validation_required
            and not measurement.activity_dynamics_validated
        )
        quota_invalid = (
            not self.quota_activity_allowed
            and measurement.activity_generation_mode == "exact_tile_quota"
        )
        if activity_invalid or quota_invalid:
            errors.append(
                "measurement activity is quota-forced or has not passed "
                "closed-loop dynamics validation"
            )
        if measurement.adaptation_execution_mode not in {
            "direct",
            "compiled_reference_field",
        }:
            errors.append(
                "measurement uses an unsupported adaptation execution mode"
            )
        if (
            measurement.adaptation_execution_mode
            == "compiled_reference_field"
            and not measurement.adaptation_reference_envelope_validated
        ):
            errors.append(
                "measurement adaptation reference envelope is not validated"
            )
        if measurement.complete_stack_peak_bytes is None:
            errors.append(
                "measurement omits complete-stack peak memory"
            )
        if measurement.complete_stack_latency_ms is None:
            errors.append(
                "measurement omits direct complete-stack latency"
            )
        valid_block_size = measurement.forecast_block_size > 0
        if (
            not valid_block_size
            or measurement.forecast_block_size
            != self.forecast_block_size
        ):
            errors.append(
                "measurement forecast block size does not match the target"
            )
        if measurement.complete_block_latency_ms is None:
            errors.append(
                "measurement omits direct complete-block latency"
            )
        elif (
            measurement.complete_stack_latency_ms is not None
            and valid_block_size
        ):
            normalized_block_latency = (
                measurement.complete_block_latency_ms
                / measurement.forecast_block_size
            )
            tolerance = max(
                0.005,
                0.02 * measurement.complete_stack_latency_ms,
            )
            if (
                abs(
                    normalized_block_latency
                    - measurement.complete_stack_latency_ms
                )
                > tolerance
            ):
                errors.append(
                    "measurement block latency is inconsistent with "
                    "its complete-step latency"
                )
        return tuple(errors)

    def evaluate(
        self,
        measurement: PopulationDeploymentMeasurement,
    ) -> dict[str, int | float | bool]:
        errors = self.validation_errors(measurement)
        if errors:
            raise ValueError("; ".join(errors))
        assert measurement.complete_stack_latency_ms is not None
        assert measurement.complete_stack_peak_bytes is not None
        assert measurement.complete_block_latency_ms is not None
        return {
            "population": measurement.population,
            "complete_step_latency_ms": (
                measurement.complete_stack_latency_ms
            ),
            "complete_block_latency_ms": (
                measurement.complete_block_latency_ms
            ),
            "forecast_block_size": measurement.forecast_block_size,
            "real_time_target_met": (
                measurement.complete_stack_latency_ms
                <= self.biological_step_ms
            ),
            "adapted_parameter_count": (
                measurement.population
                * measurement.adapted_parameters_per_neuron
            ),
            "events_per_step": (
                round(measurement.population * self.firing_rate)
                * self.fanout
            ),
            "complete_stack_peak_bytes": (
                measurement.complete_stack_peak_bytes
            ),
        }


GRU_BEHAVIOR_ADAPTATION = NeuronBehaviorAdaptation(
    width=58,
    branches=118,
    outputs=2,
    matrix_rank=2,
    adapt_recurrent=True,
)


ONE_MILLION_ADAPTED_TARGET = PopulationDeploymentTarget(
    headline_population=1_000_000,
    benchmark_population=1_048_576,
    biological_step_ms=1.0,
    adaptation=GRU_BEHAVIOR_ADAPTATION,
    shared_frozen_base=True,
    output_materialization_required=True,
    network_routing_required=True,
    local_source_identity_required=True,
    long_range_source_identity_required=True,
    independent_morphology_assignment_required=True,
    activity_dynamics_validation_required=True,
    quota_activity_allowed=False,
    forecast_block_size=4,
    firing_rate=0.01,
    fanout=1_278,
)
