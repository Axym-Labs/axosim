"""Authoritative contracts for large-population AxoSim benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LargePopulationSimulationContract:
    """Fully specified model and network workload for a population step."""

    name: str
    population_size: int
    step_ms: float
    input_step_ms: float
    output_step_ms: float
    target_latency_ms: float
    model_family: str
    patch_size: int
    temporal_execution_mode: str
    patch_phase_mode: str
    patch_phase_assignment: str
    patch_phase_storage_layout: str
    lossless_ordered_temporal_io: bool
    causal_block_feedback_required: bool
    minimum_routing_delay_steps: int
    logical_adaptation_values_per_neuron: int
    adaptation_storage_bits: int
    activation_storage_bits: int
    recurrent_state_storage_bits: int
    external_drive_storage_bits: int
    morphology_classes: int
    firing_rate: float
    fanout: int
    recurrent_contact_fraction: float
    excitatory_fraction: float
    local_connection_fraction: float
    spatial_tile_neurons: int
    delay_slots: int
    branch_count: int
    local_source_identity_required: bool
    long_range_source_identity_required: bool
    local_branch_input_mode: str
    input_channel_collision_mode: str
    local_source_summary_is_lossless_for_declared_router: bool
    posthoc_source_projection_allowed: bool
    morphology_assignment_independent: bool
    activity_must_be_model_generated: bool
    quota_activity_forcing_allowed: bool
    activity_selection_mode: str
    adaptation_reference_envelope_validation_required: bool
    morphology_aware_branch_targeting: bool = True
    source_type_aware_branch_targeting: bool = True
    source_discovery_included: bool = True
    event_delivery_included: bool = True
    delay_queue_included: bool = True
    every_neuron_has_custom_adaptation: bool = True
    empirical_connectome: bool = False

    def __post_init__(self) -> None:
        positive_fields = {
            "population_size": self.population_size,
            "step_ms": self.step_ms,
            "input_step_ms": self.input_step_ms,
            "output_step_ms": self.output_step_ms,
            "target_latency_ms": self.target_latency_ms,
            "patch_size": self.patch_size,
            "minimum_routing_delay_steps": (self.minimum_routing_delay_steps),
            "logical_adaptation_values_per_neuron": (
                self.logical_adaptation_values_per_neuron
            ),
            "adaptation_storage_bits": self.adaptation_storage_bits,
            "activation_storage_bits": self.activation_storage_bits,
            "recurrent_state_storage_bits": (self.recurrent_state_storage_bits),
            "external_drive_storage_bits": (
                self.external_drive_storage_bits
            ),
            "morphology_classes": self.morphology_classes,
            "fanout": self.fanout,
            "spatial_tile_neurons": self.spatial_tile_neurons,
            "delay_slots": self.delay_slots,
            "branch_count": self.branch_count,
        }
        for field_name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
        fractions = {
            "firing_rate": self.firing_rate,
            "recurrent_contact_fraction": (
                self.recurrent_contact_fraction
            ),
            "excitatory_fraction": self.excitatory_fraction,
            "local_connection_fraction": self.local_connection_fraction,
        }
        for field_name, value in fractions.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        storage_bits = {
            "adaptation_storage_bits": self.adaptation_storage_bits,
            "activation_storage_bits": self.activation_storage_bits,
            "recurrent_state_storage_bits": (self.recurrent_state_storage_bits),
            "external_drive_storage_bits": (
                self.external_drive_storage_bits
            ),
        }
        for field_name, value in storage_bits.items():
            if value % 8:
                raise ValueError(f"{field_name} must be byte aligned")
        if not self.every_neuron_has_custom_adaptation:
            raise ValueError(
                "the defensive population contract requires custom "
                "adaptation for every neuron"
            )
        if self.input_step_ms != self.step_ms:
            raise ValueError(
                "population inputs must retain the native simulation cadence"
            )
        if self.output_step_ms != self.step_ms:
            raise ValueError(
                "population outputs must retain the native simulation cadence"
            )
        if not self.lossless_ordered_temporal_io:
            raise ValueError(
                "temporal patching must preserve every ordered input and output"
            )
        if not self.causal_block_feedback_required:
            raise ValueError("connected simulation requires causal block feedback")
        if (
            self.patch_size > 1
            and self.temporal_execution_mode != "causal_next_block_forecast"
        ):
            raise ValueError(
                "multi-step patches must use causal next-block forecasting"
            )
        if self.patch_size > 1 and self.patch_phase_mode != "staggered":
            raise ValueError(
                "population patch updates must be staggered across native steps"
            )
        if self.patch_phase_assignment != "morphology_role_stratified_random":
            raise ValueError(
                "patch phases must be randomized and stratified by "
                "morphology and E/I role"
            )
        if self.patch_phase_storage_layout != "phase_tile_major":
            raise ValueError(
                "the fused population path requires phase-tile-major storage"
            )
        if self.minimum_routing_delay_steps < self.patch_size:
            raise ValueError(
                "connected block forecasts require routing delays at least "
                "as long as the forecast horizon"
            )
        if self.delay_slots < (
            self.minimum_routing_delay_steps + self.patch_size
        ):
            raise ValueError(
                "the delay ring must retain both the ready input block and "
                "the forecast block's delayed writes"
            )
        if (
            self.activity_must_be_model_generated
            and self.quota_activity_forcing_allowed
        ):
            raise ValueError(
                "a proper simulation cannot require model-generated activity "
                "while allowing quota-forced firing"
            )
        if (
            self.activity_must_be_model_generated
            and self.activity_selection_mode
            != "fixed_model_score_threshold_crossing"
        ):
            raise ValueError(
                "model-generated population activity must use fixed "
                "model-score threshold crossings"
            )
        if (
            self.local_source_identity_required
            or self.long_range_source_identity_required
        ) and not self.event_delivery_included:
            raise ValueError(
                "source identity requires event delivery in the workload"
            )
        if (
            self.local_source_identity_required
            and not self.local_source_summary_is_lossless_for_declared_router
        ):
            raise ValueError(
                "local source summaries must be lossless for the declared "
                "router"
            )
        if (
            self.local_source_identity_required
            and self.posthoc_source_projection_allowed
        ):
            raise ValueError(
                "post-hoc source projections cannot satisfy exact local "
                "source routing"
            )
        if self.local_branch_input_mode != (
            "binary_channels_then_exact_weighted_branch_currents"
        ):
            raise ValueError(
                "the million-neuron workload requires exact physical branch "
                "currents"
            )
        if self.input_channel_collision_mode != (
            "binary_or_before_branch_weighting"
        ):
            raise ValueError(
                "population input channels must preserve AxoBench binary OR "
                "semantics"
            )

    @property
    def steps_per_second(self) -> float:
        return 1000.0 / self.step_ms

    @property
    def target_neuron_steps_per_second(self) -> float:
        return self.population_size * self.steps_per_second

    @property
    def largest_patch_phase_population(self) -> int:
        return (
            self.population_size + self.patch_size - 1
        ) // self.patch_size

    @property
    def active_sources_per_step(self) -> int:
        return max(1, round(self.population_size * self.firing_rate))

    @property
    def events_per_step(self) -> int:
        return round(
            self.active_sources_per_step
            * self.fanout
            * self.recurrent_contact_fraction
        )

    @property
    def adaptation_bank_bytes(self) -> int:
        return (
            self.population_size
            * self.logical_adaptation_values_per_neuron
            * self.adaptation_storage_bits
            // 8
        )

    @property
    def delay_queue_bytes(self) -> int:
        return (
            self.delay_slots
            * self.population_size
            * self.branch_count
            * self.activation_storage_bits
            // 8
        )

    @property
    def external_drive_block_bytes(self) -> int:
        return (
            self.patch_size
            * self.population_size
            * self.branch_count
            * self.external_drive_storage_bits
            // 8
        )

    @staticmethod
    def bytes_to_gib(byte_count: int) -> float:
        return byte_count / (1024**3)


@dataclass(frozen=True)
class PopulationInferenceProfile:
    """Named operating point within the million-neuron contract."""

    name: str
    support_state_width: int
    support_block_rows: int
    correction_fraction: float
    correction_selection: str
    primary_metric: str
    minimum_neuron_steps_per_second: float

    def __post_init__(self) -> None:
        if self.support_state_width <= 0:
            raise ValueError("support state width must be positive")
        if self.support_block_rows not in {16, 32}:
            raise ValueError("support block rows must be 16 or 32")
        if not 0.0 <= self.correction_fraction <= 0.05:
            raise ValueError("correction fraction must be in [0, 0.05]")
        if self.minimum_neuron_steps_per_second <= 0:
            raise ValueError("minimum throughput must be positive")
        correction_disabled = self.correction_selection == "disabled"
        if correction_disabled != (self.correction_fraction == 0.0):
            raise ValueError(
                "disabled correction must use a zero correction fraction"
            )

    @property
    def correction_enabled(self) -> bool:
        return self.correction_fraction > 0.0


MILLION_NEURON_REALTIME_CONTRACT = LargePopulationSimulationContract(
    name="million-neuron-realtime-rtx5090",
    population_size=1_048_576,
    step_ms=1.0,
    input_step_ms=1.0,
    output_step_ms=1.0,
    target_latency_ms=1.0,
    model_family="adapted-causal-block-gru-regression",
    patch_size=4,
    temporal_execution_mode="causal_next_block_forecast",
    patch_phase_mode="staggered",
    patch_phase_assignment="morphology_role_stratified_random",
    patch_phase_storage_layout="phase_tile_major",
    lossless_ordered_temporal_io=True,
    causal_block_feedback_required=True,
    minimum_routing_delay_steps=4,
    logical_adaptation_values_per_neuron=1_278,
    adaptation_storage_bits=8,
    activation_storage_bits=16,
    recurrent_state_storage_bits=16,
    external_drive_storage_bits=8,
    morphology_classes=5,
    firing_rate=0.003869090909090909,
    fanout=1_278,
    recurrent_contact_fraction=0.25,
    excitatory_fraction=0.8,
    local_connection_fraction=0.9,
    spatial_tile_neurons=4_096,
    delay_slots=8,
    branch_count=118,
    local_source_identity_required=True,
    long_range_source_identity_required=True,
    local_branch_input_mode=(
        "binary_channels_then_exact_weighted_branch_currents"
    ),
    input_channel_collision_mode="binary_or_before_branch_weighting",
    local_source_summary_is_lossless_for_declared_router=True,
    posthoc_source_projection_allowed=False,
    morphology_assignment_independent=True,
    activity_must_be_model_generated=True,
    quota_activity_forcing_allowed=False,
    activity_selection_mode="fixed_model_score_threshold_crossing",
    adaptation_reference_envelope_validation_required=True,
)

MILLION_NEURON_MAX_THROUGHPUT_PROFILE = PopulationInferenceProfile(
    name="million-neuron-max-throughput",
    support_state_width=32,
    support_block_rows=32,
    correction_fraction=0.0,
    correction_selection="disabled",
    primary_metric="neuron_steps_per_second",
    minimum_neuron_steps_per_second=1_000_000_000.0,
)

MILLION_NEURON_BEST_QUALITY_REALTIME_PROFILE = PopulationInferenceProfile(
    name="million-neuron-best-quality-realtime",
    support_state_width=32,
    support_block_rows=32,
    correction_fraction=0.0175,
    correction_selection="internal_state_f1_exact_fraction",
    primary_metric="spike_f1_area_0_5ms",
    minimum_neuron_steps_per_second=1_000_000_000.0,
)
