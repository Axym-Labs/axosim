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
    lossless_ordered_temporal_io: bool
    causal_block_feedback_required: bool
    minimum_routing_delay_steps: int
    logical_adaptation_values_per_neuron: int
    adaptation_storage_bits: int
    activation_storage_bits: int
    recurrent_state_storage_bits: int
    morphology_classes: int
    firing_rate: float
    fanout: int
    excitatory_fraction: float
    local_connection_fraction: float
    spatial_tile_neurons: int
    delay_slots: int
    branch_count: int
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

    @property
    def steps_per_second(self) -> float:
        return 1000.0 / self.step_ms

    @property
    def target_neuron_steps_per_second(self) -> float:
        return self.population_size * self.steps_per_second

    @property
    def active_sources_per_step(self) -> int:
        return max(1, round(self.population_size * self.firing_rate))

    @property
    def events_per_step(self) -> int:
        return self.active_sources_per_step * self.fanout

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

    @staticmethod
    def bytes_to_gib(byte_count: int) -> float:
        return byte_count / (1024**3)


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
    lossless_ordered_temporal_io=True,
    causal_block_feedback_required=True,
    minimum_routing_delay_steps=4,
    logical_adaptation_values_per_neuron=1_808,
    adaptation_storage_bits=8,
    activation_storage_bits=16,
    recurrent_state_storage_bits=16,
    morphology_classes=5,
    firing_rate=0.01,
    fanout=1_278,
    excitatory_fraction=0.8,
    local_connection_fraction=0.9,
    spatial_tile_neurons=4_096,
    delay_slots=8,
    branch_count=118,
)
