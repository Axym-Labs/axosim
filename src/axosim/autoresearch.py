from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from axosim.checkpoint import save_checkpoint
from axosim.data import ShardedNeuronIODataset
from axosim.evaluate import _tensor_from_numpy
from axosim.train import (
    _build_optimizer,
    _build_scheduler,
    _sera_soma_relevance,
)

STANDARD_PRESENTATIONS = 120_000
LONG_PRESENTATIONS = 300_000
AXOMAMBA_DEFAULT_BURN_IN = 50
PRACTICAL_SERA_DELTA = 0.1
PRACTICAL_F1_DELTA = 0.02
PRACTICAL_EFFICIENCY_FRACTION = 0.10
POPULATION_MEMORY_REFERENCE_NEURONS = 1_000_000


def model_resident_state_metrics(model: nn.Module) -> dict[str, int]:
    """Count unique parameters and buffers that reside with a loaded model."""
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    parameter_count = sum(tensor.numel() for tensor in parameters)
    parameter_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in parameters
    )
    buffer_count = sum(tensor.numel() for tensor in buffers)
    buffer_bytes = sum(tensor.numel() * tensor.element_size() for tensor in buffers)
    return {
        "resident_parameter_count": parameter_count,
        "resident_parameter_bytes": parameter_bytes,
        "resident_buffer_count": buffer_count,
        "resident_buffer_bytes": buffer_bytes,
        "resident_state_bytes": parameter_bytes + buffer_bytes,
    }


def model_population_memory_metrics(
    model: nn.Module,
    *,
    streaming_dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Separate shared model memory from state that scales with neuron count."""
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    trainable_parameters = [
        parameter for parameter in parameters if parameter.requires_grad
    ]
    frozen_parameters = [
        parameter for parameter in parameters if not parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    frozen_count = sum(parameter.numel() for parameter in frozen_parameters)
    trainable_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in trainable_parameters
    )
    frozen_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in frozen_parameters
    )
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in buffers)
    state_element_bytes = torch.empty((), dtype=streaming_dtype).element_size()
    components_elements: dict[str, int] = defaultdict(int)

    state_model = getattr(model, "backbone", model)
    temporal_core = getattr(model, "temporal_core", None)
    recurrent_state_elements = getattr(
        temporal_core,
        "recurrent_state_elements",
        0,
    )
    if recurrent_state_elements:
        components_elements["temporal_core"] += int(recurrent_state_elements)

    for block in getattr(state_model, "blocks", ()):
        mamba = getattr(block, "mamba", block)
        d_inner = getattr(mamba, "d_inner", None)
        d_conv = getattr(mamba, "d_conv", None)
        d_state = getattr(mamba, "d_state", None)
        if all(value is not None for value in (d_inner, d_conv, d_state)):
            components_elements["mamba"] += int(d_inner) * (int(d_conv) + int(d_state))

    for block in getattr(state_model, "local_tcn_blocks", ()):
        depthwise = getattr(block, "depthwise", None)
        if isinstance(depthwise, nn.Conv1d):
            components_elements["local_tcn"] += _conv1d_streaming_history_elements(
                depthwise
            )
    temporal_refine = getattr(state_model, "temporal_refine", None)
    temporal_conv = getattr(temporal_refine, "conv", None)
    if isinstance(temporal_conv, nn.Conv1d):
        components_elements["temporal_refine"] += _conv1d_streaming_history_elements(
            temporal_conv
        )
    for name in ("integrative_pre", "integrative_post"):
        mixer = getattr(state_model, name, None)
        if mixer is None:
            continue
        for conv_name in ("short_conv", "long_conv"):
            conv = getattr(mixer, conv_name, None)
            if isinstance(conv, nn.Conv1d):
                components_elements["integrative_mixer"] += (
                    _conv1d_streaming_history_elements(conv)
                )

    branch_trace_taus = tuple(getattr(state_model, "branch_trace_taus", ()))
    if branch_trace_taus:
        config = state_model.config
        components_elements["branch_traces"] += (
            len(branch_trace_taus)
            * int(config.num_branch)
            * (int(config.branch_trace_kernel) - 1)
        )
    hidden_trace_taus = tuple(getattr(state_model, "hidden_trace_taus", ()))
    if hidden_trace_taus:
        config = state_model.config
        components_elements["hidden_traces"] += (
            len(hidden_trace_taus)
            * int(config.model_dim)
            * (int(config.hidden_trace_kernel) - 1)
        )

    component_bytes = {
        name: elements * state_element_bytes
        for name, elements in sorted(components_elements.items())
        if elements
    }
    streaming_state_bytes = sum(component_bytes.values())
    shared_fixed_state_bytes = frozen_bytes + buffer_bytes
    per_neuron_simulation_state_bytes = trainable_bytes + streaming_state_bytes
    per_neuron_adamw_training_state_bytes = 4 * trainable_bytes + streaming_state_bytes
    return {
        "trainable_parameter_count": trainable_count,
        "trainable_parameter_bytes": trainable_bytes,
        "frozen_parameter_count": frozen_count,
        "frozen_parameter_bytes": frozen_bytes,
        "shared_buffer_bytes": buffer_bytes,
        "shared_fixed_state_bytes": shared_fixed_state_bytes,
        "streaming_state_dtype": str(streaming_dtype).removeprefix("torch."),
        "streaming_state_components_bytes": component_bytes,
        "per_neuron_streaming_state_bytes": streaming_state_bytes,
        "per_neuron_trainable_plus_streaming_bytes": (
            trainable_bytes + streaming_state_bytes
        ),
        "per_neuron_simulation_state_bytes": per_neuron_simulation_state_bytes,
        "per_neuron_adamw_training_state_bytes": (
            per_neuron_adamw_training_state_bytes
        ),
        "population_memory_reference_neurons": POPULATION_MEMORY_REFERENCE_NEURONS,
        "population_simulation_state_bytes_at_reference_scale": (
            shared_fixed_state_bytes
            + POPULATION_MEMORY_REFERENCE_NEURONS * per_neuron_simulation_state_bytes
        ),
        "population_adamw_training_state_bytes_at_reference_scale": (
            shared_fixed_state_bytes
            + POPULATION_MEMORY_REFERENCE_NEURONS
            * per_neuron_adamw_training_state_bytes
        ),
        "estimated_adamw_training_state_bytes": (
            4 * trainable_bytes + shared_fixed_state_bytes
        ),
    }


def _conv1d_streaming_history_elements(conv: nn.Conv1d) -> int:
    kernel = int(conv.kernel_size[0])
    dilation = int(conv.dilation[0])
    return int(conv.in_channels) * (kernel - 1) * dilation


def practical_candidate_gate(
    deltas: dict[str, float],
    *,
    assessments: dict[str, str],
    baseline_training: dict[str, Any],
    candidate_training: dict[str, Any],
    metric_families: Iterable[str] = ("sera", "f1"),
) -> dict[str, Any]:
    """Require statistical and practical significance on at least one axis."""
    active_families = frozenset(str(family) for family in metric_families)
    unsupported_families = active_families - {"sera", "f1"}
    if not active_families or unsupported_families:
        raise ValueError("metric_families must contain one or both of 'sera' and 'f1'")
    sera_metrics = {
        "voltage_root_sera_mv",
        "dynamics_root_sera_mv_per_ms",
        "response_root_sera_mv",
    }
    f1_metrics = {
        "spike_f1_area_0_5ms",
        "spike_f1_exact",
        "spike_f1_5ms",
    }
    material_improvements = []
    material_regressions = []
    for metric, delta in deltas.items():
        if metric in sera_metrics and "sera" in active_families:
            if delta <= -PRACTICAL_SERA_DELTA and assessments.get(metric) == "improved":
                material_improvements.append(metric)
            elif (
                delta >= PRACTICAL_SERA_DELTA and assessments.get(metric) == "regressed"
            ):
                material_regressions.append(metric)
        elif metric in f1_metrics and "f1" in active_families:
            if delta >= PRACTICAL_F1_DELTA and assessments.get(metric) == "improved":
                material_improvements.append(metric)
            elif (
                delta <= -PRACTICAL_F1_DELTA and assessments.get(metric) == "regressed"
            ):
                material_regressions.append(metric)

    efficiency_improvements: dict[str, float] = {}
    baseline_parameters = float(
        baseline_training.get(
            "trainable_parameter_count",
            baseline_training.get(
                "resident_parameter_count",
                baseline_training.get("parameter_count", 0),
            ),
        )
    )
    candidate_parameters = float(
        candidate_training.get(
            "trainable_parameter_count",
            candidate_training.get(
                "resident_parameter_count",
                candidate_training.get("parameter_count", 0),
            ),
        )
    )
    if baseline_parameters > 0 and candidate_parameters > 0:
        parameter_reduction = (
            baseline_parameters - candidate_parameters
        ) / baseline_parameters
        if parameter_reduction >= PRACTICAL_EFFICIENCY_FRACTION:
            efficiency_improvements["trainable_parameter_reduction_fraction"] = (
                parameter_reduction
            )
    baseline_state_bytes = float(baseline_training.get("resident_state_bytes", 0))
    candidate_state_bytes = float(candidate_training.get("resident_state_bytes", 0))
    if baseline_state_bytes > 0 and candidate_state_bytes > 0:
        state_memory_reduction = (
            baseline_state_bytes - candidate_state_bytes
        ) / baseline_state_bytes
        if state_memory_reduction >= PRACTICAL_EFFICIENCY_FRACTION:
            efficiency_improvements["resident_state_memory_reduction_fraction"] = (
                state_memory_reduction
            )
    baseline_streaming_bytes = float(
        baseline_training.get("per_neuron_streaming_state_bytes", 0)
    )
    candidate_streaming_bytes = float(
        candidate_training.get("per_neuron_streaming_state_bytes", 0)
    )
    if baseline_streaming_bytes > 0 and candidate_streaming_bytes > 0:
        streaming_state_reduction = (
            baseline_streaming_bytes - candidate_streaming_bytes
        ) / baseline_streaming_bytes
        if streaming_state_reduction >= PRACTICAL_EFFICIENCY_FRACTION:
            efficiency_improvements["per_neuron_streaming_state_reduction_fraction"] = (
                streaming_state_reduction
            )
    baseline_rate = float(
        baseline_training.get("training_step_presentations_per_second", 0)
    )
    candidate_rate = float(
        candidate_training.get("training_step_presentations_per_second", 0)
    )
    if baseline_rate > 0 and candidate_rate > 0:
        compute_speedup = candidate_rate / baseline_rate - 1.0
        baseline_latency_p10 = float(
            baseline_training.get("training_step_latency_p10_ms", 0)
        )
        candidate_latency_p90 = float(
            candidate_training.get("training_step_latency_p90_ms", float("inf"))
        )
        robust_latency_separation = (
            baseline_latency_p10 > 0 and candidate_latency_p90 < baseline_latency_p10
        )
        if (
            compute_speedup >= PRACTICAL_EFFICIENCY_FRACTION
            and robust_latency_separation
        ):
            efficiency_improvements["training_compute_speedup_fraction"] = (
                compute_speedup
            )
    baseline_inference = float(
        baseline_training.get("inference_presentations_per_second", 0)
    )
    candidate_inference = float(
        candidate_training.get("inference_presentations_per_second", 0)
    )
    if baseline_inference > 0 and candidate_inference > 0:
        inference_speedup = candidate_inference / baseline_inference - 1.0
        baseline_latency_p10 = float(
            baseline_training.get("inference_latency_p10_ms", 0)
        )
        candidate_latency_p90 = float(
            candidate_training.get("inference_latency_p90_ms", float("inf"))
        )
        robust_latency_separation = (
            baseline_latency_p10 > 0 and candidate_latency_p90 < baseline_latency_p10
        )
        if (
            inference_speedup >= PRACTICAL_EFFICIENCY_FRACTION
            and robust_latency_separation
        ):
            efficiency_improvements["inference_speedup_fraction"] = inference_speedup
    baseline_population_memory = float(
        baseline_training.get("population_peak_allocated_bytes", 0)
    )
    candidate_population_memory = float(
        candidate_training.get("population_peak_allocated_bytes", 0)
    )
    if baseline_population_memory > 0 and candidate_population_memory > 0:
        population_memory_reduction = (
            baseline_population_memory - candidate_population_memory
        ) / baseline_population_memory
        if population_memory_reduction >= PRACTICAL_EFFICIENCY_FRACTION:
            efficiency_improvements["population_peak_memory_reduction_fraction"] = (
                population_memory_reduction
            )
    baseline_population_rate = float(
        baseline_training.get("population_neuron_steps_per_second", 0)
    )
    candidate_population_rate = float(
        candidate_training.get("population_neuron_steps_per_second", 0)
    )
    if baseline_population_rate > 0 and candidate_population_rate > 0:
        population_speedup = candidate_population_rate / baseline_population_rate - 1.0
        baseline_latency_p10 = float(
            baseline_training.get("population_step_latency_p10_ms", 0)
        )
        candidate_latency_p90 = float(
            candidate_training.get("population_step_latency_p90_ms", float("inf"))
        )
        robust_latency_separation = (
            baseline_latency_p10 > 0 and candidate_latency_p90 < baseline_latency_p10
        )
        if (
            population_speedup >= PRACTICAL_EFFICIENCY_FRACTION
            and robust_latency_separation
        ):
            efficiency_improvements["population_inference_speedup_fraction"] = (
                population_speedup
            )
    baseline_population_capacity = float(
        baseline_training.get("maximum_population_size", 0)
    )
    candidate_population_capacity = float(
        candidate_training.get("maximum_population_size", 0)
    )
    if baseline_population_capacity > 0 and candidate_population_capacity > 0:
        population_capacity_increase = (
            candidate_population_capacity / baseline_population_capacity - 1.0
        )
        if population_capacity_increase >= PRACTICAL_EFFICIENCY_FRACTION:
            efficiency_improvements["population_capacity_increase_fraction"] = (
                population_capacity_increase
            )

    automatically_rejected = not material_improvements and not efficiency_improvements
    if automatically_rejected:
        decision = "reject_tiny_metric_only"
    elif material_improvements and material_regressions:
        decision = "review_material_tradeoff"
    elif material_improvements:
        decision = "review_material_gain"
    elif material_regressions:
        decision = "review_efficiency_tradeoff"
    else:
        decision = "review_efficiency_gain"
    return {
        "decision": decision,
        "automatically_rejected": automatically_rejected,
        "thresholds": {
            "sera_absolute": PRACTICAL_SERA_DELTA,
            "f1_absolute": PRACTICAL_F1_DELTA,
            "efficiency_fraction": PRACTICAL_EFFICIENCY_FRACTION,
        },
        "metric_families": sorted(active_families),
        "material_improvements": material_improvements,
        "material_regressions": material_regressions,
        "efficiency_improvements": efficiency_improvements,
    }


@dataclass(frozen=True)
class TrainingObjective:
    spike_loss_weight: float = 0.25
    soma_loss_weight: float = 0.75
    direct_sera_loss_weight: float = 0.0
    dynamics_sera_loss_weight: float = 0.0
    f1_curve_loss_weight: float = 0.0
    one_to_one_f1_loss_weight: float = 0.0
    near_spike_voltage_loss_weight: float = 0.0
    near_spike_radius_steps: int | None = None
    raw_soma_target_mix: float = 0.0
    base_soma_loss_weight: float = 0.0
    base_soma_sera_loss_weight: float = 0.0
    base_soma_target_mix: float = 1.0
    sera_threshold_count: int = 21
    f1_tolerances_steps: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    soma_scale: float = 0.1

    def __post_init__(self) -> None:
        weights = (
            self.spike_loss_weight,
            self.soma_loss_weight,
            self.direct_sera_loss_weight,
            self.dynamics_sera_loss_weight,
            self.f1_curve_loss_weight,
            self.one_to_one_f1_loss_weight,
            self.near_spike_voltage_loss_weight,
            self.base_soma_loss_weight,
            self.base_soma_sera_loss_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("training objective weights must be non-negative")
        if self.sera_threshold_count < 2:
            raise ValueError("sera_threshold_count must be at least two")
        if not self.f1_tolerances_steps or min(self.f1_tolerances_steps) < 0:
            raise ValueError("f1_tolerances_steps must be non-empty and non-negative")
        if (
            self.near_spike_radius_steps is not None
            and self.near_spike_radius_steps < 0
        ):
            raise ValueError("near_spike_radius_steps must be non-negative")
        if self.near_spike_voltage_loss_weight and self.near_spike_radius_steps is None:
            raise ValueError(
                "near_spike_radius_steps is required when its loss is enabled"
            )
        if not 0.0 <= self.raw_soma_target_mix <= 1.0:
            raise ValueError("raw_soma_target_mix must be in [0, 1]")
        if not 0.0 <= self.base_soma_target_mix <= 1.0:
            raise ValueError("base_soma_target_mix must be in [0, 1]")
        if self.soma_scale <= 0:
            raise ValueError("soma_scale must be positive")


def default_axomamba_training_objective() -> TrainingObjective:
    """Return the replicated SERA/F1 recipe retained by the model loop."""
    return TrainingObjective(
        spike_loss_weight=0.25,
        soma_loss_weight=0.75,
        direct_sera_loss_weight=0.01,
        f1_curve_loss_weight=0.05,
        near_spike_voltage_loss_weight=0.002,
        near_spike_radius_steps=5,
        raw_soma_target_mix=1.0,
        sera_threshold_count=101,
        f1_tolerances_steps=(0, 1, 2, 3, 4, 5),
    )


def regression_axomamba_training_objective() -> TrainingObjective:
    """Return the replicated direct-SERA objective for voltage regression."""
    return TrainingObjective(
        spike_loss_weight=0.0,
        soma_loss_weight=0.0,
        direct_sera_loss_weight=0.01,
        f1_curve_loss_weight=0.0,
        near_spike_voltage_loss_weight=0.0,
        near_spike_radius_steps=5,
        raw_soma_target_mix=1.0,
        sera_threshold_count=101,
        f1_tolerances_steps=(0, 1, 2, 3, 4, 5),
    )


def spike_axomamba_training_objective() -> TrainingObjective:
    """Exploratory F1-primary recipe; not a practical-gate model lead."""
    return TrainingObjective(
        spike_loss_weight=0.25,
        soma_loss_weight=0.1,
        direct_sera_loss_weight=0.001,
        f1_curve_loss_weight=0.05,
        near_spike_voltage_loss_weight=0.0,
        near_spike_radius_steps=5,
        raw_soma_target_mix=1.0,
        sera_threshold_count=101,
        f1_tolerances_steps=(0, 1, 2, 3, 4, 5),
    )


def axomamba_training_objective(task: str) -> TrainingObjective:
    """Return the fixed dual, regression, or spike training recipe."""
    factories = {
        "dual": default_axomamba_training_objective,
        "regression": regression_axomamba_training_objective,
        "spike": spike_axomamba_training_objective,
    }
    try:
        return factories[task]()
    except KeyError as error:
        raise ValueError(f"unsupported AxoMamba task: {task}") from error


@dataclass
class PanelCandidate:
    candidate_id: str
    model: nn.Module
    batch_size: int
    checkpoint_path: str | Path
    metrics_path: str | Path
    learning_rate: float = 3e-4
    optimizer_name: str = "adamw"
    weight_decay: float = 1e-4
    lr_schedule: str = "constant"
    grad_clip_norm: float = 0.0
    objective: TrainingObjective | None = None
    burn_in: int | None = None
    output_distillation_weight: float = 0.0

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if self.batch_size <= 0:
            raise ValueError("candidate batch_size must be positive")
        if self.burn_in is not None and self.burn_in < 0:
            raise ValueError("candidate burn_in must be non-negative")
        if self.output_distillation_weight < 0:
            raise ValueError("output_distillation_weight must be non-negative")


class FrozenWindowDataset:
    """A fixed window schedule over full-trace shards."""

    def __init__(self, base: ShardedNeuronIODataset, schedule_path: str | Path) -> None:
        self.base = base
        self.schedule_path = Path(schedule_path)
        with np.load(self.schedule_path, allow_pickle=False) as payload:
            self.metadata = json.loads(str(payload["metadata_json"].item()))
            self._shard_indices = np.asarray(payload["shard_indices"], dtype=np.int32)
            self._sample_indices = np.asarray(payload["sample_indices"], dtype=np.int32)
            self._starts = np.asarray(payload["starts"], dtype=np.int32)
        if self.metadata.get("version") != 1:
            raise ValueError("unsupported frozen window schedule version")
        if self.metadata["dataset_fingerprint"] != _dataset_fingerprint(base):
            raise ValueError("frozen window schedule does not match dataset")
        if not (
            len(self._shard_indices) == len(self._sample_indices) == len(self._starts)
        ):
            raise ValueError("frozen window schedule arrays have mismatched lengths")
        self.window_size = int(self.metadata["window_size"])
        self._sample_ids = {
            (shard_idx, sample_idx): sample_id
            for shard_idx, sample_idx, sample_id in base._base_index
        }

    def __len__(self) -> int:
        return len(self._starts)

    def reshuffle(self, seed: int) -> None:
        del seed

    def get_batch(self, indices: range | list[int]) -> tuple[np.ndarray, np.ndarray]:
        selected = [int(index) for index in indices]
        entries = [
            (
                int(self._shard_indices[index]),
                int(self._sample_indices[index]),
                int(self._starts[index]),
            )
            for index in selected
        ]
        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for shard_idx, shard_entries in _group_schedule_entries(entries):
            shard_inputs, shard_targets = self.base._load_cached_shard(shard_idx)
            for sample_idx, start in shard_entries:
                stop = start + self.window_size
                inputs.append(
                    np.asarray(shard_inputs[sample_idx, start:stop], dtype=np.int8)
                )
                targets.append(
                    np.asarray(shard_targets[sample_idx, start:stop], dtype=np.float32)
                )
        return np.stack(inputs), np.stack(targets)

    def get_shared_batch(
        self, indices: range | list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        selected = [int(index) for index in indices]
        entries = [
            (
                int(self._shard_indices[index]),
                int(self._sample_indices[index]),
                int(self._starts[index]),
            )
            for index in selected
        ]
        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        raw_soma: list[np.ndarray] = []
        for shard_idx, shard_entries in _group_schedule_entries(entries):
            shard_inputs, shard_targets = self.base._load_cached_shard(shard_idx)
            shard_raw_soma = self.base._load_cached_optional_array(shard_idx, "soma_mv")
            if shard_raw_soma is None:
                raise ValueError("raw soma training requires soma_mv in every shard")
            for sample_idx, start in shard_entries:
                stop = start + self.window_size
                inputs.append(
                    np.asarray(shard_inputs[sample_idx, start:stop], dtype=np.int8)
                )
                targets.append(
                    np.asarray(shard_targets[sample_idx, start:stop], dtype=np.float32)
                )
                raw_soma.append(
                    np.asarray(shard_raw_soma[sample_idx, start:stop], dtype=np.float32)
                )
        return np.stack(inputs), np.stack(targets), np.stack(raw_soma)

    def get_sample_ids(self, indices: range | list[int]) -> list[str]:
        values = []
        for index in indices:
            position = int(index)
            shard_idx = int(self._shard_indices[position])
            sample_idx = int(self._sample_indices[position])
            start = int(self._starts[position])
            values.append(f"{self._sample_ids[(shard_idx, sample_idx)]}@{start}")
        return values


def initialize_named_parameters(
    model: nn.Module, values: dict[str, float]
) -> list[str]:
    """Fill selected parameters after common-state initialization."""
    parameters = dict(model.named_parameters())
    unknown = sorted(set(values) - set(parameters))
    if unknown:
        raise ValueError(
            f"unknown model parameters in initialization override: {unknown}"
        )
    with torch.no_grad():
        for name, value in values.items():
            parameters[name].fill_(float(value))
    return sorted(values)


def offset_named_parameters(model: nn.Module, offsets: dict[str, float]) -> list[str]:
    """Offset selected parameters after common-state initialization."""
    parameters = dict(model.named_parameters())
    unknown = sorted(set(offsets) - set(parameters))
    if unknown:
        raise ValueError(
            f"unknown model parameters in initialization offset: {unknown}"
        )
    with torch.no_grad():
        for name, offset in offsets.items():
            parameters[name].add_(float(offset))
    return sorted(offsets)


def freeze_named_parameters(model: nn.Module, names: Iterable[str]) -> list[str]:
    """Freeze an exact set of parameters for an attributable panel control."""
    requested = set(names)
    parameters = dict(model.named_parameters())
    unknown = sorted(requested - set(parameters))
    if unknown:
        raise ValueError(f"unknown model parameters in freeze request: {unknown}")
    for name in requested:
        parameters[name].requires_grad_(False)
    return sorted(requested)


def set_trainable_parameter_names(model: nn.Module, names: Iterable[str]) -> list[str]:
    """Make an exact parameter subset trainable and freeze every other tensor."""
    requested = set(names)
    parameters = dict(model.named_parameters())
    unknown = sorted(requested - set(parameters))
    if unknown:
        raise ValueError(f"unknown model parameters in trainable request: {unknown}")
    if not requested:
        raise ValueError("trainable parameter request must not be empty")
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in requested)
    return sorted(set(parameters) - requested)


def initialize_morphology_parameter_rows_from_mean(
    model: nn.Module,
    *,
    target_morphology_id: str,
    source_morphology_ids: Iterable[str],
    parameter_names: Iterable[str],
) -> dict[str, Any]:
    """Initialize held-out morphology rows from the mean trained source row."""
    configured = tuple(
        str(name)
        for name in (
            getattr(getattr(model, "config", None), "morphology_ids", None) or ()
        )
    )
    if not configured:
        raise ValueError("mean morphology initialization requires morphology_ids")
    mapping = {name: index for index, name in enumerate(configured)}
    target = str(target_morphology_id)
    sources = tuple(str(name) for name in source_morphology_ids)
    unknown = sorted(({target} | set(sources)) - set(mapping))
    if unknown:
        raise ValueError(f"unknown morphology IDs in mean initialization: {unknown}")
    if not sources:
        raise ValueError("source_morphology_ids must not be empty")
    if target in sources:
        raise ValueError("target morphology must not be a source morphology")
    if len(set(sources)) != len(sources):
        raise ValueError("source_morphology_ids must be unique")

    parameters = dict(model.named_parameters())
    names = tuple(str(name) for name in parameter_names)
    unknown_parameters = sorted(set(names) - set(parameters))
    if unknown_parameters:
        raise ValueError(
            f"unknown model parameters in mean initialization: {unknown_parameters}"
        )
    if not names:
        raise ValueError("mean initialization parameter_names must not be empty")
    target_index = mapping[target]
    source_indices = [mapping[name] for name in sources]
    with torch.no_grad():
        for name in names:
            parameter = parameters[name]
            if parameter.ndim < 1 or parameter.shape[0] != len(configured):
                raise ValueError(
                    f"mean initialization parameter {name!r} must have one row per morphology"
                )
            parameter[target_index].copy_(parameter[source_indices].mean(dim=0))
    return {
        "target_morphology_id": target,
        "target_index": target_index,
        "source_morphology_ids": list(sources),
        "source_indices": source_indices,
        "parameter_names": sorted(names),
    }


def load_named_buffer_artifact(
    model: nn.Module,
    *,
    buffer_name: str,
    artifact_path: str | Path,
    key: str,
    expected_morphology_ids: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Load one fixed NPZ tensor into an existing persistent model buffer."""
    path = Path(artifact_path).resolve()
    with np.load(path, allow_pickle=False) as artifact:
        if key not in artifact.files:
            raise ValueError(f"buffer artifact {path} has no key {key!r}")
        values = np.asarray(artifact[key])
        if expected_morphology_ids is not None:
            if "morphology_ids" not in artifact.files:
                raise ValueError(f"buffer artifact {path} has no morphology_ids")
            observed_ids = tuple(
                str(value) for value in artifact["morphology_ids"].tolist()
            )
            expected_ids = tuple(str(value) for value in expected_morphology_ids)
            if observed_ids != expected_ids:
                raise ValueError(
                    "buffer artifact morphology_ids do not match model config: "
                    f"{observed_ids} != {expected_ids}"
                )

    buffers = dict(model.named_buffers())
    if buffer_name not in buffers:
        raise ValueError(f"unknown model buffer in artifact override: {buffer_name!r}")
    target = buffers[buffer_name]
    if tuple(values.shape) != tuple(target.shape):
        raise ValueError(
            f"buffer artifact shape mismatch for {buffer_name}: "
            f"{tuple(values.shape)} != {tuple(target.shape)}"
        )
    with torch.no_grad():
        target.copy_(torch.as_tensor(values, dtype=target.dtype, device=target.device))
    return {
        "buffer_name": buffer_name,
        "path": str(path),
        "key": key,
        "shape": list(values.shape),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def create_frozen_window_schedule(
    base: ShardedNeuronIODataset,
    output_path: str | Path,
    *,
    presentations: int,
    schedule_batch_size: int,
    shard_reuse_batches: int,
    window_size: int = 500,
    start_offset: int = 500,
    sequence_length: int | None = 6000,
    seed: int = 0,
    morphology_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Create a deterministic schedule balanced across morphology groups."""
    if presentations <= 0 or schedule_batch_size <= 0 or shard_reuse_batches <= 0:
        raise ValueError("presentation, batch, and shard reuse counts must be positive")
    by_morphology: dict[str, dict[int, list[tuple[int, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for shard_idx, sample_idx, sample_id in base._base_index:
        by_morphology[_morphology_id(sample_id)][shard_idx].append(
            (sample_idx, sample_id)
        )
    if not by_morphology:
        raise ValueError("dataset contains no samples")
    if morphology_ids is not None:
        requested = tuple(str(name) for name in morphology_ids)
        if not requested:
            raise ValueError("morphology_ids must not be empty")
        if len(set(requested)) != len(requested):
            raise ValueError("morphology_ids must be unique")
        unknown = sorted(set(requested) - set(by_morphology))
        if unknown:
            raise ValueError(f"unknown morphology IDs in schedule request: {unknown}")
        by_morphology = {name: by_morphology[name] for name in requested}

    rng = np.random.default_rng(seed)
    morphology_names = sorted(by_morphology)
    group_size = schedule_batch_size * shard_reuse_batches
    shard_indices: list[int] = []
    sample_indices: list[int] = []
    starts: list[int] = []
    morphology_counts = {name: 0 for name in morphology_names}
    quota_base, quota_extra = divmod(presentations, len(morphology_names))
    morphology_quotas = {
        name: quota_base + int(position < quota_extra)
        for position, name in enumerate(morphology_names)
    }
    sequence_lengths: dict[int, int] = {}
    group_index = 0
    order = list(morphology_names)
    while len(starts) < presentations:
        if group_index % len(order) == 0:
            order = list(morphology_names)
            rng.shuffle(order)
        morphology = order[group_index % len(order)]
        if morphology_counts[morphology] >= morphology_quotas[morphology]:
            group_index += 1
            continue
        available_shards = sorted(by_morphology[morphology])
        shard_idx = int(rng.choice(available_shards))
        shard_samples = by_morphology[morphology][shard_idx]
        if shard_idx not in sequence_lengths:
            sequence_lengths[shard_idx] = base.get_shard_sequence_length(shard_idx)
        actual_length = sequence_lengths[shard_idx]
        usable_length = (
            min(actual_length, sequence_length)
            if sequence_length is not None
            else actual_length
        )
        max_start = usable_length - window_size
        if start_offset > max_start:
            raise ValueError(
                f"start_offset={start_offset} leaves no {window_size}-step windows in shard {shard_idx}"
            )
        count = min(
            group_size,
            presentations - len(starts),
            morphology_quotas[morphology] - morphology_counts[morphology],
        )
        chosen = rng.choice(
            len(shard_samples), size=count, replace=len(shard_samples) < count
        )
        chosen_starts = rng.integers(start_offset, max_start + 1, size=count)
        for chosen_index, start in zip(chosen, chosen_starts, strict=True):
            sample_idx, _ = shard_samples[int(chosen_index)]
            shard_indices.append(shard_idx)
            sample_indices.append(sample_idx)
            starts.append(int(start))
        morphology_counts[morphology] += count
        group_index += 1

    metadata: dict[str, Any] = {
        "version": 1,
        "dataset_fingerprint": _dataset_fingerprint(base),
        "presentations": presentations,
        "schedule_batch_size": schedule_batch_size,
        "shard_reuse_batches": shard_reuse_batches,
        "window_size": window_size,
        "start_offset": start_offset,
        "sequence_length": sequence_length,
        "seed": seed,
        "morphology_ids": morphology_names,
        "morphology_counts": morphology_counts,
    }
    output = Path(output_path)
    if output.suffix.lower() != ".npz":
        raise ValueError("frozen window schedule path must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        shard_indices=np.asarray(shard_indices, dtype=np.int32),
        sample_indices=np.asarray(sample_indices, dtype=np.int32),
        starts=np.asarray(starts, dtype=np.int32),
    )
    return metadata


def train_candidate_panel(
    candidates: Iterable[PanelCandidate],
    dataset: Any,
    *,
    objective: TrainingObjective | None = None,
    device: str = "cuda",
    burn_in: int = AXOMAMBA_DEFAULT_BURN_IN,
    seed: int = 0,
    prefetch_batches: int = 1,
    pin_memory: bool = False,
    non_blocking: bool = True,
    warmup_cuda: bool = True,
    distillation_teacher: nn.Module | None = None,
    resume_path: str | Path | None = None,
    checkpoint_every_super_batches: int = 0,
    snapshot_every_super_batches: int = 0,
    snapshot_root: str | Path | None = None,
    max_super_batches: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Train several candidates from one loader and one transferred super-batch."""
    candidates = list(candidates)
    if not candidates:
        raise ValueError("candidate panel must not be empty")
    if (
        burn_in < 0
        or prefetch_batches < 0
        or checkpoint_every_super_batches < 0
        or snapshot_every_super_batches < 0
    ):
        raise ValueError(
            "burn_in, prefetch_batches, and checkpoint interval must be non-negative"
        )
    if max_super_batches is not None and max_super_batches <= 0:
        raise ValueError("max_super_batches must be positive when provided")
    if checkpoint_every_super_batches and resume_path is None:
        raise ValueError("checkpoint_every_super_batches requires resume_path")
    if snapshot_every_super_batches and snapshot_root is None:
        raise ValueError("snapshot_every_super_batches requires snapshot_root")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate_id values must be unique")
    torch.manual_seed(seed)
    default_objective = objective or default_axomamba_training_objective()
    super_batch_size = max(candidate.batch_size for candidate in candidates)
    states: dict[str, dict[str, Any]] = {}
    if distillation_teacher is None and any(
        candidate.output_distillation_weight for candidate in candidates
    ):
        raise ValueError("output distillation requires a teacher")
    teacher = (
        distillation_teacher.to(device).eval()
        if distillation_teacher is not None
        else None
    )
    for candidate in candidates:
        model = candidate.model.to(device).train()
        optimizer = _build_optimizer(
            model,
            candidate.optimizer_name,
            candidate.learning_rate,
            candidate.weight_decay,
        )
        total_steps = math.ceil(len(dataset) / candidate.batch_size)
        scheduler = _build_scheduler(optimizer, candidate.lr_schedule, total_steps)
        states[candidate.candidate_id] = {
            "candidate": candidate,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "objective": candidate.objective or default_objective,
            "burn_in": burn_in if candidate.burn_in is None else candidate.burn_in,
            "steps": 0,
            "presentations": 0,
            "last_components": {},
            "loss_sum": 0.0,
            "loss_observations": 0,
            "gradient_norm_sum": 0.0,
            "gradient_norm_observations": 0,
            "preclip_gradient_norm_max": 0.0,
            "gradient_clipped_steps": 0,
            "compute_timing": [],
            "compute_seconds": 0.0,
        }
    total_shared_batches = math.ceil(len(dataset) / super_batch_size)
    resolved_resume = Path(resume_path) if resume_path is not None else None
    resolved_snapshot_root = Path(snapshot_root) if snapshot_root is not None else None
    start_shared_batch = 0
    cumulative_panel_seconds = 0.0
    cumulative_teacher_seconds = 0.0
    if resolved_resume is not None and resolved_resume.exists():
        payload = torch.load(
            resolved_resume,
            map_location=device,
            weights_only=False,
        )
        if payload.get("version") != 1:
            raise ValueError("unsupported panel resume version")
        if payload.get("candidate_ids") != list(states):
            raise ValueError("panel resume candidate IDs do not match")
        if int(payload.get("total_shared_batches", -1)) != total_shared_batches:
            raise ValueError("panel resume dataset length does not match")
        start_shared_batch = int(payload["next_shared_batch"])
        cumulative_panel_seconds = float(payload.get("cumulative_panel_seconds", 0.0))
        cumulative_teacher_seconds = float(
            payload.get("cumulative_teacher_seconds", 0.0)
        )
        for candidate_id, state in states.items():
            saved = payload["states"][candidate_id]
            state["model"].load_state_dict(saved["model"])
            state["optimizer"].load_state_dict(saved["optimizer"])
            if state["scheduler"] is not None:
                if saved["scheduler"] is None:
                    raise ValueError("panel resume is missing scheduler state")
                state["scheduler"].load_state_dict(saved["scheduler"])
            state["steps"] = int(saved["steps"])
            state["presentations"] = int(saved["presentations"])
            state["last_components"] = dict(saved["last_components"])
            state["loss_sum"] = float(saved["loss_sum"])
            state["loss_observations"] = int(
                saved.get("loss_observations", saved["steps"])
            )
            state["gradient_norm_sum"] = float(
                saved.get("gradient_norm_sum", 0.0)
            )
            state["gradient_norm_observations"] = int(
                saved.get("gradient_norm_observations", 0)
            )
            state["preclip_gradient_norm_max"] = float(
                saved.get("preclip_gradient_norm_max", 0.0)
            )
            state["gradient_clipped_steps"] = int(
                saved.get("gradient_clipped_steps", 0)
            )
            state["compute_seconds"] = float(saved.get("compute_seconds", 0.0))
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in payload["cuda_rng_state_all"]]
            )
    warmup_seconds = 0.0
    if (
        start_shared_batch == 0
        and warmup_cuda
        and device.startswith("cuda")
        and torch.cuda.is_available()
    ):
        warmup_started = time.perf_counter()
        window_size = int(getattr(dataset, "window_size", 500))
        for state in states.values():
            candidate = state["candidate"]
            model = state["model"]
            input_dim = int(getattr(model, "num_input", getattr(model, "input_dim", 0)))
            if input_dim <= 0:
                raise ValueError(
                    "CUDA warmup requires model.num_input or model.input_dim"
                )
            x = torch.zeros(candidate.batch_size, window_size, input_dim, device=device)
            y = torch.zeros(candidate.batch_size, window_size, 2, device=device)
            warmup_names = (
                [str(model.config.morphology_ids[0])] * candidate.batch_size
                if getattr(getattr(model, "config", None), "morphology_ids", None)
                else None
            )
            morphology_indices = _morphology_index_tensor(
                model, warmup_names, device=device
            )
            prediction = _forward_model(model, x, morphology_indices)
            loss, _ = training_objective_loss(
                prediction,
                y,
                objective=state["objective"],
                burn_in=state["burn_in"],
                raw_soma_mv=torch.full(
                    (candidate.batch_size, window_size), -67.7, device=device
                ),
                base_soma_prediction=getattr(model, "base_soma_prediction", None),
            )
            loss.backward()
            state["optimizer"].zero_grad(set_to_none=True)
            del x, y, prediction, loss
        _synchronize(device)
        warmup_seconds = time.perf_counter() - warmup_started
    _synchronize(device)
    started_at = time.perf_counter()
    context_required = any(
        getattr(getattr(state["model"], "config", None), "morphology_ids", None)
        for state in states.values()
    ) or bool(getattr(getattr(teacher, "config", None), "morphology_ids", None))
    teacher_compute_timing: list[Any] = []
    next_shared_batch = start_shared_batch
    processed_this_call = 0
    for processed_this_call, (
        batch_index,
        batch_indices,
        batch_arrays,
    ) in enumerate(
        _iter_shared_batches(
            dataset,
            super_batch_size,
            prefetch_batches=prefetch_batches,
            start_batch=start_shared_batch,
        ),
        start=1,
    ):
        if len(batch_arrays) == 2:
            x_np, y_np = batch_arrays
            raw_soma_np = None
        elif len(batch_arrays) == 3:
            x_np, y_np, raw_soma_np = batch_arrays
        else:
            raise ValueError(
                "shared batch must contain inputs, targets, and optional raw soma"
            )
        x_super = _tensor_from_numpy(
            x_np, device, pin_memory=pin_memory, non_blocking=non_blocking
        ).to(dtype=torch.float32)
        y_super = _tensor_from_numpy(
            y_np, device, pin_memory=pin_memory, non_blocking=non_blocking
        )
        raw_soma_super = (
            _tensor_from_numpy(
                raw_soma_np,
                device,
                pin_memory=pin_memory,
                non_blocking=non_blocking,
            )
            if raw_soma_np is not None
            else None
        )
        sample_ids = dataset.get_sample_ids(batch_indices) if context_required else None
        teacher_prediction = None
        if teacher is not None:
            teacher_morphology = _morphology_index_tensor(
                teacher, sample_ids, device=device
            )
            if device.startswith("cuda") and torch.cuda.is_available():
                teacher_start = torch.cuda.Event(enable_timing=True)
                teacher_stop = torch.cuda.Event(enable_timing=True)
                teacher_start.record()
            else:
                teacher_start = time.perf_counter()
                teacher_stop = None
            with torch.no_grad():
                teacher_prediction = _forward_model(
                    teacher,
                    x_super,
                    teacher_morphology,
                )
            if teacher_stop is not None:
                teacher_stop.record()
                teacher_compute_timing.append((teacher_start, teacher_stop))
            else:
                teacher_compute_timing.append(time.perf_counter() - teacher_start)
        for state in states.values():
            candidate = state["candidate"]
            model = state["model"]
            optimizer = state["optimizer"]
            scheduler = state["scheduler"]
            if device.startswith("cuda") and torch.cuda.is_available():
                timing_start = torch.cuda.Event(enable_timing=True)
                timing_end = torch.cuda.Event(enable_timing=True)
                timing_start.record()
            else:
                timing_start = time.perf_counter()
                timing_end = None
            morphology_super = _morphology_index_tensor(
                model, sample_ids, device=device
            )
            for start in range(0, x_super.shape[0], candidate.batch_size):
                x = x_super[start : start + candidate.batch_size]
                y = y_super[start : start + candidate.batch_size]
                raw_soma = (
                    raw_soma_super[start : start + candidate.batch_size]
                    if raw_soma_super is not None
                    else None
                )
                morphology_indices = (
                    morphology_super[start : start + candidate.batch_size]
                    if morphology_super is not None
                    else None
                )
                optimizer.zero_grad(set_to_none=True)
                prediction = _forward_model(model, x, morphology_indices)
                loss, components = training_objective_loss(
                    prediction,
                    y,
                    objective=state["objective"],
                    burn_in=state["burn_in"],
                    raw_soma_mv=raw_soma,
                    base_soma_prediction=getattr(model, "base_soma_prediction", None),
                )
                if candidate.output_distillation_weight:
                    assert teacher_prediction is not None
                    distillation = output_distillation_loss(
                        prediction,
                        teacher_prediction[start : start + candidate.batch_size],
                        burn_in=state["burn_in"],
                    )
                    loss = loss + candidate.output_distillation_weight * distillation
                    components["output_distillation"] = distillation
                    components["total"] = loss
                if not bool(torch.isfinite(loss).detach().cpu()):
                    raise FloatingPointError(
                        "non-finite training loss for "
                        f"{candidate.candidate_id} before optimizer step "
                        f"{state['steps'] + 1}"
                    )
                loss.backward()
                if candidate.grad_clip_norm:
                    gradient_norm = nn.utils.clip_grad_norm_(
                        model.parameters(),
                        candidate.grad_clip_norm,
                        error_if_nonfinite=True,
                    )
                    gradient_norm_value = float(gradient_norm.detach().cpu())
                    state["gradient_norm_sum"] += gradient_norm_value
                    state["gradient_norm_observations"] += 1
                    state["preclip_gradient_norm_max"] = max(
                        state["preclip_gradient_norm_max"],
                        gradient_norm_value,
                    )
                    if gradient_norm_value > candidate.grad_clip_norm:
                        state["gradient_clipped_steps"] += 1
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                state["steps"] += 1
                state["presentations"] += int(x.shape[0])
                loss_value = float(loss.detach().cpu())
                state["loss_sum"] += loss_value
                state["loss_observations"] += 1
                state["last_components"] = {
                    name: float(value.detach().cpu())
                    for name, value in components.items()
                }
            if timing_end is not None:
                timing_end.record()
                state["compute_timing"].append((timing_start, timing_end))
            else:
                state["compute_timing"].append(time.perf_counter() - timing_start)
        next_shared_batch = batch_index + 1
        should_checkpoint = (
            resolved_resume is not None
            and checkpoint_every_super_batches > 0
            and (
                next_shared_batch % checkpoint_every_super_batches == 0
                or next_shared_batch == total_shared_batches
            )
        )
        should_stop = (
            max_super_batches is not None and processed_this_call >= max_super_batches
        )
        should_snapshot = (
            resolved_snapshot_root is not None
            and snapshot_every_super_batches > 0
            and (
                next_shared_batch % snapshot_every_super_batches == 0
                or next_shared_batch == total_shared_batches
            )
        )
        if should_checkpoint or should_stop:
            cumulative_teacher_seconds += _flush_panel_timings(
                states,
                teacher_compute_timing,
                device=device,
            )
            _synchronize(device)
            elapsed_segment = time.perf_counter() - started_at
            if resolved_resume is not None:
                _write_panel_resume(
                    resolved_resume,
                    states,
                    next_shared_batch=next_shared_batch,
                    total_shared_batches=total_shared_batches,
                    cumulative_panel_seconds=(
                        cumulative_panel_seconds + elapsed_segment
                    ),
                    cumulative_teacher_seconds=cumulative_teacher_seconds,
                )
        if should_snapshot:
            _write_panel_snapshot(
                resolved_snapshot_root,
                states,
                shared_batch=next_shared_batch,
                total_shared_batches=total_shared_batches,
            )
        if should_stop:
            break
    _synchronize(device)
    cumulative_teacher_seconds += _flush_panel_timings(
        states,
        teacher_compute_timing,
        device=device,
    )
    panel_seconds = cumulative_panel_seconds + time.perf_counter() - started_at
    training_complete = next_shared_batch == total_shared_batches
    if resolved_resume is not None:
        _write_panel_resume(
            resolved_resume,
            states,
            next_shared_batch=next_shared_batch,
            total_shared_batches=total_shared_batches,
            cumulative_panel_seconds=panel_seconds,
            cumulative_teacher_seconds=cumulative_teacher_seconds,
        )

    results: dict[str, dict[str, Any]] = {}
    for candidate_id, state in states.items():
        candidate = state["candidate"]
        model = state["model"]
        compute_seconds = state["compute_seconds"]
        resident_state = model_resident_state_metrics(model)
        population_memory = model_population_memory_metrics(model)
        metrics = {
            "candidate_id": candidate_id,
            "train_presentations": state["presentations"],
            "optimizer_steps": state["steps"],
            "batch_size": candidate.batch_size,
            "learning_rate": candidate.learning_rate,
            "optimizer": candidate.optimizer_name,
            "weight_decay": candidate.weight_decay,
            "lr_schedule": candidate.lr_schedule,
            "burn_in": state["burn_in"],
            "output_distillation_weight": candidate.output_distillation_weight,
            "seed": seed,
            "device": device,
            "parameter_count": resident_state["resident_parameter_count"],
            **resident_state,
            **population_memory,
            "mean_training_loss": (
                state["loss_sum"] / max(state["loss_observations"], 1)
            ),
            "loss_observations": state["loss_observations"],
            "gradient_norm_observations": state[
                "gradient_norm_observations"
            ],
            "preclip_gradient_norm_mean": (
                state["gradient_norm_sum"]
                / state["gradient_norm_observations"]
                if state["gradient_norm_observations"]
                else None
            ),
            "preclip_gradient_norm_max": (
                state["preclip_gradient_norm_max"]
                if state["gradient_norm_observations"]
                else None
            ),
            "gradient_clipped_steps": state["gradient_clipped_steps"],
            "last_components": state["last_components"],
            "panel_candidates": [item.candidate_id for item in candidates],
            "shared_super_batch_size": super_batch_size,
            "prefetch_batches": prefetch_batches,
            "panel_wall_seconds": panel_seconds,
            "warmup_seconds": warmup_seconds,
            "shared_teacher_compute_seconds": cumulative_teacher_seconds,
            "candidate_compute_seconds": compute_seconds,
            "candidate_compute_presentations_per_second": (
                state["presentations"] / compute_seconds
                if compute_seconds > 0
                else float("inf")
            ),
            "panel_presentations_per_second": (
                len(dataset) * len(candidates) / panel_seconds
                if panel_seconds > 0
                else float("inf")
            ),
            "objective": asdict(state["objective"]),
            "morphology_conditioning": getattr(
                getattr(model, "config", None), "morphology_ids", None
            ),
            "training_complete": training_complete,
            "processed_shared_batches": next_shared_batch,
            "total_shared_batches": total_shared_batches,
            "resume_path": (
                str(resolved_resume) if resolved_resume is not None else None
            ),
        }
        checkpoint_path = Path(candidate.checkpoint_path)
        metrics_path = Path(candidate.metrics_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model, checkpoint_path, metadata=metrics)
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        results[candidate_id] = metrics
    return results


def output_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    burn_in: int,
) -> torch.Tensor:
    """Match a mature teacher's event probabilities and normalized voltage."""
    if student.shape != teacher.shape or student.ndim != 3 or student.shape[-1] != 2:
        raise ValueError("student and teacher outputs must have shape (batch, time, 2)")
    student_trimmed = student[:, burn_in:]
    teacher_trimmed = teacher[:, burn_in:].detach()
    spike_loss = F.binary_cross_entropy_with_logits(
        student_trimmed[..., 0],
        teacher_trimmed[..., 0].sigmoid(),
    )
    soma_loss = F.mse_loss(
        student_trimmed[..., 1],
        teacher_trimmed[..., 1],
    )
    return 0.25 * spike_loss + 0.75 * soma_loss


def benchmark_model_inference(
    model: nn.Module,
    *,
    device: str,
    batch_size: int = 32,
    window_size: int = 500,
    warmup_repeats: int = 20,
    repeats: int = 100,
) -> dict[str, float | int]:
    """Measure repeated fixed-shape inference latency for practical gating."""
    if batch_size <= 0 or window_size <= 0 or warmup_repeats < 0 or repeats <= 0:
        raise ValueError("inference benchmark sizes and repeats must be positive")
    model = model.to(device).eval()
    input_dim = int(getattr(model, "num_input", getattr(model, "input_dim", 0)))
    if input_dim <= 0:
        raise ValueError(
            "inference benchmark requires model.num_input or model.input_dim"
        )
    x = torch.zeros(batch_size, window_size, input_dim, device=device)
    morphology_indices = (
        torch.zeros(batch_size, dtype=torch.long, device=device)
        if getattr(getattr(model, "config", None), "morphology_ids", None)
        else None
    )
    timings: list[float] = []
    needs_normal_no_grad = any(
        getattr(module, "learn_memory_tau", False)
        or hasattr(module, "branch_gain_enabled")
        for module in model.modules()
    )
    eval_context = torch.no_grad() if needs_normal_no_grad else torch.inference_mode()
    with eval_context:
        for _ in range(warmup_repeats):
            _forward_model(model, x, morphology_indices)
        _synchronize(device)
        if device.startswith("cuda") and torch.cuda.is_available():
            events = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                _forward_model(model, x, morphology_indices)
                stop.record()
                events.append((start, stop))
            _synchronize(device)
            timings = [start.elapsed_time(stop) for start, stop in events]
        else:
            for _ in range(repeats):
                started = time.perf_counter()
                _forward_model(model, x, morphology_indices)
                timings.append((time.perf_counter() - started) * 1000.0)
    latency = np.asarray(timings, dtype=np.float64)
    median_ms = float(np.median(latency))
    return {
        "inference_batch_size": batch_size,
        "inference_window_size": window_size,
        "inference_benchmark_repeats": repeats,
        "inference_latency_median_ms": median_ms,
        "inference_latency_p10_ms": float(np.quantile(latency, 0.1)),
        "inference_latency_p90_ms": float(np.quantile(latency, 0.9)),
        "inference_presentations_per_second": batch_size * 1000.0 / median_ms,
    }


def benchmark_model_training_step(
    model: nn.Module,
    *,
    device: str,
    batch_size: int = 32,
    window_size: int = 500,
    warmup_repeats: int = 20,
    repeats: int = 100,
) -> dict[str, float | int]:
    """Measure repeated forward/backward latency without optimizer or loader noise."""
    if batch_size <= 0 or window_size <= 0 or warmup_repeats < 0 or repeats <= 0:
        raise ValueError("training benchmark sizes and repeats must be positive")
    model = model.to(device).train()
    input_dim = int(getattr(model, "num_input", getattr(model, "input_dim", 0)))
    if input_dim <= 0:
        raise ValueError(
            "training benchmark requires model.num_input or model.input_dim"
        )
    x = torch.randn(batch_size, window_size, input_dim, device=device)
    morphology_indices = (
        torch.zeros(batch_size, dtype=torch.long, device=device)
        if getattr(getattr(model, "config", None), "morphology_ids", None)
        else None
    )

    def training_step() -> None:
        model.zero_grad(set_to_none=True)
        prediction = _forward_model(model, x, morphology_indices)
        prediction.square().mean().backward()

    timings: list[float] = []
    for _ in range(warmup_repeats):
        training_step()
    _synchronize(device)
    if device.startswith("cuda") and torch.cuda.is_available():
        events = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            training_step()
            stop.record()
            events.append((start, stop))
        _synchronize(device)
        timings = [start.elapsed_time(stop) for start, stop in events]
    else:
        for _ in range(repeats):
            started = time.perf_counter()
            training_step()
            timings.append((time.perf_counter() - started) * 1000.0)
    latency = np.asarray(timings, dtype=np.float64)
    median_ms = float(np.median(latency))
    return {
        "training_step_batch_size": batch_size,
        "training_step_window_size": window_size,
        "training_step_benchmark_repeats": repeats,
        "training_step_latency_median_ms": median_ms,
        "training_step_latency_p10_ms": float(np.quantile(latency, 0.1)),
        "training_step_latency_p90_ms": float(np.quantile(latency, 0.9)),
        "training_step_presentations_per_second": batch_size * 1000.0 / median_ms,
    }


def training_objective_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    objective: TrainingObjective,
    burn_in: int,
    raw_soma_mv: torch.Tensor | None = None,
    base_soma_prediction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction.shape != target.shape or prediction.shape[-1] != 2:
        raise ValueError("prediction and target must match with final dimension two")
    if burn_in >= prediction.shape[1]:
        raise ValueError("burn_in leaves no training timesteps")
    clipped_target = target
    raw_soma_target = None
    if raw_soma_mv is not None:
        if raw_soma_mv.shape != target.shape[:2]:
            raise ValueError(
                "raw_soma_mv must match prediction batch and time dimensions"
            )
        target = target.clone()
        raw_soma_target = (raw_soma_mv + 67.7) * objective.soma_scale
        target[..., 1] = torch.lerp(
            target[..., 1], raw_soma_target, objective.raw_soma_target_mix
        )
    elif objective.raw_soma_target_mix:
        raise ValueError("raw_soma_mv is required when raw_soma_target_mix is nonzero")
    pred = prediction[:, burn_in:, :]
    tgt = target[:, burn_in:, :]
    spike_bce = F.binary_cross_entropy_with_logits(pred[..., 0], tgt[..., 0])
    soma_mse = F.mse_loss(pred[..., 1], tgt[..., 1])
    total = (
        objective.spike_loss_weight * spike_bce + objective.soma_loss_weight * soma_mse
    )
    components = {"spike_bce": spike_bce, "soma_mse": soma_mse}
    if objective.direct_sera_loss_weight:
        sera = differentiable_root_sera(
            pred[..., 1],
            tgt,
            threshold_count=objective.sera_threshold_count,
            soma_scale=objective.soma_scale,
        )
        total = total + objective.direct_sera_loss_weight * sera
        components["direct_sera"] = sera
    if objective.dynamics_sera_loss_weight:
        if pred.shape[1] < 2:
            raise ValueError("dynamics SERA requires at least two training timesteps")
        dynamics_target = tgt[:, 1:, :].clone()
        dynamics_target[..., 1] = tgt[:, 1:, 1] - tgt[:, :-1, 1]
        dynamics_sera = differentiable_root_sera(
            pred[:, 1:, 1] - pred[:, :-1, 1],
            dynamics_target,
            threshold_count=objective.sera_threshold_count,
            soma_scale=objective.soma_scale,
        )
        total = total + objective.dynamics_sera_loss_weight * dynamics_sera
        components["dynamics_sera"] = dynamics_sera
    if objective.f1_curve_loss_weight:
        f1_loss = differentiable_f1_curve_loss(
            pred[..., 0],
            tgt[..., 0],
            tolerances_steps=objective.f1_tolerances_steps,
        )
        total = total + objective.f1_curve_loss_weight * f1_loss
        components["f1_curve_loss"] = f1_loss
    if objective.one_to_one_f1_loss_weight:
        one_to_one = probabilistic_one_to_one_f1_loss(
            pred[..., 0],
            tgt[..., 0],
            tolerances_steps=objective.f1_tolerances_steps,
        )
        total = total + objective.one_to_one_f1_loss_weight * one_to_one
        components["one_to_one_f1_loss"] = one_to_one
    if objective.near_spike_radius_steps is not None:
        near_spike_loss = differentiable_near_spike_voltage_loss(
            pred[..., 1],
            tgt,
            radius_steps=objective.near_spike_radius_steps,
            soma_scale=objective.soma_scale,
        )
        total = total + objective.near_spike_voltage_loss_weight * near_spike_loss
        components["near_spike_voltage_loss"] = near_spike_loss
    if objective.base_soma_loss_weight or objective.base_soma_sera_loss_weight:
        if base_soma_prediction is None:
            raise ValueError("base_soma_prediction is required for base soma loss")
        if base_soma_prediction.shape != (*prediction.shape[:2], 1):
            raise ValueError(
                "base_soma_prediction must match prediction batch and time"
            )
        base_target = clipped_target.clone()
        if raw_soma_target is not None:
            base_target[..., 1] = torch.lerp(
                clipped_target[..., 1],
                raw_soma_target,
                objective.base_soma_target_mix,
            )
        elif objective.base_soma_target_mix:
            raise ValueError("raw_soma_mv is required for nonzero base_soma_target_mix")
        base_prediction = base_soma_prediction[:, burn_in:, 0]
        base_trimmed_target = base_target[:, burn_in:, :]
        base_mse = F.mse_loss(base_prediction, base_trimmed_target[..., 1])
        total = total + objective.base_soma_loss_weight * base_mse
        components["base_soma_mse"] = base_mse
        if objective.base_soma_sera_loss_weight:
            base_sera = differentiable_root_sera(
                base_prediction,
                base_trimmed_target,
                threshold_count=objective.sera_threshold_count,
                soma_scale=objective.soma_scale,
            )
            total = total + objective.base_soma_sera_loss_weight * base_sera
            components["base_soma_sera"] = base_sera
    components["total"] = total
    return total, components


def differentiable_near_spike_voltage_loss(
    prediction_soma: torch.Tensor,
    target: torch.Tensor,
    *,
    radius_steps: int,
    soma_scale: float,
) -> torch.Tensor:
    if prediction_soma.shape != target[..., 1].shape:
        raise ValueError("prediction_soma must match the target time axes")
    spikes = (target[..., 0] > 0.5).to(dtype=prediction_soma.dtype)
    if radius_steps:
        kernel = 2 * radius_steps + 1
        mask = F.max_pool1d(
            spikes.unsqueeze(1), kernel_size=kernel, stride=1, padding=radius_steps
        ).squeeze(1)
    else:
        mask = spikes
    squared_error = ((prediction_soma - target[..., 1]) / soma_scale).square()
    count = mask.sum()
    if not bool(count):
        return squared_error.sum() * 0.0
    return ((mask * squared_error).sum() / count).clamp_min(0.0).sqrt()


def differentiable_root_sera(
    prediction_soma: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold_count: int = 21,
    soma_scale: float = 0.1,
) -> torch.Tensor:
    relevance = _sera_soma_relevance(target).detach()
    error_mv = (prediction_soma - target[..., 1]) / soma_scale
    return _differentiable_root_sera_from_error(
        error_mv,
        relevance,
        threshold_count=threshold_count,
    )


def _differentiable_root_sera_from_error(
    error: torch.Tensor,
    relevance: torch.Tensor,
    *,
    threshold_count: int,
) -> torch.Tensor:
    squared_error = error.square()
    thresholds = torch.linspace(
        0.0, 1.0, threshold_count, device=error.device, dtype=error.dtype
    )
    mask = relevance.unsqueeze(0) >= thresholds.view(-1, 1, 1)
    counts = mask.sum(dim=(1, 2))
    sums = (mask * squared_error.unsqueeze(0)).sum(dim=(1, 2))
    valid = counts > 0
    values = sums[valid] / counts[valid].to(dtype=sums.dtype)
    used_thresholds = thresholds[valid]
    if values.numel() == 0:
        return squared_error.mean().sqrt()
    if values.numel() == 1:
        return values[0].sqrt()
    area = torch.trapezoid(values, used_thresholds)
    span = (used_thresholds[-1] - used_thresholds[0]).clamp_min(
        torch.finfo(values.dtype).eps
    )
    return (area / span).clamp_min(0.0).sqrt()


def probabilistic_one_to_one_f1_loss(
    spike_logits: torch.Tensor,
    spike_target: torch.Tensor,
    *,
    tolerances_steps: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
) -> torch.Tensor:
    """Soft one-to-one F1 loss (1 - trapezoid F1 area over tolerances).

    Every timestep within a tolerance band belongs to its nearest true event;
    each event is hit with probability 1 - prod(1 - p) over its zone, and
    probability beyond that hit or outside all zones counts as false-positive
    mass, so duplicate predictions are penalized (ported from I130).
    """
    probability = torch.sigmoid(spike_logits)
    exact = spike_target > 0.5
    event_ids = exact.reshape(-1).to(torch.long).cumsum(0).reshape_as(exact)
    event_ids = event_ids * exact.to(torch.long)
    event_count = exact.sum().to(probability.dtype)
    epsilon = torch.finfo(probability.dtype).eps
    f1_values = []
    for radius in tolerances_steps:
        zones = event_ids.clone()
        unassigned = zones == 0
        for distance in range(1, radius + 1):
            earlier = torch.zeros_like(event_ids)
            earlier[:, distance:] = event_ids[:, :-distance]
            take = unassigned & (earlier > 0)
            zones = torch.where(take, earlier, zones)
            unassigned = unassigned & ~take
            later = torch.zeros_like(event_ids)
            later[:, :-distance] = event_ids[:, distance:]
            take = unassigned & (later > 0)
            zones = torch.where(take, later, zones)
            unassigned = unassigned & ~take
        flat_zone = zones.reshape(-1)
        flat_probability = probability.reshape(-1)
        masses = torch.zeros(probability.numel() + 1, dtype=probability.dtype, device=probability.device)
        masses = masses.scatter_add(0, flat_zone, flat_probability)
        log_misses = torch.zeros_like(masses).scatter_add(
            0, flat_zone, torch.log1p(-flat_probability.clamp(max=1.0 - epsilon))
        )
        hits = 1.0 - torch.exp(log_misses[1:])
        true_positive = hits.sum()
        false_negative = event_count - true_positive
        false_positive = masses[0] + (masses[1:] - hits).clamp_min(0.0).sum()
        f1_values.append(
            2.0 * true_positive / (2.0 * true_positive + false_positive + false_negative).clamp_min(epsilon)
        )
    values = torch.as_tensor(tolerances_steps, dtype=probability.dtype, device=probability.device)
    if len(tolerances_steps) == 1 or tolerances_steps[-1] == tolerances_steps[0]:
        weights = torch.ones_like(values) / values.numel()
    else:
        widths = values[1:] - values[:-1]
        weights = torch.zeros_like(values)
        weights[:-1] += 0.5 * widths
        weights[1:] += 0.5 * widths
        weights = weights / (values[-1] - values[0])
    return 1.0 - (weights * torch.stack(f1_values)).sum()


def differentiable_f1_curve_loss(
    spike_logits: torch.Tensor,
    spike_target: torch.Tensor,
    *,
    tolerances_steps: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
) -> torch.Tensor:
    probability = torch.sigmoid(spike_logits)
    target = (spike_target > 0.5).to(dtype=probability.dtype)
    epsilon = torch.finfo(probability.dtype).eps
    f1_values = []
    for tolerance in tolerances_steps:
        if tolerance:
            kernel = 2 * tolerance + 1
            target_near = F.max_pool1d(
                target.unsqueeze(1), kernel, stride=1, padding=tolerance
            ).squeeze(1)
            probability_near = F.max_pool1d(
                probability.unsqueeze(1), kernel, stride=1, padding=tolerance
            ).squeeze(1)
        else:
            target_near = target
            probability_near = probability
        precision = (probability * target_near).sum() / probability.sum().clamp_min(
            epsilon
        )
        recall = (target * probability_near).sum() / target.sum().clamp_min(epsilon)
        f1_values.append(
            2.0 * precision * recall / (precision + recall).clamp_min(epsilon)
        )
    return 1.0 - torch.stack(f1_values).mean()


def resolve_presentation_budget(*, iteration: int, force_long: bool) -> int:
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    return (
        LONG_PRESENTATIONS
        if force_long or (iteration > 0 and iteration % 10 == 0)
        else STANDARD_PRESENTATIONS
    )


def _iter_shared_batches(
    dataset: Any,
    batch_size: int,
    *,
    prefetch_batches: int,
    start_batch: int = 0,
):
    indexed_ranges = list(
        enumerate(
            range(start, min(start + batch_size, len(dataset)))
            for start in range(0, len(dataset), batch_size)
        )
    )
    if start_batch < 0 or start_batch > len(indexed_ranges):
        raise ValueError("start_batch lies outside the shared batch range")
    indexed_ranges = indexed_ranges[start_batch:]
    load = getattr(dataset, "get_shared_batch", None)
    if load is None:
        load = dataset.get_batch
    if prefetch_batches <= 0:
        for batch_index, indices in indexed_ranges:
            yield batch_index, indices, load(indices)
        return
    queue_depth = min(prefetch_batches, len(indexed_ranges))
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = deque(
            executor.submit(load, indexed_ranges[index][1])
            for index in range(queue_depth)
        )
        next_index = queue_depth
        for batch_index, indices in indexed_ranges:
            batch_arrays = futures.popleft().result()
            if next_index < len(indexed_ranges):
                futures.append(executor.submit(load, indexed_ranges[next_index][1]))
                next_index += 1
            yield batch_index, indices, batch_arrays


def _flush_panel_timings(
    states: dict[str, dict[str, Any]],
    teacher_timing: list[Any],
    *,
    device: str,
) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        _synchronize(device)
        for state in states.values():
            state["compute_seconds"] += (
                sum(start.elapsed_time(stop) for start, stop in state["compute_timing"])
                / 1000.0
            )
            state["compute_timing"].clear()
        teacher_seconds = (
            sum(start.elapsed_time(stop) for start, stop in teacher_timing) / 1000.0
        )
    else:
        for state in states.values():
            state["compute_seconds"] += sum(state["compute_timing"])
            state["compute_timing"].clear()
        teacher_seconds = sum(teacher_timing)
    teacher_timing.clear()
    return teacher_seconds


def _write_panel_resume(
    path: Path,
    states: dict[str, dict[str, Any]],
    *,
    next_shared_batch: int,
    total_shared_batches: int,
    cumulative_panel_seconds: float,
    cumulative_teacher_seconds: float,
) -> None:
    for candidate_id, state in states.items():
        _require_finite_checkpoint_state(candidate_id, state)
    payload = {
        "version": 1,
        "candidate_ids": list(states),
        "next_shared_batch": next_shared_batch,
        "total_shared_batches": total_shared_batches,
        "cumulative_panel_seconds": cumulative_panel_seconds,
        "cumulative_teacher_seconds": cumulative_teacher_seconds,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "states": {
            candidate_id: {
                "model": state["model"].state_dict(),
                "optimizer": state["optimizer"].state_dict(),
                "scheduler": (
                    state["scheduler"].state_dict()
                    if state["scheduler"] is not None
                    else None
                ),
                "steps": state["steps"],
                "presentations": state["presentations"],
                "last_components": state["last_components"],
                "loss_sum": state["loss_sum"],
                "loss_observations": state["loss_observations"],
                "gradient_norm_sum": state["gradient_norm_sum"],
                "gradient_norm_observations": state[
                    "gradient_norm_observations"
                ],
                "preclip_gradient_norm_max": state[
                    "preclip_gradient_norm_max"
                ],
                "gradient_clipped_steps": state["gradient_clipped_steps"],
                "compute_seconds": state["compute_seconds"],
            }
            for candidate_id, state in states.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    for state in states.values():
        candidate = state["candidate"]
        save_checkpoint(
            state["model"],
            candidate.checkpoint_path,
            metadata={
                "training_in_progress": next_shared_batch < total_shared_batches,
                "processed_shared_batches": next_shared_batch,
                "total_shared_batches": total_shared_batches,
                "train_presentations": state["presentations"],
                "optimizer_steps": state["steps"],
            },
        )


def _require_finite_checkpoint_state(
    candidate_id: str,
    state: dict[str, Any],
) -> None:
    def walk(value: Any, path: str):
        if torch.is_tensor(value):
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise FloatingPointError(
                    f"non-finite checkpoint state for {candidate_id}: {path}"
                )
            return
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(state["model"].state_dict(), "model")
    walk(state["optimizer"].state_dict(), "optimizer")
    if state["scheduler"] is not None:
        walk(state["scheduler"].state_dict(), "scheduler")
    for name in (
        "loss_sum",
        "gradient_norm_sum",
        "preclip_gradient_norm_max",
        "compute_seconds",
    ):
        value = float(state[name])
        if not math.isfinite(value):
            raise FloatingPointError(
                f"non-finite checkpoint state for {candidate_id}: {name}"
            )


def _write_panel_snapshot(
    root: Path,
    states: dict[str, dict[str, Any]],
    *,
    shared_batch: int,
    total_shared_batches: int,
) -> None:
    for candidate_id, state in states.items():
        _require_finite_checkpoint_state(candidate_id, state)
    output = root / f"batch-{shared_batch:07d}"
    for candidate_id, state in states.items():
        save_checkpoint(
            state["model"],
            output / f"{candidate_id}.pt",
            metadata={
                "training_snapshot": True,
                "processed_shared_batches": shared_batch,
                "total_shared_batches": total_shared_batches,
                "train_presentations": state["presentations"],
                "optimizer_steps": state["steps"],
            },
        )


def _forward_model(
    model: nn.Module,
    inputs: torch.Tensor,
    morphology_indices: torch.Tensor | None,
) -> torch.Tensor:
    if morphology_indices is None:
        return model(inputs)
    return model(inputs, morphology_indices=morphology_indices)


def _morphology_index_tensor(
    model: nn.Module,
    sample_ids: list[str] | None,
    *,
    device: str,
) -> torch.Tensor | None:
    configured = getattr(getattr(model, "config", None), "morphology_ids", None)
    if configured is None:
        return None
    if sample_ids is None:
        raise ValueError("conditioned model requires dataset sample IDs")
    mapping = {str(name): index for index, name in enumerate(configured)}
    names = [_morphology_id(sample_id) for sample_id in sample_ids]
    unknown = sorted(set(names) - set(mapping))
    if unknown:
        raise ValueError(f"unknown morphology IDs for conditioned model: {unknown}")
    return torch.tensor(
        [mapping[name] for name in names], dtype=torch.long, device=device
    )


def _group_schedule_entries(
    entries: list[tuple[int, int, int]],
) -> list[tuple[int, list[tuple[int, int]]]]:
    groups: list[tuple[int, list[tuple[int, int]]]] = []
    for shard_idx, sample_idx, start in entries:
        if not groups or groups[-1][0] != shard_idx:
            groups.append((shard_idx, []))
        groups[-1][1].append((sample_idx, start))
    return groups


def _morphology_id(sample_id: str) -> str:
    return str(sample_id).split(":", 1)[0]


def _dataset_fingerprint(dataset: ShardedNeuronIODataset) -> str:
    payload = {
        "root": str(dataset.root.resolve()),
        "shards": [shard.name for shard in dataset.shards],
        "samples": len(dataset._base_index),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
