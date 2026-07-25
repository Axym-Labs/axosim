"""Morphology-aware procedural routing for large GPU populations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from axosim.simulation_contract import LargePopulationSimulationContract

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised in CPU-only installations
    triton = None
    tl = None


@dataclass(frozen=True)
class DelayBucket:
    delay_steps: int
    first_fanout_offset: int
    event_count_per_source: int
    local_targets: bool

    @property
    def fanout_offsets(self) -> tuple[int, ...]:
        return tuple(
            self.first_fanout_offset + 2 * index
            for index in range(self.event_count_per_source)
        )


def procedural_delay_buckets(
    *,
    fanout: int,
    local_fanout: int,
) -> tuple[DelayBucket, ...]:
    """Partition fanout offsets into local/long-range delay classes."""

    if fanout <= 0:
        raise ValueError("fanout must be positive")
    if not 0 <= local_fanout <= fanout:
        raise ValueError("local_fanout must be in [0, fanout]")
    buckets = []
    for delay, parity, is_local in (
        (1, 0, True),
        (2, 1, True),
        (3, 0, False),
        (4, 1, False),
    ):
        lower = 0 if is_local else local_fanout
        upper = local_fanout if is_local else fanout
        first = lower + ((parity - lower) & 1)
        count = max(0, (upper - first + 1) // 2)
        buckets.append(
            DelayBucket(
                delay_steps=delay,
                first_fanout_offset=first,
                event_count_per_source=count,
                local_targets=is_local,
            )
        )
    return tuple(buckets)


def compact_active_sources(
    outputs: torch.Tensor,
    *,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Return source-sorted int32 indices whose scalar output fires."""

    if outputs.ndim != 1:
        raise ValueError("outputs must have shape (population,)")
    return torch.nonzero(
        outputs > threshold,
        as_tuple=False,
    ).flatten().to(dtype=torch.int32)


if triton is not None:

    @triton.jit
    def _delay_bucket_kernel(
        active,
        source_is_inhibitory,
        slot_map,
        morphology,
        morphology_gain,
        feature_gain,
        branch_output,
        active_count,
        fanout: tl.constexpr,
        first_fanout_offset: tl.constexpr,
        bucket_fanout: tl.constexpr,
        local_targets: tl.constexpr,
        population_mask: tl.constexpr,
        spatial_tile_mask: tl.constexpr,
        num_branch: tl.constexpr,
        synapses_per_branch: tl.constexpr,
        slot_count: tl.constexpr,
        block_size: tl.constexpr,
    ):
        source_row = tl.program_id(0)
        bucket_offsets = (
            tl.program_id(1) * block_size
            + tl.arange(0, block_size)
        )
        mask = (source_row < active_count) & (
            bucket_offsets < bucket_fanout
        )
        source = tl.load(active + source_row)
        fanout_offsets = (
            first_fanout_offset + 2 * bucket_offsets
        )
        if local_targets:
            targets = (
                (source & ~spatial_tile_mask)
                + (
                    (source * 131 + fanout_offsets * 127)
                    & spatial_tile_mask
                )
            )
        else:
            targets = (
                source * 413_293 + fanout_offsets * 489_905
            ) & population_mask
        target_morphology = tl.load(
            morphology + targets,
            mask=mask,
            other=0,
        )
        inhibitory = (
            tl.load(source_is_inhibitory + source) != 0
        )
        channels = (
            source * 37
            + fanout_offsets * 97
            + target_morphology * 53
            + inhibitory.to(tl.int32) * 191
        ) % fanout
        slots = tl.load(slot_map + channels, mask=mask, other=0)
        gains = tl.load(
            morphology_gain + slots,
            mask=mask,
            other=0.0,
        )
        gains *= tl.load(
            feature_gain
            + target_morphology * slot_count
            + slots,
            mask=mask,
            other=0.0,
        )
        gains *= tl.where(inhibitory, -1.2, 1.0)
        branches = slots // synapses_per_branch
        output_offsets = targets * num_branch + branches
        tl.atomic_add(
            branch_output + output_offsets,
            gains,
            mask=mask,
        )


class ProceduralMorphologyConnectomeRouter:
    """Persistent delay-queue router for the scalable connectome control."""

    def __init__(
        self,
        contract: LargePopulationSimulationContract,
        *,
        slot_map: torch.Tensor,
        morphology: torch.Tensor,
        source_is_inhibitory: torch.Tensor,
        morphology_gain: torch.Tensor,
        feature_gain: torch.Tensor,
        synapses_per_branch: int,
    ) -> None:
        if triton is None:
            raise RuntimeError(
                "ProceduralMorphologyConnectomeRouter requires Triton"
            )
        if contract.delay_slots != 4:
            raise ValueError("procedural routing currently requires 4 delays")
        if not _is_power_of_two(contract.population_size):
            raise ValueError("population_size must be a power of two")
        if not _is_power_of_two(contract.spatial_tile_neurons):
            raise ValueError("spatial_tile_neurons must be a power of two")
        if contract.population_size % contract.spatial_tile_neurons:
            raise ValueError(
                "spatial tiles must evenly divide the population"
            )
        if synapses_per_branch <= 0:
            raise ValueError("synapses_per_branch must be positive")
        self.contract = contract
        self.synapses_per_branch = synapses_per_branch
        self.slot_count = contract.branch_count * synapses_per_branch
        self.local_fanout = round(
            contract.fanout * contract.local_connection_fraction
        )
        self.delay_buckets = procedural_delay_buckets(
            fanout=contract.fanout,
            local_fanout=self.local_fanout,
        )

        tensors = {
            "slot_map": slot_map,
            "morphology": morphology,
            "source_is_inhibitory": source_is_inhibitory,
            "morphology_gain": morphology_gain,
            "feature_gain": feature_gain,
        }
        devices = {tensor.device for tensor in tensors.values()}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise ValueError("routing tensors must share one CUDA device")
        expected_shapes = {
            "slot_map": (contract.fanout,),
            "morphology": (contract.population_size,),
            "source_is_inhibitory": (contract.population_size,),
            "morphology_gain": (self.slot_count,),
            "feature_gain": (
                contract.morphology_classes,
                self.slot_count,
            ),
        }
        for name, expected_shape in expected_shapes.items():
            if tensors[name].shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}"
                )
        if slot_map.dtype != torch.int32:
            raise ValueError("slot_map must use int32")
        if morphology.dtype != torch.int32:
            raise ValueError("morphology must use int32")
        if source_is_inhibitory.dtype != torch.uint8:
            raise ValueError("source_is_inhibitory must use uint8")
        if morphology_gain.dtype != torch.float16:
            raise ValueError("morphology_gain must use float16")
        if feature_gain.dtype != torch.float16:
            raise ValueError("feature_gain must use float16")

        self.device = slot_map.device
        self.slot_map = slot_map.contiguous()
        self.morphology = morphology.contiguous()
        self.source_is_inhibitory = (
            source_is_inhibitory.contiguous()
        )
        self.morphology_gain = morphology_gain.contiguous()
        self.feature_gain = feature_gain.contiguous()
        self.delay_queue = torch.zeros(
            contract.delay_slots,
            contract.population_size,
            contract.branch_count,
            device=self.device,
            dtype=torch.float16,
        )

    def current_inputs(self, current_slot: int) -> torch.Tensor:
        self._validate_slot(current_slot)
        return self.delay_queue[current_slot]

    def clear_and_route(
        self,
        active_sources: torch.Tensor,
        *,
        current_slot: int,
    ) -> torch.Tensor:
        """Clear a consumed queue slot and schedule new delayed events."""

        self._validate_slot(current_slot)
        if (
            active_sources.ndim != 1
            or active_sources.dtype != torch.int32
            or active_sources.device != self.device
        ):
            raise ValueError(
                "active_sources must be a CUDA int32 vector"
            )
        self.delay_queue[current_slot].zero_()
        active_count = active_sources.numel()
        if active_count == 0:
            return self.delay_queue
        block_size = 1_024
        for bucket in self.delay_buckets:
            if bucket.event_count_per_source == 0:
                continue
            output_slot = (
                current_slot + bucket.delay_steps
            ) & (self.contract.delay_slots - 1)
            _delay_bucket_kernel[
                (
                    active_count,
                    triton.cdiv(
                        bucket.event_count_per_source,
                        block_size,
                    ),
                )
            ](
                active_sources,
                self.source_is_inhibitory,
                self.slot_map,
                self.morphology,
                self.morphology_gain,
                self.feature_gain,
                self.delay_queue[output_slot],
                active_count,
                fanout=self.contract.fanout,
                first_fanout_offset=(
                    bucket.first_fanout_offset
                ),
                bucket_fanout=bucket.event_count_per_source,
                local_targets=bucket.local_targets,
                population_mask=self.contract.population_size - 1,
                spatial_tile_mask=(
                    self.contract.spatial_tile_neurons - 1
                ),
                num_branch=self.contract.branch_count,
                synapses_per_branch=self.synapses_per_branch,
                slot_count=self.slot_count,
                block_size=block_size,
                num_warps=8,
            )
        return self.delay_queue

    def _validate_slot(self, current_slot: int) -> None:
        if not 0 <= current_slot < self.contract.delay_slots:
            raise ValueError("current_slot is outside the delay ring")


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0
