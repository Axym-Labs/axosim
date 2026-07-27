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
    minimum_delay_steps: int = 1,
) -> tuple[DelayBucket, ...]:
    """Partition fanout offsets into local/long-range delay classes."""

    if fanout <= 0:
        raise ValueError("fanout must be positive")
    if not 0 <= local_fanout <= fanout:
        raise ValueError("local_fanout must be in [0, fanout]")
    if minimum_delay_steps <= 0:
        raise ValueError("minimum_delay_steps must be positive")
    buckets = []
    for delay_offset, parity, is_local in (
        (0, 0, True),
        (1, 1, True),
        (2, 0, False),
        (3, 1, False),
    ):
        lower = 0 if is_local else local_fanout
        upper = local_fanout if is_local else fanout
        first = lower + ((parity - lower) & 1)
        count = max(0, (upper - first + 1) // 2)
        buckets.append(
            DelayBucket(
                delay_steps=minimum_delay_steps + delay_offset,
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


def build_typed_tile_pools(
    source_is_inhibitory: torch.Tensor,
    physical_to_logical: torch.Tensor,
    *,
    spatial_tile_neurons: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build invertible logical E/I ranks within every spatial tile."""

    if (
        source_is_inhibitory.ndim != 1
        or source_is_inhibitory.dtype != torch.uint8
    ):
        raise ValueError(
            "source_is_inhibitory must be a uint8 vector"
        )
    if (
        physical_to_logical.shape != source_is_inhibitory.shape
        or physical_to_logical.dtype != torch.int32
        or physical_to_logical.device
        != source_is_inhibitory.device
    ):
        raise ValueError(
            "physical_to_logical must be a matching int32 vector"
        )
    population = source_is_inhibitory.numel()
    if (
        spatial_tile_neurons <= 0
        or population % spatial_tile_neurons
    ):
        raise ValueError(
            "spatial_tile_neurons must divide the population"
        )
    logical_role = torch.empty_like(source_is_inhibitory)
    logical_role[physical_to_logical.long()] = source_is_inhibitory
    logical_ids = torch.arange(
        population,
        device=source_is_inhibitory.device,
        dtype=torch.int64,
    )
    tile_count = population // spatial_tile_neurons
    tiles = torch.div(
        logical_ids,
        spatial_tile_neurons,
        rounding_mode="floor",
    )
    groups = logical_role.long() * tile_count + tiles
    order = torch.argsort(groups * population + logical_ids)
    sorted_groups = groups.index_select(0, order)
    counts = torch.bincount(
        sorted_groups, minlength=2 * tile_count
    )
    offsets = torch.empty(
        2 * tile_count + 1,
        device=source_is_inhibitory.device,
        dtype=torch.int64,
    )
    offsets[0] = 0
    offsets[1:] = counts.cumsum(0)
    ranks_sorted = torch.arange(
        population,
        device=source_is_inhibitory.device,
        dtype=torch.int64,
    ) - torch.repeat_interleave(offsets[:-1], counts)
    logical_ranks = torch.empty(
        population,
        device=source_is_inhibitory.device,
        dtype=torch.int32,
    )
    logical_ranks[order] = ranks_sorted.to(dtype=torch.int32)
    return (
        logical_ids.index_select(0, order)
        .to(dtype=torch.int32)
        .contiguous(),
        offsets.to(dtype=torch.int32).contiguous(),
        logical_ranks.contiguous(),
    )


def build_external_count_branch_bank(
    *,
    slot_map: torch.Tensor,
    morphology_gain: torch.Tensor,
    feature_gain: torch.Tensor,
    synapses_per_branch: int,
    max_event_count: int = 32,
    channels_per_type: int | None = None,
) -> torch.Tensor:
    """Precompute exact branch currents for a consecutive event pattern."""

    if slot_map.ndim != 1 or slot_map.dtype != torch.int32:
        raise ValueError("slot_map must be an int32 vector")
    if morphology_gain.ndim != 1:
        raise ValueError("morphology_gain must be a vector")
    if feature_gain.ndim != 2:
        raise ValueError("feature_gain must be a matrix")
    if slot_map.device != morphology_gain.device:
        raise ValueError("slot_map and morphology_gain must share a device")
    if slot_map.device != feature_gain.device:
        raise ValueError("slot_map and feature_gain must share a device")
    if slot_map.numel() % 2:
        raise ValueError("external channels must have equal E/I halves")
    if channels_per_type is None:
        channels_per_type = slot_map.numel() // 2
    if channels_per_type * 2 != slot_map.numel():
        raise ValueError("channels_per_type must cover the E/I channel halves")
    if synapses_per_branch <= 0:
        raise ValueError("synapses_per_branch must be positive")
    if max_event_count <= 0:
        raise ValueError("max_event_count must be positive")
    slot_count = morphology_gain.numel()
    if feature_gain.shape[1] != slot_count:
        raise ValueError("feature_gain width must match morphology_gain")
    if slot_count % synapses_per_branch:
        raise ValueError("synapse slots must divide evenly into branches")

    device = slot_map.device
    morphology_classes = feature_gain.shape[0]
    branch_count = slot_count // synapses_per_branch
    bases = torch.arange(
        channels_per_type, device=device, dtype=torch.long
    ).unsqueeze(1)
    offsets = torch.arange(
        max_event_count, device=device, dtype=torch.long
    ).unsqueeze(0)
    bank = torch.zeros(
        2,
        morphology_classes,
        channels_per_type,
        max_event_count + 1,
        branch_count,
        device=device,
        dtype=torch.float32,
    )
    for source_type in range(2):
        channels = (
            bases + offsets * 127
        ) % channels_per_type
        channels = channels + source_type * channels_per_type
        slots = slot_map.index_select(
            0, channels.flatten()
        ).view_as(channels).long()
        branches = torch.div(
            slots,
            synapses_per_branch,
            rounding_mode="floor",
        )
        for morphology_index in range(morphology_classes):
            gains = morphology_gain.float().index_select(
                0, slots.flatten()
            ).view_as(slots)
            gains = gains * feature_gain[
                morphology_index
            ].float().index_select(
                0, slots.flatten()
            ).view_as(slots)
            if source_type:
                gains = -gains
            increments = torch.zeros(
                channels_per_type,
                max_event_count,
                branch_count,
                device=device,
                dtype=torch.float32,
            )
            increments.scatter_add_(
                2,
                branches.unsqueeze(-1),
                gains.unsqueeze(-1),
            )
            bank[source_type, morphology_index, :, 1:] = (
                increments.cumsum(dim=1)
            )
    return bank.to(dtype=torch.float16).contiguous()


if triton is not None:

    @triton.jit
    def _delay_bucket_kernel(
        active,
        physical_to_logical,
        logical_to_physical,
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
        storage_reordered: tl.constexpr,
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
        physical_source = tl.load(active + source_row)
        if storage_reordered:
            logical_source = tl.load(
                physical_to_logical + physical_source
            )
        else:
            logical_source = physical_source
        fanout_offsets = (
            first_fanout_offset + 2 * bucket_offsets
        )
        if local_targets:
            logical_targets = (
                (logical_source & ~spatial_tile_mask)
                + (
                    (
                        logical_source * 131
                        + fanout_offsets * 127
                    )
                    & spatial_tile_mask
                )
            )
        else:
            logical_targets = (
                logical_source * 413_293
                + fanout_offsets * 489_905
            ) & population_mask
        if storage_reordered:
            targets = tl.load(
                logical_to_physical + logical_targets,
                mask=mask,
                other=0,
            )
        else:
            targets = logical_targets
        target_morphology = tl.load(
            morphology + targets,
            mask=mask,
            other=0,
        )
        inhibitory = (
            tl.load(
                source_is_inhibitory + physical_source
            )
            != 0
        )
        channels = (
            logical_source * 37
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

    @triton.jit
    def _delay_block_kernel(
        active,
        physical_to_logical,
        logical_to_physical,
        source_is_inhibitory,
        slot_map,
        morphology,
        morphology_gain,
        feature_gain,
        delay_queue,
        touched_offsets,
        active_count,
        current_slot: tl.constexpr,
        fanout: tl.constexpr,
        routed_fanout: tl.constexpr,
        recurrent_stride: tl.constexpr,
        local_fanout: tl.constexpr,
        minimum_delay_steps: tl.constexpr,
        delay_slot_mask: tl.constexpr,
        population: tl.constexpr,
        population_mask: tl.constexpr,
        spatial_tile_mask: tl.constexpr,
        num_branch: tl.constexpr,
        synapses_per_branch: tl.constexpr,
        slot_count: tl.constexpr,
        storage_reordered: tl.constexpr,
        record_touched: tl.constexpr,
        block_size: tl.constexpr,
    ):
        source_row = tl.program_id(0)
        compact_offsets = (
            tl.program_id(1) * block_size
            + tl.arange(0, block_size)
        )
        mask = (source_row < active_count) & (
            compact_offsets < routed_fanout
        )
        physical_source = tl.load(active + source_row)
        if storage_reordered:
            logical_source = tl.load(
                physical_to_logical + physical_source
            )
        else:
            logical_source = physical_source
        fanout_offsets = (
            compact_offsets * recurrent_stride
            + logical_source % recurrent_stride
        )
        mask &= fanout_offsets < fanout
        local_targets = fanout_offsets < local_fanout
        local_target_ids = (
            (logical_source & ~spatial_tile_mask)
            + (
                (
                    logical_source * 131
                    + fanout_offsets * 127
                )
                & spatial_tile_mask
            )
        )
        long_target_ids = (
            logical_source * 413_293
            + fanout_offsets * 489_905
        ) & population_mask
        logical_targets = tl.where(
            local_targets,
            local_target_ids,
            long_target_ids,
        )
        if storage_reordered:
            targets = tl.load(
                logical_to_physical + logical_targets,
                mask=mask,
                other=0,
            )
        else:
            targets = logical_targets
        target_morphology = tl.load(
            morphology + targets,
            mask=mask,
            other=0,
        )
        inhibitory = (
            tl.load(
                source_is_inhibitory + physical_source
            )
            != 0
        )
        channels = (
            logical_source * 37
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
        delay_steps = (
            minimum_delay_steps
            + (fanout_offsets & 1)
            + tl.where(local_targets, 0, 2)
        )
        output_slots = (
            current_slot + delay_steps
        ) & delay_slot_mask
        output_offsets = (
            (output_slots * population + targets)
            * num_branch
            + branches
        )
        if record_touched:
            touched_index = (
                source_row * routed_fanout + compact_offsets
            )
            tl.store(
                touched_offsets + touched_index,
                tl.where(mask, output_offsets, -1),
                mask=compact_offsets < routed_fanout,
            )
        tl.atomic_add(
            delay_queue + output_offsets,
            gains,
            mask=mask,
        )

    @triton.jit
    def _clear_touched_kernel(
        touched_offsets,
        delay_queue,
        count,
        block_size: tl.constexpr,
    ):
        offsets = (
            tl.program_id(0) * block_size
            + tl.arange(0, block_size)
        )
        mask = offsets < count
        destinations = tl.load(
            touched_offsets + offsets,
            mask=mask,
            other=-1,
        )
        tl.store(
            delay_queue + destinations,
            0.0,
            mask=mask & (destinations >= 0),
        )

    @triton.jit
    def _binary_delay_block_kernel(
        active,
        physical_to_logical,
        logical_to_physical,
        source_is_inhibitory,
        morphology,
        channel_queue,
        active_count,
        current_slot: tl.constexpr,
        fanout: tl.constexpr,
        channels_per_type: tl.constexpr,
        routed_fanout: tl.constexpr,
        recurrent_stride: tl.constexpr,
        local_fanout: tl.constexpr,
        minimum_delay_steps: tl.constexpr,
        delay_slot_mask: tl.constexpr,
        population: tl.constexpr,
        population_mask: tl.constexpr,
        spatial_tile_mask: tl.constexpr,
        channel_words: tl.constexpr,
        storage_reordered: tl.constexpr,
        block_size: tl.constexpr,
    ):
        source_row = tl.program_id(0)
        compact_offsets = (
            tl.program_id(1) * block_size
            + tl.arange(0, block_size)
        )
        mask = (source_row < active_count) & (
            compact_offsets < routed_fanout
        )
        physical_source = tl.load(active + source_row)
        if storage_reordered:
            logical_source = tl.load(
                physical_to_logical + physical_source
            )
        else:
            logical_source = physical_source
        fanout_offsets = (
            compact_offsets * recurrent_stride
            + logical_source % recurrent_stride
        )
        mask &= fanout_offsets < fanout
        local_targets = fanout_offsets < local_fanout
        local_target_ids = (
            (logical_source & ~spatial_tile_mask)
            + (
                (
                    logical_source * 131
                    + fanout_offsets * 127
                )
                & spatial_tile_mask
            )
        )
        long_target_ids = (
            logical_source * 413_293
            + fanout_offsets * 489_905
        ) & population_mask
        logical_targets = tl.where(
            local_targets,
            local_target_ids,
            long_target_ids,
        )
        if storage_reordered:
            targets = tl.load(
                logical_to_physical + logical_targets,
                mask=mask,
                other=0,
            )
        else:
            targets = logical_targets
        target_morphology = tl.load(
            morphology + targets,
            mask=mask,
            other=0,
        )
        inhibitory = (
            tl.load(source_is_inhibitory + physical_source) != 0
        )
        channels = (
            logical_source * 37
            + fanout_offsets * 97
            + target_morphology * 53
        ) % channels_per_type
        channels += inhibitory.to(tl.int32) * channels_per_type
        delay_steps = (
            minimum_delay_steps
            + (fanout_offsets & 1)
            + tl.where(local_targets, 0, 2)
        )
        output_slots = (
            current_slot + delay_steps
        ) & delay_slot_mask
        words = channels >> 5
        bits = 1 << (channels & 31)
        output_offsets = (
            (output_slots * population + targets) * channel_words
            + words
        )
        tl.atomic_or(
            channel_queue + output_offsets,
            bits,
            mask=mask,
        )

    @triton.jit
    def _exact_branch_delay_block_kernel(
        active,
        physical_to_logical,
        logical_to_physical,
        source_is_inhibitory,
        slot_map,
        morphology,
        morphology_gain,
        feature_gain,
        external_trajectories,
        channel_queue,
        branch_queue,
        active_count,
        current_step,
        external_seed,
        external_excitatory_gain,
        external_inhibitory_gain,
        current_slot: tl.constexpr,
        fanout: tl.constexpr,
        channels_per_type: tl.constexpr,
        routed_fanout: tl.constexpr,
        recurrent_stride: tl.constexpr,
        local_fanout: tl.constexpr,
        minimum_delay_steps: tl.constexpr,
        delay_slot_mask: tl.constexpr,
        population: tl.constexpr,
        population_mask: tl.constexpr,
        spatial_tile_mask: tl.constexpr,
        channel_words: tl.constexpr,
        branches: tl.constexpr,
        synapses_per_branch: tl.constexpr,
        slot_count: tl.constexpr,
        trajectory_count: tl.constexpr,
        trajectory_steps: tl.constexpr,
        max_external_event_count: tl.constexpr,
        external_channel_stride_inverse: tl.constexpr,
        external_collision_aware: tl.constexpr,
        storage_reordered: tl.constexpr,
        block_size: tl.constexpr,
    ):
        source_row = tl.program_id(0)
        compact_offsets = (
            tl.program_id(1) * block_size
            + tl.arange(0, block_size)
        )
        mask = (source_row < active_count) & (
            compact_offsets < routed_fanout
        )
        physical_source = tl.load(active + source_row)
        if storage_reordered:
            logical_source = tl.load(
                physical_to_logical + physical_source
            )
        else:
            logical_source = physical_source
        fanout_offsets = (
            compact_offsets * recurrent_stride
            + logical_source % recurrent_stride
        )
        mask &= fanout_offsets < fanout
        local_targets = fanout_offsets < local_fanout
        local_target_ids = (
            (logical_source & ~spatial_tile_mask)
            + (
                (
                    logical_source * 131
                    + fanout_offsets * 127
                )
                & spatial_tile_mask
            )
        )
        long_target_ids = (
            logical_source * 413_293
            + fanout_offsets * 489_905
        ) & population_mask
        logical_targets = tl.where(
            local_targets,
            local_target_ids,
            long_target_ids,
        )
        if storage_reordered:
            targets = tl.load(
                logical_to_physical + logical_targets,
                mask=mask,
                other=0,
            )
        else:
            targets = logical_targets
        target_morphology = tl.load(
            morphology + targets,
            mask=mask,
            other=0,
        )
        inhibitory = (
            tl.load(source_is_inhibitory + physical_source) != 0
        )
        channels = (
            logical_source * 37
            + fanout_offsets * 97
            + target_morphology * 53
        ) % channels_per_type
        channels += inhibitory.to(tl.int32) * channels_per_type
        delay_steps = (
            minimum_delay_steps
            + (fanout_offsets & 1)
            + tl.where(local_targets, 0, 2)
        )
        output_slots = (
            current_slot + delay_steps
        ) & delay_slot_mask
        words = channels >> 5
        bits = 1 << (channels & 31)
        channel_offsets = (
            (output_slots * population + targets) * channel_words
            + words
        )
        previous = tl.atomic_or(
            channel_queue + channel_offsets,
            bits,
            mask=mask,
        )
        first_delivery = mask & ((previous & bits) == 0)
        if external_collision_aware:
            source_type = inhibitory.to(tl.int32)
            future_step = current_step + delay_steps
            trajectory_index = (
                logical_targets * 37 + 11
            ) % trajectory_count
            trajectory_time = (
                future_step
                + (
                    logical_targets * 277 + external_seed
                )
                % trajectory_steps
            ) % trajectory_steps
            trajectory_offset = (
                (
                    source_type * trajectory_count
                    + trajectory_index
                )
                * trajectory_steps
                + trajectory_time
            )
            raw_count = tl.load(
                external_trajectories + trajectory_offset,
                mask=first_delivery,
                other=0,
            ).to(tl.float32)
            keep = tl.where(
                inhibitory,
                external_inhibitory_gain,
                external_excitatory_gain,
            )
            expected_count = raw_count * keep
            integer_count = tl.floor(expected_count).to(tl.int32)
            fraction = expected_count - integer_count.to(tl.float32)
            hash_value = (
                logical_targets.to(tl.uint32) * 1_103_515_245
                + (future_step + 1) * 12_345
                + source_type.to(tl.uint32) * 2_654_435_761
            ) & 0xFFFFFF
            draw = hash_value.to(tl.float32) / 16_777_216.0
            integer_count += (draw < fraction).to(tl.int32)
            integer_count = tl.minimum(
                integer_count, max_external_event_count
            )
            base_channel = (
                logical_targets * 131
                + (future_step + 1) * 193
                + source_type * 977
            ) % channels_per_type
            relative_channel = channels % channels_per_type
            external_offset = (
                (
                    relative_channel
                    - base_channel
                    + channels_per_type
                )
                * external_channel_stride_inverse
            ) % channels_per_type
            first_delivery &= external_offset >= integer_count
        slots = tl.load(
            slot_map + channels,
            mask=first_delivery,
            other=0,
        )
        gains = tl.load(
            morphology_gain + slots,
            mask=first_delivery,
            other=0.0,
        ).to(tl.float32)
        gains *= tl.load(
            feature_gain
            + target_morphology * slot_count
            + slots,
            mask=first_delivery,
            other=0.0,
        ).to(tl.float32)
        gains *= tl.where(inhibitory, -1.0, 1.0)
        branch = slots // synapses_per_branch
        branch_offsets = (
            (output_slots * population + targets) * branches
            + branch
        )
        tl.atomic_add(
            branch_queue + branch_offsets,
            gains,
            mask=first_delivery,
        )

    @triton.jit(
        do_not_specialize=(
            "active_count",
            "current_step",
            "touched_base",
        )
    )
    def _unique_channel_branch_route_kernel(
        active,
        physical_to_logical,
        logical_to_physical,
        source_is_inhibitory,
        logical_type_rank,
        typed_pool_ids,
        typed_pool_offsets,
        slot_map,
        morphology,
        morphology_gain,
        feature_gain,
        external_trajectories,
        branch_queue,
        touched_offsets,
        unique_rows,
        unique_row_flags,
        unique_row_count,
        active_count,
        current_step,
        external_seed,
        external_excitatory_gain,
        external_inhibitory_gain,
        current_slot: tl.constexpr,
        channels_per_type: tl.constexpr,
        channel_lower: tl.constexpr,
        channel_upper: tl.constexpr,
        selected_channel_count: tl.constexpr,
        recurrent_stride: tl.constexpr,
        target_role_count: tl.constexpr,
        max_target_multiplicity: tl.constexpr,
        candidate_count: tl.constexpr,
        delay_slot_mask: tl.constexpr,
        spatial_tile_neurons: tl.constexpr,
        tile_count: tl.constexpr,
        tile_mask: tl.constexpr,
        branches: tl.constexpr,
        synapses_per_branch: tl.constexpr,
        slot_count: tl.constexpr,
        trajectory_count: tl.constexpr,
        trajectory_steps: tl.constexpr,
        max_external_event_count: tl.constexpr,
        external_channel_stride_inverse: tl.constexpr,
        external_collision_aware: tl.constexpr,
        touched_base,
        record_touched: tl.constexpr,
        record_unique_rows: tl.constexpr,
        local_connections: tl.constexpr,
        block_size: tl.constexpr,
    ):
        source_row = tl.program_id(0)
        candidate = (
            tl.program_id(1) * block_size
            + tl.arange(0, block_size)
        )
        candidate_mask = (source_row < active_count) & (
            candidate < candidate_count
        )
        mask = candidate_mask
        physical_source = tl.load(active + source_row)
        logical_source = tl.load(
            physical_to_logical + physical_source
        )
        source_role = tl.load(
            source_is_inhibitory + physical_source
        ).to(tl.int32)
        source_tile = logical_source // spatial_tile_neurons
        source_rank = tl.load(
            logical_type_rank + logical_source
        )
        source_group = source_role * tile_count + source_tile
        source_begin = tl.load(
            typed_pool_offsets + source_group
        )
        source_end = tl.load(
            typed_pool_offsets + source_group + 1
        )
        source_count = source_end - source_begin
        per_channel = (
            target_role_count * max_target_multiplicity
        )
        channel_index = candidate // per_channel
        candidate_remainder = candidate % per_channel
        target_role = (
            candidate_remainder // max_target_multiplicity
        )
        target_copy = (
            candidate_remainder % max_target_multiplicity
        )
        first_channel = channel_lower + (
            (source_rank - channel_lower) & (recurrent_stride - 1)
        )
        relative_channel = (
            first_channel + recurrent_stride * channel_index
        )
        mask &= (
            (channel_index < selected_channel_count)
            & (relative_channel < channel_upper)
        )
        if local_connections:
            target_tile = source_tile
            delay_steps = 4 + (relative_channel & 1)
        else:
            target_tile = (
                source_tile
                + tile_count // 2
                + relative_channel
                - channel_lower
            ) & tile_mask
            delay_steps = 6 + (relative_channel & 1)
        target_group = target_role * tile_count + target_tile
        target_begin = tl.load(
            typed_pool_offsets + target_group,
            mask=mask,
            other=0,
        )
        target_end = tl.load(
            typed_pool_offsets + target_group + 1,
            mask=mask,
            other=0,
        )
        target_count = target_end - target_begin
        target_rank_base = (
            source_rank
            - relative_channel
            - 1
            + source_count
        ) % source_count
        target_rank = (
            target_rank_base + target_copy * source_count
        )
        mask &= target_rank < target_count
        logical_target = tl.load(
            typed_pool_ids + target_begin + target_rank,
            mask=mask,
            other=0,
        )
        physical_target = tl.load(
            logical_to_physical + logical_target,
            mask=mask,
            other=0,
        )
        target_morphology = tl.load(
            morphology + physical_target,
            mask=mask,
            other=0,
        )
        channel = (
            relative_channel + source_role * channels_per_type
        )
        deliver = mask
        if external_collision_aware:
            future_step = current_step + delay_steps
            trajectory_index = (
                logical_target * 37 + 11
            ) % trajectory_count
            trajectory_time = (
                future_step
                + (
                    logical_target * 277 + external_seed
                )
                % trajectory_steps
            ) % trajectory_steps
            trajectory_offset = (
                (
                    source_role * trajectory_count
                    + trajectory_index
                )
                * trajectory_steps
                + trajectory_time
            )
            raw_count = tl.load(
                external_trajectories + trajectory_offset,
                mask=deliver,
                other=0,
            ).to(tl.float32)
            keep = tl.where(
                source_role != 0,
                external_inhibitory_gain,
                external_excitatory_gain,
            )
            expected_count = raw_count * keep
            integer_count = tl.floor(expected_count).to(tl.int32)
            fraction = expected_count - integer_count.to(tl.float32)
            hash_value = (
                logical_target.to(tl.uint32) * 1_103_515_245
                + (future_step + 1) * 12_345
                + source_role.to(tl.uint32) * 2_654_435_761
            ) & 0xFFFFFF
            draw = hash_value.to(tl.float32) / 16_777_216.0
            integer_count += (draw < fraction).to(tl.int32)
            integer_count = tl.minimum(
                integer_count, max_external_event_count
            )
            base_channel = (
                logical_target * 131
                + (future_step + 1) * 193
                + source_role * 977
            ) % channels_per_type
            external_offset = (
                (
                relative_channel
                - base_channel
                    + channels_per_type
                )
                * external_channel_stride_inverse
            ) % channels_per_type
            deliver &= external_offset >= integer_count
        slots = tl.load(
            slot_map + channel,
            mask=deliver,
            other=0,
        )
        gains = tl.load(
            morphology_gain + slots,
            mask=deliver,
            other=0.0,
        ).to(tl.float32)
        gains *= tl.load(
            feature_gain
            + target_morphology * slot_count
            + slots,
            mask=deliver,
            other=0.0,
        ).to(tl.float32)
        gains *= tl.where(source_role != 0, -1.0, 1.0)
        branch = slots // synapses_per_branch
        output_slot = (
            current_slot + delay_steps
        ) & delay_slot_mask
        branch_offset = (
            (output_slot * tile_count * spatial_tile_neurons
             + physical_target)
            * branches
            + branch
        )
        if record_touched:
            touched_index = (
                touched_base
                + source_row * candidate_count
                + candidate
            )
            tl.store(
                touched_offsets + touched_index,
                tl.where(deliver, branch_offset, -1),
                mask=candidate_mask,
            )
        previous_branch_value = tl.atomic_add(
            branch_queue + branch_offset,
            gains,
            mask=deliver,
        )
        if record_unique_rows:
            first_branch_delivery = deliver & (
                previous_branch_value == 0
            )
            previous_row_flag = tl.atomic_xchg(
                unique_row_flags + physical_target,
                1,
                mask=first_branch_delivery,
            )
            first_row_delivery = first_branch_delivery & (
                previous_row_flag == 0
            )
            unique_index = tl.atomic_add(
                unique_row_count
                + tl.zeros((block_size,), dtype=tl.int32),
                1,
                mask=first_row_delivery,
            )
            tl.store(
                unique_rows + unique_index,
                physical_target,
                mask=first_row_delivery,
            )

    @triton.jit
    def _binary_channels_to_branches_kernel(
        channel_bits,
        slot_to_channel,
        morphology,
        morphology_gain,
        feature_gain,
        output,
        population: tl.constexpr,
        channels_per_type: tl.constexpr,
        channel_words: tl.constexpr,
        morphology_classes: tl.constexpr,
        branches: tl.constexpr,
        synapses_per_branch: tl.constexpr,
        slot_count: tl.constexpr,
        block_rows: tl.constexpr,
        block_branches: tl.constexpr,
    ):
        rows = (
            tl.program_id(0) * block_rows
            + tl.arange(0, block_rows)
        )
        branch = (
            tl.program_id(1) * block_branches
            + tl.arange(0, block_branches)
        )
        row_mask = rows < population
        branch_mask = branch < branches
        matrix_mask = row_mask[:, None] & branch_mask[None, :]
        morphology_index = tl.load(
            morphology + rows,
            mask=row_mask,
            other=0,
        )
        values = tl.zeros(
            (block_rows, block_branches),
            dtype=tl.float32,
        )
        for synapse in tl.static_range(0, synapses_per_branch):
            slots = branch * synapses_per_branch + synapse
            channels = tl.load(
                slot_to_channel + slots,
                mask=branch_mask,
                other=-1,
            )
            valid = matrix_mask & (channels[None, :] >= 0)
            words = channels >> 5
            packed = tl.load(
                channel_bits
                + rows[:, None] * channel_words
                + words[None, :],
                mask=valid,
                other=0,
            )
            active = (
                (packed >> (channels[None, :] & 31)) & 1
            ).to(tl.float32)
            gains = tl.load(
                morphology_gain + slots,
                mask=branch_mask,
                other=0.0,
            ).to(tl.float32)
            feature_offsets = (
                morphology_index[:, None] * slot_count
                + slots[None, :]
            )
            gains = gains[None, :] * tl.load(
                feature_gain + feature_offsets,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            signs = tl.where(
                channels[None, :] < channels_per_type,
                1.0,
                -1.0,
            )
            values += active * gains * signs
        output_offsets = rows[:, None] * branches + branch[None, :]
        tl.store(
            output + output_offsets,
            values,
            mask=matrix_mask,
        )

    @triton.jit
    def _external_branch_lookup_kernel(
        logical_ids,
        trajectory_indices,
        trajectory_offsets,
        morphology,
        trajectories,
        branch_bank,
        existing,
        output,
        step,
        excitatory_gain,
        inhibitory_gain,
        population: tl.constexpr,
        trajectory_count: tl.constexpr,
        trajectory_steps: tl.constexpr,
        morphology_classes: tl.constexpr,
        channels_per_type: tl.constexpr,
        count_values: tl.constexpr,
        branches: tl.constexpr,
        add_existing: tl.constexpr,
        block_rows: tl.constexpr,
        block_branches: tl.constexpr,
    ):
        rows = (
            tl.program_id(0) * block_rows
            + tl.arange(0, block_rows)
        )
        branch = (
            tl.program_id(1) * block_branches
            + tl.arange(0, block_branches)
        )
        row_mask = rows < population
        branch_mask = branch < branches
        matrix_mask = row_mask[:, None] & branch_mask[None, :]
        logical = tl.load(
            logical_ids + rows,
            mask=row_mask,
            other=0,
        )
        trajectory_index = tl.load(
            trajectory_indices + rows,
            mask=row_mask,
            other=0,
        )
        trajectory_time = (
            step
            + tl.load(
                trajectory_offsets + rows,
                mask=row_mask,
                other=0,
            )
        ) % trajectory_steps
        morphology_index = tl.load(
            morphology + rows,
            mask=row_mask,
            other=0,
        )
        output_offset = rows[:, None] * branches + branch[None, :]
        if add_existing:
            values = tl.load(
                existing + output_offset,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            values = tl.zeros(
                (block_rows, block_branches),
                dtype=tl.float32,
            )
        logical_u32 = logical.to(tl.uint32)
        step_u32 = step + 1
        for source_type in tl.static_range(0, 2):
            trajectory_offset = (
                (
                    source_type * trajectory_count
                    + trajectory_index
                )
                * trajectory_steps
                + trajectory_time
            )
            raw_count = tl.load(
                trajectories + trajectory_offset,
                mask=row_mask,
                other=0,
            ).to(tl.float32)
            if source_type == 0:
                keep = excitatory_gain
            else:
                keep = inhibitory_gain
            expected_count = raw_count * keep
            integer_count = tl.floor(expected_count).to(tl.int32)
            fraction = expected_count - integer_count.to(tl.float32)
            hash_value = (
                logical_u32 * 1_103_515_245
                + step_u32 * 12_345
                + source_type * 2_654_435_761
            ) & 0xFFFFFF
            draw = hash_value.to(tl.float32) / 16_777_216.0
            integer_count += (draw < fraction).to(tl.int32)
            integer_count = tl.minimum(
                integer_count, count_values - 1
            )
            base_channel = (
                logical * 131
                + (step + 1) * 193
                + source_type * 977
            ) % channels_per_type
            bank_row = (
                (
                    (
                        (
                            source_type * morphology_classes
                            + morphology_index
                        )
                        * channels_per_type
                        + base_channel
                    )
                    * count_values
                    + integer_count
                )
                * branches
            )
            bank_offset = bank_row[:, None] + branch[None, :]
            values += tl.load(
                branch_bank + bank_offset,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            output + output_offset,
            values,
            mask=matrix_mask,
        )
    @triton.jit
    def _external_recurrent_collision_kernel(
        logical_ids,
        trajectory_indices,
        trajectory_offsets,
        morphology,
        trajectories,
        recurrent_bits,
        slot_map,
        morphology_gain,
        feature_gain,
        output,
        step,
        excitatory_gain,
        inhibitory_gain,
        population: tl.constexpr,
        trajectory_count: tl.constexpr,
        trajectory_steps: tl.constexpr,
        channels_per_type: tl.constexpr,
        count_values: tl.constexpr,
        channel_words: tl.constexpr,
        branches: tl.constexpr,
        synapses_per_branch: tl.constexpr,
        slot_count: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        rows = (
            tl.program_id(0) * block_rows
            + tl.arange(0, block_rows)
        )
        row_mask = rows < population
        logical = tl.load(
            logical_ids + rows,
            mask=row_mask,
            other=0,
        )
        trajectory_index = tl.load(
            trajectory_indices + rows,
            mask=row_mask,
            other=0,
        )
        trajectory_time = (
            step
            + tl.load(
                trajectory_offsets + rows,
                mask=row_mask,
                other=0,
            )
        ) % trajectory_steps
        morphology_index = tl.load(
            morphology + rows,
            mask=row_mask,
            other=0,
        )
        logical_u32 = logical.to(tl.uint32)
        step_u32 = step + 1
        for source_type in tl.static_range(0, 2):
            trajectory_offset = (
                (
                    source_type * trajectory_count
                    + trajectory_index
                )
                * trajectory_steps
                + trajectory_time
            )
            raw_count = tl.load(
                trajectories + trajectory_offset,
                mask=row_mask,
                other=0,
            ).to(tl.float32)
            if source_type == 0:
                keep = excitatory_gain
            else:
                keep = inhibitory_gain
            expected_count = raw_count * keep
            integer_count = tl.floor(expected_count).to(tl.int32)
            fraction = expected_count - integer_count.to(tl.float32)
            hash_value = (
                logical_u32 * 1_103_515_245
                + step_u32 * 12_345
                + source_type * 2_654_435_761
            ) & 0xFFFFFF
            draw = hash_value.to(tl.float32) / 16_777_216.0
            integer_count += (draw < fraction).to(tl.int32)
            integer_count = tl.minimum(
                integer_count, count_values - 1
            )
            base_channel = (
                logical * 131
                + (step + 1) * 193
                + source_type * 977
            ) % channels_per_type
            for event_offset in tl.static_range(0, count_values - 1):
                active_event = row_mask & (event_offset < integer_count)
                channels = (
                    base_channel + event_offset * 127
                ) % channels_per_type
                channels += source_type * channels_per_type
                words = channels >> 5
                bits = 1 << (channels & 31)
                packed = tl.load(
                    recurrent_bits + rows * channel_words + words,
                    mask=active_event,
                    other=0,
                )
                collision = active_event & ((packed & bits) != 0)
                slots = tl.load(
                    slot_map + channels,
                    mask=collision,
                    other=0,
                )
                gains = tl.load(
                    morphology_gain + slots,
                    mask=collision,
                    other=0.0,
                ).to(tl.float32)
                gains *= tl.load(
                    feature_gain
                    + morphology_index * slot_count
                    + slots,
                    mask=collision,
                    other=0.0,
                ).to(tl.float32)
                if source_type == 0:
                    gains *= -1.0
                branch = slots // synapses_per_branch
                tl.atomic_add(
                    output + rows * branches + branch,
                    gains,
                    mask=collision,
                )


class TrajectoryExternalBranchDrive:
    """Exact native-time trajectory drive at the learned branch boundary."""

    def __init__(
        self,
        *,
        trajectories: torch.Tensor,
        branch_bank: torch.Tensor,
        logical_ids: torch.Tensor,
        morphology: torch.Tensor,
        seed: int,
        excitatory_gain: float,
        inhibitory_gain: float,
        slot_map: torch.Tensor | None = None,
        morphology_gain: torch.Tensor | None = None,
        feature_gain: torch.Tensor | None = None,
        synapses_per_branch: int | None = None,
        block_rows: int = 8,
        block_branches: int = 128,
    ) -> None:
        if triton is None:
            raise RuntimeError(
                "TrajectoryExternalBranchDrive requires Triton"
            )
        if trajectories.ndim != 3 or trajectories.shape[0] != 2:
            raise ValueError("trajectories must have shape (2, traces, time)")
        if branch_bank.ndim != 5 or branch_bank.shape[0] != 2:
            raise ValueError(
                "branch_bank must have shape (2, morphologies, channels, "
                "counts, branches)"
            )
        if logical_ids.ndim != 1 or logical_ids.dtype != torch.int32:
            raise ValueError("logical_ids must be an int32 vector")
        if morphology.shape != logical_ids.shape:
            raise ValueError("morphology must match logical_ids")
        if morphology.dtype != torch.int32:
            raise ValueError("morphology must use int32")
        collision_values = (
            slot_map,
            morphology_gain,
            feature_gain,
            synapses_per_branch,
        )
        if any(value is not None for value in collision_values) and not all(
            value is not None for value in collision_values
        ):
            raise ValueError(
                "collision correction requires slot_map, morphology_gain, "
                "feature_gain, and synapses_per_branch together"
            )
        tensors = [trajectories, branch_bank, logical_ids, morphology]
        tensors.extend(
            value
            for value in (slot_map, morphology_gain, feature_gain)
            if value is not None
        )
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise ValueError("external-drive tensors must share one CUDA device")
        if branch_bank.dtype != torch.float16:
            raise ValueError("branch_bank must use float16")
        if block_rows not in {1, 2, 4, 8, 16, 32}:
            raise ValueError(
                "block_rows must be one of 1, 2, 4, 8, 16, or 32"
            )
        if block_branches not in {32, 64, 128}:
            raise ValueError(
                "block_branches must be one of 32, 64, or 128"
            )
        if branch_bank.shape[1] <= int(morphology.max()):
            raise ValueError("branch_bank does not cover every morphology")
        if trajectories.shape[2] <= 0:
            raise ValueError("trajectory time dimension must be nonempty")
        if slot_map is not None:
            assert morphology_gain is not None
            assert feature_gain is not None
            assert synapses_per_branch is not None
            slot_count = branch_bank.shape[4] * synapses_per_branch
            if (
                slot_map.shape != (branch_bank.shape[2] * 2,)
                or slot_map.dtype != torch.int32
            ):
                raise ValueError(
                    "collision slot_map must cover all E/I channels"
                )
            if (
                morphology_gain.shape != (slot_count,)
                or morphology_gain.dtype != torch.float16
            ):
                raise ValueError(
                    "collision morphology_gain must cover every slot"
                )
            if (
                feature_gain.shape
                != (branch_bank.shape[1], slot_count)
                or feature_gain.dtype != torch.float16
            ):
                raise ValueError(
                    "collision feature_gain must cover morphologies and slots"
                )
        self.device = logical_ids.device
        self.trajectories = trajectories.to(
            dtype=torch.int16
        ).contiguous()
        self.branch_bank = branch_bank.contiguous()
        self.logical_ids = logical_ids.contiguous()
        self.morphology = morphology.contiguous()
        logical_long = self.logical_ids.long()
        self.trajectory_indices = (
            (logical_long * 37 + 11) % trajectories.shape[1]
        ).to(dtype=torch.int32)
        self.trajectory_offsets = (
            (logical_long * 277 + seed) % trajectories.shape[2]
        ).to(dtype=torch.int32)
        self.excitatory_gain = float(excitatory_gain)
        self.inhibitory_gain = float(inhibitory_gain)
        self.block_rows = block_rows
        self.block_branches = block_branches
        self.population = logical_ids.numel()
        self.morphology_classes = branch_bank.shape[1]
        self.channels_per_type = branch_bank.shape[2]
        self.count_values = branch_bank.shape[3]
        self.branches = branch_bank.shape[4]
        self.collision_slot_map = slot_map
        self.collision_morphology_gain = morphology_gain
        self.collision_feature_gain = feature_gain
        self.synapses_per_branch = synapses_per_branch

    def materialize(
        self,
        step: int,
        *,
        output: torch.Tensor | None = None,
        add_existing: bool = False,
        existing: torch.Tensor | None = None,
        recurrent_bits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Materialize exact branch currents for one native timestep."""

        if output is None:
            output = torch.empty(
                self.population,
                self.branches,
                device=self.device,
                dtype=torch.float16,
            )
        if (
            output.shape != (self.population, self.branches)
            or output.device != self.device
            or output.dtype
            not in {torch.float16, torch.float8_e4m3fn}
        ):
            raise ValueError(
                "output must be a CUDA float16 or float8_e4m3fn "
                "population-by-branch matrix"
            )
        if existing is not None:
            add_existing = True
        elif add_existing:
            existing = output
        else:
            existing = output
        if (
            existing.shape != output.shape
            or existing.device != self.device
            or existing.dtype
            not in {torch.float16, torch.float8_e4m3fn}
        ):
            raise ValueError(
                "existing branch currents must match the output matrix"
            )
        if recurrent_bits is not None:
            if self.collision_slot_map is None:
                raise ValueError(
                    "recurrent collision correction was not configured"
                )
            expected_words = (self.channels_per_type * 2 + 31) // 32
            if (
                recurrent_bits.shape
                != (self.population, expected_words)
                or recurrent_bits.device != self.device
                or recurrent_bits.dtype != torch.int32
            ):
                raise ValueError(
                    "recurrent_bits must be a packed int32 channel matrix"
                )
            if output.dtype != torch.float16:
                raise ValueError(
                    "collision-corrected output currently requires float16"
                )
        _external_branch_lookup_kernel[
            (
                triton.cdiv(self.population, self.block_rows),
                triton.cdiv(self.branches, self.block_branches),
            )
        ](
            self.logical_ids,
            self.trajectory_indices,
            self.trajectory_offsets,
            self.morphology,
            self.trajectories,
            self.branch_bank,
            existing,
            output,
            step,
            self.excitatory_gain,
            self.inhibitory_gain,
            population=self.population,
            trajectory_count=self.trajectories.shape[1],
            trajectory_steps=self.trajectories.shape[2],
            morphology_classes=self.morphology_classes,
            channels_per_type=self.channels_per_type,
            count_values=self.count_values,
            branches=self.branches,
            add_existing=add_existing,
            block_rows=self.block_rows,
            block_branches=self.block_branches,
            num_warps=4,
        )
        if recurrent_bits is not None:
            assert self.collision_slot_map is not None
            assert self.collision_morphology_gain is not None
            assert self.collision_feature_gain is not None
            assert self.synapses_per_branch is not None
            _external_recurrent_collision_kernel[
                (triton.cdiv(self.population, self.block_rows),)
            ](
                self.logical_ids,
                self.trajectory_indices,
                self.trajectory_offsets,
                self.morphology,
                self.trajectories,
                recurrent_bits,
                self.collision_slot_map,
                self.collision_morphology_gain,
                self.collision_feature_gain,
                output,
                step,
                self.excitatory_gain,
                self.inhibitory_gain,
                population=self.population,
                trajectory_count=self.trajectories.shape[1],
                trajectory_steps=self.trajectories.shape[2],
                channels_per_type=self.channels_per_type,
                count_values=self.count_values,
                channel_words=recurrent_bits.shape[1],
                branches=self.branches,
                synapses_per_branch=self.synapses_per_branch,
                slot_count=(
                    self.branches * self.synapses_per_branch
                ),
                block_rows=self.block_rows,
                num_warps=4,
            )
        return output


class ProceduralBinaryChannelConnectomeRouter:
    """Exact delayed routing in AxoBench's signed binary-channel space."""

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
        physical_to_logical: torch.Tensor | None = None,
        logical_to_physical: torch.Tensor | None = None,
        route_block_size: int = 512,
        route_num_warps: int = 4,
        branch_block_rows: int = 8,
        branch_block_size: int = 128,
    ) -> None:
        if triton is None:
            raise RuntimeError(
                "ProceduralBinaryChannelConnectomeRouter requires Triton"
            )
        if not _is_power_of_two(contract.delay_slots):
            raise ValueError("delay_slots must be a power of two")
        if not _is_power_of_two(contract.population_size):
            raise ValueError("population_size must be a power of two")
        if not _is_power_of_two(contract.spatial_tile_neurons):
            raise ValueError("spatial_tile_neurons must be a power of two")
        if contract.population_size % contract.spatial_tile_neurons:
            raise ValueError(
                "spatial tiles must evenly divide the population"
            )
        if contract.fanout % 2:
            raise ValueError("binary input channels require equal E/I halves")
        if synapses_per_branch <= 0:
            raise ValueError("synapses_per_branch must be positive")
        if route_block_size <= 0 or not _is_power_of_two(route_block_size):
            raise ValueError("route_block_size must be a power of two")
        if route_num_warps not in {1, 2, 4, 8}:
            raise ValueError(
                "route_num_warps must be one of 1, 2, 4, or 8"
            )
        if branch_block_rows not in {1, 2, 4, 8, 16, 32}:
            raise ValueError(
                "branch_block_rows must be one of 1, 2, 4, 8, 16, or 32"
            )
        if branch_block_size not in {32, 64, 128}:
            raise ValueError(
                "branch_block_size must be one of 32, 64, or 128"
            )

        self.contract = contract
        self.synapses_per_branch = synapses_per_branch
        self.slot_count = contract.branch_count * synapses_per_branch
        self.channels_per_type = contract.fanout // 2
        self.channel_words = (contract.fanout + 31) // 32
        self.local_fanout = round(
            contract.fanout * contract.local_connection_fraction
        )
        recurrent_stride = round(
            1.0 / contract.recurrent_contact_fraction
        )
        if abs(
            contract.recurrent_contact_fraction
            - 1.0 / recurrent_stride
        ) > 1e-9:
            raise ValueError(
                "procedural recurrent contact fraction must be the "
                "reciprocal of an integer"
            )
        self.recurrent_stride = recurrent_stride
        self.routed_fanout = (
            contract.fanout + recurrent_stride - 1
        ) // recurrent_stride
        if (
            contract.minimum_routing_delay_steps + 3
            >= contract.delay_slots
        ):
            raise ValueError(
                "delay_slots must exceed the largest procedural delay"
            )

        if (physical_to_logical is None) != (
            logical_to_physical is None
        ):
            raise ValueError(
                "both storage permutations must be provided together"
            )
        storage_reordered = physical_to_logical is not None
        if physical_to_logical is None:
            physical_to_logical = torch.arange(
                contract.population_size,
                device=slot_map.device,
                dtype=torch.int32,
            )
            logical_to_physical = physical_to_logical
        assert logical_to_physical is not None
        tensors = {
            "slot_map": slot_map,
            "morphology": morphology,
            "source_is_inhibitory": source_is_inhibitory,
            "morphology_gain": morphology_gain,
            "feature_gain": feature_gain,
            "physical_to_logical": physical_to_logical,
            "logical_to_physical": logical_to_physical,
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
            "physical_to_logical": (contract.population_size,),
            "logical_to_physical": (contract.population_size,),
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
        if (
            physical_to_logical.dtype != torch.int32
            or logical_to_physical.dtype != torch.int32
        ):
            raise ValueError("storage permutations must use int32")
        if int(slot_map.min()) < 0 or int(slot_map.max()) >= self.slot_count:
            raise ValueError("slot_map contains a slot outside the branch bank")
        if torch.unique(slot_map).numel() != contract.fanout:
            raise ValueError("slot_map must assign each channel once")

        self.device = slot_map.device
        self.storage_reordered = storage_reordered
        self.route_block_size = route_block_size
        self.route_num_warps = route_num_warps
        self.branch_block_rows = branch_block_rows
        self.branch_block_size = branch_block_size
        self.slot_map = slot_map.contiguous()
        self.morphology = morphology.contiguous()
        self.source_is_inhibitory = source_is_inhibitory.contiguous()
        self.morphology_gain = morphology_gain.contiguous()
        self.feature_gain = feature_gain.contiguous()
        self.physical_to_logical = physical_to_logical.contiguous()
        self.logical_to_physical = logical_to_physical.contiguous()
        self.slot_to_channel = torch.full(
            (self.slot_count,),
            -1,
            device=self.device,
            dtype=torch.int32,
        )
        self.slot_to_channel[self.slot_map.long()] = torch.arange(
            contract.fanout,
            device=self.device,
            dtype=torch.int32,
        )
        self.channel_queue = torch.zeros(
            contract.delay_slots,
            contract.population_size,
            self.channel_words,
            device=self.device,
            dtype=torch.int32,
        )

    @property
    def queue_bytes(self) -> int:
        return self.channel_queue.numel() * self.channel_queue.element_size()

    def current_channel_bits(self, current_slot: int) -> torch.Tensor:
        self._validate_slot(current_slot)
        return self.channel_queue[current_slot]

    def current_inputs(
        self,
        current_slot: int,
        *,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert one exact binary channel slot to learned branch currents."""

        self._validate_slot(current_slot)
        if output is None:
            output = torch.empty(
                self.contract.population_size,
                self.contract.branch_count,
                device=self.device,
                dtype=torch.float16,
            )
        if (
            output.shape
            != (
                self.contract.population_size,
                self.contract.branch_count,
            )
            or output.device != self.device
            or output.dtype
            not in {torch.float16, torch.float8_e4m3fn}
        ):
            raise ValueError(
                "output must be a CUDA float16 or float8_e4m3fn "
                "population-by-branch matrix"
            )
        _binary_channels_to_branches_kernel[
            (
                triton.cdiv(
                    self.contract.population_size,
                    self.branch_block_rows,
                ),
                triton.cdiv(
                    self.contract.branch_count,
                    self.branch_block_size,
                ),
            )
        ](
            self.channel_queue[current_slot],
            self.slot_to_channel,
            self.morphology,
            self.morphology_gain,
            self.feature_gain,
            output,
            population=self.contract.population_size,
            channels_per_type=self.channels_per_type,
            channel_words=self.channel_words,
            morphology_classes=self.contract.morphology_classes,
            branches=self.contract.branch_count,
            synapses_per_branch=self.synapses_per_branch,
            slot_count=self.slot_count,
            block_rows=self.branch_block_rows,
            block_branches=self.branch_block_size,
            num_warps=4,
        )
        return output

    def clear_and_route(
        self,
        active_sources: torch.Tensor,
        *,
        current_slot: int,
        clear_consumed: bool = True,
    ) -> torch.Tensor:
        """Clear a consumed slot and OR new delayed recurrent channels."""

        self._validate_slot(current_slot)
        if (
            active_sources.ndim != 1
            or active_sources.dtype != torch.int32
            or active_sources.device != self.device
        ):
            raise ValueError(
                "active_sources must be a CUDA int32 vector"
            )
        if clear_consumed:
            self.channel_queue[current_slot].zero_()
        active_count = active_sources.numel()
        if active_count == 0:
            return self.channel_queue
        _binary_delay_block_kernel[
            (
                active_count,
                triton.cdiv(
                    self.routed_fanout,
                    self.route_block_size,
                ),
            )
        ](
            active_sources,
            self.physical_to_logical,
            self.logical_to_physical,
            self.source_is_inhibitory,
            self.morphology,
            self.channel_queue,
            active_count,
            current_slot=current_slot,
            fanout=self.contract.fanout,
            channels_per_type=self.channels_per_type,
            routed_fanout=self.routed_fanout,
            recurrent_stride=self.recurrent_stride,
            local_fanout=self.local_fanout,
            minimum_delay_steps=(
                self.contract.minimum_routing_delay_steps
            ),
            delay_slot_mask=self.contract.delay_slots - 1,
            population=self.contract.population_size,
            population_mask=self.contract.population_size - 1,
            spatial_tile_mask=(
                self.contract.spatial_tile_neurons - 1
            ),
            channel_words=self.channel_words,
            storage_reordered=self.storage_reordered,
            block_size=self.route_block_size,
            num_warps=self.route_num_warps,
        )
        return self.channel_queue

    def _validate_slot(self, current_slot: int) -> None:
        if not 0 <= current_slot < self.contract.delay_slots:
            raise ValueError("current_slot is outside the delay ring")


class ProceduralExactBranchConnectomeRouter(
    ProceduralBinaryChannelConnectomeRouter
):
    """Sparse exact-OR routing with ready-to-consume branch currents."""

    def __init__(
        self,
        *args,
        external_trajectories: torch.Tensor | None = None,
        external_seed: int = 0,
        external_excitatory_gain: float = 1.0,
        external_inhibitory_gain: float = 1.0,
        max_external_event_count: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if external_trajectories is not None:
            if (
                external_trajectories.ndim != 3
                or external_trajectories.shape[0] != 2
                or external_trajectories.device != self.device
            ):
                raise ValueError(
                    "external_trajectories must have shape "
                    "(2, traces, time) on the routing device"
                )
            if max_external_event_count <= 0:
                raise ValueError(
                    "max_external_event_count must be positive"
                )
            try:
                stride_inverse = pow(
                    127, -1, self.channels_per_type
                )
            except ValueError as error:
                raise ValueError(
                    "external channel stride must be invertible"
                ) from error
            self.external_trajectories = external_trajectories.to(
                dtype=torch.int16
            ).contiguous()
            self.external_channel_stride_inverse = stride_inverse
        else:
            self.external_trajectories = None
            self.external_channel_stride_inverse = 1
        self.external_seed = int(external_seed)
        self.external_excitatory_gain = float(
            external_excitatory_gain
        )
        self.external_inhibitory_gain = float(
            external_inhibitory_gain
        )
        self.max_external_event_count = int(
            max_external_event_count
        )
        self.branch_queue = torch.zeros(
            self.contract.delay_slots,
            self.contract.population_size,
            self.contract.branch_count,
            device=self.device,
            dtype=torch.float16,
        )

    @property
    def queue_bytes(self) -> int:
        return (
            self.channel_queue.numel()
            * self.channel_queue.element_size()
            + self.branch_queue.numel()
            * self.branch_queue.element_size()
        )

    def current_inputs(
        self,
        current_slot: int,
        *,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_slot(current_slot)
        current = self.branch_queue[current_slot]
        if output is None:
            return current
        if (
            output.shape != current.shape
            or output.device != self.device
            or output.dtype
            not in {torch.float16, torch.float8_e4m3fn}
        ):
            raise ValueError(
                "output must be a CUDA float16 or float8_e4m3fn "
                "population-by-branch matrix"
            )
        output.copy_(current)
        return output

    def clear_and_route(
        self,
        active_sources: torch.Tensor,
        *,
        current_slot: int,
        current_step: int | None = None,
        clear_consumed: bool = True,
    ) -> torch.Tensor:
        """Clear consumed state and route each binary channel at most once."""

        self._validate_slot(current_slot)
        if (
            active_sources.ndim != 1
            or active_sources.dtype != torch.int32
            or active_sources.device != self.device
        ):
            raise ValueError(
                "active_sources must be a CUDA int32 vector"
            )
        if clear_consumed:
            self.channel_queue[current_slot].zero_()
            self.branch_queue[current_slot].zero_()
        if current_step is None:
            current_step = current_slot
        active_count = active_sources.numel()
        if active_count == 0:
            return self.branch_queue
        _exact_branch_delay_block_kernel[
            (
                active_count,
                triton.cdiv(
                    self.routed_fanout,
                    self.route_block_size,
                ),
            )
        ](
            active_sources,
            self.physical_to_logical,
            self.logical_to_physical,
            self.source_is_inhibitory,
            self.slot_map,
            self.morphology,
            self.morphology_gain,
            self.feature_gain,
            (
                self.external_trajectories
                if self.external_trajectories is not None
                else self.channel_queue
            ),
            self.channel_queue,
            self.branch_queue,
            active_count,
            current_step,
            self.external_seed,
            self.external_excitatory_gain,
            self.external_inhibitory_gain,
            current_slot=current_slot,
            fanout=self.contract.fanout,
            channels_per_type=self.channels_per_type,
            routed_fanout=self.routed_fanout,
            recurrent_stride=self.recurrent_stride,
            local_fanout=self.local_fanout,
            minimum_delay_steps=(
                self.contract.minimum_routing_delay_steps
            ),
            delay_slot_mask=self.contract.delay_slots - 1,
            population=self.contract.population_size,
            population_mask=self.contract.population_size - 1,
            spatial_tile_mask=(
                self.contract.spatial_tile_neurons - 1
            ),
            channel_words=self.channel_words,
            branches=self.contract.branch_count,
            synapses_per_branch=self.synapses_per_branch,
            slot_count=self.slot_count,
            trajectory_count=(
                1
                if self.external_trajectories is None
                else self.external_trajectories.shape[1]
            ),
            trajectory_steps=(
                1
                if self.external_trajectories is None
                else self.external_trajectories.shape[2]
            ),
            max_external_event_count=self.max_external_event_count,
            external_channel_stride_inverse=(
                self.external_channel_stride_inverse
            ),
            external_collision_aware=(
                self.external_trajectories is not None
            ),
            storage_reordered=self.storage_reordered,
            block_size=self.route_block_size,
            num_warps=self.route_num_warps,
        )
        return self.branch_queue


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
        physical_to_logical: torch.Tensor | None = None,
        logical_to_physical: torch.Tensor | None = None,
        route_block_size: int = 512,
        route_num_warps: int = 4,
    ) -> None:
        if triton is None:
            raise RuntimeError(
                "ProceduralMorphologyConnectomeRouter requires Triton"
            )
        if not _is_power_of_two(contract.delay_slots):
            raise ValueError("delay_slots must be a power of two")
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
        if not _is_power_of_two(route_block_size):
            raise ValueError("route_block_size must be a power of two")
        if route_num_warps not in {1, 2, 4, 8}:
            raise ValueError(
                "route_num_warps must be one of 1, 2, 4, or 8"
            )
        self.contract = contract
        self.synapses_per_branch = synapses_per_branch
        self.route_block_size = route_block_size
        self.route_num_warps = route_num_warps
        self.slot_count = contract.branch_count * synapses_per_branch
        self.local_fanout = round(
            contract.fanout * contract.local_connection_fraction
        )
        recurrent_stride = round(
            1.0 / contract.recurrent_contact_fraction
        )
        if abs(
            contract.recurrent_contact_fraction
            - 1.0 / recurrent_stride
        ) > 1e-9:
            raise ValueError(
                "procedural recurrent contact fraction must be the "
                "reciprocal of an integer"
            )
        self.recurrent_stride = recurrent_stride
        self.routed_fanout = (
            contract.fanout + recurrent_stride - 1
        ) // recurrent_stride
        self.delay_buckets = procedural_delay_buckets(
            fanout=contract.fanout,
            local_fanout=self.local_fanout,
            minimum_delay_steps=contract.minimum_routing_delay_steps,
        )
        if max(
            bucket.delay_steps for bucket in self.delay_buckets
        ) >= contract.delay_slots:
            raise ValueError(
                "delay_slots must exceed the largest procedural delay"
            )

        if (physical_to_logical is None) != (
            logical_to_physical is None
        ):
            raise ValueError(
                "both storage permutations must be provided together"
            )
        storage_reordered = physical_to_logical is not None
        if physical_to_logical is None:
            physical_to_logical = torch.arange(
                contract.population_size,
                device=slot_map.device,
                dtype=torch.int32,
            )
            logical_to_physical = physical_to_logical
        assert logical_to_physical is not None
        tensors = {
            "slot_map": slot_map,
            "morphology": morphology,
            "source_is_inhibitory": source_is_inhibitory,
            "morphology_gain": morphology_gain,
            "feature_gain": feature_gain,
            "physical_to_logical": physical_to_logical,
            "logical_to_physical": logical_to_physical,
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
        if storage_reordered:
            expected_shapes.update(
                {
                    "physical_to_logical": (
                        contract.population_size,
                    ),
                    "logical_to_physical": (
                        contract.population_size,
                    ),
                }
            )
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
        if storage_reordered and (
            physical_to_logical.dtype != torch.int32
            or logical_to_physical.dtype != torch.int32
        ):
            raise ValueError("storage permutations must use int32")

        self.device = slot_map.device
        self.storage_reordered = storage_reordered
        self.slot_map = slot_map.contiguous()
        self.morphology = morphology.contiguous()
        self.source_is_inhibitory = (
            source_is_inhibitory.contiguous()
        )
        self.morphology_gain = morphology_gain.contiguous()
        self.feature_gain = feature_gain.contiguous()
        self.physical_to_logical = (
            physical_to_logical.contiguous()
        )
        self.logical_to_physical = (
            logical_to_physical.contiguous()
        )
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
        clear_consumed: bool = True,
        touched_offsets: torch.Tensor | None = None,
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
        if clear_consumed:
            self.delay_queue[current_slot].zero_()
        active_count = active_sources.numel()
        if active_count == 0:
            return self.delay_queue
        if self.recurrent_stride != 1:
            record_touched = touched_offsets is not None
            if touched_offsets is None:
                touched_offsets = active_sources
            elif (
                touched_offsets.device != self.device
                or touched_offsets.dtype != torch.int32
                or touched_offsets.ndim != 1
                or touched_offsets.numel()
                < active_count * self.routed_fanout
            ):
                raise ValueError(
                    "touched_offsets must be a sufficiently large CUDA "
                    "int32 vector"
                )
            block_size = self.route_block_size
            _delay_block_kernel[
                (
                    active_count,
                    triton.cdiv(
                        self.routed_fanout,
                        block_size,
                    ),
                )
            ](
                active_sources,
                self.physical_to_logical,
                self.logical_to_physical,
                self.source_is_inhibitory,
                self.slot_map,
                self.morphology,
                self.morphology_gain,
                self.feature_gain,
                self.delay_queue,
                touched_offsets,
                active_count,
                current_slot=current_slot,
                fanout=self.contract.fanout,
                routed_fanout=self.routed_fanout,
                recurrent_stride=self.recurrent_stride,
                local_fanout=self.local_fanout,
                minimum_delay_steps=(
                    self.contract.minimum_routing_delay_steps
                ),
                delay_slot_mask=self.contract.delay_slots - 1,
                population=self.contract.population_size,
                population_mask=self.contract.population_size - 1,
                spatial_tile_mask=(
                    self.contract.spatial_tile_neurons - 1
                ),
                num_branch=self.contract.branch_count,
                synapses_per_branch=self.synapses_per_branch,
                slot_count=self.slot_count,
                storage_reordered=self.storage_reordered,
                record_touched=record_touched,
                block_size=block_size,
                num_warps=self.route_num_warps,
            )
            return self.delay_queue
        block_size = self.route_block_size
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
                self.physical_to_logical,
                self.logical_to_physical,
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
                storage_reordered=self.storage_reordered,
                block_size=block_size,
                num_warps=self.route_num_warps,
            )
        return self.delay_queue

    def clear_recorded_branches(
        self,
        touched_offsets: torch.Tensor,
    ) -> None:
        """Clear branch queue destinations recorded by event delivery."""

        if (
            touched_offsets.device != self.device
            or touched_offsets.dtype != torch.int32
            or touched_offsets.ndim != 1
        ):
            raise ValueError(
                "touched_offsets must be a CUDA int32 vector"
            )
        block_size = 256
        _clear_touched_kernel[
            (triton.cdiv(touched_offsets.numel(), block_size),)
        ](
            touched_offsets,
            self.delay_queue,
            touched_offsets.numel(),
            block_size=block_size,
            num_warps=4,
        )

    def _validate_slot(self, current_slot: int) -> None:
        if not 0 <= current_slot < self.contract.delay_slots:
            raise ValueError("current_slot is outside the delay ring")


class ProceduralUniqueChannelConnectomeRouter(
    ProceduralMorphologyConnectomeRouter
):
    """Collision-free typed fan-in with invertible event-wise routing."""

    def __init__(
        self,
        *args,
        external_trajectories: torch.Tensor | None = None,
        external_seed: int = 0,
        external_excitatory_gain: float = 1.0,
        external_inhibitory_gain: float = 1.0,
        max_external_event_count: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not _is_power_of_two(self.recurrent_stride):
            raise ValueError(
                "unique-channel routing requires a power-of-two recurrent "
                "stride"
            )
        self.channels_per_type = self.contract.fanout // 2
        if self.channels_per_type * 2 != self.contract.fanout:
            raise ValueError("typed channels require equal E/I halves")
        self.tile_count = (
            self.contract.population_size
            // self.contract.spatial_tile_neurons
        )
        if (
            not _is_power_of_two(self.tile_count)
            or self.tile_count < 2
        ):
            raise ValueError(
                "unique-channel long routing requires an even power-of-two "
                "tile count"
            )
        (
            self.typed_pool_ids,
            self.typed_pool_offsets,
            self.logical_type_rank,
        ) = build_typed_tile_pools(
            self.source_is_inhibitory,
            self.physical_to_logical,
            spatial_tile_neurons=self.contract.spatial_tile_neurons,
        )
        typed_counts = (
            self.typed_pool_offsets[1:]
            - self.typed_pool_offsets[:-1]
        )
        if int(typed_counts.min()) <= self.channels_per_type:
            raise ValueError(
                "every tile/type pool must exceed the typed channel count "
                "to guarantee self-edge-free cyclic fan-in"
            )
        self.max_target_multiplicity = int(
            torch.ceil(
                typed_counts.max().float()
                / typed_counts.min().float()
            )
        )
        if self.max_target_multiplicity > 5:
            raise ValueError(
                "typed population imbalance exceeds the supported 80/20 "
                "envelope"
            )
        self.local_channels_per_type = round(
            self.channels_per_type
            * self.contract.local_connection_fraction
        )
        if not 0 < self.local_channels_per_type < self.channels_per_type:
            raise ValueError(
                "unique-channel routing requires local and long channels"
            )
        if external_trajectories is not None:
            if (
                external_trajectories.ndim != 3
                or external_trajectories.shape[0] != 2
                or external_trajectories.device != self.device
            ):
                raise ValueError(
                    "external_trajectories must have shape "
                    "(2, traces, time) on the routing device"
                )
            try:
                stride_inverse = pow(
                    127, -1, self.channels_per_type
                )
            except ValueError as error:
                raise ValueError(
                    "external channel stride must be invertible"
                ) from error
            self.external_trajectories = external_trajectories.to(
                dtype=torch.int16
            ).contiguous()
            self.external_channel_stride_inverse = stride_inverse
        else:
            self.external_trajectories = None
            self.external_channel_stride_inverse = 1
        self.external_seed = int(external_seed)
        self.external_excitatory_gain = float(
            external_excitatory_gain
        )
        self.external_inhibitory_gain = float(
            external_inhibitory_gain
        )
        self.max_external_event_count = int(
            max_external_event_count
        )

    @property
    def queue_bytes(self) -> int:
        return self.delay_queue.numel() * self.delay_queue.element_size()

    @property
    def recorded_offsets_per_source(self) -> int:
        """Candidate offset slots needed to record one routed source."""

        total = 0
        for lower, upper in (
            (0, self.local_channels_per_type),
            (
                self.local_channels_per_type,
                self.channels_per_type,
            ),
        ):
            selected_channels = (
                upper - lower + self.recurrent_stride - 1
            ) // self.recurrent_stride
            total += (
                selected_channels
                * 2
                * self.max_target_multiplicity
            )
        return total

    def clear_and_route(
        self,
        active_sources: torch.Tensor,
        *,
        current_slot: int,
        current_step: int | None = None,
        clear_consumed: bool = True,
        touched_offsets: torch.Tensor | None = None,
        unique_rows: torch.Tensor | None = None,
        unique_row_flags: torch.Tensor | None = None,
        unique_row_count: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route the exact selected typed channel for each active source."""

        self._validate_slot(current_slot)
        if (
            active_sources.ndim != 1
            or active_sources.dtype != torch.int32
            or active_sources.device != self.device
        ):
            raise ValueError(
                "active_sources must be a CUDA int32 vector"
            )
        if clear_consumed:
            self.delay_queue[current_slot].zero_()
        if current_step is None:
            current_step = current_slot
        active_count = active_sources.numel()
        if active_count == 0:
            return self.delay_queue
        candidate_counts = []
        for lower, upper in (
            (0, self.local_channels_per_type),
            (
                self.local_channels_per_type,
                self.channels_per_type,
            ),
        ):
            selected_channels = (
                upper - lower + self.recurrent_stride - 1
            ) // self.recurrent_stride
            candidate_counts.append(
                selected_channels
                * 2
                * self.max_target_multiplicity
            )
        required_touched = active_count * sum(candidate_counts)
        if touched_offsets is not None and (
            touched_offsets.device != self.device
            or touched_offsets.dtype != torch.int32
            or touched_offsets.ndim != 1
            or touched_offsets.numel() < required_touched
        ):
            raise ValueError(
                "touched_offsets must be a sufficiently large CUDA "
                "int32 vector"
            )
        record_unique_rows = any(
            value is not None
            for value in (
                unique_rows,
                unique_row_flags,
                unique_row_count,
            )
        )
        if record_unique_rows and (
            unique_rows is None
            or unique_row_flags is None
            or unique_row_count is None
            or unique_rows.device != self.device
            or unique_rows.dtype != torch.int32
            or unique_rows.ndim != 1
            or unique_rows.numel() < self.contract.population_size
            or unique_row_flags.device != self.device
            or unique_row_flags.dtype != torch.int32
            or unique_row_flags.shape
            != (self.contract.population_size,)
            or unique_row_count.device != self.device
            or unique_row_count.dtype != torch.int32
            or unique_row_count.shape != (1,)
        ):
            raise ValueError(
                "unique row recording requires population-sized CUDA int32 "
                "rows/flags and one CUDA int32 count"
            )
        touched_base = 0
        for candidate_count, (
            local_connections,
            lower,
            upper,
        ) in zip(
            candidate_counts,
            (
            (True, 0, self.local_channels_per_type),
            (
                False,
                self.local_channels_per_type,
                self.channels_per_type,
            ),
            ),
            strict=True,
        ):
            selected_channels = (
                upper - lower + self.recurrent_stride - 1
            ) // self.recurrent_stride
            _unique_channel_branch_route_kernel[
                (
                    active_count,
                    triton.cdiv(
                        candidate_count,
                        self.route_block_size,
                    ),
                )
            ](
                active_sources,
                self.physical_to_logical,
                self.logical_to_physical,
                self.source_is_inhibitory,
                self.logical_type_rank,
                self.typed_pool_ids,
                self.typed_pool_offsets,
                self.slot_map,
                self.morphology,
                self.morphology_gain,
                self.feature_gain,
                (
                    self.external_trajectories
                    if self.external_trajectories is not None
                    else self.delay_queue
                ),
                self.delay_queue,
                (
                    touched_offsets
                    if touched_offsets is not None
                    else active_sources
                ),
                (
                    unique_rows
                    if unique_rows is not None
                    else active_sources
                ),
                (
                    unique_row_flags
                    if unique_row_flags is not None
                    else active_sources
                ),
                (
                    unique_row_count
                    if unique_row_count is not None
                    else active_sources
                ),
                active_count,
                current_step,
                self.external_seed,
                self.external_excitatory_gain,
                self.external_inhibitory_gain,
                current_slot=current_slot,
                channels_per_type=self.channels_per_type,
                channel_lower=lower,
                channel_upper=upper,
                selected_channel_count=selected_channels,
                recurrent_stride=self.recurrent_stride,
                target_role_count=2,
                max_target_multiplicity=(
                    self.max_target_multiplicity
                ),
                candidate_count=candidate_count,
                delay_slot_mask=self.contract.delay_slots - 1,
                spatial_tile_neurons=(
                    self.contract.spatial_tile_neurons
                ),
                tile_count=self.tile_count,
                tile_mask=self.tile_count - 1,
                branches=self.contract.branch_count,
                synapses_per_branch=self.synapses_per_branch,
                slot_count=self.slot_count,
                trajectory_count=(
                    1
                    if self.external_trajectories is None
                    else self.external_trajectories.shape[1]
                ),
                trajectory_steps=(
                    1
                    if self.external_trajectories is None
                    else self.external_trajectories.shape[2]
                ),
                max_external_event_count=(
                    self.max_external_event_count
                ),
                external_channel_stride_inverse=(
                    self.external_channel_stride_inverse
                ),
                external_collision_aware=(
                    self.external_trajectories is not None
                ),
                touched_base=touched_base,
                record_touched=touched_offsets is not None,
                record_unique_rows=record_unique_rows,
                local_connections=local_connections,
                block_size=self.route_block_size,
                num_warps=self.route_num_warps,
            )
            touched_base += active_count * candidate_count
        return self.delay_queue


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0
