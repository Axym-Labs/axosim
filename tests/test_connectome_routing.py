from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from axosim.simulation_contract import MILLION_NEURON_REALTIME_CONTRACT


def test_delay_buckets_cover_every_fanout_offset_once() -> None:
    from axosim.connectome import procedural_delay_buckets

    buckets = procedural_delay_buckets(fanout=8, local_fanout=6)

    assert [(row.delay_steps, row.local_targets) for row in buckets] == [
        (1, True),
        (2, True),
        (3, False),
        (4, False),
    ]
    offsets = [
        offset
        for bucket in buckets
        for offset in bucket.fanout_offsets
    ]
    assert sorted(offsets) == list(range(8))
    assert len(offsets) == len(set(offsets))


def test_delay_buckets_respect_causal_block_horizon() -> None:
    from axosim.connectome import procedural_delay_buckets

    buckets = procedural_delay_buckets(
        fanout=8,
        local_fanout=6,
        minimum_delay_steps=4,
    )

    assert [row.delay_steps for row in buckets] == [4, 5, 6, 7]


def test_dense_spike_compaction_returns_sorted_int32_sources() -> None:
    from axosim.connectome import compact_active_sources

    outputs = torch.tensor([-0.2, 0.4, 0.1, -0.1, 2.0])

    active = compact_active_sources(outputs, threshold=0.15)

    assert active.dtype == torch.int32
    torch.testing.assert_close(active, torch.tensor([1, 4], dtype=torch.int32))


def test_unique_channel_route_keeps_dynamic_scalars_unspecialized() -> None:
    from axosim.connectome import _unique_channel_branch_route_kernel

    parameters = {
        parameter.name: parameter
        for parameter in _unique_channel_branch_route_kernel.params
    }

    for name in ("active_count", "current_step", "touched_base"):
        assert parameters[name].do_not_specialize


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton external branch drive requires CUDA",
)
def test_external_trajectory_drive_matches_dense_channel_reference() -> None:
    from axosim.connectome import (
        TrajectoryExternalBranchDrive,
        build_external_count_branch_bank,
    )

    slot_map = torch.tensor(
        [3, 0, 7, 2, 5, 1, 6, 4],
        device="cuda",
        dtype=torch.int32,
    )
    morphology_gain = torch.linspace(
        0.8, 1.2, 8, device="cuda", dtype=torch.float16
    )
    feature_gain = torch.stack(
        (torch.ones(8), torch.linspace(0.7, 1.3, 8))
    ).to(device="cuda", dtype=torch.float16)
    branch_bank = build_external_count_branch_bank(
        slot_map=slot_map,
        morphology_gain=morphology_gain,
        feature_gain=feature_gain,
        synapses_per_branch=2,
        max_event_count=8,
        channels_per_type=4,
    )
    trajectories = (
        torch.arange(2 * 3 * 20, device="cuda")
        .reshape(2, 3, 20)
        .remainder(6)
        .to(dtype=torch.int16)
    )
    logical_ids = torch.tensor(
        [3, 0, 1, 2], device="cuda", dtype=torch.int32
    )
    morphology = torch.tensor(
        [1, 0, 1, 0], device="cuda", dtype=torch.int32
    )
    seed = 73
    step = 7
    gains = (1.05, 0.6)
    drive = TrajectoryExternalBranchDrive(
        trajectories=trajectories,
        branch_bank=branch_bank,
        logical_ids=logical_ids,
        morphology=morphology,
        seed=seed,
        excitatory_gain=gains[0],
        inhibitory_gain=gains[1],
        block_rows=2,
    )

    actual = drive.materialize(step).float()

    expected = torch.zeros(4, 4, device="cuda")
    for physical_row, logical_id in enumerate(logical_ids.tolist()):
        morphology_index = int(morphology[physical_row])
        trajectory_index = (logical_id * 37 + 11) % 3
        trajectory_time = (
            step + (logical_id * 277 + seed) % 20
        ) % 20
        for source_type, keep in enumerate(gains):
            raw_count = int(
                trajectories[
                    source_type, trajectory_index, trajectory_time
                ]
            )
            count = int(raw_count * keep)
            fraction = raw_count * keep - count
            hashed = (
                logical_id * 1_103_515_245
                + (step + 1) * 12_345
                + source_type * 2_654_435_761
            ) & 0xFFFFFF
            count += int(hashed / float(0x1000000) < fraction)
            for event_offset in range(count):
                channel = (
                    logical_id * 131
                    + (step + 1) * 193
                    + event_offset * 127
                    + source_type * 977
                ) % 4
                channel += source_type * 4
                slot = int(slot_map[channel])
                value = (
                    morphology_gain[slot].float()
                    * feature_gain[morphology_index, slot].float()
                )
                if source_type:
                    value = -value
                expected[physical_row, slot // 2] += value

    torch.testing.assert_close(actual, expected, atol=4e-3, rtol=4e-3)
    fp8 = drive.materialize(
        step,
        output=torch.empty(
            4, 4, device="cuda", dtype=torch.float8_e4m3fn
        ),
    ).float()
    assert float((fp8 - expected).abs().mean()) < 0.04

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton external collision correction requires CUDA",
)
def test_external_drive_or_combines_recurrent_binary_channels() -> None:
    from axosim.connectome import (
        TrajectoryExternalBranchDrive,
        build_external_count_branch_bank,
    )

    slot_map = torch.tensor(
        [3, 0, 7, 2, 5, 1, 6, 4],
        device="cuda",
        dtype=torch.int32,
    )
    morphology_gain = torch.linspace(
        0.8, 1.2, 8, device="cuda", dtype=torch.float16
    )
    feature_gain = torch.stack(
        (torch.ones(8), torch.linspace(0.7, 1.3, 8))
    ).to(device="cuda", dtype=torch.float16)
    branch_bank = build_external_count_branch_bank(
        slot_map=slot_map,
        morphology_gain=morphology_gain,
        feature_gain=feature_gain,
        synapses_per_branch=2,
        max_event_count=2,
        channels_per_type=4,
    )
    trajectories = torch.ones(
        2, 3, 20, device="cuda", dtype=torch.int16
    )
    logical_ids = torch.tensor(
        [3, 0, 1, 2], device="cuda", dtype=torch.int32
    )
    morphology = torch.tensor(
        [1, 0, 1, 0], device="cuda", dtype=torch.int32
    )
    step = 7
    drive = TrajectoryExternalBranchDrive(
        trajectories=trajectories,
        branch_bank=branch_bank,
        logical_ids=logical_ids,
        morphology=morphology,
        seed=73,
        excitatory_gain=1.0,
        inhibitory_gain=1.0,
        slot_map=slot_map,
        morphology_gain=morphology_gain,
        feature_gain=feature_gain,
        synapses_per_branch=2,
        block_rows=2,
    )
    recurrent_bits = torch.zeros(
        4, 1, device="cuda", dtype=torch.int32
    )
    recurrent_channels = []
    external_channels = []
    for row, logical_id in enumerate(logical_ids.tolist()):
        external = {
            (
                logical_id * 131
                + (step + 1) * 193
                + source_type * 977
            )
            % 4
            + source_type * 4
            for source_type in range(2)
        }
        recurrent = set(external)
        recurrent.add(next(channel for channel in range(4) if channel not in external))
        external_channels.append(external)
        recurrent_channels.append(recurrent)
        packed = sum(1 << channel for channel in recurrent)
        recurrent_bits[row, 0] = packed

    existing = torch.zeros(4, 4, device="cuda", dtype=torch.float16)
    expected = torch.zeros(4, 4, device="cuda")
    for row in range(4):
        morphology_index = int(morphology[row])
        for channel in recurrent_channels[row]:
            slot = int(slot_map[channel])
            value = (
                morphology_gain[slot].float()
                * feature_gain[morphology_index, slot].float()
            )
            if channel >= 4:
                value = -value
            existing[row, slot // 2] += value
        for channel in recurrent_channels[row] | external_channels[row]:
            slot = int(slot_map[channel])
            value = (
                morphology_gain[slot].float()
                * feature_gain[morphology_index, slot].float()
            )
            if channel >= 4:
                value = -value
            expected[row, slot // 2] += value

    actual = drive.materialize(
        step,
        existing=existing,
        recurrent_bits=recurrent_bits,
    ).float()

    torch.testing.assert_close(
        actual,
        expected,
        atol=4e-3,
        rtol=4e-3,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton binary connectome router requires CUDA",
)
def test_binary_channel_router_preserves_or_semantics_and_typed_channels() -> None:
    from axosim.connectome import (
        ProceduralBinaryChannelConnectomeRouter,
    )

    contract = replace(
        MILLION_NEURON_REALTIME_CONTRACT,
        population_size=32,
        morphology_classes=2,
        firing_rate=2 / 32,
        fanout=8,
        recurrent_contact_fraction=1.0,
        local_connection_fraction=0.75,
        spatial_tile_neurons=8,
        branch_count=4,
    )
    slot_map = torch.tensor(
        [3, 0, 7, 2, 5, 1, 6, 4],
        device="cuda",
        dtype=torch.int32,
    )
    morphology = (
        torch.arange(32, device="cuda", dtype=torch.int32) % 2
    )
    source_is_inhibitory = torch.zeros(
        32, device="cuda", dtype=torch.uint8
    )
    source_is_inhibitory[2] = 1
    morphology_gain = torch.linspace(
        0.8, 1.2, 8, device="cuda", dtype=torch.float16
    )
    feature_gain = torch.stack(
        (torch.ones(8), torch.linspace(0.7, 1.3, 8))
    ).to(device="cuda", dtype=torch.float16)
    router = ProceduralBinaryChannelConnectomeRouter(
        contract,
        slot_map=slot_map,
        morphology=morphology,
        source_is_inhibitory=source_is_inhibitory,
        morphology_gain=morphology_gain,
        feature_gain=feature_gain,
        synapses_per_branch=2,
    )

    router.clear_and_route(
        torch.tensor([1, 2], device="cuda", dtype=torch.int32),
        current_slot=0,
    )

    expected_channels: dict[tuple[int, int], set[int]] = {}
    for source in (1, 2):
        inhibitory = bool(source_is_inhibitory[source])
        for fanout_offset in range(8):
            is_local = fanout_offset < 6
            if is_local:
                target = (
                    (source & ~7)
                    + ((source * 131 + fanout_offset * 127) & 7)
                )
                delay = 4 + (fanout_offset & 1)
            else:
                target = (
                    source * 413_293 + fanout_offset * 489_905
                ) & 31
                delay = 6 + (fanout_offset & 1)
            target_morphology = target % 2
            channel = (
                source * 37
                + fanout_offset * 97
                + target_morphology * 53
            ) % 4
            channel += int(inhibitory) * 4
            expected_channels.setdefault((delay, target), set()).add(
                channel
            )

    actual_bits = router.channel_queue.cpu()
    for (delay, target), channels in expected_channels.items():
        packed = int(actual_bits[delay, target, 0])
        decoded = {
            channel for channel in range(8) if packed & (1 << channel)
        }
        assert decoded == channels
        if 1 in (1, 2) and target >= 0:
            assert all(0 <= channel < 8 for channel in decoded)
    assert int(torch.count_nonzero(router.channel_queue)) == len(
        expected_channels
    )

    for delay in range(4, 8):
        actual = router.current_inputs(delay).float()
        expected = torch.zeros(32, 4, device="cuda")
        for (event_delay, target), channels in expected_channels.items():
            if event_delay != delay:
                continue
            target_morphology = int(morphology[target])
            for channel in channels:
                slot = int(slot_map[channel])
                value = (
                    morphology_gain[slot].float()
                    * feature_gain[target_morphology, slot].float()
                )
                if channel >= 4:
                    value = -value
                expected[target, slot // 2] += value
        torch.testing.assert_close(
            actual,
            expected,
            atol=4e-3,
            rtol=4e-3,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton exact branch router requires CUDA",
)
def test_exact_branch_router_matches_binary_reference_with_duplicate_events() -> None:
    from axosim.connectome import (
        ProceduralBinaryChannelConnectomeRouter,
        ProceduralExactBranchConnectomeRouter,
    )

    contract = replace(
        MILLION_NEURON_REALTIME_CONTRACT,
        population_size=32,
        morphology_classes=2,
        firing_rate=2 / 32,
        fanout=8,
        recurrent_contact_fraction=1.0,
        local_connection_fraction=0.75,
        spatial_tile_neurons=8,
        branch_count=4,
    )
    kwargs = {
        "slot_map": torch.tensor(
            [3, 0, 7, 2, 5, 1, 6, 4],
            device="cuda",
            dtype=torch.int32,
        ),
        "morphology": (
            torch.arange(32, device="cuda", dtype=torch.int32) % 2
        ),
        "source_is_inhibitory": torch.tensor(
            [0, 0, 1] + [0] * 29,
            device="cuda",
            dtype=torch.uint8,
        ),
        "morphology_gain": torch.linspace(
            0.8, 1.2, 8, device="cuda", dtype=torch.float16
        ),
        "feature_gain": torch.stack(
            (torch.ones(8), torch.linspace(0.7, 1.3, 8))
        ).to(device="cuda", dtype=torch.float16),
        "synapses_per_branch": 2,
    }
    binary = ProceduralBinaryChannelConnectomeRouter(
        contract, **kwargs
    )
    exact = ProceduralExactBranchConnectomeRouter(
        contract, **kwargs
    )
    active = torch.tensor(
        [1, 1, 2], device="cuda", dtype=torch.int32
    )

    binary.clear_and_route(active, current_slot=0)
    exact.clear_and_route(active, current_slot=0)

    torch.testing.assert_close(
        exact.channel_queue,
        binary.channel_queue,
    )
    for delay in range(4, 8):
        torch.testing.assert_close(
            exact.current_inputs(delay),
            binary.current_inputs(delay),
            atol=4e-3,
            rtol=4e-3,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton exact composed router requires CUDA",
)
def test_exact_router_prefilters_future_external_channel_collisions() -> None:
    from axosim.connectome import (
        ProceduralBinaryChannelConnectomeRouter,
        ProceduralExactBranchConnectomeRouter,
        TrajectoryExternalBranchDrive,
        build_external_count_branch_bank,
    )

    contract = replace(
        MILLION_NEURON_REALTIME_CONTRACT,
        population_size=32,
        morphology_classes=2,
        firing_rate=2 / 32,
        fanout=8,
        recurrent_contact_fraction=1.0,
        local_connection_fraction=0.75,
        spatial_tile_neurons=8,
        branch_count=4,
    )
    slot_map = torch.tensor(
        [3, 0, 7, 2, 5, 1, 6, 4],
        device="cuda",
        dtype=torch.int32,
    )
    morphology = (
        torch.arange(32, device="cuda", dtype=torch.int32) % 2
    )
    source_is_inhibitory = torch.tensor(
        [0, 0, 1] + [0] * 29,
        device="cuda",
        dtype=torch.uint8,
    )
    morphology_gain = torch.linspace(
        0.8, 1.2, 8, device="cuda", dtype=torch.float16
    )
    feature_gain = torch.stack(
        (torch.ones(8), torch.linspace(0.7, 1.3, 8))
    ).to(device="cuda", dtype=torch.float16)
    trajectories = torch.ones(
        2, 3, 20, device="cuda", dtype=torch.int16
    )
    kwargs = {
        "slot_map": slot_map,
        "morphology": morphology,
        "source_is_inhibitory": source_is_inhibitory,
        "morphology_gain": morphology_gain,
        "feature_gain": feature_gain,
        "synapses_per_branch": 2,
    }
    binary = ProceduralBinaryChannelConnectomeRouter(
        contract, **kwargs
    )
    exact = ProceduralExactBranchConnectomeRouter(
        contract,
        **kwargs,
        external_trajectories=trajectories,
        external_seed=73,
        external_excitatory_gain=1.0,
        external_inhibitory_gain=1.0,
        max_external_event_count=2,
    )
    active = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    binary.clear_and_route(active, current_slot=0)
    exact.clear_and_route(
        active,
        current_slot=0,
        current_step=0,
    )
    branch_bank = build_external_count_branch_bank(
        slot_map=slot_map,
        morphology_gain=morphology_gain,
        feature_gain=feature_gain,
        synapses_per_branch=2,
        max_event_count=2,
        channels_per_type=4,
    )
    drive = TrajectoryExternalBranchDrive(
        trajectories=trajectories,
        branch_bank=branch_bank,
        logical_ids=torch.arange(
            32, device="cuda", dtype=torch.int32
        ),
        morphology=morphology,
        seed=73,
        excitatory_gain=1.0,
        inhibitory_gain=1.0,
        block_rows=2,
    )

    for step in range(4, 8):
        for target in range(32):
            for source_type in range(2):
                channel = (
                    target * 131
                    + (step + 1) * 193
                    + source_type * 977
                ) % 4
                channel += source_type * 4
                binary.channel_queue[
                    step, target, 0
                ] |= 1 << channel
        expected = binary.current_inputs(step)
        actual = drive.materialize(
            step,
            existing=exact.branch_queue[step],
        )
        torch.testing.assert_close(
            actual,
            expected,
            atol=4e-3,
            rtol=4e-3,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton unique-channel router requires CUDA",
)
@pytest.mark.parametrize("contact_fraction", [0.25, 0.0625])
def test_unique_channel_router_matches_invertible_typed_fanin_reference(
    contact_fraction: float,
) -> None:
    from axosim.connectome import (
        ProceduralUniqueChannelConnectomeRouter,
    )

    contract = replace(
        MILLION_NEURON_REALTIME_CONTRACT,
        population_size=64,
        morphology_classes=2,
        firing_rate=1.0,
        fanout=8,
        recurrent_contact_fraction=contact_fraction,
        local_connection_fraction=0.75,
        spatial_tile_neurons=32,
        branch_count=4,
    )
    slot_map = torch.tensor(
        [3, 0, 7, 2, 5, 1, 6, 4],
        device="cuda",
        dtype=torch.int32,
    )
    morphology = (
        torch.arange(64, device="cuda", dtype=torch.int32) % 2
    )
    source_is_inhibitory = (
        torch.arange(64, device="cuda") % 4 == 0
    ).to(dtype=torch.uint8)
    morphology_gain = torch.linspace(
        0.8, 1.2, 8, device="cuda", dtype=torch.float16
    )
    feature_gain = torch.stack(
        (torch.ones(8), torch.linspace(0.7, 1.3, 8))
    ).to(device="cuda", dtype=torch.float16)
    router = ProceduralUniqueChannelConnectomeRouter(
        contract,
        slot_map=slot_map,
        morphology=morphology,
        source_is_inhibitory=source_is_inhibitory,
        morphology_gain=morphology_gain,
        feature_gain=feature_gain,
        synapses_per_branch=2,
    )

    router.clear_and_route(
        torch.arange(64, device="cuda", dtype=torch.int32),
        current_slot=0,
        current_step=0,
    )

    pool = router.typed_pool_ids.cpu()
    offsets = router.typed_pool_offsets.cpu()
    ranks = router.logical_type_rank.cpu()
    roles = source_is_inhibitory.cpu()
    expected = torch.zeros_like(router.delay_queue)
    target_channels = set()
    for target in range(64):
        target_tile = target // 32
        target_role = int(roles[target])
        target_rank = int(ranks[target])
        for source_role in range(2):
            for relative_channel in range(4):
                if relative_channel < 3:
                    source_tile = target_tile
                    delay = 4 + (relative_channel & 1)
                else:
                    source_tile = (
                        target_tile
                        - 1
                        - (relative_channel - 3)
                    ) & 1
                    delay = 6 + (relative_channel & 1)
                group = source_role * 2 + source_tile
                begin = int(offsets[group])
                end = int(offsets[group + 1])
                source_count = end - begin
                source_rank = (
                    target_rank + relative_channel + 1
                ) % source_count
                if (
                    relative_channel % router.recurrent_stride
                    != source_rank % router.recurrent_stride
                ):
                    continue
                source = int(pool[begin + source_rank])
                assert source != target
                channel = relative_channel + source_role * 4
                assert (target, channel) not in target_channels
                target_channels.add((target, channel))
                slot = int(slot_map[channel])
                value = (
                    morphology_gain[slot]
                    * feature_gain[int(morphology[target]), slot]
                )
                if source_role:
                    value = -value
                expected[delay, target, slot // 2] += value

    torch.testing.assert_close(
        router.delay_queue,
        expected,
        atol=4e-3,
        rtol=4e-3,
    )
    router.delay_queue.zero_()
    active = torch.arange(64, device="cuda", dtype=torch.int32)
    touched = torch.empty(
        active.numel() * router.recorded_offsets_per_source,
        device="cuda",
        dtype=torch.int32,
    )
    router.clear_and_route(
        active,
        current_slot=0,
        current_step=0,
        touched_offsets=touched,
    )
    torch.testing.assert_close(
        router.delay_queue,
        expected,
        atol=4e-3,
        rtol=4e-3,
    )
    router.clear_recorded_branches(touched)
    torch.testing.assert_close(
        router.delay_queue,
        torch.zeros_like(router.delay_queue),
    )
    unique_rows = torch.empty(
        contract.population_size,
        device="cuda",
        dtype=torch.int32,
    )
    unique_row_flags = torch.zeros_like(unique_rows)
    unique_row_count = torch.zeros(
        1, device="cuda", dtype=torch.int32
    )
    router.clear_and_route(
        active,
        current_slot=0,
        current_step=0,
        unique_rows=unique_rows,
        unique_row_flags=unique_row_flags,
        unique_row_count=unique_row_count,
    )
    expected_rows = torch.nonzero(
        router.delay_queue.abs().sum(dim=(0, 2)) > 0,
        as_tuple=False,
    ).flatten()
    recorded_rows = unique_rows[
        : int(unique_row_count.item())
    ].long()
    torch.testing.assert_close(
        recorded_rows.sort().values,
        expected_rows.sort().values,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton connectome router requires CUDA",
)
def test_procedural_router_matches_reference_delivery() -> None:
    from axosim.connectome import ProceduralMorphologyConnectomeRouter

    contract = replace(
        MILLION_NEURON_REALTIME_CONTRACT,
        population_size=32,
        morphology_classes=2,
        firing_rate=1 / 32,
        fanout=8,
        recurrent_contact_fraction=1.0,
        local_connection_fraction=0.75,
        spatial_tile_neurons=8,
        branch_count=4,
    )
    slot_map = torch.arange(8, device="cuda", dtype=torch.int32)
    morphology = (
        torch.arange(32, device="cuda", dtype=torch.int32) % 2
    )
    source_is_inhibitory = torch.zeros(
        32, device="cuda", dtype=torch.uint8
    )
    morphology_gain = torch.linspace(
        0.8, 1.2, 8, device="cuda", dtype=torch.float16
    )
    feature_gain = torch.stack(
        (torch.ones(8), torch.full((8,), 1.5))
    ).to(device="cuda", dtype=torch.float16)
    router = ProceduralMorphologyConnectomeRouter(
        contract,
        slot_map=slot_map,
        morphology=morphology,
        source_is_inhibitory=source_is_inhibitory,
        morphology_gain=morphology_gain,
        feature_gain=feature_gain,
        synapses_per_branch=2,
    )
    active = torch.tensor([1], device="cuda", dtype=torch.int32)

    router.clear_and_route(active, current_slot=0)

    expected = torch.zeros(
        8, 32, 4, device="cuda", dtype=torch.float16
    )
    source = 1
    for fanout_offset in range(8):
        is_local = fanout_offset < 6
        if is_local:
            target = (
                (source & ~7)
                + ((source * 131 + fanout_offset * 127) & 7)
            )
            delay = 4 + (fanout_offset & 1)
        else:
            target = (
                source * 413_293 + fanout_offset * 489_905
            ) & 31
            delay = 6 + (fanout_offset & 1)
        target_morphology = target % 2
        channel = (
            source * 37
            + fanout_offset * 97
            + target_morphology * 53
        ) % 8
        slot = int(slot_map[channel].item())
        gain = morphology_gain[slot] * feature_gain[
            target_morphology, slot
        ]
        expected[delay, target, slot // 2] += gain

    torch.testing.assert_close(
        router.delay_queue,
        expected,
        atol=2e-3,
        rtol=2e-3,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton connectome router requires CUDA",
)
def test_procedural_router_preserves_logical_routes_after_storage_reorder() -> None:
    from axosim.connectome import ProceduralMorphologyConnectomeRouter

    contract = replace(
        MILLION_NEURON_REALTIME_CONTRACT,
        population_size=32,
        morphology_classes=2,
        firing_rate=1 / 32,
        fanout=8,
        recurrent_contact_fraction=1.0,
        local_connection_fraction=0.75,
        spatial_tile_neurons=8,
        branch_count=4,
    )
    physical_to_logical = torch.tensor(
        [2, 0, 3, 1, 6, 4, 7, 5]
        + list(range(8, 32)),
        device="cuda",
        dtype=torch.int32,
    )
    logical_to_physical = torch.empty_like(physical_to_logical)
    logical_to_physical[physical_to_logical.long()] = torch.arange(
        32, device="cuda", dtype=torch.int32
    )
    logical_morphology = (
        torch.arange(32, device="cuda", dtype=torch.int32) % 2
    )
    morphology = logical_morphology[
        physical_to_logical.long()
    ].contiguous()
    router = ProceduralMorphologyConnectomeRouter(
        contract,
        slot_map=torch.arange(8, device="cuda", dtype=torch.int32),
        morphology=morphology,
        source_is_inhibitory=torch.zeros(
            32, device="cuda", dtype=torch.uint8
        ),
        morphology_gain=torch.ones(
            8, device="cuda", dtype=torch.float16
        ),
        feature_gain=torch.ones(
            2, 8, device="cuda", dtype=torch.float16
        ),
        synapses_per_branch=2,
        physical_to_logical=physical_to_logical,
        logical_to_physical=logical_to_physical,
    )
    physical_source = int(logical_to_physical[1])

    router.clear_and_route(
        torch.tensor(
            [physical_source], device="cuda", dtype=torch.int32
        ),
        current_slot=0,
    )

    expected = torch.zeros_like(router.delay_queue)
    for fanout_offset in range(8):
        if fanout_offset < 6:
            logical_target = (
                (1 & ~7) + ((1 * 131 + fanout_offset * 127) & 7)
            )
            delay = 4 + (fanout_offset & 1)
        else:
            logical_target = (
                1 * 413_293 + fanout_offset * 489_905
            ) & 31
            delay = 6 + (fanout_offset & 1)
        physical_target = int(logical_to_physical[logical_target])
        target_morphology = int(logical_morphology[logical_target])
        channel = (
            1 * 37
            + fanout_offset * 97
            + target_morphology * 53
        ) % 8
        expected[delay, physical_target, channel // 2] += 1
    torch.testing.assert_close(router.delay_queue, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton connectome router requires CUDA",
)
def test_sparse_touched_destinations_clear_routed_branch_values() -> None:
    from axosim.connectome import ProceduralMorphologyConnectomeRouter

    contract = replace(
        MILLION_NEURON_REALTIME_CONTRACT,
        population_size=32,
        morphology_classes=2,
        firing_rate=1 / 32,
        fanout=8,
        recurrent_contact_fraction=0.5,
        local_connection_fraction=0.75,
        spatial_tile_neurons=8,
        branch_count=4,
    )
    router = ProceduralMorphologyConnectomeRouter(
        contract,
        slot_map=torch.arange(8, device="cuda", dtype=torch.int32),
        morphology=(
            torch.arange(32, device="cuda", dtype=torch.int32) % 2
        ),
        source_is_inhibitory=torch.zeros(
            32, device="cuda", dtype=torch.uint8
        ),
        morphology_gain=torch.ones(
            8, device="cuda", dtype=torch.float16
        ),
        feature_gain=torch.ones(
            2, 8, device="cuda", dtype=torch.float16
        ),
        synapses_per_branch=2,
    )
    touched = torch.empty(
        router.routed_fanout,
        device="cuda",
        dtype=torch.int32,
    )

    router.clear_and_route(
        torch.tensor([1], device="cuda", dtype=torch.int32),
        current_slot=0,
        clear_consumed=False,
        touched_offsets=touched,
    )

    assert int((touched >= 0).sum()) == 4
    assert int(torch.count_nonzero(router.delay_queue)) > 0
    router.clear_recorded_branches(touched)
    assert int(torch.count_nonzero(router.delay_queue)) == 0
