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


def test_dense_spike_compaction_returns_sorted_int32_sources() -> None:
    from axosim.connectome import compact_active_sources

    outputs = torch.tensor([-0.2, 0.4, 0.1, -0.1, 2.0])

    active = compact_active_sources(outputs, threshold=0.15)

    assert active.dtype == torch.int32
    torch.testing.assert_close(active, torch.tensor([1, 4], dtype=torch.int32))


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
        4, 32, 4, device="cuda", dtype=torch.float16
    )
    source = 1
    for fanout_offset in range(8):
        is_local = fanout_offset < 6
        if is_local:
            target = (
                (source & ~7)
                + ((source * 131 + fanout_offset * 127) & 7)
            )
            delay = 1 + (fanout_offset & 1)
        else:
            target = (
                source * 413_293 + fanout_offset * 489_905
            ) & 31
            delay = 3 + (fanout_offset & 1)
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
        expected[delay % 4, target, slot // 2] += gain

    torch.testing.assert_close(
        router.delay_queue,
        expected,
        atol=2e-3,
        rtol=2e-3,
    )
