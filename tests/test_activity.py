from dataclasses import replace

import pytest
import torch

import axosim
from axosim.activity import (
    FixedActivityThresholds,
    FixedBlockActivityThresholds,
    HomeostaticThresholdController,
    build_stratified_patch_phase_layout,
    threshold_block_model_activity,
    threshold_model_activity,
)
from axosim.simulation_contract import (
    MILLION_NEURON_REALTIME_CONTRACT,
)


def test_threshold_model_activity_uses_role_and_morphology() -> None:
    scores = torch.tensor(
        [
            [0.0, 0.6, 0.9, 0.1],
            [0.8, 0.4, 0.2, 1.1],
        ]
    )
    morphology = torch.tensor([0, 1, 0, 1])
    inhibitory = torch.tensor([False, False, True, True])
    thresholds = FixedActivityThresholds(
        excitatory=(0.5, 0.5),
        inhibitory=(0.8, 1.0),
    )

    activity = threshold_model_activity(
        scores,
        morphology,
        inhibitory,
        thresholds,
    )

    assert activity.tolist() == [
        [False, True, True, False],
        [True, False, False, True],
    ]


def test_threshold_activity_has_no_fixed_cardinality() -> None:
    scores = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    thresholds = FixedActivityThresholds(
        excitatory=(0.0,),
        inhibitory=(0.0,),
    )

    activity = threshold_model_activity(
        scores,
        torch.zeros(3, dtype=torch.long),
        torch.zeros(3, dtype=torch.bool),
        thresholds,
    )

    assert activity.sum(dim=-1).tolist() == [0, 2, 3]


def test_homeostatic_threshold_controller_opposes_group_rate_errors() -> None:
    controller = HomeostaticThresholdController(
        group_ids=torch.tensor([0, 0, 1, 1]),
        target_rates_hz=torch.tensor([10.0, 10.0]),
        dt_ms=10.0,
        update_interval_ms=100.0,
        time_constant_ms=100.0,
        learning_rate=0.1,
        max_abs_offset=0.5,
    )

    for step in range(10):
        updated = controller.observe(
            torch.tensor([True, True, False, False])
        )
        assert updated is (step == 9)

    assert controller.threshold_offsets[0] > 0
    assert controller.threshold_offsets[1] < 0
    assert controller.threshold_offsets.abs().max() <= 0.5
    assert torch.equal(
        controller.threshold_offsets_per_neuron(),
        controller.threshold_offsets[torch.tensor([0, 0, 1, 1])],
    )


def test_homeostatic_threshold_controller_uses_population_rate() -> None:
    controller = HomeostaticThresholdController(
        group_ids=torch.tensor([0, 0]),
        target_rates_hz=torch.tensor([5.0]),
        dt_ms=100.0,
        update_interval_ms=200.0,
        time_constant_ms=200.0,
        learning_rate=0.05,
        max_abs_offset=0.2,
    )

    controller.observe(torch.tensor([True, False]))
    controller.observe(torch.tensor([False, True]))

    assert controller.last_window_rates_hz.tolist() == pytest.approx([5.0])
    assert controller.threshold_offsets.tolist() == pytest.approx([0.0])


def test_homeostatic_threshold_controller_rejects_missing_groups() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        HomeostaticThresholdController(
            group_ids=torch.tensor([0, 2]),
            target_rates_hz=torch.tensor([5.0, 5.0, 5.0]),
            dt_ms=1.0,
            update_interval_ms=100.0,
            time_constant_ms=500.0,
            learning_rate=0.01,
            max_abs_offset=1.0,
        )


def test_homeostatic_threshold_controller_is_a_public_entrypoint() -> None:
    assert (
        axosim.HomeostaticThresholdController
        is HomeostaticThresholdController
    )


def test_block_thresholds_are_fixed_per_forecast_position() -> None:
    scores = torch.tensor(
        [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )
    thresholds = FixedBlockActivityThresholds(
        forecast_steps=(
            FixedActivityThresholds(
                excitatory=(0.4,),
                inhibitory=(0.4,),
            ),
            FixedActivityThresholds(
                excitatory=(0.6,),
                inhibitory=(0.6,),
            ),
        )
    )

    activity = threshold_block_model_activity(
        scores,
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.bool),
        thresholds,
    )

    assert activity.tolist() == [[True, True], [False, False]]


def test_threshold_activity_rejects_unknown_morphology() -> None:
    with pytest.raises(ValueError, match="unknown morphology"):
        threshold_model_activity(
            torch.zeros(2),
            torch.tensor([0, 1]),
            torch.zeros(2, dtype=torch.bool),
            FixedActivityThresholds(
                excitatory=(0.0,),
                inhibitory=(0.0,),
            ),
        )


def test_million_neuron_contract_requires_threshold_crossings() -> None:
    assert (
        MILLION_NEURON_REALTIME_CONTRACT.activity_selection_mode
        == "fixed_model_score_threshold_crossing"
    )
    with pytest.raises(ValueError, match="threshold crossings"):
        replace(
            MILLION_NEURON_REALTIME_CONTRACT,
            activity_selection_mode="tile_topk",
        )


def test_patch_phases_are_balanced_within_morphology_and_role() -> None:
    morphology = torch.tensor([0] * 17 + [1] * 19)
    inhibitory = torch.tensor(
        ([False] * 9 + [True] * 8)
        + ([False] * 10 + [True] * 9)
    )

    layout = build_stratified_patch_phase_layout(
        morphology,
        inhibitory,
        patch_size=4,
        seed=17,
    )

    assert layout.phase_sizes == (9, 9, 9, 9)
    assert torch.equal(
        layout.inverse_permutation[layout.storage_permutation],
        torch.arange(morphology.numel()),
    )
    for morphology_id in (0, 1):
        for role in (False, True):
            group = (morphology == morphology_id) & (inhibitory == role)
            counts = torch.bincount(
                layout.phase_ids[group],
                minlength=4,
            )
            assert int(counts.max() - counts.min()) <= 1


def test_patch_phase_layout_is_seeded_and_not_index_modulo() -> None:
    morphology = torch.arange(80).remainder(5)
    inhibitory = torch.arange(80).remainder(5) == 4

    first = build_stratified_patch_phase_layout(
        morphology,
        inhibitory,
        patch_size=4,
        seed=11,
    )
    repeat = build_stratified_patch_phase_layout(
        morphology,
        inhibitory,
        patch_size=4,
        seed=11,
    )
    changed = build_stratified_patch_phase_layout(
        morphology,
        inhibitory,
        patch_size=4,
        seed=12,
    )

    assert torch.equal(first.phase_ids, repeat.phase_ids)
    assert not torch.equal(first.phase_ids, changed.phase_ids)
    assert not torch.equal(
        first.phase_ids,
        torch.arange(80).remainder(4),
    )


def test_phase_tile_major_layout_preserves_contiguous_phase_cohorts() -> None:
    population = 160
    tile_ids = torch.arange(population).div(40, rounding_mode="floor")
    morphology = torch.arange(population).remainder(5)
    inhibitory = torch.arange(population).remainder(5) == 4

    layout = build_stratified_patch_phase_layout(
        morphology,
        inhibitory,
        patch_size=4,
        seed=23,
        tile_ids=tile_ids,
    )

    ordered_tiles = tile_ids[layout.storage_permutation]
    ordered_phases = layout.phase_ids[layout.storage_permutation]
    storage_group = ordered_phases * 4 + ordered_tiles
    assert torch.all(storage_group[1:] >= storage_group[:-1])
    assert layout.storage_group_shape == (4, 4)
    assert layout.phase_sizes == (40, 40, 40, 40)
    for phase in range(4):
        start = phase * 40
        stop = start + 40
        assert torch.all(ordered_phases[start:stop] == phase)
    assert all(
        stop - start == 10
        for start, stop in zip(
            layout.storage_group_offsets[:-1],
            layout.storage_group_offsets[1:],
            strict=True,
        )
    )
