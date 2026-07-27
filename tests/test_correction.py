from __future__ import annotations

import pytest
import torch

from axosim.correction import (
    apply_block_corrections,
    block_disagreement_scores,
    top_fraction_correction_indices,
    top_fraction_correction_mask,
)


def test_block_disagreement_separates_voltage_and_event_errors():
    teacher = torch.zeros(2, 8, 2)
    surrogate = teacher.clone()
    surrogate[0, 0:4, 1] = 0.1
    surrogate[1, 4:8, 0] = 2.0

    scores = block_disagreement_scores(
        surrogate,
        teacher,
        patch_size=4,
        spike_threshold=1.0,
    )

    assert scores.voltage_rmse_mv.shape == (2, 2)
    assert scores.event_mismatch_fraction.shape == (2, 2)
    torch.testing.assert_close(
        scores.voltage_rmse_mv,
        torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
    )
    torch.testing.assert_close(
        scores.event_mismatch_fraction,
        torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
    )
    assert scores.combined[0, 0] > 0
    assert scores.combined[1, 1] > 0


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [(0.0, 0), (0.25, 1), (0.5, 2), (1.0, 4)],
)
def test_top_fraction_mask_has_exact_global_cardinality(
    fraction,
    expected,
):
    scores = torch.tensor([[1.0, 4.0], [3.0, 2.0]])

    mask = top_fraction_correction_mask(scores, fraction)

    assert mask.dtype == torch.bool
    assert int(mask.sum()) == expected
    if expected:
        selected = scores[mask]
        rejected = scores[~mask]
        if rejected.numel():
            assert selected.min() >= rejected.max()


def test_top_fraction_indices_match_mask_and_budget():
    scores = torch.tensor([[1.0, 4.0], [3.0, 2.0]])

    indices = top_fraction_correction_indices(scores, 0.5)
    mask = top_fraction_correction_mask(scores, 0.5)

    assert indices.shape == (2,)
    assert torch.equal(
        indices.sort().values,
        torch.nonzero(mask.flatten(), as_tuple=False).flatten(),
    )


def test_apply_block_corrections_replaces_only_selected_blocks():
    surrogate = torch.zeros(1, 8, 2)
    teacher = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    mask = torch.tensor([[False, True]])

    hybrid = apply_block_corrections(
        surrogate,
        teacher,
        mask,
        patch_size=4,
    )

    torch.testing.assert_close(hybrid[:, :4], surrogate[:, :4])
    torch.testing.assert_close(hybrid[:, 4:], teacher[:, 4:])


def test_block_diagnostics_reject_incomplete_shapes():
    with pytest.raises(ValueError, match="divisible"):
        block_disagreement_scores(
            torch.zeros(1, 7, 2),
            torch.zeros(1, 7, 2),
            patch_size=4,
        )
