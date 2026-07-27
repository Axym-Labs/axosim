from __future__ import annotations

import pytest
import torch

from axosim.checkpoint import load_checkpoint, save_checkpoint
from axosim.correction_mamba import (
    CompactCorrectionMamba,
    CorrectionMambaConfig,
)


def _model() -> CompactCorrectionMamba:
    return CompactCorrectionMamba(
        CorrectionMambaConfig(
            input_dim=16,
            model_dim=16,
            state_dim=8,
            history_blocks=4,
        )
    )


def test_correction_mamba_validates_history_shape():
    model = _model()

    with pytest.raises(ValueError, match="history"):
        model(torch.randn(2, 3, 16))


def test_correction_mamba_checkpoint_round_trip(tmp_path):
    model = _model()
    path = tmp_path / "corrector.pt"

    save_checkpoint(model, path, metadata={"test": True})
    restored, metadata = load_checkpoint(path)

    assert isinstance(restored, CompactCorrectionMamba)
    assert restored.config == model.config
    assert metadata["test"] is True
    for expected, actual in zip(
        model.state_dict().values(),
        restored.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_recurrent_replay_matches_parallel_mamba():
    model = _model().cuda().half().eval()
    history = torch.randn(8, 4, 16, device="cuda", dtype=torch.float16)
    state = model.allocate_replay_state(8, dtype=torch.float16)

    with torch.inference_mode():
        expected = model(history)
        observed = model.recurrent_replay(history, state)

    torch.testing.assert_close(
        observed.float(),
        expected.float(),
        atol=5e-2,
        rtol=5e-2,
    )
