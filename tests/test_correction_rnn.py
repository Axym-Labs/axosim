from __future__ import annotations

import pytest
import torch

from axosim.checkpoint import load_checkpoint, save_checkpoint
from axosim.correction_rnn import CompactCorrectionGRU, CorrectionGRUConfig


def _model() -> CompactCorrectionGRU:
    return CompactCorrectionGRU(
        CorrectionGRUConfig(
            input_dim=16,
            hidden_dim=12,
            history_blocks=4,
        )
    )


def test_correction_gru_predicts_one_native_block():
    model = _model()
    output = model(torch.randn(3, 4, 16))

    assert output.shape == (3, 4, 2)


def test_correction_gru_validates_history_shape():
    model = _model()

    with pytest.raises(ValueError, match="history"):
        model(torch.randn(2, 3, 16))


def test_correction_gru_checkpoint_round_trip(tmp_path):
    model = _model()
    path = tmp_path / "corrector-gru.pt"

    save_checkpoint(model, path, metadata={"test": True})
    restored, metadata = load_checkpoint(path)

    assert isinstance(restored, CompactCorrectionGRU)
    assert restored.config == model.config
    assert metadata["test"] is True
    for expected, actual in zip(
        model.state_dict().values(),
        restored.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
