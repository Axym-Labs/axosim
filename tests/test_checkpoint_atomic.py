from __future__ import annotations

from pathlib import Path

import pytest

from axosim import checkpoint
from axosim.model_family import create_axosim_profile


def test_checkpoint_save_preserves_destination_on_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = create_axosim_profile(
        "axosim-gru-small",
        use_pytorch_fallback=True,
    )
    destination = tmp_path / "model.pt"
    destination.write_bytes(b"previous-valid-checkpoint")

    def interrupted_save(payload: object, path: Path) -> None:
        del payload
        Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(checkpoint.torch, "save", interrupted_save)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        checkpoint.save_checkpoint(model, destination)

    assert destination.read_bytes() == b"previous-valid-checkpoint"
    assert not (tmp_path / "model.pt.tmp").exists()
