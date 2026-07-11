from __future__ import annotations

import numpy as np

from axosim.axobench_iteration import make_checkpoint_predictor
from axosim.checkpoint import save_checkpoint
from axosim.model import BranchELM, BranchELMConfig


def test_checkpoint_predictor_is_batch_callable_and_returns_numpy(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        BranchELM(BranchELMConfig(input_dim=4, memory_units=3, num_branches=2, hidden_units=4)),
        checkpoint,
    )

    predictor, metadata = make_checkpoint_predictor(checkpoint, device="cpu")
    output = predictor(np.zeros((2, 7, 4), dtype=np.int8))

    assert output.shape == (2, 7, 2)
    assert output.dtype == np.float32
    assert metadata["model_kind"] == "baseline"
