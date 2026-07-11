from __future__ import annotations

import json

import numpy as np

from axosim.axobench_iteration import make_checkpoint_predictor
from axosim.axobench_iteration import main as evaluate_main
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


def test_cli_saves_only_when_output_path_is_given(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        BranchELM(BranchELMConfig(input_dim=4, memory_units=3, num_branches=2, hidden_units=4)),
        checkpoint,
    )
    val = tmp_path / "dataset" / "val"
    val.mkdir(parents=True)
    inputs = np.zeros((4, 12, 4), dtype=np.float32)
    targets = np.zeros((4, 12, 2), dtype=np.float32)
    np.savez_compressed(val / "val.npz", inputs=inputs, targets=targets)

    assert evaluate_main(
        [
            str(checkpoint),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--device",
            "cpu",
            "--max-samples",
            "4",
            "--max-interventions",
            "0",
            "--ignore-start",
            "0",
        ]
    ) == 0
    assert not (tmp_path / "report.csv").exists()

    assert evaluate_main(
        [
            str(checkpoint),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--device",
            "cpu",
            "--max-samples",
            "4",
            "--max-interventions",
            "0",
            "--ignore-start",
            "0",
            "--output",
            str(tmp_path / "report.csv"),
        ]
    ) == 0
    assert (tmp_path / "report.csv").exists()
    assert (tmp_path / "report.csv").read_text(encoding="utf-8").startswith("model_id,")


def test_cli_selects_requested_morphology(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        BranchELM(BranchELMConfig(input_dim=4, memory_units=3, num_branches=2, hidden_units=4)),
        checkpoint,
    )
    val = tmp_path / "dataset" / "val"
    val.mkdir(parents=True)
    inputs = np.zeros((8, 12, 4), dtype=np.float32)
    targets = np.zeros((8, 12, 2), dtype=np.float32)
    sample_ids = np.asarray(
        [f"specimen_100:trace_{index}" for index in range(4)]
        + [f"specimen_200:trace_{index}" for index in range(4)]
    )
    np.savez_compressed(val / "val.npz", inputs=inputs, targets=targets, sample_ids=sample_ids)
    output = tmp_path / "report.json"

    assert evaluate_main(
        [
            str(checkpoint),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--device",
            "cpu",
            "--morphology-id",
            "specimen_200",
            "--max-samples",
            "4",
            "--max-interventions",
            "0",
            "--ignore-start",
            "0",
            "--output",
            str(output),
        ]
    ) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["morphology_mode"] == "single"
    assert report["morphology_ids"] == ["specimen_200"]
