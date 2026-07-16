from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from axosim.axobench_iteration import make_checkpoint_predictor
from axosim.axobench_iteration import main as evaluate_main
from axosim.checkpoint import save_checkpoint
from axosim.axomamba import AxoMambaConfig, AxoPyTorchMamba
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
    assert metadata["inference_dtype"] == "float32"


def test_checkpoint_predictor_supports_explicit_inference_dtype(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        BranchELM(BranchELMConfig(input_dim=4, memory_units=3, num_branches=2, hidden_units=4)),
        checkpoint,
    )

    predictor, metadata = make_checkpoint_predictor(
        checkpoint,
        device="cpu",
        dtype=torch.bfloat16,
    )
    output = predictor(np.zeros((2, 7, 4), dtype=np.float32))

    assert output.shape == (2, 7, 2)
    assert output.dtype == np.float32
    assert metadata["inference_dtype"] == "bfloat16"
    with pytest.raises(ValueError, match="inference dtype"):
        make_checkpoint_predictor(checkpoint, device="cpu", dtype="float64")


def test_checkpoint_predictor_maps_morphology_ids_for_conditioned_model(tmp_path) -> None:
    checkpoint = tmp_path / "conditioned.pt"
    model = AxoPyTorchMamba(
        AxoMambaConfig(
            num_input=4,
            num_branch=2,
            num_synapse_per_branch=2,
            model_dim=4,
            num_layers=1,
            state_dim=2,
            conv_kernel=2,
            expansion=2,
            morphology_ids=["specimen_a", "specimen_b"],
            morphology_embedding_scale=1.0,
        )
    )
    save_checkpoint(model, checkpoint)
    predictor, _ = make_checkpoint_predictor(checkpoint, device="cpu")

    output = predictor(
        np.zeros((2, 7, 4), dtype=np.int8),
        morphology_ids=["specimen_a", "specimen_b"],
    )

    assert output.shape == (2, 7, 2)
    assert predictor.supports_morphology_ids is True


def test_checkpoint_predictor_streaming_mode_matches_full_sequence(tmp_path) -> None:
    checkpoint = tmp_path / "conditioned.pt"
    model = AxoPyTorchMamba(
        AxoMambaConfig(
            num_input=4,
            num_branch=2,
            num_synapse_per_branch=2,
            model_dim=4,
            num_layers=1,
            state_dim=2,
            conv_kernel=2,
            expansion=2,
            morphology_ids=["specimen_a", "specimen_b"],
            morphology_embedding_scale=1.0,
        )
    ).eval()
    save_checkpoint(model, checkpoint)
    full_predictor, _ = make_checkpoint_predictor(checkpoint, device="cpu")
    streaming_predictor, metadata = make_checkpoint_predictor(
        checkpoint,
        device="cpu",
        streaming=True,
        streaming_chunk_size=1,
    )
    inputs = np.zeros((2, 7, 4), dtype=np.float32)
    morphology_ids = ["specimen_a", "specimen_b"]

    full = full_predictor(inputs, morphology_ids=morphology_ids)
    streamed = streaming_predictor(inputs, morphology_ids=morphology_ids)

    np.testing.assert_allclose(streamed, full, atol=2e-5, rtol=2e-5)
    assert metadata["streaming_inference"] is True
    assert metadata["streaming_chunk_size"] == 1


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


def test_cli_compares_candidate_with_baseline_on_paired_sample(tmp_path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    baseline = tmp_path / "baseline.pt"
    model = BranchELM(BranchELMConfig(input_dim=4, memory_units=3, num_branches=2, hidden_units=4))
    save_checkpoint(model, checkpoint)
    save_checkpoint(model, baseline)
    val = tmp_path / "dataset" / "val"
    val.mkdir(parents=True)
    inputs = np.zeros((8, 12, 4), dtype=np.float32)
    targets = np.zeros((8, 12, 2), dtype=np.float32)
    np.savez_compressed(val / "val.npz", inputs=inputs, targets=targets)
    output = tmp_path / "comparison.json"

    assert evaluate_main(
        [
            str(checkpoint),
            "--baseline-checkpoint",
            str(baseline),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--device",
            "cpu",
            "--max-samples",
            "8",
            "--max-interventions",
            "0",
            "--ignore-start",
            "0",
            "--bootstrap-replicates",
            "10",
            "--output",
            str(output),
        ]
    ) == 0

    comparison = json.loads(output.read_text(encoding="utf-8"))
    assert comparison["candidate"]["samples"] == comparison["baseline"]["samples"] == 8
    assert {row["assessment"] for row in comparison["comparison"]} == {"uncertain"}


def test_cli_saves_and_reuses_baseline_cache(tmp_path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    baseline = tmp_path / "baseline.pt"
    model = BranchELM(BranchELMConfig(input_dim=4, memory_units=3, num_branches=2, hidden_units=4))
    save_checkpoint(model, checkpoint)
    save_checkpoint(model, baseline)
    val = tmp_path / "dataset" / "val"
    val.mkdir(parents=True)
    inputs = np.zeros((8, 12, 4), dtype=np.float32)
    targets = np.zeros((8, 12, 2), dtype=np.float32)
    np.savez_compressed(val / "val.npz", inputs=inputs, targets=targets)
    cache = tmp_path / "baseline-cache.npz"

    assert evaluate_main(
        [
            str(baseline),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--device",
            "cpu",
            "--max-samples",
            "8",
            "--max-interventions",
            "0",
            "--ignore-start",
            "0",
            "--save-cache",
            str(cache),
        ]
    ) == 0
    assert cache.exists()

    output = tmp_path / "comparison.json"
    assert evaluate_main(
        [
            str(checkpoint),
            "--baseline-cache",
            str(cache),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--device",
            "cpu",
            "--max-samples",
            "8",
            "--max-interventions",
            "0",
            "--ignore-start",
            "0",
            "--bootstrap-replicates",
            "10",
            "--output",
            str(output),
        ]
    ) == 0

    comparison = json.loads(output.read_text(encoding="utf-8"))
    assert comparison["baseline_cache"] == str(cache.resolve())
    assert {row["assessment"] for row in comparison["comparison"]} == {"uncertain"}
