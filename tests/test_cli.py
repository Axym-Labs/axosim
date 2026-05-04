from pathlib import Path

import torch

from bnn_sim.cli import main
from bnn_sim.checkpoint import load_checkpoint
from bnn_sim.official import create_official_elm, load_official_elm_config


def _write_small_official_config(path: Path) -> None:
    path.write_text(
        """{
  "input_to_synapse_routing": "neuronio_routing",
  "learn_memory_tau": false,
  "memory_tau_max": 150.0,
  "memory_tau_min": 1.0,
  "num_branch": 3,
  "num_input": 6,
  "num_memory": 4,
  "num_output": 2,
  "num_synapse_per_branch": 2
}
""",
        encoding="utf-8",
    )


def test_cli_smoke_data_and_evaluate(tmp_path):
    data_dir = tmp_path / "data"
    metrics_path = tmp_path / "metrics.json"

    assert main(["make-demo-data", "--output", str(data_dir), "--samples", "2", "--time-steps", "4", "--input-dim", "6"]) == 0
    assert main(["evaluate", "--data", str(data_dir), "--input-dim", "6", "--memory-units", "4", "--branches", "2", "--output", str(metrics_path)]) == 0

    assert Path(metrics_path).exists()


def test_cli_evaluate_accepts_window_options(tmp_path):
    data_dir = tmp_path / "data"
    metrics_path = tmp_path / "metrics.json"

    main(["make-demo-data", "--output", str(data_dir), "--samples", "2", "--time-steps", "10", "--input-dim", "6"])

    result = main(
        [
            "evaluate",
            "--data",
            str(data_dir),
            "--input-dim",
            "6",
            "--memory-units",
            "4",
            "--branches",
            "2",
            "--window-size",
            "4",
            "--window-stride",
            "4",
            "--blocking-transfer",
            "--output",
            str(metrics_path),
        ]
    )

    assert result == 0
    assert Path(metrics_path).exists()


def test_cli_benchmark_writes_json_report(tmp_path):
    report_path = tmp_path / "benchmark.json"

    assert main(
        [
            "benchmark",
            "--input-dim",
            "6",
            "--memory-units",
            "4",
            "--branches",
            "2",
            "--batch-size",
            "1",
            "--time-steps",
            "3",
            "--runs",
            "1",
            "--output",
            str(report_path),
        ]
    ) == 0
    assert Path(report_path).exists()


def test_cli_evaluate_accepts_official_model_config(tmp_path):
    data_dir = tmp_path / "data"
    metrics_path = tmp_path / "metrics.json"
    config_path = tmp_path / "official.json"
    _write_small_official_config(config_path)

    main(["make-demo-data", "--output", str(data_dir), "--samples", "2", "--time-steps", "4", "--input-dim", "6"])

    result = main(
        [
            "evaluate",
            "--data",
            str(data_dir),
            "--model-kind",
            "official",
            "--model-config",
            str(config_path),
            "--output",
            str(metrics_path),
        ]
    )

    assert result == 0
    assert metrics_path.exists()


def test_cli_train_saves_official_checkpoint(tmp_path):
    data_dir = tmp_path / "data"
    checkpoint_path = tmp_path / "official.pt"
    metrics_path = tmp_path / "metrics.json"
    config_path = tmp_path / "official.json"
    _write_small_official_config(config_path)

    main(["make-demo-data", "--output", str(data_dir), "--samples", "2", "--time-steps", "4", "--input-dim", "6"])

    result = main(
        [
            "train",
            "--data",
            str(data_dir),
            "--model-kind",
            "official",
            "--model-config",
            str(config_path),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--shuffle-mode",
            "shard",
            "--cache-shards",
            "1",
            "--checkpoint",
            str(checkpoint_path),
            "--metrics",
            str(metrics_path),
        ]
    )

    assert result == 0
    model, metadata = load_checkpoint(checkpoint_path)
    assert model.num_input == 6
    assert model.num_memory == 4
    assert metadata["model_kind"] == "official"


def test_cli_evaluate_loads_raw_official_state_dict(tmp_path):
    data_dir = tmp_path / "data"
    checkpoint_path = tmp_path / "raw-official.pt"
    metrics_path = tmp_path / "metrics.json"
    config_path = tmp_path / "official.json"
    _write_small_official_config(config_path)

    main(["make-demo-data", "--output", str(data_dir), "--samples", "2", "--time-steps", "4", "--input-dim", "6"])
    model = create_official_elm(load_official_elm_config(config_path))
    torch_state = model.state_dict()

    torch.save(torch_state, checkpoint_path)

    result = main(
        [
            "evaluate",
            "--data",
            str(data_dir),
            "--model-kind",
            "official",
            "--model-config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(metrics_path),
        ]
    )

    assert result == 0
    assert metrics_path.exists()


def test_cli_convert_neuronio_requires_existing_raw_path(tmp_path):
    output_dir = tmp_path / "out"

    result = main(["convert-neuronio", "--input", str(tmp_path / "missing"), "--output", str(output_dir)])

    assert result == 2


def test_cli_repack_shards_writes_sliceable_dataset(tmp_path):
    data_dir = tmp_path / "data"
    repacked_dir = tmp_path / "repacked"

    main(["make-demo-data", "--output", str(data_dir), "--samples", "2", "--time-steps", "4", "--input-dim", "6"])

    result = main(["repack-shards", "--input", str(data_dir), "--output", str(repacked_dir)])

    assert result == 0
    assert (repacked_dir / "manifest.json").exists()
    assert len(list(repacked_dir.glob("*_inputs.npy"))) == 1
