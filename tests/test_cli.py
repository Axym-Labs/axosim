from pathlib import Path
import json

import torch

from axosim.cli import main
from axosim.checkpoint import load_checkpoint
from axosim.experiments import apply_preset
from axosim.official import create_official_elm, load_official_elm_config


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
    assert main(
        [
            "evaluate",
            "--preset",
            "legacy-baseline-smoke",
            "--data",
            str(data_dir),
            "--input-dim",
            "6",
            "--memory-units",
            "4",
            "--branches",
            "2",
            "--output",
            str(metrics_path),
        ]
    ) == 0

    assert Path(metrics_path).exists()


def test_cli_evaluate_accepts_window_options(tmp_path):
    data_dir = tmp_path / "data"
    metrics_path = tmp_path / "metrics.json"

    main(["make-demo-data", "--output", str(data_dir), "--samples", "2", "--time-steps", "10", "--input-dim", "6"])

    result = main(
        [
            "evaluate",
            "--preset",
            "legacy-baseline-smoke",
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
            "--model-kind",
            "baseline",
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


def test_cli_inference_benchmark_writes_deployment_report(tmp_path):
    report_path = tmp_path / "inference-benchmark.json"

    assert main(
        [
            "inference-benchmark",
            "--model-kind",
            "baseline",
            "--input-dim",
            "6",
            "--memory-units",
            "4",
            "--branches",
            "2",
            "--batch-sizes",
            "1,2",
            "--time-steps",
            "3",
            "--runs",
            "1",
            "--warmup-runs",
            "0",
            "--output",
            str(report_path),
        ]
    ) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["model_parameter_count"] > 0
    assert len(report["measurements"]) == 2
    assert report["measurements"][0]["neuron_steps_per_second"] > 0


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
            "--registry",
            "off",
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


def test_cli_preset_and_registry_write_jsonl(tmp_path):
    data_dir = tmp_path / "data"
    metrics_path = tmp_path / "metrics.json"
    registry_path = tmp_path / "registry.jsonl"
    config_path = tmp_path / "official.json"
    _write_small_official_config(config_path)

    main(["make-demo-data", "--output", str(data_dir), "--samples", "2", "--time-steps", "8", "--input-dim", "6"])

    result = main(
        [
            "evaluate",
            "--preset",
            "fulltrace-paper-metric",
            "--data",
            str(data_dir),
            "--model-config",
            str(config_path),
            "--metric-ignore-start",
            "1",
            "--registry",
            str(registry_path),
            "--output",
            str(metrics_path),
        ]
    )

    assert result == 0
    lines = registry_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["command"] == "evaluate"
    assert entry["preset"] == "fulltrace-paper-metric"


def test_mamba_presets_use_soma075_objective():
    class Args:
        spike_loss_weight = 0.5
        soma_loss_weight = 0.5

    for preset in ("mamba-official-probe", "mamba2-official-probe"):
        args = Args()
        args.preset = preset

        apply_preset("train", preset, args, set())

        assert args.spike_loss_weight == 0.25
        assert args.soma_loss_weight == 0.75


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
