from pathlib import Path
import json

import numpy as np
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


def test_cli_prints_benchmark_axis_profile(capsys):
    assert main(["benchmark-profile", "--name", "paper-default", "--plan"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["profile"]["name"] == "paper-default"
    assert payload["diagnostic_state_evaluation"]["mode"] == "dense"
    assert (
        payload["morphology_intervention_sanity"]["selected_pairs"]
        < payload["morphology_intervention_sanity"]["full_cartesian_pairs"]
    )
    assert payload["capacity_estimate"]["within_target_wallclock"] is True


def test_cli_prints_benchmark_profile_cost(capsys):
    assert main(["benchmark-profile", "--name", "paper-default", "--cost"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["total_hours"] > 0
    assert payload["teacher_total_seconds"] > payload["model_eval_seconds"]


def test_cli_prints_dataset_schema(capsys):
    assert main(["dataset-schema", "--kind", "manifest"]) == 0
    manifest = json.loads(capsys.readouterr().out)

    assert manifest["schema_version"] == 2
    assert manifest["morphology_count"] == 5
    assert "interventions" in {entry["name"] for entry in manifest["categories"]}


def test_cli_prints_diagnostic_catalog(capsys):
    assert main(["dataset-schema", "--kind", "diagnostics"]) == 0
    catalog = json.loads(capsys.readouterr().out)

    assert catalog["plotter_policy"]["shape"] == "variable_length_list"
    assert "clipping_distortion" in catalog["suites"]["paper-summary"]


def test_cli_writes_teacher_qc_benchmark_rows(tmp_path, capsys):
    data_dir = tmp_path / "v1"
    output = tmp_path / "rows.jsonl"
    assert main(["make-demo-data", "--output", str(data_dir / "val"), "--samples", "2", "--time-steps", "8", "--input-dim", "4"]) == 0

    result = main(
        [
            "benchmark-rows",
            "--mode",
            "teacher-qc",
            "--dataset-root",
            str(data_dir),
            "--selector",
            "val",
            "--output",
            str(output),
            "--ignore-start",
            "0",
            "--morphology-id",
            "unit-morph",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] > 0
    assert payload["output"] == str(output)
    assert output.exists()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert any(row["diagnostic"] == "teacher_distribution_qc" for row in rows)
    assert all(row["morphology_id"] == "unit-morph" for row in rows)


def test_cli_writes_model_benchmark_rows(tmp_path, capsys):
    data_dir = tmp_path / "v1"
    output = tmp_path / "model_rows.json"
    summary = tmp_path / "model_summary.md"
    assert main(["make-demo-data", "--output", str(data_dir / "val"), "--samples", "2", "--time-steps", "8", "--input-dim", "4"]) == 0

    result = main(
        [
            "benchmark-rows",
            "--mode",
            "model",
            "--dataset-root",
            str(data_dir),
            "--selector",
            "val",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
            "--ignore-start",
            "0",
            "--suites",
            "trace-core,paper-summary",
            "--model-kind",
            "baseline",
            "--input-dim",
            "4",
            "--memory-units",
            "4",
            "--branches",
            "2",
            "--batch-size",
            "1",
            "--model-id",
            "unit-baseline",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] > 0
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert any(row["diagnostic"] == "trace_core" for row in rows)
    assert all(row["model_id"] == "unit-baseline" for row in rows)
    assert "Benchmark Diagnostic Summary" in summary.read_text(encoding="utf-8")


def test_cli_writes_intervention_benchmark_rows_from_pair_path(tmp_path, capsys):
    pair_path = tmp_path / "intervention_pair.npz"
    output = tmp_path / "intervention_rows.jsonl"
    baseline_inputs = np.zeros((1, 8, 4), dtype=np.int8)
    intervention_inputs = baseline_inputs.copy()
    intervention_inputs[:, 2, 0] = 1
    baseline_targets = np.zeros((1, 8, 2), dtype=np.float32)
    intervention_targets = baseline_targets.copy()
    baseline_targets[..., 1] = 1.0
    intervention_targets[..., 1] = 0.5
    np.savez_compressed(
        pair_path,
        baseline_inputs=baseline_inputs,
        baseline_targets=baseline_targets,
        intervention_inputs=intervention_inputs,
        intervention_targets=intervention_targets,
        sample_ids=np.asarray(["trace-a"]),
    )

    result = main(
        [
            "benchmark-rows",
            "--mode",
            "intervention",
            "--dataset-root",
            str(tmp_path / "unused-v1-root"),
            "--selector",
            "intervention",
            "--intervention",
            "event_dropout",
            "--pair-path",
            str(pair_path),
            "--output",
            str(output),
            "--ignore-start",
            "0",
            "--suites",
            "teacher-qc,intervention-response,paper-summary",
            "--morphology-id",
            "unit-morph",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] > 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert any(row["diagnostic"] == "intervention_response" for row in rows)
    assert any(row["metric"] == "teacher_direction" for row in rows)
    assert all(row["morphology_id"] == "unit-morph" for row in rows)
    assert all(row["condition"] == "interventions/event_dropout" for row in rows)
    assert all(row["intervention"] == "event_dropout" for row in rows)


def test_cli_runs_v1_benchmark_bundle(tmp_path, capsys):
    data_dir = tmp_path / "v1"
    output_dir = tmp_path / "bundle"
    pair_dir = tmp_path / "pilot" / "specimen_488683425_event_dropout_simbase8"
    pair_path = pair_dir / "intervention_pair.npz"
    assert main(["make-demo-data", "--output", str(data_dir / "val"), "--samples", "1", "--time-steps", "8", "--input-dim", "4"]) == 0
    pair_dir.mkdir(parents=True)
    baseline_inputs = np.zeros((1, 8, 4), dtype=np.int8)
    intervention_inputs = baseline_inputs.copy()
    intervention_inputs[:, 2, 0] = 1
    baseline_targets = np.zeros((1, 8, 2), dtype=np.float32)
    intervention_targets = baseline_targets.copy()
    intervention_targets[..., 1] = 0.5
    np.savez_compressed(
        pair_path,
        baseline_inputs=baseline_inputs,
        baseline_targets=baseline_targets,
        intervention_inputs=intervention_inputs,
        intervention_targets=intervention_targets,
        sample_ids=np.asarray(["trace-a"]),
    )

    result = main(
        [
            "benchmark-v1",
            "--dataset-root",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--intervention-pair-glob",
            str(tmp_path / "pilot" / "*" / "intervention_pair.npz"),
            "--max-val-samples",
            "1",
            "--max-intervention-samples",
            "1",
            "--ignore-start",
            "0",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["artifacts"]) == 2
    assert (output_dir / "teacher_qc_val.md").exists()
    assert (output_dir / "intervention_specimen_488683425_event_dropout.md").exists()


def test_cli_make_arbor_hetero_data_writes_shards_and_report(tmp_path):
    data_dir = tmp_path / "arbor-hetero"
    report_path = tmp_path / "report.json"

    result = main(
        [
            "make-arbor-hetero-data",
            "--output",
            str(data_dir),
            "--samples",
            "1",
            "--shards",
            "1",
            "--time-steps",
            "16",
            "--input-dim",
            "8",
            "--branches",
            "2",
            "--integration-dt-ms",
            "0.1",
            "--report",
            str(report_path),
        ]
    )

    assert result == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["samples"] == 1
    assert report["backend"] == "arbor"
    assert report["simulator"]["name"] == "arbor"
    assert "bnn::NMDA_CA1_pyr_SC" in report["simulator"]["custom_catalogues"]["mechanisms"]
    assert report["simulator"]["custom_catalogues"]["nmda_source"]["modeldb_model"] == "2019343"
    assert "nmda_weight_scale" in report["simulator"]["heterogeneity_names"]
    assert (data_dir / "manifest.json").exists()
    assert (data_dir / "generation_report.json").exists()


def test_cli_make_arbor_hetero_splits(tmp_path):
    data_dir = tmp_path / "arbor-hetero-splits"

    result = main(
        [
            "make-arbor-hetero-data",
            "--output",
            str(data_dir),
            "--splits",
            "train:2,val:1",
            "--shards-per-split",
            "1",
            "--time-steps",
            "8",
            "--input-dim",
            "6",
            "--branches",
            "2",
            "--integration-dt-ms",
            "0.1",
        ]
    )

    assert result == 0
    summary = json.loads((data_dir / "generation_summary.json").read_text(encoding="utf-8"))
    assert set(summary["splits"]) == {"train", "val"}
    assert summary["backend"] == "arbor"
    assert (data_dir / "train" / "manifest.json").exists()


def test_cli_make_arbor_hetero_data_accepts_config_file(tmp_path):
    data_dir = tmp_path / "configured-arbor-hetero"
    report_path = tmp_path / "configured-report.json"
    config_path = tmp_path / "sim-config.json"
    config_path.write_text(
        json.dumps(
            {
                "output": str(data_dir),
                "samples": 1,
                "shards": 1,
                "time_steps": 8,
                "input_dim": 6,
                "branches": 2,
                "integration_dt_ms": 0.1,
                "report": str(report_path),
            }
        ),
        encoding="utf-8",
    )

    result = main(["make-arbor-hetero-data", "--config", str(config_path)])

    assert result == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["samples"] == 1
    assert report["input_dim"] == 6
    assert (data_dir / "manifest.json").exists()


def test_cli_make_arbor_hetero_data_accepts_param_overrides_config(tmp_path):
    data_dir = tmp_path / "fixed-arbor-hetero"
    config_path = tmp_path / "sim-config.json"
    config_path.write_text(
        json.dumps(
            {
                "output": str(data_dir),
                "samples": 1,
                "shards": 1,
                "time_steps": 8,
                "input_dim": 6,
                "branches": 2,
                "integration_dt_ms": 0.1,
                "param_overrides": {
                    "v_init_mv": -72.0,
                    "input_rate_scale": 1.0,
                    "spike_threshold_mv": -38.0,
                },
            }
        ),
        encoding="utf-8",
    )

    result = main(["make-arbor-hetero-data", "--config", str(config_path)])

    assert result == 0
    report = json.loads((data_dir / "generation_report.json").read_text(encoding="utf-8"))
    assert report["param_overrides"]["v_init_mv"] == -72.0
    assert report["param_overrides"]["input_rate_scale"] == 1.0
    assert report["param_overrides"]["spike_threshold_mv"] == -38.0


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
