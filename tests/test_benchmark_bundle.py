import json

import numpy as np

from bnn_sim.benchmark.bundle import infer_pair_metadata, run_v1_benchmark_bundle
from bnn_sim.benchmark.diagnostic_rows import DiagnosticContext
from bnn_sim.data import write_demo_shards


def _write_pair(path):
    baseline_inputs = np.zeros((1, 8, 4), dtype=np.int8)
    intervention_inputs = baseline_inputs.copy()
    intervention_inputs[:, 2, 0] = 1
    baseline_targets = np.zeros((1, 8, 2), dtype=np.float32)
    intervention_targets = baseline_targets.copy()
    baseline_targets[..., 1] = 1.0
    intervention_targets[..., 1] = 0.5
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        baseline_inputs=baseline_inputs,
        baseline_targets=baseline_targets,
        intervention_inputs=intervention_inputs,
        intervention_targets=intervention_targets,
        sample_ids=np.asarray(["trace-a"]),
    )


def test_infer_pair_metadata_from_pilot_path():
    intervention, morphology = infer_pair_metadata(
        "pilot/specimen_488683425_inh_dropout_simbase8/intervention_pair.npz"
    )

    assert intervention == "inh_dropout"
    assert morphology == "specimen_488683425"


def test_run_v1_benchmark_bundle_writes_manifest_rows_and_summaries(tmp_path):
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "bundle"
    write_demo_shards(dataset_root / "val", shard_count=1, samples_per_shard=1, time_steps=8, input_dim=4)
    pair_path = tmp_path / "pilot" / "specimen_488683425_event_dropout_simbase8" / "intervention_pair.npz"
    _write_pair(pair_path)

    manifest = run_v1_benchmark_bundle(
        dataset_root=dataset_root,
        output_dir=output_dir,
        base_context=DiagnosticContext(ignore_start=0),
        intervention_pair_paths=[pair_path],
        max_val_samples=1,
        max_intervention_samples=1,
    )

    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "teacher_qc_val.jsonl").exists()
    assert (output_dir / "teacher_qc_val.md").exists()
    assert (output_dir / "morphology_routing_qc.jsonl").exists()
    assert (output_dir / "intervention_specimen_488683425_event_dropout.jsonl").exists()
    intervention_rows = [
        json.loads(line)
        for line in (output_dir / "intervention_specimen_488683425_event_dropout.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["model_id"] == "teacher" for row in intervention_rows)
    assert len(manifest["artifacts"]) == 3
    assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))["artifacts"]


def test_run_v1_benchmark_bundle_can_skip_summaries(tmp_path):
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "bundle"
    write_demo_shards(dataset_root / "val", shard_count=1, samples_per_shard=1, time_steps=8, input_dim=4)
    pair_path = tmp_path / "pilot" / "specimen_488683425_event_dropout_simbase8" / "intervention_pair.npz"
    _write_pair(pair_path)

    manifest = run_v1_benchmark_bundle(
        dataset_root=dataset_root,
        output_dir=output_dir,
        base_context=DiagnosticContext(ignore_start=0),
        intervention_pair_paths=[pair_path],
        max_val_samples=1,
        max_intervention_samples=1,
        write_summaries=False,
    )

    assert (output_dir / "teacher_qc_val.jsonl").exists()
    assert not (output_dir / "teacher_qc_val.md").exists()
    assert manifest["write_summaries"] is False
    assert all("summary_path" not in artifact for artifact in manifest["artifacts"])


def test_run_v1_benchmark_bundle_expands_sharded_intervention_selector_by_morphology(tmp_path):
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "bundle"
    write_demo_shards(dataset_root / "val", shard_count=1, samples_per_shard=1, time_steps=8, input_dim=4)
    _write_pair(dataset_root / "interventions" / "event_dropout" / "specimen_111_intervention_pair_00000.npz")
    _write_pair(dataset_root / "interventions" / "event_dropout" / "specimen_222_intervention_pair_00001.npz")

    manifest = run_v1_benchmark_bundle(
        dataset_root=dataset_root,
        output_dir=output_dir,
        base_context=DiagnosticContext(ignore_start=0),
        max_val_samples=1,
        max_intervention_samples=1,
    )

    assert (output_dir / "intervention_specimen_111_event_dropout.jsonl").exists()
    assert (output_dir / "intervention_specimen_222_event_dropout.jsonl").exists()
    assert (output_dir / "morphology_contrast_interventions.jsonl").exists()
    assert len(manifest["artifacts"]) == 5


def test_run_v1_benchmark_bundle_groups_intervention_shards_by_morphology(tmp_path):
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "bundle"
    write_demo_shards(dataset_root / "val", shard_count=1, samples_per_shard=1, time_steps=8, input_dim=4)
    _write_pair(dataset_root / "interventions" / "event_dropout" / "specimen_111_intervention_pair_00000.npz")
    _write_pair(dataset_root / "interventions" / "event_dropout" / "specimen_111_intervention_pair_00001.npz")
    _write_pair(dataset_root / "interventions" / "event_dropout" / "specimen_222_intervention_pair_00002.npz")

    manifest = run_v1_benchmark_bundle(
        dataset_root=dataset_root,
        output_dir=output_dir,
        base_context=DiagnosticContext(ignore_start=0),
        max_val_samples=1,
        max_intervention_samples=1,
        write_summaries=False,
    )

    specimen_111_files = list(output_dir.glob("intervention_specimen_111_event_dropout.jsonl"))
    assert len(specimen_111_files) == 1
    rows_111 = [json.loads(line) for line in specimen_111_files[0].read_text(encoding="utf-8").splitlines()]
    assert rows_111
    intervention_artifacts = [
        artifact
        for artifact in manifest["artifacts"]
        if "intervention_specimen_111_event_dropout.jsonl" in artifact["rows_path"]
    ]
    assert len(intervention_artifacts) == 1
    assert len(manifest["artifacts"]) == 5


def test_run_v1_benchmark_bundle_teacher_qc_suite_skips_model_rows(tmp_path):
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "bundle"
    write_demo_shards(dataset_root / "val", shard_count=1, samples_per_shard=1, time_steps=8, input_dim=4)

    def model_fn(inputs):
        return np.zeros((inputs.shape[0], inputs.shape[1], 2), dtype=np.float32)

    manifest = run_v1_benchmark_bundle(
        dataset_root=dataset_root,
        output_dir=output_dir,
        base_context=DiagnosticContext(ignore_start=0),
        model_fn=model_fn,
        model_id="unit-model",
        suite="teacher-qc",
        max_val_samples=1,
    )

    assert manifest["suite"] == "teacher-qc"
    assert manifest["suite_config"]["model"] is None
    assert not list(output_dir.glob("model_val_*.jsonl"))
