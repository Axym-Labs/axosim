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
    assert (output_dir / "intervention_specimen_488683425_event_dropout.jsonl").exists()
    assert len(manifest["artifacts"]) == 2
    assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))["artifacts"]
