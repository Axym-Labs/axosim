import json

import numpy as np

from bnn_sim.benchmark.diagnostic_rows import DiagnosticContext
from bnn_sim.benchmark.runner import (
    load_selector_batch,
    run_paired_intervention_on_npz,
    run_paired_intervention_on_selector,
    run_model_diagnostics_on_selector,
    run_teacher_qc_on_selector,
)
from bnn_sim.benchmark.selectors import V1DatasetLayout
from bnn_sim.data import write_demo_shards


def test_runner_loads_selector_batch(tmp_path):
    write_demo_shards(tmp_path / "val", shard_count=1, samples_per_shard=3, time_steps=6, input_dim=4)
    selector = V1DatasetLayout(tmp_path).default_validation()

    inputs, targets, trace_ids = load_selector_batch(selector, max_samples=2)

    assert inputs.shape == (2, 6, 4)
    assert targets.shape == (2, 6, 2)
    assert len(trace_ids) == 2


def test_runner_writes_model_diagnostic_rows_for_selector(tmp_path):
    write_demo_shards(tmp_path / "val", shard_count=1, samples_per_shard=2, time_steps=8, input_dim=4)
    selector = V1DatasetLayout(tmp_path).default_validation()
    output = tmp_path / "rows.json"

    def model_fn(inputs):
        return np.zeros((inputs.shape[0], inputs.shape[1], 2), dtype=np.float32)

    rows = run_model_diagnostics_on_selector(
        model_fn,
        selector,
        context=DiagnosticContext(suites=("trace-core", "paper-summary"), model_id="unit-model", ignore_start=0),
        output=output,
    )

    assert rows
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload[0]["condition"] == "val"
    assert any(row["model_id"] == "unit-model" for row in payload)


def test_runner_writes_teacher_qc_rows_for_selector(tmp_path):
    write_demo_shards(tmp_path / "val", shard_count=1, samples_per_shard=1, time_steps=8, input_dim=4)
    selector = V1DatasetLayout(tmp_path).default_validation()
    output = tmp_path / "teacher.csv"

    rows = run_teacher_qc_on_selector(
        selector,
        context=DiagnosticContext(suites=("teacher-qc",), model_id="teacher", ignore_start=0),
        output=output,
    )

    assert rows
    assert output.exists()
    assert any(row["diagnostic"] == "teacher_distribution_qc" for row in rows)


def test_runner_auto_loads_raw_soma_for_clipping_qc(tmp_path):
    val = tmp_path / "val"
    val.mkdir()
    inputs = np.zeros((1, 8, 4), dtype=np.int8)
    targets = np.zeros((1, 8, 2), dtype=np.float32)
    targets[..., 1] = (-67.0 + 67.7) * 0.1
    targets[0, 4, 0] = 1.0
    targets[0, 4, 1] = (-55.0 + 67.7) * 0.1
    soma_mv = np.full((1, 8), -67.0, dtype=np.float32)
    soma_mv[0, 4] = -20.0
    np.savez_compressed(val / "raw_000.npz", inputs=inputs, targets=targets, soma_mv=soma_mv)
    selector = V1DatasetLayout(tmp_path).default_validation()

    rows = run_teacher_qc_on_selector(
        selector,
        context=DiagnosticContext(suites=("teacher-qc", "paper-summary"), model_id="teacher", ignore_start=0),
    )

    assert any(row["diagnostic"] == "clipping_distortion" for row in rows)


def test_runner_writes_paired_intervention_rows_from_npz(tmp_path):
    pair_path = tmp_path / "intervention_pair.npz"
    baseline_inputs = np.zeros((2, 8, 4), dtype=np.int8)
    intervention_inputs = baseline_inputs.copy()
    intervention_inputs[:, 2, 0] = 1
    baseline_targets = np.zeros((2, 8, 2), dtype=np.float32)
    intervention_targets = baseline_targets.copy()
    baseline_targets[..., 1] = 1.0
    intervention_targets[..., 1] = 0.5
    np.savez_compressed(
        pair_path,
        baseline_inputs=baseline_inputs,
        baseline_targets=baseline_targets,
        intervention_inputs=intervention_inputs,
        intervention_targets=intervention_targets,
        sample_ids=np.asarray(["trace-a", "trace-b"]),
    )

    rows = run_paired_intervention_on_npz(
        pair_path,
        context=DiagnosticContext(
            suites=("teacher-qc", "intervention-response", "paper-summary"),
            condition="interventions/event_dropout",
            intervention="event_dropout",
            ignore_start=0,
        ),
        output=tmp_path / "intervention_rows.json",
    )

    assert any(row["diagnostic"] == "intervention_response" for row in rows)
    assert any(row["metric"] == "teacher_direction" for row in rows)
    assert all(row["condition"] == "interventions/event_dropout" for row in rows)


def test_runner_reads_sharded_paired_intervention_selector(tmp_path):
    pair_dir = tmp_path / "interventions" / "event_dropout"
    pair_dir.mkdir(parents=True)
    for shard_idx, trace_id in enumerate(("trace-a", "trace-b")):
        baseline_inputs = np.zeros((1, 8, 4), dtype=np.int8)
        intervention_inputs = baseline_inputs.copy()
        intervention_inputs[:, 2, 0] = 1
        baseline_targets = np.zeros((1, 8, 2), dtype=np.float32)
        intervention_targets = baseline_targets.copy()
        intervention_targets[..., 1] = 0.5 + shard_idx
        np.savez_compressed(
            pair_dir / f"intervention_pair_{shard_idx:05d}.npz",
            baseline_inputs=baseline_inputs,
            baseline_targets=baseline_targets,
            intervention_inputs=intervention_inputs,
            intervention_targets=intervention_targets,
            sample_ids=np.asarray([trace_id]),
        )

    rows = run_paired_intervention_on_selector(
        V1DatasetLayout(tmp_path).intervention("event_dropout"),
        context=DiagnosticContext(
            suites=("teacher-qc", "intervention-response", "paper-summary"),
            condition="interventions/event_dropout",
            intervention="event_dropout",
            ignore_start=0,
        ),
        max_samples=2,
    )

    delta_row = next(row for row in rows if row["metric"] == "delta_soma_mean_mv")
    assert delta_row["value"] == 10.0
    assert any(row["diagnostic"] == "intervention_response" for row in rows)
