import json

import numpy as np

from bnn_sim.benchmark.diagnostic_rows import DiagnosticContext
from bnn_sim.benchmark.runner import (
    load_selector_batch,
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
