import numpy as np
import json

from bnn_sim.benchmark.diagnostic_rows import (
    DiagnosticContext,
    evaluate_model_diagnostic_rows,
    paired_intervention_diagnostic_rows,
    teacher_qc_diagnostic_rows,
    validate_diagnostic_rows,
    write_diagnostic_rows,
)


def test_model_diagnostics_emit_dataframe_rows_for_overlapping_suites():
    inputs = np.zeros((2, 8, 4), dtype=np.float32)
    targets = np.zeros((2, 8, 2), dtype=np.float32)
    predictions = targets.copy()
    targets[..., 1] = -1.0
    predictions[..., 1] = -1.0
    targets[0, 4, 0] = 1.0
    predictions[0, 4, 0] = 8.0

    rows = evaluate_model_diagnostic_rows(
        lambda _: predictions,
        inputs,
        targets,
        context=DiagnosticContext(
            suites=("trace-core", "state-dynamics", "paper-summary"),
            model_id="unit-model",
            morphology_id="morph-a",
            ignore_start=2,
        ),
        trace_ids=["trace-a", "trace-b"],
    )

    validate_diagnostic_rows(rows)
    assert {row["diagnostic"] for row in rows}.issuperset({"trace_core", "state_dynamics", "spike_behavior"})
    assert any(row["suite"] == "trace-core|paper-summary" for row in rows if row["diagnostic"] == "trace_core")
    assert any(row["trace_id"] == "trace-a" and row["metric"] == "soma_rmse_mv" for row in rows)
    assert any(row["state"] == "spiking" for row in rows if row["diagnostic"] == "state_dynamics")
    assert all(row["morphology_id"] == "morph-a" for row in rows)


def test_teacher_qc_rows_include_distribution_and_clipping_distortion():
    inputs = np.zeros((1, 8, 4), dtype=np.float32)
    targets = np.zeros((1, 8, 2), dtype=np.float32)
    targets[..., 1] = (-67.0 + 67.7) * 0.1
    targets[0, 4, 0] = 1.0
    raw_soma_mv = np.full((1, 8), -67.0, dtype=np.float32)
    raw_soma_mv[0, 4] = -20.0
    targets[0, 4, 1] = (-55.0 + 67.7) * 0.1

    rows = teacher_qc_diagnostic_rows(
        inputs,
        targets,
        raw_soma_mv=raw_soma_mv,
        context=DiagnosticContext(
            suites=("teacher-qc", "paper-summary"),
            model_id="teacher",
            morphology_id="morph-a",
            ignore_start=0,
        ),
        trace_ids=["trace-a"],
    )

    validate_diagnostic_rows(rows)
    metrics = {(row["diagnostic"], row["metric"]): row for row in rows}
    assert metrics[("teacher_distribution_qc", "spike_count")]["value"] == 1
    assert metrics[("clipping_distortion", "clipped_fraction")]["value"] > 0
    assert metrics[("clipping_distortion", "distortion_max_mv")]["value"] > 30
    assert metrics[("clipping_distortion", "near_spike_distortion_rmse_mv")]["n"] > 0


def test_paired_intervention_rows_include_teacher_and_model_delta_metrics():
    baseline_inputs = np.zeros((1, 5, 4), dtype=np.float32)
    intervention_inputs = baseline_inputs.copy()
    intervention_inputs[0, 1, 0] = 1.0
    baseline_targets = np.zeros((1, 5, 2), dtype=np.float32)
    intervention_targets = np.zeros_like(baseline_targets)
    baseline_targets[..., 1] = 1.0
    intervention_targets[..., 1] = 0.5
    baseline_targets[0, 2, 0] = 1.0

    def model_fn(x):
        if np.array_equal(x, intervention_inputs):
            return intervention_targets.copy()
        return baseline_targets.copy()

    rows = paired_intervention_diagnostic_rows(
        baseline_inputs,
        baseline_targets,
        intervention_inputs,
        intervention_targets,
        model_fn=model_fn,
        context=DiagnosticContext(
            suites=("teacher-qc", "intervention-response", "paper-summary"),
            condition="interventions/event_dropout",
            intervention="event_dropout",
            morphology_id="morph-a",
            ignore_start=0,
        ),
        trace_ids=["trace-a"],
    )

    validate_diagnostic_rows(rows)
    metrics = {row["metric"]: row["value"] for row in rows}
    assert metrics["delta_soma_mean_mv"] < 0
    assert metrics["response_detected"] == 1.0
    assert metrics["soma_delta_rmse_mv"] == 0.0
    assert "teacher-qc|intervention-response|paper-summary" in {row["suite"] for row in rows}


def test_write_diagnostic_rows_supports_dataframe_friendly_formats(tmp_path):
    rows = [
        {
            "diagnostic": "trace_core",
            "suite": "trace-core",
            "model_id": "m",
            "teacher_id": "t",
            "morphology_id": "morph",
            "split": "val",
            "condition": "val",
            "intervention": None,
            "trace_id": None,
            "state": None,
            "metric": "soma_rmse_mv",
            "value": 0.1,
            "unit": "mV",
            "n": 10,
            "target_view": "neuronio_clipped_normalized",
        }
    ]

    json_path = tmp_path / "rows.json"
    jsonl_path = tmp_path / "rows.jsonl"
    csv_path = tmp_path / "rows.csv"
    write_diagnostic_rows(rows, json_path)
    write_diagnostic_rows(rows, jsonl_path)
    write_diagnostic_rows(rows, csv_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["metric"] == "soma_rmse_mv"
    assert json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])["unit"] == "mV"
    assert "diagnostic,suite,model_id" in csv_path.read_text(encoding="utf-8").splitlines()[0]
