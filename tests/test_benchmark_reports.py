import json

from bnn_sim.benchmark.reports import summarize_diagnostic_rows, write_diagnostic_summary


def _rows():
    return [
        {
            "diagnostic": "trace_core",
            "suite": "trace-core|paper-summary",
            "model_id": "model-a",
            "teacher_id": "teacher",
            "morphology_id": "morph-a",
            "split": "val",
            "condition": "val",
            "intervention": None,
            "trace_id": "trace-a",
            "state": None,
            "metric": "soma_rmse_mv",
            "value": 0.8,
            "unit": "mV",
            "n": 10,
        },
        {
            "diagnostic": "intervention_response",
            "suite": "teacher-qc|intervention-response|paper-summary",
            "model_id": "model-a",
            "teacher_id": "teacher",
            "morphology_id": "morph-a",
            "split": "",
            "condition": "interventions/event_dropout",
            "intervention": "event_dropout",
            "trace_id": None,
            "state": None,
            "metric": "teacher_direction",
            "value": "hyperpolarizing",
            "unit": "label",
            "n": None,
        },
    ]


def test_summarize_diagnostic_rows_groups_numeric_and_label_metrics():
    summary = summarize_diagnostic_rows(_rows())

    assert summary["row_count"] == 2
    assert summary["diagnostics"] == ["intervention_response", "trace_core"]
    assert summary["trace_ids"] == ["trace-a"]
    assert summary["numeric_metrics"][0]["mean"] == 0.8
    assert summary["label_metrics"][0]["counts"] == {"hyperpolarizing": 1}


def test_write_diagnostic_summary_supports_json_and_markdown(tmp_path):
    json_path = tmp_path / "summary.json"
    md_path = tmp_path / "summary.md"

    write_diagnostic_summary(_rows(), json_path)
    write_diagnostic_summary(_rows(), md_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["row_count"] == 2
    assert "Benchmark Diagnostic Summary" in md_path.read_text(encoding="utf-8")
