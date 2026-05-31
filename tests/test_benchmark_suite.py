import numpy as np

from bnn_sim.benchmark.regime_stratified import compute_regime_stratified_metrics
from bnn_sim.benchmark.suite import BenchmarkSuite


def test_benchmark_suite_reports_mvp_schema_and_masked_m1(tmp_path):
    inputs = np.zeros((2, 5, 3), dtype=np.float32)
    targets = np.zeros((2, 5, 2), dtype=np.float32)
    predictions = np.zeros_like(targets)

    targets[..., 1] = 1.0
    predictions[..., 1] = 1.0
    predictions[:, :2, 1] = 11.0
    predictions[..., 0] = -8.0
    targets[0, 3, 0] = 1.0
    predictions[0, 3, 0] = 8.0

    suite = BenchmarkSuite(output_dir=tmp_path)
    report = suite.evaluate_model(
        lambda x: predictions,
        inputs,
        targets,
        soma_scale=0.1,
        ignore_start=2,
        model_name="unit-model",
        model_config={"kind": "toy"},
        dataset_path="unit-dataset",
        trace_ids=["trace-a", "trace-b"],
        run_perturbation=False,
    )

    assert set(["metadata", "m1_trace_fidelity", "m1b_feature_fidelity", "m2_regimes", "m4_intervention", "limitations"]).issubset(report)
    assert report["metadata"]["model_name"] == "unit-model"
    assert report["metadata"]["model_config"] == {"kind": "toy"}
    assert report["metadata"]["dataset_path"] == "unit-dataset"
    assert report["metadata"]["trace_ids"] == ["trace-a", "trace-b"]
    assert report["metadata"]["mask"]["ignore_start"] == 2
    assert report["metadata"]["mask"]["evaluated_timesteps_per_trace"] == 3
    assert report["m1_trace_fidelity"]["soma_rmse_mv"] == 0.0
    assert report["m1_trace_fidelity"]["spike_target_count"] == 1
    assert report["m1_trace_fidelity"]["spike_brier"] < 0.001
    assert report["m1b_feature_fidelity"]["features"]["spike_count"]["teacher_mean"] == 0.5
    assert report["m1b_feature_fidelity"]["worst_trace_ids"]["spike_count_abs_error"] == []
    assert report["m4_intervention"]["status"] == "not_run"
    assert (tmp_path / "benchmark_results.json").exists()


def test_regime_metrics_do_not_leak_spike_windows_across_trace_boundaries():
    predictions = np.zeros((2, 5, 2), dtype=np.float32)
    targets = np.zeros((2, 5, 2), dtype=np.float32)
    inputs = np.zeros((2, 5, 2), dtype=np.float32)

    targets[0, 4, 0] = 1.0
    predictions[0, 2:5, 1] = 1.0
    predictions[1, 0:2, 1] = 100.0

    metrics = compute_regime_stratified_metrics(
        predictions,
        targets,
        inputs,
        soma_scale=1.0,
        ignore_start=0,
    )

    near = next(row for row in metrics["spike_proximity"] if row["regime_label"] == "near_spike")
    far = next(row for row in metrics["spike_proximity"] if row["regime_label"] == "far_from_spike")
    mvp = {row["regime_label"]: row for row in metrics["mvp_regimes"]}

    assert near["count"] == 3
    assert far["count"] == 7
    assert mvp["near_spike"]["count"] == 3
    assert mvp["far_from_spike_subthreshold"]["count"] == 7
