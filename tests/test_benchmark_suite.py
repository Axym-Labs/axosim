import numpy as np

from bnn_sim.benchmark.regime_stratified import compute_regime_stratified_metrics
from bnn_sim.benchmark.suite import BenchmarkSuite
from bnn_sim.benchmark.structured_intervention import (
    compute_paired_event_intervention_metrics,
    make_event_dropout_inputs,
)


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
    assert "M4 structured intervention was not run for this report." in report["limitations"]
    assert (tmp_path / "benchmark_results.json").exists()


def test_benchmark_suite_reports_calibrated_spike_threshold():
    inputs = np.zeros((1, 4, 2), dtype=np.float32)
    targets = np.zeros((1, 4, 2), dtype=np.float32)
    predictions = np.zeros_like(targets)

    targets[0, 1, 0] = 1.0
    predictions[0, :, 0] = np.array([-0.7, -0.2, -0.4, -0.6], dtype=np.float32)

    report = BenchmarkSuite().evaluate_model(
        lambda x: predictions,
        inputs,
        targets,
        soma_scale=1.0,
        ignore_start=0,
        run_perturbation=False,
        calibrated_spike_threshold=-0.3,
    )

    m1 = report["m1_trace_fidelity"]
    assert m1["spike_threshold_zero"]["predicted_count"] == 0
    assert m1["spike_calibrated_threshold"]["predicted_count"] == 1
    assert m1["spike_calibrated_threshold"]["count_mae_per_trace"] == 0.0
    assert report["m1b_feature_fidelity"]["calibrated_threshold"]["features"]["spike_count"]["prediction_mean"] == 1.0
    assert report["m2_regimes"]["spike_timing_calibrated_threshold"]["threshold"] == -0.3


def test_benchmark_voltage_features_and_regime_bins_include_soma_bias():
    inputs = np.zeros((1, 4, 2), dtype=np.float32)
    targets = np.zeros((1, 4, 2), dtype=np.float32)
    predictions = np.zeros_like(targets)

    # Canonical target encoding: normalized = (mV - bias) * scale.
    # These values should decode to -70, -65, -60, and -55 mV.
    targets[0, :, 1] = np.array([-0.3, 0.2, 0.7, 1.2], dtype=np.float32)
    predictions[0, :, 1] = targets[0, :, 1]

    report = BenchmarkSuite().evaluate_model(
        lambda x: predictions,
        inputs,
        targets,
        soma_scale=0.1,
        soma_bias_mv=-67.0,
        ignore_start=0,
        run_perturbation=False,
        trace_ids=["trace-a"],
    )

    features = report["m1b_feature_fidelity"]["features"]
    assert np.isclose(features["peak_voltage_mv"]["teacher_mean"], -55.0)
    assert np.isclose(features["trough_voltage_mv"]["teacher_mean"], -70.0)

    voltage_rows = report["m2_regimes"]["voltage_regime"]
    assert np.isclose(voltage_rows[0]["bin_low"], -70.0)
    assert np.isclose(voltage_rows[-1]["bin_high"], -55.0)


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
    assert "near_spike" in metrics["what_failure_means"]
    warned = {row["regime_label"] for row in metrics["support_warnings"]}
    assert {"post_spike_refractory", "post_input_event"}.issubset(warned)


def test_regime_post_event_uses_high_drive_for_many_input_channels():
    predictions = np.zeros((1, 10, 2), dtype=np.float32)
    targets = np.zeros((1, 10, 2), dtype=np.float32)
    inputs = np.zeros((1, 10, 128), dtype=np.float32)

    inputs[:, :, :2] = 1.0
    inputs[:, 4, :64] = 1.0

    metrics = compute_regime_stratified_metrics(
        predictions,
        targets,
        inputs,
        soma_scale=1.0,
        soma_bias_mv=0.0,
        post_event_quantile=0.95,
    )
    post = next(row for row in metrics["post_event"] if row["regime_label"] == "post_event")
    no_recent = next(row for row in metrics["post_event"] if row["regime_label"] == "no_recent_event")
    mvp = {row["regime_label"]: row for row in metrics["mvp_regimes"]}

    assert metrics["post_event_definition"]["event_count_threshold"] > 2
    assert post["count"] == 3
    assert no_recent["count"] == 7
    assert mvp["far_from_spike_subthreshold"]["count"] == 7
    assert "post_input_event" in metrics["what_failure_means"]


def test_event_dropout_only_removes_existing_events_deterministically():
    inputs = np.zeros((1, 6, 4), dtype=np.int8)
    inputs[0, 1, 0] = 1
    inputs[0, 2, 1] = -1
    inputs[0, 3, 2] = 1

    dropped_a, mask_a = make_event_dropout_inputs(inputs, dropout_probability=0.5, seed=7)
    dropped_b, mask_b = make_event_dropout_inputs(inputs, dropout_probability=0.5, seed=7)

    assert np.array_equal(dropped_a, dropped_b)
    assert np.array_equal(mask_a, mask_b)
    assert np.count_nonzero(dropped_a) <= np.count_nonzero(inputs)
    assert np.all(dropped_a[inputs == 0] == 0)
    assert set(np.unique(dropped_a)).issubset({-1, 0, 1})


def test_paired_event_intervention_scores_teacher_and_model_delta_alignment():
    baseline_inputs = np.zeros((2, 4, 3), dtype=np.float32)
    intervention_inputs = baseline_inputs.copy()
    baseline_inputs[:, 1:, 0] = 1.0

    baseline_targets = np.zeros((2, 4, 2), dtype=np.float32)
    intervention_targets = np.zeros_like(baseline_targets)
    baseline_targets[..., 1] = 2.0
    intervention_targets[..., 1] = 1.0
    baseline_targets[:, 2, 0] = 1.0

    def aligned_model(x):
        out = np.zeros((x.shape[0], x.shape[1], 2), dtype=np.float32)
        active = np.count_nonzero(x, axis=-1) > 0
        time_is_spike = np.arange(x.shape[1])[None, :] == 2
        out[..., 1] = np.where(active, 2.0, 1.0)
        out[..., 0] = np.where(active & time_is_spike, 8.0, -8.0)
        return out

    report = compute_paired_event_intervention_metrics(
        aligned_model,
        baseline_inputs,
        baseline_targets,
        intervention_inputs,
        intervention_targets,
        intervention_name="unit_dropout",
        soma_scale=1.0,
        ignore_start=1,
        trace_ids=["a", "b"],
    )

    assert report["status"] == "model_compared"
    assert report["teacher_response"]["baseline_event_fraction"] > report["teacher_response"]["intervention_event_fraction"]
    assert report["teacher_response"]["delta_spike_rate_per_1000_steps"] < 0
    assert report["teacher_response"]["n_traces"] == 2
    assert report["teacher_response"]["delta_soma_trace_mean_mv"]["n"] == 2
    assert report["model_response"]["soma_delta_rmse_mv"] == 0.0
    assert report["model_response"]["spike_count_delta_mae_per_trace"] == 0.0
    assert report["model_response"]["soma_delta_rmse_per_trace"]["mean"] == 0.0
    assert report["model_response"]["spike_count_delta_abs_error_per_trace"]["n"] == 2
    assert report["representative_trace_ids"]


def test_paired_event_intervention_uses_configured_model_spike_threshold():
    baseline_inputs = np.zeros((1, 4, 2), dtype=np.float32)
    intervention_inputs = baseline_inputs.copy()
    baseline_targets = np.zeros((1, 4, 2), dtype=np.float32)
    intervention_targets = np.zeros_like(baseline_targets)
    intervention_targets[:, 2, 0] = 1.0

    def below_zero_spike_model(x):
        out = np.zeros((x.shape[0], x.shape[1], 2), dtype=np.float32)
        out[:, 2, 0] = -0.2
        return out

    report = compute_paired_event_intervention_metrics(
        below_zero_spike_model,
        baseline_inputs,
        baseline_targets,
        intervention_inputs,
        intervention_targets,
        soma_scale=1.0,
        model_spike_threshold=-0.3,
    )

    model = report["model_response"]
    assert model["model_spike_threshold"] == -0.3
    assert model["spike_count_delta_model_mean"] == 0.0
    assert model["spike_count_delta_mae_per_trace"] == 1.0


def test_benchmark_suite_accepts_paired_m4_intervention():
    inputs = np.zeros((1, 4, 2), dtype=np.float32)
    targets = np.zeros((1, 4, 2), dtype=np.float32)
    intervention_inputs = inputs.copy()
    intervention_targets = targets.copy()
    inputs[:, 1:, 0] = 1.0
    targets[..., 1] = 2.0
    intervention_targets[..., 1] = 1.0

    def model_fn(x):
        out = np.zeros((x.shape[0], x.shape[1], 2), dtype=np.float32)
        out[..., 1] = np.where(np.count_nonzero(x, axis=-1) > 0, 2.0, 1.0)
        out[..., 0] = -8.0
        return out

    report = BenchmarkSuite().evaluate_model(
        model_fn,
        inputs,
        targets,
        soma_scale=1.0,
        ignore_start=1,
        run_perturbation=False,
        m4_paired_intervention={
            "baseline_inputs": inputs,
            "baseline_targets": targets,
            "intervention_inputs": intervention_inputs,
            "intervention_targets": intervention_targets,
            "intervention_name": "unit_event_dropout",
        },
    )

    assert report["m4_intervention"]["status"] == "model_compared"
    assert report["m4_intervention"]["kind"] == "paired_event_intervention"
    assert report["m4_intervention"]["model_response"]["soma_delta_rmse_mv"] == 0.0
    assert "M4 structured intervention results are limited to the supplied paired traces and should be expanded before robustness claims." in report["limitations"]
