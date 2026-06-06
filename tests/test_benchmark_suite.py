import numpy as np

from bnn_sim.benchmark.regime_stratified import compute_regime_stratified_metrics
from bnn_sim.benchmark.morphology_transfer import MorphologyTransferEvaluator
from bnn_sim.benchmark.state_metrics import compute_state_conditioned_metrics
from bnn_sim.benchmark.suite import BenchmarkSuite
from bnn_sim.benchmark.trace_shape import compute_trace_shape_metrics
from bnn_sim.benchmark.structured_intervention import (
    compute_paired_event_intervention_metrics,
    make_event_dropout_inputs,
    make_selective_event_dropout_inputs,
    make_site_silence_inputs,
    make_structured_intervention_inputs,
    make_temporal_jitter_inputs,
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

    assert set(["metadata", "m1_trace_fidelity", "m1b_feature_fidelity", "m2_regimes", "state_conditioned_fidelity", "trace_shape_fidelity", "m4_intervention", "limitations"]).issubset(report)
    assert report["metadata"]["model_name"] == "unit-model"
    assert report["metadata"]["model_config"] == {"kind": "toy"}
    assert report["metadata"]["dataset_path"] == "unit-dataset"
    assert report["metadata"]["trace_ids"] == ["trace-a", "trace-b"]
    assert report["metadata"]["mask"]["ignore_start"] == 2
    assert report["metadata"]["mask"]["evaluated_timesteps_per_trace"] == 3
    assert report["m1_trace_fidelity"]["soma_rmse_mv"] == 0.0
    assert "state_weighted_trace_distance_mv" in report["m1_trace_fidelity"]
    assert "dynamic_distance" in report["m1_trace_fidelity"]
    assert "trace_shape_distance" in report["m1_trace_fidelity"]
    assert "state_trace_shape_distance" in report["m1_trace_fidelity"]
    assert "spike_train_distance" in report["m1_trace_fidelity"]
    assert report["state_conditioned_fidelity"]["kind"] == "state_conditioned_dynamic_fidelity"
    assert report["trace_shape_fidelity"]["kind"] == "multiscale_trace_shape_fidelity"
    assert report["m1_trace_fidelity"]["spike_target_count"] == 1
    assert report["m1_trace_fidelity"]["spike_brier"] < 0.001
    assert report["m1_trace_fidelity"]["worst_trace_ids"]["soma_rmse"] == ["trace-a", "trace-b"]
    assert report["m1_trace_fidelity"]["per_trace_summary"]["rows"][0]["trace_id"] == "trace-a"
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
    assert metrics["mvp_regime_representatives"]["near_spike"]["worst_trace_ids"] == ["0"]
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
    assert metrics["mvp_regime_representatives"]["post_input_event"]["worst_trace_ids"] == ["0"]
    assert "post_input_event" in metrics["what_failure_means"]


def test_state_conditioned_metrics_weight_rare_states_separately():
    predictions = np.zeros((1, 20, 2), dtype=np.float32)
    targets = np.zeros((1, 20, 2), dtype=np.float32)
    inputs = np.zeros((1, 20, 4), dtype=np.float32)

    # Most of the trace is perfectly predicted resting voltage. A single
    # spiking/near-threshold region has a large voltage error that global RMSE
    # would dilute.
    targets[..., 1] = -10.0
    predictions[..., 1] = targets[..., 1]
    targets[0, 10, 0] = 1.0
    targets[0, 9:11, 1] = 8.0
    predictions[0, 9:11, 1] = -10.0
    inputs[0, 8:11, :2] = 1.0

    metrics = compute_state_conditioned_metrics(
        predictions,
        targets,
        inputs,
        soma_scale=1.0,
        soma_bias_mv=0.0,
        ignore_start=0,
    )

    rows = {row["state"]: row for row in metrics["states"]}
    assert rows["spiking"]["count"] == 1
    assert rows["spiking"]["voltage_rmse_mv"] == 18.0
    assert metrics["state_weighted_trace_distance_mv"] > np.sqrt(np.mean((predictions[..., 1] - targets[..., 1]) ** 2))
    assert metrics["dynamic_distance"] is not None


def test_trace_shape_metric_is_lag_tolerant_for_shifted_events():
    targets = np.zeros((1, 30, 2), dtype=np.float32)
    predictions = np.zeros_like(targets)
    waveform = np.array([0.0, 2.0, 8.0, 2.0, -1.0], dtype=np.float32)
    targets[0, 10:15, 1] = waveform
    predictions[0, 11:16, 1] = waveform
    targets[0, 12, 0] = 1.0
    predictions[0, 13, 0] = 8.0

    metrics = compute_trace_shape_metrics(
        predictions,
        targets,
        soma_scale=1.0,
        soma_bias_mv=0.0,
        ignore_start=0,
        lag_tolerance_steps=1,
    )

    assert metrics["lag_tolerant_voltage"]["symmetric_rmse_mv"] < metrics["pointwise"]["rmse_mv"]
    assert metrics["spike_triggered_waveform"]["matched_spikes"] == 1
    assert metrics["spike_triggered_waveform"]["waveform_rmse_mv"] == 0.0
    assert metrics["spike_train_distance"]["mean_normalized_van_rossum_distance"] > 0.0
    assert metrics["state_conditioned_trace_shape"]["state_trace_shape_distance"] >= 0.0
    assert metrics["trace_shape_distance"] >= 0.0


def test_spike_train_distance_rewards_nearby_spikes_more_than_missed_spikes():
    targets = np.zeros((1, 40, 2), dtype=np.float32)
    near_predictions = np.zeros_like(targets)
    missed_predictions = np.zeros_like(targets)
    targets[0, 20, 0] = 1.0
    near_predictions[0, 21, 0] = 8.0

    near = compute_trace_shape_metrics(
        near_predictions,
        targets,
        soma_scale=1.0,
        soma_bias_mv=0.0,
        spike_train_tau_steps=(10,),
    )
    missed = compute_trace_shape_metrics(
        missed_predictions,
        targets,
        soma_scale=1.0,
        soma_bias_mv=0.0,
        spike_train_tau_steps=(10,),
    )

    assert (
        near["spike_train_distance"]["mean_normalized_van_rossum_distance"]
        < missed["spike_train_distance"]["mean_normalized_van_rossum_distance"]
    )


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


def test_structured_intervention_input_transforms_are_targeted():
    inputs = np.zeros((1, 6, 8), dtype=np.int8)
    inputs[0, 1, 0] = 1
    inputs[0, 2, 1] = 1
    inputs[0, 3, 4] = -1
    inputs[0, 4, 5] = -1

    exc_dropped, exc_mask = make_selective_event_dropout_inputs(
        inputs,
        dropout_probability=1.0,
        event_sign=1,
        seed=0,
    )
    assert np.count_nonzero(exc_dropped == 1) == 0
    assert np.count_nonzero(exc_dropped == -1) == 2
    assert np.count_nonzero(exc_mask) == 2

    jittered, changed = make_temporal_jitter_inputs(inputs, max_jitter_steps=1, seed=2)
    assert np.count_nonzero(jittered) <= np.count_nonzero(inputs)
    assert changed.shape == inputs.shape

    silenced, silence_mask = make_site_silence_inputs(inputs, start_fraction=0.0, site_fraction=0.5)
    assert np.count_nonzero(silenced[..., [0, 1, 4, 5]]) == 0
    assert np.count_nonzero(silence_mask) == 4

    transformed, transformed_mask, meta = make_structured_intervention_inputs(
        inputs,
        intervention="inh_dropout",
        dropout_probability=1.0,
        seed=0,
    )
    assert np.count_nonzero(transformed == -1) == 0
    assert np.count_nonzero(transformed == 1) == 2
    assert transformed_mask.sum() == 2
    assert meta["intervention"] == "inh_dropout"


def test_morphology_transfer_reports_feature_and_route_mismatch(tmp_path):
    train_swc = tmp_path / "train.swc"
    test_swc = tmp_path / "test.swc"
    train_swc.write_text(
        "\n".join([
            "1 1 0 0 0 5 -1",
            "2 3 10 0 0 1 1",
            "3 3 20 0 0 1 2",
            "4 4 0 10 0 1 1",
            "5 4 0 20 0 1 4",
        ]) + "\n",
        encoding="utf-8",
    )
    test_swc.write_text(
        "\n".join([
            "1 1 0 0 0 5 -1",
            "2 3 30 0 0 1 1",
            "3 3 60 0 0 1 2",
            "4 4 0 30 0 1 1",
            "5 4 0 60 0 1 4",
            "6 4 0 90 0 1 5",
        ]) + "\n",
        encoding="utf-8",
    )

    evaluator = MorphologyTransferEvaluator(cache_dir=tmp_path)
    train = evaluator.load_morphology(train_swc, n_branches=2)
    test = evaluator.load_morphology(test_swc, n_branches=2)
    report = evaluator.compare_morphology_set(train, [test])

    assert report["status"] == "morphology_feature_comparison"
    assert report["train_features"]["total_dendritic_length_um"] > 0
    mismatch = report["test_morphologies"][0]["route_mismatch"]
    assert mismatch["routing_shape_match"] is True
    assert mismatch["total_length_ratio"] > 1.0
    assert "what_failure_means" in report


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
    assert report["model_response"]["soma_delta_gain"] == 1.0
    assert report["model_response"]["spike_count_delta_gain"] == 1.0
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


def test_compare_models_uses_publishable_metric_schema():
    inputs = np.zeros((1, 8, 4), dtype=np.float32)
    targets = np.zeros((1, 8, 2), dtype=np.float32)
    targets[..., 1] = 1.0
    targets[0, 4, 0] = 1.0
    baseline_inputs = inputs.copy()
    intervention_inputs = inputs.copy()
    intervention_inputs[:, 2:, 0] = 1.0
    baseline_targets = targets.copy()
    intervention_targets = targets.copy()
    intervention_targets[..., 1] = 0.5

    def perfect_model(x):
        if np.count_nonzero(x) > 0:
            return intervention_targets.copy()
        return baseline_targets.copy()

    def offset_model(x):
        out = perfect_model(x)
        out[..., 1] += 0.25
        return out

    suite = BenchmarkSuite()
    paired = {
        "baseline_inputs": baseline_inputs,
        "baseline_targets": baseline_targets,
        "intervention_inputs": intervention_inputs,
        "intervention_targets": intervention_targets,
        "intervention_name": "unit_compare",
    }
    reports = {
        "perfect": suite.evaluate_model(
            perfect_model,
            baseline_inputs,
            baseline_targets,
            soma_scale=1.0,
            ignore_start=0,
            run_perturbation=False,
            m4_paired_intervention=paired,
        ),
        "offset": suite.evaluate_model(
            offset_model,
            baseline_inputs,
            baseline_targets,
            soma_scale=1.0,
            ignore_start=0,
            run_perturbation=False,
            m4_paired_intervention=paired,
        ),
    }

    comparison = suite.compare_models(reports)

    assert comparison["m1_fidelity"]["perfect"]["soma_rmse_mv"] == 0.0
    assert comparison["m1_fidelity"]["offset"]["soma_rmse_mv"] > 0.0
    assert comparison["m1_fidelity"]["perfect"]["trace_shape_distance"] == 0.0
    assert comparison["m1_fidelity"]["perfect"]["state_trace_shape_distance"] == 0.0
    assert comparison["m1_fidelity"]["perfect"]["spike_train_distance"] == 0.0
    assert "dynamic_distance" in comparison["state_conditioned_fidelity"]["perfect"]
    assert "trace_shape_distance" in comparison["trace_shape_fidelity"]["perfect"]
    assert "spike_train_distance" in comparison["trace_shape_fidelity"]["perfect"]
    assert "spiking" in comparison["state_conditioned_fidelity"]["perfect"]["per_state_voltage_rmse_mv"]
    assert comparison["m1b_feature_fidelity"]["perfect"]["spike_count_mae"] == 0.0
    assert comparison["m4_intervention"]["perfect"]["status"] == "model_compared"
    assert comparison["m4_intervention"]["perfect"]["soma_delta_rmse_mv"] == 0.0
