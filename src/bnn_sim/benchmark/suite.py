"""NEURO-BENCH Suite: Combines all metric suites.

Usage:
    from bnn_sim.benchmark import BenchmarkSuite

    suite = BenchmarkSuite()
    results = suite.evaluate_model(
        model_fn=model.predict,
        inputs=test_inputs,
        targets=test_targets,
        output_dir="runs/benchmark_eval",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from bnn_sim.benchmark.regime_stratified import compute_regime_stratified_metrics
from bnn_sim.benchmark.state_metrics import compute_state_conditioned_metrics
from bnn_sim.benchmark.perturbation_stability import compute_perturbation_stability
from bnn_sim.benchmark.structured_intervention import compute_paired_event_intervention_metrics
from bnn_sim.benchmark.trace_shape import compute_trace_shape_metrics
from bnn_sim.official.constants import DEFAULT_Y_TRAIN_SOMA_BIAS


class BenchmarkSuite:
    """Combined benchmark suite for neuron surrogate evaluation."""

    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir) if output_dir else None

    def evaluate_model(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        inputs: np.ndarray,
        targets: np.ndarray,
        soma_scale: float = 0.1,
        soma_bias_mv: float = DEFAULT_Y_TRAIN_SOMA_BIAS,
        ignore_start: int = 500,
        run_perturbation: bool = True,
        perturbation_epsilons: list[float] | None = None,
        model_name: str | None = None,
        model_config: dict[str, Any] | None = None,
        dataset_path: str | None = None,
        trace_ids: list[str] | None = None,
        m4_paired_intervention: dict[str, Any] | None = None,
        calibrated_spike_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Run full benchmark suite on a model.

        Args:
            model_fn: Function(inputs) -> predictions.
            inputs: Input array (batch, time, channels).
            targets: Target array (batch, time, 2).
            soma_scale: Scale factor for soma voltage.
            soma_bias_mv: Bias added after inverse soma scaling to recover mV.
            ignore_start: Timesteps to ignore at start.
            run_perturbation: Whether to run perturbation stability.
            perturbation_epsilons: Epsilon values for perturbation test.

        Returns:
            Dictionary with all benchmark results.
        """
        inputs = np.asarray(inputs)
        targets = np.asarray(targets)

        # Get predictions
        predictions = np.asarray(model_fn(inputs))

        trace_ids = trace_ids or [str(i) for i in range(inputs.shape[0])]
        m1 = _compute_m1_trace_fidelity(
            predictions,
            targets,
            soma_scale,
            ignore_start,
            calibrated_spike_threshold=calibrated_spike_threshold,
        )
        m1b = _compute_m1b_feature_fidelity(
            predictions,
            targets,
            trace_ids,
            soma_scale,
            soma_bias_mv,
            ignore_start,
            calibrated_spike_threshold=calibrated_spike_threshold,
        )

        # M2: Regime-stratified errors
        m2 = compute_regime_stratified_metrics(
            predictions, targets, inputs,
            soma_scale=soma_scale,
            soma_bias_mv=soma_bias_mv,
            ignore_start=ignore_start,
            calibrated_spike_threshold=calibrated_spike_threshold,
        )
        state_metrics = compute_state_conditioned_metrics(
            predictions,
            targets,
            inputs,
            soma_scale=soma_scale,
            soma_bias_mv=soma_bias_mv,
            ignore_start=ignore_start,
            calibrated_spike_threshold=calibrated_spike_threshold,
        )
        m1["state_weighted_trace_distance_mv"] = state_metrics["state_weighted_trace_distance_mv"]
        m1["dynamic_distance"] = state_metrics["dynamic_distance"]
        m1["dynamic_distance_components"] = state_metrics["dynamic_distance_components"]
        trace_shape = compute_trace_shape_metrics(
            predictions,
            targets,
            inputs,
            soma_scale=soma_scale,
            soma_bias_mv=soma_bias_mv,
            ignore_start=ignore_start,
            calibrated_spike_threshold=calibrated_spike_threshold,
        )
        m1["trace_shape_distance"] = trace_shape["trace_shape_distance"]
        m1["lag_tolerant_voltage_rmse_mv"] = trace_shape["lag_tolerant_voltage"]["symmetric_rmse_mv"]
        m1["state_trace_shape_distance"] = trace_shape["state_conditioned_trace_shape"]["state_trace_shape_distance"]
        m1["spike_train_distance"] = trace_shape["spike_train_distance"]["mean_normalized_van_rossum_distance"]

        # M4: Perturbation stability
        m4 = {}
        if run_perturbation:
            # Use a subset for perturbation to save time
            subset_size = min(8, inputs.shape[0])
            m4 = compute_perturbation_stability(
                model_fn,
                inputs[:subset_size],
                epsilon_values=perturbation_epsilons,
            )

        if m4_paired_intervention is not None:
            m4_intervention = compute_paired_event_intervention_metrics(
                model_fn if m4_paired_intervention.get("compare_model", True) else None,
                m4_paired_intervention["baseline_inputs"],
                m4_paired_intervention["baseline_targets"],
                m4_paired_intervention["intervention_inputs"],
                m4_paired_intervention["intervention_targets"],
                intervention_name=m4_paired_intervention.get("intervention_name", "event_intervention"),
                soma_scale=soma_scale,
                soma_bias_mv=soma_bias_mv,
                ignore_start=m4_paired_intervention.get("ignore_start", ignore_start),
                trace_ids=m4_paired_intervention.get("trace_ids", trace_ids),
                model_spike_threshold=m4_paired_intervention.get(
                    "model_spike_threshold",
                    calibrated_spike_threshold if calibrated_spike_threshold is not None else 0.0,
                ),
            )
        else:
            m4_intervention = {
                "status": "debug_only" if run_perturbation else "not_run",
                "kind": "gaussian_perturbation_sanity_check" if run_perturbation else None,
                "what_failure_means": (
                    "This is only a numerical brittleness smoke test. It is not "
                    "evidence of biological robustness or global smoothness."
                ),
                "structured_intervention_required": True,
            }

        metadata = {
            "model_name": model_name,
            "model_config": model_config,
            "dataset_path": dataset_path,
            "trace_ids": trace_ids,
            "n_samples": inputs.shape[0],
            "n_timesteps": inputs.shape[1],
            "soma_scale": soma_scale,
            "soma_bias_mv": soma_bias_mv,
            "mask": {
                "ignore_start": ignore_start,
                "evaluated_timesteps_per_trace": max(0, inputs.shape[1] - ignore_start),
                "semantics": "drop the first ignore_start timesteps from every trace before metrics",
            },
            "calibrated_spike_threshold": calibrated_spike_threshold,
        }

        limitations = [
            "Gaussian perturbation output is debug-only when present.",
        ]
        if m4_paired_intervention is None:
            limitations.append("M4 structured intervention was not run for this report.")
        else:
            limitations.append(
                "M4 structured intervention results are limited to the supplied paired traces and should be expanded before robustness claims."
            )

        results = {
            "benchmark_version": "0.1.0",
            "metadata": metadata,
            "m1_trace_fidelity": m1,
            "m1b_feature_fidelity": m1b,
            "m2_regimes": m2,
            "state_conditioned_fidelity": state_metrics,
            "trace_shape_fidelity": trace_shape,
            "m4_intervention": m4_intervention,
            "limitations": limitations,
            # Legacy aliases retained for existing scripts while the benchmark
            # report moves to the MVP schema from spec.md.
            "m1_open_loop_fidelity": {
                "rmse_normalized": m1["soma_rmse_normalized"],
                "rmse_mv": m1["soma_rmse_mv"],
                "auc": m1["spike_auc"],
            },
            "m2_regime_stratified": m2,
            "m4_perturbation_stability": m4,
        }

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.output_dir / "benchmark_results.json", "w") as f:
                json.dump(results, f, indent=2)

        return results

    def compare_models(
        self,
        model_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Compare multiple models on benchmark metrics.

        Args:
            model_results: Dict mapping model_name -> benchmark results.

        Returns:
            Comparison table.
        """
        comparison = {
            "models": list(model_results.keys()),
            "m1_fidelity": {},
            "m1b_feature_fidelity": {},
            "state_conditioned_fidelity": {},
            "trace_shape_fidelity": {},
            "m2_regime_summary": {},
            "m4_intervention": {},
            "m4_stability": {},
        }

        for name, results in model_results.items():
            m1 = results.get("m1_trace_fidelity", {})
            if not m1:
                legacy_m1 = results.get("m1_open_loop_fidelity", {})
                m1 = {
                    "soma_rmse_mv": legacy_m1.get("rmse_mv"),
                    "spike_auc": legacy_m1.get("auc"),
                }
            comparison["m1_fidelity"][name] = {
                "soma_rmse_mv": m1.get("soma_rmse_mv"),
                "soma_mae_mv": m1.get("soma_mae_mv"),
                "state_weighted_trace_distance_mv": m1.get("state_weighted_trace_distance_mv"),
                "dynamic_distance": m1.get("dynamic_distance"),
                "trace_shape_distance": m1.get("trace_shape_distance"),
                "lag_tolerant_voltage_rmse_mv": m1.get("lag_tolerant_voltage_rmse_mv"),
                "state_trace_shape_distance": m1.get("state_trace_shape_distance"),
                "spike_train_distance": m1.get("spike_train_distance"),
                "spike_auc": m1.get("spike_auc"),
                "spike_brier": m1.get("spike_brier"),
                "spike_count_mae_per_trace": m1.get("spike_count_mae_per_trace"),
                "calibrated_spike_count_mae_per_trace": (
                    m1.get("spike_calibrated_threshold", {}).get("count_mae_per_trace")
                    if isinstance(m1.get("spike_calibrated_threshold"), dict)
                    else None
                ),
            }

            m1b = results.get("m1b_feature_fidelity", {})
            feature_block = (
                m1b.get("calibrated_threshold", {}).get("features")
                or m1b.get("features")
                or {}
            )
            comparison["m1b_feature_fidelity"][name] = {
                "spike_count_mae": _feature_mae(feature_block, "spike_count"),
                "firing_rate_mae_per_1000_steps": _feature_mae(feature_block, "firing_rate_per_1000_steps"),
                "peak_voltage_mae_mv": _feature_mae(feature_block, "peak_voltage_mv"),
                "trough_voltage_mae_mv": _feature_mae(feature_block, "trough_voltage_mv"),
                "non_spiking_mae": _feature_mae(feature_block, "non_spiking"),
            }

            state_metrics = results.get("state_conditioned_fidelity", {})
            comparison["state_conditioned_fidelity"][name] = {
                "state_weighted_trace_distance_mv": state_metrics.get("state_weighted_trace_distance_mv"),
                "dynamic_distance": state_metrics.get("dynamic_distance"),
                "dynamic_distance_components": state_metrics.get("dynamic_distance_components"),
                "spike_timing_f1_at_2steps": state_metrics.get("spike_timing", {}).get("f1_at_2steps"),
                "spike_count_mae_per_trace": state_metrics.get("spike_count_mae_per_trace"),
                "per_state_voltage_rmse_mv": _state_metric_map(state_metrics, "voltage_rmse_mv"),
                "per_state_fraction": _state_metric_map(state_metrics, "fraction"),
            }
            trace_shape = results.get("trace_shape_fidelity", {})
            comparison["trace_shape_fidelity"][name] = {
                "trace_shape_distance": trace_shape.get("trace_shape_distance"),
                "trace_shape_distance_components": trace_shape.get("trace_shape_distance_components"),
                "state_trace_shape_distance": (
                    trace_shape.get("state_conditioned_trace_shape", {}).get("state_trace_shape_distance")
                    if isinstance(trace_shape.get("state_conditioned_trace_shape"), dict)
                    else None
                ),
                "per_state_trace_shape_distance": _state_metric_map(
                    trace_shape.get("state_conditioned_trace_shape", {}),
                    "state_trace_shape_distance",
                ),
                "lag_tolerant_voltage_rmse_mv": (
                    trace_shape.get("lag_tolerant_voltage", {}).get("symmetric_rmse_mv")
                    if isinstance(trace_shape.get("lag_tolerant_voltage"), dict)
                    else None
                ),
                "lowpass_rmse_mv": {
                    str(row.get("window_steps")): row.get("rmse_mv")
                    for row in trace_shape.get("multiscale_lowpass", [])
                    if isinstance(row, dict)
                },
                "dvdt_mae_mv_per_step": trace_shape.get("pointwise", {}).get("dvdt_mae_mv_per_step"),
                "voltage_quantile_mae_mv": (
                    trace_shape.get("voltage_quantile_distance", {}).get("mean_abs_quantile_error_mv")
                    if isinstance(trace_shape.get("voltage_quantile_distance"), dict)
                    else None
                ),
                "spike_waveform_rmse_mv": (
                    trace_shape.get("spike_triggered_waveform", {}).get("waveform_rmse_mv")
                    if isinstance(trace_shape.get("spike_triggered_waveform"), dict)
                    else None
                ),
                "spike_waveform_match_fraction": (
                    trace_shape.get("spike_triggered_waveform", {}).get("match_fraction")
                    if isinstance(trace_shape.get("spike_triggered_waveform"), dict)
                    else None
                ),
                "spike_train_distance": (
                    trace_shape.get("spike_train_distance", {}).get("mean_normalized_van_rossum_distance")
                    if isinstance(trace_shape.get("spike_train_distance"), dict)
                    else None
                ),
            }

            m2 = results.get("m2_regimes") or results.get("m2_regime_stratified", {})
            comparison["m2_regime_summary"][name] = {
                "near_spike_rmse": _regime_rmse(m2, "spike_proximity", "near_spike"),
                "far_from_spike_rmse": _regime_rmse(m2, "spike_proximity", "far_from_spike"),
                "post_spike_refractory_rmse": _regime_rmse(m2, "mvp_regimes", "post_spike_refractory"),
                "post_input_event_rmse": _regime_rmse(m2, "mvp_regimes", "post_input_event"),
                "high_dvdt_rmse": _regime_rmse(m2, "slope_regime", "spike_rise"),
            }

            comparison["m4_intervention"][name] = _m4_intervention_summary(results.get("m4_intervention", {}))
            m4 = results.get("m4_perturbation_stability", {})
            sc = m4.get("stability_coefficient", {})
            comparison["m4_stability"][name] = {
                "soma_stability_coeff": sc.get("soma"),
                "spike_stability_coeff": sc.get("spike"),
            }

        return comparison


def _feature_mae(features: dict[str, Any], name: str) -> float | None:
    row = features.get(name)
    return row.get("mean_abs_error") if isinstance(row, dict) else None


def _state_metric_map(state_metrics: dict[str, Any], field: str) -> dict[str, Any]:
    return {
        row["state"]: row.get(field)
        for row in state_metrics.get("states", [])
        if isinstance(row, dict) and "state" in row
    }


def _regime_rmse(m2: dict[str, Any], section: str, label: str) -> float | None:
    return next(
        (
            row.get("rmse")
            for row in m2.get(section, [])
            if row.get("regime_label") == label
        ),
        None,
    )


def _m4_intervention_summary(m4: dict[str, Any]) -> dict[str, Any]:
    teacher = m4.get("teacher_response", {}) if isinstance(m4, dict) else {}
    model = m4.get("model_response", {}) if isinstance(m4, dict) else {}
    direction = model.get("direction_agreement", {}) if isinstance(model, dict) else {}
    return {
        "status": m4.get("status") if isinstance(m4, dict) else None,
        "intervention_name": m4.get("intervention_name") if isinstance(m4, dict) else None,
        "teacher_response_detected": teacher.get("response_detected"),
        "teacher_direction": teacher.get("teacher_direction"),
        "teacher_delta_soma_mean_mv": teacher.get("delta_soma_mean_mv"),
        "teacher_delta_spike_rate_per_1000_steps": teacher.get("delta_spike_rate_per_1000_steps"),
        "soma_delta_rmse_mv": model.get("soma_delta_rmse_mv") if isinstance(model, dict) else None,
        "spike_count_delta_mae_per_trace": (
            model.get("spike_count_delta_mae_per_trace") if isinstance(model, dict) else None
        ),
        "soma_direction_agreement": direction.get("soma_mean"),
        "spike_direction_agreement": direction.get("spike_count"),
    }


def _apply_time_mask(array: np.ndarray, ignore_start: int) -> np.ndarray:
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim != 3:
        leading = int(np.prod(array.shape[:-2]))
        array = array.reshape(leading, array.shape[-2], array.shape[-1])
    return array[:, ignore_start:, :]


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -60.0, 60.0)))


def _compute_m1_trace_fidelity(
    predictions: np.ndarray,
    targets: np.ndarray,
    soma_scale: float,
    ignore_start: int,
    *,
    calibrated_spike_threshold: float | None = None,
) -> dict[str, Any]:
    pred = _apply_time_mask(np.asarray(predictions), ignore_start)
    tgt = _apply_time_mask(np.asarray(targets), ignore_start)
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction and target shapes must match, got {pred.shape} and {tgt.shape}")
    if pred.shape[1] == 0:
        raise ValueError("ignore_start leaves no timesteps to evaluate")

    residual = pred[..., 1] - tgt[..., 1]
    residual_mv = residual / soma_scale
    spike_scores = pred[..., 0]
    spike_labels = tgt[..., 0]
    spike_probs = _sigmoid(spike_scores)

    from bnn_sim.metrics import binary_auc

    clipped = np.clip(spike_probs, 1e-7, 1.0 - 1e-7)
    target_binary = spike_labels > 0.5
    spike_binary = spike_scores > 0.0
    matched_threshold = _prevalence_matched_threshold(spike_scores, int(target_binary.sum()))
    matched_binary = spike_scores >= matched_threshold
    pred_counts = spike_binary.sum(axis=1)
    matched_counts = matched_binary.sum(axis=1)
    target_counts = target_binary.sum(axis=1)

    result = {
        "soma_rmse_normalized": float(np.sqrt(np.mean(residual ** 2))),
        "soma_rmse_mv": float(np.sqrt(np.mean(residual_mv ** 2))),
        "soma_mae_mv": float(np.mean(np.abs(residual_mv))),
        "spike_auc": float(binary_auc(spike_scores, spike_labels)),
        "spike_brier": float(np.mean((spike_probs - spike_labels) ** 2)),
        "spike_log_loss": float(-np.mean(spike_labels * np.log(clipped) + (1.0 - spike_labels) * np.log(1.0 - clipped))),
        "spike_target_count": int(target_binary.sum()),
        "spike_predicted_count": int(spike_binary.sum()),
        "spike_count_mae_per_trace": float(np.mean(np.abs(pred_counts - target_counts))),
        "spike_threshold_zero": {
            "threshold": 0.0,
            "predicted_count": int(spike_binary.sum()),
            "count_mae_per_trace": float(np.mean(np.abs(pred_counts - target_counts))),
            "semantics": "fixed logit threshold; use this as a calibration/operating-point check",
        },
        "spike_prevalence_matched": {
            "threshold": float(matched_threshold),
            "predicted_count": int(matched_binary.sum()),
            "count_mae_per_trace": float(np.mean(np.abs(matched_counts - target_counts))),
            "semantics": "diagnostic threshold chosen so total predicted spike count matches the teacher under the current mask",
        },
        "what_failure_means": {
            "soma": "Large RMSE/MAE means the surrogate misses open-loop voltage traces under the stated mask.",
            "spike": "Poor AUC/Brier/log-loss means spike scores do not rank or calibrate event likelihoods well.",
        },
    }
    if calibrated_spike_threshold is not None:
        calibrated_binary = spike_scores >= calibrated_spike_threshold
        calibrated_counts = calibrated_binary.sum(axis=1)
        result["spike_calibrated_threshold"] = {
            "threshold": float(calibrated_spike_threshold),
            "predicted_count": int(calibrated_binary.sum()),
            "count_mae_per_trace": float(np.mean(np.abs(calibrated_counts - target_counts))),
            "semantics": "fixed logit threshold learned on a separate calibration split and frozen before evaluation",
        }
    return result


def _trace_features(spikes: np.ndarray, soma_mv: np.ndarray) -> dict[str, float]:
    spike_idx = np.flatnonzero(spikes)
    isis = np.diff(spike_idx)
    return {
        "spike_count": float(spike_idx.size),
        "firing_rate_per_1000_steps": float(1000.0 * spike_idx.size / max(1, spikes.size)),
        "first_isi_steps": float(isis[0]) if isis.size else float("nan"),
        "last_isi_steps": float(isis[-1]) if isis.size else float("nan"),
        "mean_isi_steps": float(np.mean(isis)) if isis.size else float("nan"),
        "adaptation_ratio": float(isis[-1] / isis[0]) if isis.size >= 2 and isis[0] != 0 else float("nan"),
        "peak_voltage_mv": float(np.max(soma_mv)),
        "trough_voltage_mv": float(np.min(soma_mv)),
        "non_spiking": float(spike_idx.size == 0),
    }


def _nanmean(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    if np.isnan(arr).all():
        return None
    return float(np.nanmean(arr))


def _compute_m1b_feature_fidelity(
    predictions: np.ndarray,
    targets: np.ndarray,
    trace_ids: list[str],
    soma_scale: float,
    soma_bias_mv: float,
    ignore_start: int,
    *,
    calibrated_spike_threshold: float | None = None,
) -> dict[str, Any]:
    pred = _apply_time_mask(np.asarray(predictions), ignore_start)
    tgt = _apply_time_mask(np.asarray(targets), ignore_start)
    pred_spikes = pred[..., 0] > 0.0
    target_spikes = tgt[..., 0] > 0.5
    matched_threshold = _prevalence_matched_threshold(pred[..., 0], int(target_spikes.sum()))
    matched_pred_spikes = pred[..., 0] >= matched_threshold
    pred_soma_mv = pred[..., 1] / soma_scale + soma_bias_mv
    target_soma_mv = tgt[..., 1] / soma_scale + soma_bias_mv

    features, worst = _feature_fidelity_rows(pred_spikes, target_spikes, pred_soma_mv, target_soma_mv, trace_ids)
    matched_features, matched_worst = _feature_fidelity_rows(
        matched_pred_spikes,
        target_spikes,
        pred_soma_mv,
        target_soma_mv,
        trace_ids,
    )

    result = {
        "features": features,
        "worst_trace_ids": worst,
        "threshold_zero": {
            "threshold": 0.0,
            "features": features,
            "worst_trace_ids": worst,
            "semantics": "fixed logit threshold; use this as a calibration/operating-point check",
        },
        "prevalence_matched": {
            "threshold": float(matched_threshold),
            "features": matched_features,
            "worst_trace_ids": matched_worst,
            "semantics": "diagnostic threshold chosen so total predicted spike count matches the teacher under the current mask",
        },
        "what_failure_means": (
            "Feature errors indicate mismatches in practitioner-visible trace "
            "properties such as spike count, firing rate, adaptation, and voltage extrema."
        ),
    }
    if calibrated_spike_threshold is not None:
        calibrated_pred_spikes = pred[..., 0] >= calibrated_spike_threshold
        calibrated_features, calibrated_worst = _feature_fidelity_rows(
            calibrated_pred_spikes,
            target_spikes,
            pred_soma_mv,
            target_soma_mv,
            trace_ids,
        )
        result["calibrated_threshold"] = {
            "threshold": float(calibrated_spike_threshold),
            "features": calibrated_features,
            "worst_trace_ids": calibrated_worst,
            "semantics": "fixed logit threshold learned on a separate calibration split and frozen before evaluation",
        }
    return result


def _feature_fidelity_rows(
    pred_spikes: np.ndarray,
    target_spikes: np.ndarray,
    pred_soma_mv: np.ndarray,
    target_soma_mv: np.ndarray,
    trace_ids: list[str],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    pred_features = []
    target_features = []
    for index in range(pred_spikes.shape[0]):
        pred_features.append(_trace_features(pred_spikes[index], pred_soma_mv[index]))
        target_features.append(_trace_features(target_spikes[index], target_soma_mv[index]))

    feature_names = list(target_features[0]) if target_features else []
    features: dict[str, Any] = {}
    worst: dict[str, list[str]] = {}
    for name in feature_names:
        target_values = [row[name] for row in target_features]
        pred_values = [row[name] for row in pred_features]
        abs_errors = np.abs(np.asarray(pred_values, dtype=np.float64) - np.asarray(target_values, dtype=np.float64))
        features[name] = {
            "teacher_mean": _nanmean(target_values),
            "prediction_mean": _nanmean(pred_values),
            "mean_abs_error": _nanmean(abs_errors.tolist()),
        }
        bad = np.flatnonzero(np.nan_to_num(abs_errors, nan=0.0) > 0)
        worst[f"{name}_abs_error"] = [trace_ids[i] for i in bad[:5]]
    return features, worst


def _prevalence_matched_threshold(scores: np.ndarray, positive_count: int) -> float:
    flat = np.asarray(scores).reshape(-1)
    if positive_count <= 0:
        return float("inf")
    if positive_count >= flat.size:
        return float("-inf")
    return float(np.partition(flat, flat.size - positive_count)[flat.size - positive_count])
