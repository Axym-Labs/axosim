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
from bnn_sim.benchmark.perturbation_stability import compute_perturbation_stability
from bnn_sim.benchmark.structured_intervention import compute_paired_event_intervention_metrics
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
            "m2_regime_summary": {},
            "m4_stability": {},
        }

        for name, results in model_results.items():
            m1 = results.get("m1_open_loop_fidelity", {})
            comparison["m1_fidelity"][name] = {
                "rmse_mv": m1.get("rmse_mv"),
                "auc": m1.get("auc"),
            }

            m2 = results.get("m2_regime_stratified", {})
            comparison["m2_regime_summary"][name] = {
                "near_spike_rmse": next(
                    (r["rmse"] for r in m2.get("spike_proximity", [])
                     if r.get("regime_label") == "near_spike"),
                    None
                ),
                "high_dvdt_rmse": next(
                    (r["rmse"] for r in m2.get("slope_regime", [])
                     if r.get("regime_label") == "spike_rise"),
                    None
                ),
            }

            m4 = results.get("m4_perturbation_stability", {})
            sc = m4.get("stability_coefficient", {})
            comparison["m4_stability"][name] = {
                "soma_stability_coeff": sc.get("soma"),
                "spike_stability_coeff": sc.get("spike"),
            }

        return comparison


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
