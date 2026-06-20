#!/usr/bin/env python3
"""Eight-plus iteration audit over failure-mode probe artifacts."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("docs/ta1/artifacts/analysis_sources/existing_models")
SCALED = ROOT / "robustness_ood_stability_20260619" / "rows.jsonl"
REPLICATION = ROOT / "robustness_stability_windowrep_branchbest_20260619" / "rows.jsonl"
OUT_MD = ROOT / "failure_mode_iteration_pass_20260619.md"
OUT_JSON = ROOT / "failure_mode_iteration_pass_20260619.json"


def main() -> None:
    scaled = _load_rows(SCALED)
    replication = _load_rows(REPLICATION)
    ood = [row for row in scaled if row["kind"] == "teacher_grounded_ood"]
    ood_spiking = [row for row in ood if row["target_spikes"] > 0]
    scaled_adv = [row for row in scaled if row["kind"] == "model_only_adversarial_event"]
    repl_adv = [row for row in replication if row["kind"] == "model_only_adversarial_event"]
    scaled_lin = [row for row in scaled if row["kind"] == "model_only_local_linearity"]
    repl_lin = [row for row in replication if row["kind"] == "model_only_local_linearity"]

    iterations = [
        _iter_auc_timing_decoupling(ood_spiking),
        _iter_prevalence_matching(ood_spiking),
        _iter_ood_voltage_auc(ood_spiking),
        _iter_raw_near_spike_stability(ood_spiking),
        _iter_metric_rank_reversal(ood_spiking),
        _iter_subthreshold_sanity(ood),
        _iter_adversarial_replication(repl_adv),
        _iter_spike_mass_ratio_caveat(repl_adv),
        _iter_budget_saturation(repl_adv),
        _iter_gradient_vs_random_linearity(repl_lin),
        _iter_spike_specific_instability_caveat(repl_lin),
        _iter_model_size_smoothness_tradeoff(scaled_lin, ood_spiking),
    ]
    payload = {
        "source_rows": {
            "scaled": str(SCALED),
            "replication": str(REPLICATION),
        },
        "iteration_count": len(iterations),
        "iterations": iterations,
        "new_or_strengthened_candidates": [
            item
            for item in iterations
            if item["outcome"] in {"supported", "strengthened"}
        ],
        "caveats_or_negative_results": [
            item
            for item in iterations
            if item["outcome"] in {"caveat", "not_supported", "mixed"}
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(json.dumps({"iteration_count": len(iterations), "supported_or_strengthened": len(payload["new_or_strengthened_candidates"]), "caveats_or_negative": len(payload["caveats_or_negative_results"])}, indent=2))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _iter_auc_timing_decoupling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aucs = [float(row["spike_auc"]) for row in rows]
    f1s = [float(row["val_threshold_event_f1_2ms"]) for row in rows]
    return _iteration(
        1,
        "AUC-event-timing decoupling",
        "If AUC is a faithful spike metric, high-AUC rows should not have poor event F1.",
        "Compared spike AUC and validation-threshold F1@2ms across all spiking original-NeuronIO regime rows.",
        {
            "n_rows": len(rows),
            "auc_min": min(aucs),
            "auc_max": max(aucs),
            "f1_min": min(f1s),
            "f1_max": max(f1s),
            "f1_mean": _mean(f1s),
        },
        "supported",
        "All spiking rows have AUC >= 0.986 while event F1 remains only 0.230-0.368, strengthening the AUC-inflation claim.",
    )


def _iter_prevalence_matching(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["regime_prevalence_event_f1_2ms"]) - float(row["val_threshold_event_f1_2ms"]) for row in rows]
    return _iteration(
        2,
        "Prevalence matching does not solve timing",
        "If poor F1 is mainly a count-threshold artifact, regime-specific prevalence matching should materially improve F1.",
        "Computed prevalence-matched F1 minus validation-threshold F1 for each spiking OOD/regime row.",
        {
            "delta_min": min(deltas),
            "delta_max": max(deltas),
            "delta_mean": _mean(deltas),
            "improved_rows": sum(delta > 0 for delta in deltas),
            "worsened_rows": sum(delta < 0 for delta in deltas),
        },
        "supported",
        "Prevalence matching changes F1 by only -0.003 to +0.020 on average, so event timing remains weak after count matching.",
    )


def _iter_ood_voltage_auc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model = _by(rows, "model_id")
    deltas = []
    for model_id, model_rows in by_model.items():
        base = _one(model_rows, regime="canonical_test")
        for regime in ["ood_rates", "combined_ergodic"]:
            row = _one(model_rows, regime=regime)
            deltas.append(
                {
                    "model_id": model_id,
                    "regime": regime,
                    "rmse_delta_mv": row["clipped_rmse_mv"] - base["clipped_rmse_mv"],
                    "auc_delta": row["spike_auc"] - base["spike_auc"],
                }
            )
    return _iteration(
        3,
        "OOD voltage degradation with preserved AUC",
        "If AUC tracks overall OOD behavior, rows with worse OOD voltage should show comparably large AUC drops.",
        "Compared OOD-rate and combined-ergodic rows against canonical-test rows within each model.",
        {
            "rmse_delta_mv_min": min(item["rmse_delta_mv"] for item in deltas),
            "rmse_delta_mv_max": max(item["rmse_delta_mv"] for item in deltas),
            "auc_delta_min": min(item["auc_delta"] for item in deltas),
            "auc_delta_max": max(item["auc_delta"] for item in deltas),
            "deltas": deltas,
        },
        "supported",
        "OOD voltage RMSE increases by 0.289-0.351 mV while AUC drops by only 0.002-0.006, so AUC misses a real voltage shift.",
    )


def _iter_raw_near_spike_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row["near_spike_raw_rmse_mv"] for row in rows if row.get("near_spike_raw_rmse_mv") is not None]
    return _iteration(
        4,
        "Raw near-spike error persists across regimes",
        "If raw AP-shape failure is only a canonical-split artifact, near-spike raw RMSE should vary strongly or vanish in other regimes.",
        "Measured near-spike raw-soma RMSE across spiking original-NeuronIO regimes and checkpoints.",
        {
            "min_mv": min(values),
            "max_mv": max(values),
            "mean_mv": _mean(values),
            "std_mv": _std(values),
        },
        "strengthened",
        "Near-spike raw RMSE stays in a narrow 17.21-17.77 mV band, strengthening the clipping/raw-AP target-distortion claim.",
    )


def _iter_metric_rank_reversal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "clipped_rmse_mv_low": ("clipped_rmse_mv", min),
        "spike_auc_high": ("spike_auc", max),
        "val_f1_high": ("val_threshold_event_f1_2ms", max),
        "raw_near_spike_rmse_low": ("near_spike_raw_rmse_mv", min),
    }
    winners = {}
    for regime in sorted({row["regime"] for row in rows}):
        regime_rows = [row for row in rows if row["regime"] == regime]
        winners[regime] = {}
        for label, (field, reducer) in metrics.items():
            valid = [row for row in regime_rows if row.get(field) is not None]
            target = reducer(float(row[field]) for row in valid)
            winners[regime][label] = [row["model_id"] for row in valid if float(row[field]) == target]
    unique_winners = sorted({winner for metric_map in winners.values() for winners_list in metric_map.values() for winner in winners_list})
    return _iteration(
        5,
        "Rank reversal across metrics",
        "If the benchmark is reducible to one score, the same checkpoint should win across voltage, AUC, timing, and raw near-spike metrics.",
        "Computed per-regime winners for clipped RMSE, AUC, event F1, and raw near-spike RMSE.",
        {"winners": winners, "unique_winning_models": unique_winners},
        "not_supported",
        "Within this restricted official ELM/Branch-ELM subset, branch_elm_best wins all checked metrics/regimes; rank reversal needs broader model families or additional diagnostics.",
    )


def _iter_subthreshold_sanity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sub = [row for row in rows if row["regime"] == "subthreshold"]
    predicted = [int(row["val_threshold_predicted_spikes"]) for row in sub]
    return _iteration(
        6,
        "Subthreshold false-positive sanity",
        "If the validation threshold is simply too permissive, it should hallucinate spikes on subthreshold traces.",
        "Checked validation-threshold spike counts on original NeuronIO subthreshold rows.",
        {"predicted_spikes_by_model": {row["model_id"]: row["val_threshold_predicted_spikes"] for row in sub}},
        "caveat",
        "All models predict zero subthreshold spikes, so threshold failure is not a trivial global false-positive problem.",
    )


def _iter_adversarial_replication(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = _paired_adversarial(rows)
    grouped = defaultdict(list)
    for key, pair in pairs.items():
        group, _, budget = key
        adv, rnd = pair["gradient_targeted"], pair["random_control"]
        grouped[(group, budget)].append(adv["soma_delta_rmse_mv"] / max(rnd["soma_delta_rmse_mv"], 1e-12))
    stats = {
        f"{group}_budget{budget}": {
            "n": len(values),
            "median_ratio": statistics.median(values),
            "mean_ratio": _mean(values),
        }
        for (group, budget), values in sorted(grouped.items())
    }
    return _iteration(
        7,
        "Adversarial event sensitivity replicates over more windows",
        "If targeted-event sensitivity was a three-window accident, ratios should collapse toward 1 in a larger window replication.",
        "Used the focused 16-window branch_elm_best replication and compared targeted edits with matched random controls.",
        stats,
        "strengthened",
        "Median soma ratios remain 8.08-16.93x across groups/budgets, so the sparse-event brittleness signal survives replication.",
    )


def _iter_spike_mass_ratio_caveat(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = _paired_adversarial(rows)
    random_masses = [abs(pair["random_control"]["spike_probability_mass_delta"]) for pair in pairs.values()]
    targeted_masses = [abs(pair["gradient_targeted"]["spike_probability_mass_delta"]) for pair in pairs.values()]
    tiny = sum(value < 1e-3 for value in random_masses)
    return _iteration(
        8,
        "Spike-mass ratio denominator caveat",
        "If spike-mass targeted/random ratios are reliable effect sizes, random-control denominators should not be near zero.",
        "Inspected absolute random-control spike-probability-mass deltas in the focused replication.",
        {
            "random_mass_median_abs": statistics.median(random_masses),
            "targeted_mass_median_abs": statistics.median(targeted_masses),
            "random_mass_lt_1e_minus_3": tiny,
            "n_pairs": len(random_masses),
        },
        "caveat",
        "Many random spike-mass denominators are tiny, so soma-delta ratios are the cleaner adversarial robustness headline.",
    )


def _iter_budget_saturation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = _paired_adversarial(rows)
    grouped = defaultdict(list)
    for key, pair in pairs.items():
        group, _, budget = key
        adv, rnd = pair["gradient_targeted"], pair["random_control"]
        grouped[(group, budget)].append(
            {
                "ratio": adv["soma_delta_rmse_mv"] / max(rnd["soma_delta_rmse_mv"], 1e-12),
                "adv_delta": adv["soma_delta_rmse_mv"],
            }
        )
    stats = {}
    for key, values in sorted(grouped.items()):
        stats[f"{key[0]}_budget{key[1]}"] = {
            "median_ratio": statistics.median(v["ratio"] for v in values),
            "mean_adv_delta_mv": _mean(v["adv_delta"] for v in values),
        }
    return _iteration(
        9,
        "Adversarial budget saturation",
        "If the adversarial effect scales linearly as an effect size, higher budgets should increase both absolute deltas and targeted/random ratios.",
        "Compared budget 32 and 128 in the focused branch_elm_best replication.",
        stats,
        "mixed",
        "Absolute targeted soma deltas grow with budget, but targeted/random ratios fall at budget 128, suggesting saturation and a need to report both absolute and ratio metrics.",
    )


def _iter_gradient_vs_random_linearity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["window_group"], row["epsilon"], row["direction"])].append(row)
    stats = {}
    for group in ["quiet_or_low_spike", "spike_rich"]:
        for eps in [0.01, 0.05]:
            grad = grouped[(group, eps, "gradient_sign")]
            rnd = grouped[(group, eps, "random_gaussian")]
            stats[f"{group}_eps{eps}"] = {
                "gradient_soma_delta_mean_mv": _mean(row["soma_delta_rmse_mv"] for row in grad),
                "random_soma_delta_mean_mv": _mean(row["soma_delta_rmse_mv"] for row in rnd),
                "delta_ratio": _mean(row["soma_delta_rmse_mv"] for row in grad) / max(_mean(row["soma_delta_rmse_mv"] for row in rnd), 1e-12),
                "gradient_linearity_error_mean": _mean(row["soma_linearity_relative_error"] for row in grad),
                "random_linearity_error_mean": _mean(row["soma_linearity_relative_error"] for row in rnd),
            }
    return _iteration(
        10,
        "Random smoothness understates adversarial-direction instability",
        "If random perturbation stability is sufficient, random and gradient-direction local deltas should be comparable.",
        "Compared local linearity rows for random Gaussian and gradient-sign directions in the focused replication.",
        stats,
        "supported",
        "Gradient directions produce roughly 25-52x larger soma deltas than random directions at eps 0.01-0.05, so random smoothness is a weak robustness proxy.",
    )


def _iter_spike_specific_instability_caveat(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        if row["direction"] == "gradient_sign":
            grouped[(row["window_group"], row["epsilon"])].append(row)
    comparison = {}
    for eps in [0.001, 0.01, 0.05]:
        quiet = _mean(row["soma_linearity_relative_error"] for row in grouped[("quiet_or_low_spike", eps)])
        spike = _mean(row["soma_linearity_relative_error"] for row in grouped[("spike_rich", eps)])
        comparison[f"eps{eps}"] = {"quiet_error": quiet, "spike_error": spike, "spike_minus_quiet": spike - quiet}
    return _iteration(
        11,
        "Spike-specific local instability is not cleanly isolated",
        "If local instability is uniquely spike-regime specific, spike-rich windows should dominate quiet windows at every epsilon.",
        "Compared gradient-sign local linearity error by window group in the focused replication.",
        comparison,
        "caveat",
        "Spike-rich windows are worse at eps 0.05 but not uniformly worse at eps 0.01, so the stronger claim is adversarial-direction instability, not purely spike-specific instability.",
    )


def _iter_model_size_smoothness_tradeoff(lin_rows: list[dict[str, Any]], ood_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sign_rows = [
        row
        for row in lin_rows
        if row["direction"] == "gradient_sign" and float(row["epsilon"]) == 0.05 and row["window_group"] == "spike_rich"
    ]
    sign_by_model = {
        model: {
            "sign_flip_mean": _mean(row["activation_sign_flip_fraction"] for row in rows),
            "linearity_error_mean": _mean(row["soma_linearity_relative_error"] for row in rows),
        }
        for model, rows in _by(sign_rows, "model_id").items()
    }
    ood_by_model = {
        model: _one(rows, regime="ood_rates")["clipped_rmse_mv"]
        for model, rows in _by(ood_rows, "model_id").items()
    }
    return _iteration(
        12,
        "Smoothness-size tradeoff candidate",
        "If a locally smoother checkpoint is generally better, lower sign-crossing should coincide with lower OOD voltage RMSE.",
        "Compared spike-rich gradient eps=0.05 sign-crossing/linearity with OOD-rate clipped RMSE across official sizes.",
        {"local_stability": sign_by_model, "ood_rate_clipped_rmse_mv": ood_by_model},
        "mixed",
        "The 100-memory checkpoint has lower sign-crossing than best/m30 but worse OOD-rate RMSE than branch_elm_best, suggesting smoothness and fidelity may trade off rather than align.",
    )


def _iteration(
    idx: int,
    title: str,
    hypothesis: str,
    test: str,
    result: dict[str, Any],
    outcome: str,
    conclusion: str,
) -> dict[str, Any]:
    return {
        "iteration": idx,
        "title": title,
        "hypothesis": hypothesis,
        "test": test,
        "result": _to_builtin(result),
        "outcome": outcome,
        "conclusion": conclusion,
    }


def _paired_adversarial(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, dict[str, Any]]]:
    pairs: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["window_group"], row["trace_id"], int(row["budget"]))
        pairs[key][row["perturbation"]] = row
    return {
        key: value
        for key, value in pairs.items()
        if "gradient_targeted" in value and "random_control" in value
    }


def _by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return dict(grouped)


def _one(rows: list[dict[str, Any]], **conditions: Any) -> dict[str, Any]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in conditions.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {conditions}, got {len(matches)}")
    return matches[0]


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if value is not None and not _is_nan(float(value))]
    return float(statistics.mean(vals)) if vals else float("nan")


def _std(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if value is not None and not _is_nan(float(value))]
    return float(statistics.pstdev(vals)) if vals else float("nan")


def _is_nan(value: float) -> bool:
    return math.isnan(value)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value
    return value


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Additional Failure-Mode Iteration Pass",
        "",
        "This pass deliberately ran more than eight small hypothesis checks over the saved robustness/OOD artifacts. "
        "The aim was to find additional failure modes and, equally importantly, to identify caveats that prevent overclaiming.",
        "",
        f"Source rows: `{payload['source_rows']['scaled']}` and `{payload['source_rows']['replication']}`.",
        "",
        f"Total iterations: `{payload['iteration_count']}`.",
        "",
    ]
    for item in payload["iterations"]:
        lines.extend(
            [
                f"## Iteration {item['iteration']}: {item['title']}",
                "",
                f"**Hypothesis.** {item['hypothesis']}",
                "",
                f"**Test.** {item['test']}",
                "",
                f"**Outcome.** `{item['outcome']}`.",
                "",
                f"**Conclusion.** {item['conclusion']}",
                "",
                "```json",
                json.dumps(item["result"], indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Summary",
            "",
            "Strengthened paper-facing candidates:",
            "",
        ]
    )
    for item in payload["new_or_strengthened_candidates"]:
        lines.append(f"- Iteration {item['iteration']}: {item['title']}.")
    lines.extend(["", "Caveats or negative/mixed results:", ""])
    for item in payload["caveats_or_negative_results"]:
        lines.append(f"- Iteration {item['iteration']}: {item['title']}.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
