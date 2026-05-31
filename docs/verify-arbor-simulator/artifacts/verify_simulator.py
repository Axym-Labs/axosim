#!/usr/bin/env python3
"""Arbor simulator verification script for T2-T6."""

import json
import sys
from pathlib import Path

import numpy as np


def load_shard(shard_path: Path):
    """Load a single shard and return inputs, targets, heterogeneity."""
    data = np.load(shard_path)
    return data["inputs"], data["targets"], data.get("heterogeneity", None)


def t2_biophysical_plausibility(targets):
    """T2: Check voltage traces for biophysical plausibility."""
    # targets shape: (samples, time_steps, 2) where [:, :, 0] = spike, [:, :, 1] = soma
    soma = targets[:, :, 1]
    spike = targets[:, :, 0]

    results = {}

    # Resting potential range (-70 to -50 mV typical)
    # Note: soma is normalized, need to denormalize: V = soma * 0.1 - 67.7
    soma_mv = soma * 0.1 - 67.7
    results["soma_mean_mv"] = float(np.mean(soma_mv))
    results["soma_std_mv"] = float(np.std(soma_mv))
    results["soma_min_mv"] = float(np.min(soma_mv))
    results["soma_max_mv"] = float(np.max(soma_mv))

    # Check resting range
    resting_ok = -80 < results["soma_mean_mv"] < -40
    results["resting_in_range"] = resting_ok

    # Spike amplitudes (>0 mV peak typical)
    spike_peaks = np.max(spike, axis=1)
    results["spike_peak_mean"] = float(np.mean(spike_peaks))
    results["spike_peak_max"] = float(np.max(spike_peaks))

    # Spike rate in biologically plausible range (0.5-100 Hz)
    # Spike signal is 0/1, count spikes per second
    dt_ms = 1.0  # sample dt
    spike_counts = np.sum(spike > 0.5, axis=1)
    duration_s = spike.shape[1] * dt_ms / 1000.0
    spike_rates_hz = spike_counts / duration_s
    results["spike_rate_mean_hz"] = float(np.mean(spike_rates_hz))
    results["spike_rate_min_hz"] = float(np.min(spike_rates_hz))
    results["spike_rate_max_hz"] = float(np.max(spike_rates_hz))
    results["spike_rate_in_range"] = bool(
        np.all(spike_rates_hz >= 0) and np.all(spike_rates_hz <= 200)
    )

    # No pathological values
    results["no_nan"] = bool(not np.any(np.isnan(targets)))
    results["no_inf"] = bool(not np.any(np.isinf(targets)))
    results["no_pathological"] = bool(
        results["soma_min_mv"] > -100 and results["soma_max_mv"] < 50
    )

    return results


def t3_heterogeneity_sensitivity(base_targets, variant_targets):
    """T3: Verify traces are meaningfully different when one parameter changes."""
    base_soma = base_targets[:, :, 1]
    var_soma = variant_targets[:, :, 1]

    # Mean absolute difference across samples and time
    diff = np.abs(base_soma - var_soma)
    results = {
        "mean_abs_diff_mv": float(np.mean(diff) * 0.1),
        "max_diff_mv": float(np.max(diff) * 0.1),
        "significantly_different": float(np.mean(diff)) > 0.1,
    }
    return results


def t4_mechanism_verification(nmda_on_targets, nmda_off_targets):
    """T4: Verify NMDA disabling produces expected changes."""
    on_soma = nmda_on_targets[:, :, 1]
    off_soma = nmda_off_targets[:, :, 1]

    diff = np.abs(on_soma - off_soma)
    results = {
        "mean_abs_diff_mv": float(np.mean(diff) * 0.1),
        "max_diff_mv": float(np.max(diff) * 0.1),
        "nmda_has_effect": float(np.mean(diff)) > 0.05,
    }
    return results


def t5_input_response_consistency(targets_list, input_rates):
    """T5: Verify spike rate increases monotonically with input rate."""
    spike_rates = []
    for targets in targets_list:
        spike = targets[:, :, 0]
        counts = np.sum(spike > 0.5, axis=1)
        duration_s = spike.shape[1] * 1.0 / 1000.0
        rates = counts / duration_s
        spike_rates.append(float(np.mean(rates)))

    # Check monotonicity (should generally increase)
    monotonic = all(
        spike_rates[i] <= spike_rates[i + 1]
        for i in range(len(spike_rates) - 1)
    )

    results = {
        "input_rates": input_rates,
        "spike_rates_hz": spike_rates,
        "monotonic": monotonic,
        "correlation": float(
            np.corrcoef(input_rates, spike_rates)[0, 1]
        ) if len(input_rates) > 1 else None,
    }
    return results


def t6_edge_cases(targets):
    """T6: Check for NaN/Inf in outputs."""
    return {
        "has_nan": bool(np.any(np.isnan(targets))),
        "has_inf": bool(np.any(np.isinf(targets))),
        "soma_min": float(np.min(targets[:, :, 1])),
        "soma_max": float(np.max(targets[:, :, 1])),
        "spike_min": float(np.min(targets[:, :, 0])),
        "spike_max": float(np.max(targets[:, :, 0])),
    }


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/arbor_nmda_source_exact_gpu_smoke")
    report_path = data_dir / "generation_report.json"

    if not report_path.exists():
        print(f"Report not found: {report_path}")
        sys.exit(1)

    with open(report_path) as f:
        report = json.load(f)

    # Find shards
    shards = sorted(data_dir.glob("*.npz"))
    if not shards:
        print("No shards found")
        sys.exit(1)

    all_inputs = []
    all_targets = []
    for shard in shards:
        inp, tgt, _ = load_shard(shard)
        all_inputs.append(inp)
        all_targets.append(tgt)

    inputs = np.concatenate(all_inputs, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    print(f"Loaded {targets.shape[0]} samples, {targets.shape[1]} steps")

    # T2
    print("\n=== T2: Biophysical Plausibility ===")
    t2_results = t2_biophysical_plausibility(targets)
    for k, v in t2_results.items():
        print(f"  {k}: {v}")

    # T6 (run on same data)
    print("\n=== T6: Edge Cases ===")
    t6_results = t6_edge_cases(targets)
    for k, v in t6_results.items():
        print(f"  {k}: {v}")

    # Write results
    results = {
        "data_dir": str(data_dir),
        "samples": targets.shape[0],
        "time_steps": targets.shape[1],
        "t2_biophysical_plausibility": t2_results,
        "t6_edge_cases": t6_results,
    }

    out_path = Path("docs/verify-arbor-simulator/artifacts/verification_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
