"""M4: Perturbation Stability Metrics.

Measures how small input perturbations affect output predictions.
Lipschitz-like analysis for surrogate robustness.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def compute_perturbation_stability(
    model_fn: Callable[[np.ndarray], np.ndarray],
    inputs: np.ndarray,
    epsilon_values: list[float] | None = None,
    n_perturbations: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute M4: Perturbation stability metrics.

    Args:
        model_fn: Function that takes inputs array and returns predictions.
        inputs: Input array of shape (batch, time, channels) or (time, channels).
        epsilon_values: List of relative perturbation magnitudes.
            Defaults to [0.001, 0.005, 0.01, 0.02, 0.05].
        n_perturbations: Number of random perturbations per epsilon.
        seed: Random seed.

    Returns:
        Dictionary with stability metrics.
    """
    rng = np.random.default_rng(seed)
    inputs = np.asarray(inputs)

    if epsilon_values is None:
        epsilon_values = [0.001, 0.005, 0.01, 0.02, 0.05]

    # Baseline prediction
    baseline_pred = np.asarray(model_fn(inputs))

    # Compute input norm for relative scaling
    input_norm = float(np.linalg.norm(inputs))
    if input_norm == 0:
        input_norm = 1.0

    results = []
    for eps in epsilon_values:
        abs_eps = eps * input_norm

        soma_ratio_list = []
        spike_ratio_list = []
        local_lipschitz_list = []

        for _ in range(n_perturbations):
            # Random perturbation
            noise = rng.standard_normal(inputs.shape)
            noise = noise / (np.linalg.norm(noise) + 1e-10) * abs_eps
            perturbed_inputs = inputs + noise

            perturbed_pred = np.asarray(model_fn(perturbed_inputs))

            # Output changes
            delta_pred = perturbed_pred - baseline_pred
            delta_input = noise

            # Soma voltage change ratio
            pred_soma = baseline_pred[..., 1]
            delta_soma = delta_pred[..., 1]
            soma_ratio = float(np.linalg.norm(delta_soma) / (np.linalg.norm(pred_soma) + 1e-10))
            soma_ratio_list.append(soma_ratio)

            # Spike score change ratio
            pred_spike = baseline_pred[..., 0]
            delta_spike = delta_pred[..., 0]
            spike_ratio = float(np.linalg.norm(delta_spike) / (np.linalg.norm(pred_spike) + 1e-10))
            spike_ratio_list.append(spike_ratio)

            # Local Lipschitz constant (output change / input change)
            output_norm = float(np.linalg.norm(delta_pred))
            input_norm_delta = float(np.linalg.norm(delta_input))
            if input_norm_delta > 0:
                local_lipschitz_list.append(output_norm / input_norm_delta)

        results.append({
            "epsilon_relative": eps,
            "epsilon_absolute": abs_eps,
            "soma_relative_change_mean": float(np.mean(soma_ratio_list)),
            "soma_relative_change_std": float(np.std(soma_ratio_list)),
            "spike_relative_change_mean": float(np.mean(spike_ratio_list)),
            "spike_relative_change_std": float(np.std(spike_ratio_list)),
            "local_lipschitz_mean": float(np.mean(local_lipschitz_list)) if local_lipschitz_list else None,
            "local_lipschitz_std": float(np.std(local_lipschitz_list)) if local_lipschitz_list else None,
        })

    # Compute stability coefficient (linear fit slope of output change vs epsilon)
    epsilons = np.array([r["epsilon_relative"] for r in results])
    soma_changes = np.array([r["soma_relative_change_mean"] for r in results])
    spike_changes = np.array([r["spike_relative_change_mean"] for r in results])

    stability_coefficient = {
        "soma": float(np.polyfit(epsilons, soma_changes, 1)[0]) if len(epsilons) > 1 else None,
        "spike": float(np.polyfit(epsilons, spike_changes, 1)[0]) if len(epsilons) > 1 else None,
    }

    return {
        "epsilon_values": epsilon_values,
        "n_perturbations": n_perturbations,
        "input_norm": input_norm,
        "per_epsilon": results,
        "stability_coefficient": stability_coefficient,
        "interpretation": (
            "Lower stability coefficient indicates better robustness. "
            "A model with stability_coefficient > 10 is likely too sensitive for closed-loop use."
        ),
    }
