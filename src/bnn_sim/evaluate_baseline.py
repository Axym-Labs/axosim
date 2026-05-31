#!/usr/bin/env python3
"""Evaluate Branch-ELM baseline on NEURO-BENCH M1-M4.

Note: This script evaluates the model architecture. For publication,
use a properly trained checkpoint (40k+ presentations).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from bnn_sim.benchmark import BenchmarkSuite
from bnn_sim.model import BranchELM, BranchELMConfig


def load_or_create_branch_elm(
    checkpoint_path: str | None = "runs/branch-elm.pt",
    input_dim: int = 1278,
    num_branches: int = 45,
) -> BranchELM:
    """Load Branch-ELM model from checkpoint or create fresh.
    
    If checkpoint is incompatible (wrong input_dim), creates fresh model.
    """
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        config = BranchELMConfig(**ckpt["config"])
        
        if config.input_dim != input_dim or config.num_branches != num_branches:
            print(f"Checkpoint incompatible ({config.input_dim}, {config.num_branches} branches)")
            print(f"Creating fresh Branch-ELM-30 ({input_dim}, {num_branches} branches)")
            config = BranchELMConfig.branch_elm_30(
                input_dim=input_dim, num_branches=num_branches
            )
            model = BranchELM(config)
        else:
            model = BranchELM(config)
            model.load_state_dict(ckpt["state_dict"])
            print(f"Loaded checkpoint: {sum(p.numel() for p in model.parameters()):,} params")
    else:
        print(f"Creating fresh Branch-ELM-30 ({input_dim}, {num_branches} branches)")
        config = BranchELMConfig.branch_elm_30(
            input_dim=input_dim, num_branches=num_branches
        )
        model = BranchELM(config)
    
    model.eval()
    return model


def evaluate_on_benchmark(
    model: BranchELM,
    data_dir: str = "data/arbor-allen-nmda-source-medium-20260510",
    n_val_samples: int = 64,
    device: str = "cuda",
    output_dir: str = "runs/benchmark_branch_elm",
) -> dict:
    """Evaluate model on benchmark suite."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    # Load validation data
    val_dir = Path(data_dir) / "val"
    val_files = sorted(val_dir.glob("*.npz"))

    all_inputs = []
    all_targets = []
    for f in val_files:
        data = np.load(f)
        all_inputs.append(data["inputs"])
        all_targets.append(data["targets"])

    inputs = np.concatenate(all_inputs, axis=0)[:n_val_samples]
    targets = np.concatenate(all_targets, axis=0)[:n_val_samples]

    print(f"Evaluating on {len(inputs)} validation samples")

    # Convert to tensors
    inputs_t = torch.from_numpy(inputs).float().to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        preds = model(inputs_t)

    preds_np = preds.cpu().numpy()
    inputs_np = inputs_t.cpu().numpy()

    # Run benchmark suite
    suite = BenchmarkSuite(output_dir=output_dir)

    def model_fn(x: np.ndarray) -> np.ndarray:
        """Wrapper for benchmark suite."""
        with torch.no_grad():
            x_t = torch.from_numpy(x).float().to(device)
            out = model(x_t)
            return out.cpu().numpy()

    results = suite.evaluate_model(
        model_fn,
        inputs_np,
        targets,
        soma_scale=0.1,
        ignore_start=500,
        run_perturbation=True,
        perturbation_epsilons=[0.001, 0.005, 0.01, 0.02, 0.05],
    )

    # Save results
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def print_results(results: dict, model_name: str = "Branch-ELM") -> None:
    """Pretty print benchmark results."""
    print("\n" + "=" * 60)
    print(f"{model_name} BASELINE BENCHMARK RESULTS")
    print("=" * 60)

    m1 = results["m1_open_loop_fidelity"]
    print(f"\nM1: Open-Loop Fidelity")
    print(f"  RMSE: {m1['rmse_mv']:.2f} mV")
    print(f"  AUC:  {m1['auc']:.4f}")

    m2 = results["m2_regime_stratified"]
    print(f"\nM2: Regime-Stratified Errors")
    print(f"  Voltage Regimes:")
    for r in m2["voltage_regime"]:
        rmse_str = f"{r['rmse']:.2f}" if r['rmse'] is not None else "N/A"
        print(f"    {r['regime_label']:15s}: RMSE={rmse_str:>6s} mV, count={r['count']}")

    print(f"\n  Slope Regimes:")
    for r in m2["slope_regime"]:
        rmse_str = f"{r['rmse']:.2f}" if r['rmse'] is not None else "N/A"
        print(f"    {r['regime_label']:15s}: RMSE={rmse_str:>6s} mV, count={r['count']}")

    print(f"\n  Spike Proximity:")
    for r in m2["spike_proximity"]:
        rmse_str = f"{r['rmse']:.2f}" if r['rmse'] is not None else "N/A"
        print(f"    {r['regime_label']:15s}: RMSE={rmse_str:>6s} mV, count={r['count']}")

    print(f"\n  Spike Timing:")
    timing = m2["spike_timing"]
    mean_err = f"{timing['mean_timing_error_steps']:.2f}" if timing['mean_timing_error_steps'] is not None else "N/A"
    print(f"    Mean timing error: {mean_err} steps")
    print(f"    Precision@2: {timing['precision_at_2steps']:.4f}")
    print(f"    Recall@2: {timing['recall_at_2steps']:.4f}")
    print(f"    F1@2: {timing['f1_at_2steps']:.4f}")

    m4 = results["m4_perturbation_stability"]
    print(f"\nM4: Perturbation Stability")
    soma_sc = m4['stability_coefficient']['soma']
    spike_sc = m4['stability_coefficient']['spike']
    soma_str = f"{soma_sc:.4f}" if soma_sc is not None else "N/A"
    spike_str = f"{spike_sc:.4f}" if spike_sc is not None else "N/A"
    print(f"  Soma stability coeff: {soma_str}")
    print(f"  Spike stability coeff: {spike_str}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Loading Branch-ELM baseline...")
    model = load_or_create_branch_elm(
        checkpoint_path="runs/branch-elm.pt",
        input_dim=1278,
        num_branches=45,
    )
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    results = evaluate_on_benchmark(model, n_val_samples=64, device="cuda")
    print_results(results, model_name="Branch-ELM-30 (untrained)")
    
    print("\nNOTE: This is an UNTRAINED model for benchmark validation.")
    print("For publication results, train with: python -m bnn_sim.cli train ...")
