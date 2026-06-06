#!/usr/bin/env python3
"""Comprehensive benchmark runner for NEURO-BENCH.

Evaluates a model on all active metric suites (M1-M4) and saves results.

Usage:
    python -m bnn_sim.run_benchmark \
        --model-path runs/my_model.pt \
        --data-dir data/my_dataset \
        --output-dir runs/benchmark_my_model \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from bnn_sim.benchmark import BenchmarkSuite, load_available_morphologies, MorphologyTransferEvaluator
from bnn_sim.model import BranchELM, BranchELMConfig


def load_model(model_path: str, device: str = "cuda", input_dim: int = 1278, num_branches: int = 45) -> tuple[torch.nn.Module, str]:
    """Load a model from checkpoint.

    Supports Branch-ELM checkpoints. If checkpoint is incompatible,
    creates a fresh model.
    """
    ckpt = torch.load(model_path, map_location=device)

    if ckpt.get("model_kind") == "baseline":
        config = BranchELMConfig(**ckpt["config"])
        
        # Check compatibility
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
        
        model_name = f"Branch-ELM-{config.memory_units}"
    else:
        raise ValueError(f"Unknown model kind: {ckpt.get('model_kind')}")

    model = model.to(device)
    model.eval()
    return model, model_name


def run_benchmark(
    model: torch.nn.Module,
    model_name: str,
    data_dir: str,
    output_dir: str,
    device: str = "cuda",
    n_val_samples: int = 64,
    run_morphology_transfer: bool = False,
) -> dict:
    """Run full benchmark suite."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Benchmarking {model_name} on {len(inputs)} validation samples")

    # Create model function wrapper
    def model_fn(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(x).float().to(device)
            out = model(x_t)
            return out.cpu().numpy()

    # Run M1-M4 benchmark
    suite = BenchmarkSuite(output_dir=output_dir)
    results = suite.evaluate_model(
        model_fn,
        inputs,
        targets,
        soma_scale=0.1,
        ignore_start=500,
        run_perturbation=True,
        model_name=model_name,
        dataset_path=data_dir,
    )

    # Optionally run M3 (morphology transfer)
    if run_morphology_transfer:
        print("\nRunning morphology transfer evaluation...")
        morphologies = load_available_morphologies()
        if len(morphologies) >= 2:
            evaluator = MorphologyTransferEvaluator()
            train_morph = evaluator.load_morphology(morphologies[0], n_branches=45)
            test_morphs = [evaluator.load_morphology(m, n_branches=45) for m in morphologies[1:]]

            results["m3_morphology_transfer"] = evaluator.compare_morphology_set(train_morph, test_morphs)
        else:
            print("  Skipped: Need >=2 morphologies for transfer")

    # Save comprehensive results
    results["metadata"].update({
        "model_name": model_name,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "data_dir": data_dir,
        "dataset_path": data_dir,
        "n_val_samples": len(inputs),
        "device": device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    with open(output_dir / "full_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def print_summary(results: dict) -> None:
    """Print benchmark summary."""
    meta = results.get("metadata", {})
    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS: {meta.get('model_name', 'Unknown')}")
    print("=" * 70)

    m1 = results["m1_trace_fidelity"]
    print(f"\nM1: Open-Loop Fidelity")
    print(f"  RMSE: {m1['soma_rmse_mv']:.2f} mV")
    print(f"  AUC:  {m1['spike_auc']:.4f}")
    print(f"  State-weighted trace distance: {m1['state_weighted_trace_distance_mv']:.2f} mV")

    m2 = results["m2_regimes"]
    print(f"\nM2: Regime-Stratified Errors")
    near = next((r for r in m2.get("spike_proximity", []) if r["regime_label"] == "near_spike"), None)
    far = next((r for r in m2.get("spike_proximity", []) if r["regime_label"] == "far_from_spike"), None)
    if near is not None and near.get("rmse") is not None:
        print(f"  Near-spike RMSE: {near['rmse']:.2f} mV")
    if far is not None and far.get("rmse") is not None:
        print(f"  Far-from-spike RMSE: {far['rmse']:.2f} mV")

    m4 = results["m4_intervention"]
    print(f"\nM4: Structured Intervention")
    print(f"  Status: {m4.get('status')}")
    if m4.get("status") == "model_compared":
        model_response = m4.get("model_response", {})
        soma_delta = model_response.get("soma_delta_rmse_mv")
        spike_delta = model_response.get("spike_count_delta_mae_per_trace")
        print(f"  Soma-delta RMSE: {soma_delta:.4f} mV" if soma_delta is not None else "  Soma-delta RMSE: N/A")
        print(f"  Spike-delta MAE: {spike_delta:.4f}" if spike_delta is not None else "  Spike-delta MAE: N/A")

    if "m3_morphology_transfer" in results:
        m3 = results["m3_morphology_transfer"]
        print(f"\nM3: Morphology Transfer")
        print(f"  Train: {m3['train_features']['n_branches']} branches")
        for test in m3.get("test_morphologies", []):
            features = test["features"]
            mismatch = test["route_mismatch"]
            print(
                "  Test:  "
                f"{features['n_branches']} branches "
                f"({Path(test['swc_path']).name}), "
                f"length ratio {mismatch['total_length_ratio']:.3f}"
            )

    print("\n" + "=" * 70)
    print(f"Full results saved to: {meta.get('output_dir', 'N/A')}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run NEURO-BENCH evaluation")
    parser.add_argument("--model-path", type=str, default="runs/branch-elm.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/arbor-allen-nmda-source-medium-20260510",
                        help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="runs/benchmark_results",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--n-val-samples", type=int, default=64,
                        help="Number of validation samples")
    parser.add_argument("--run-morphology-transfer", action="store_true",
                        help="Run morphology transfer evaluation (M3)")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, model_name = load_model(args.model_path, device=args.device)
    print(f"Loaded {model_name} with {sum(p.numel() for p in model.parameters()):,} parameters")

    results = run_benchmark(
        model,
        model_name,
        args.data_dir,
        args.output_dir,
        device=args.device,
        n_val_samples=args.n_val_samples,
        run_morphology_transfer=args.run_morphology_transfer,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
