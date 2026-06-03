"""NEURO-BENCH: Neuron Surrogate Benchmark Suite.

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

from bnn_sim.benchmark.regime_stratified import compute_regime_stratified_metrics
from bnn_sim.benchmark.perturbation_stability import compute_perturbation_stability
from bnn_sim.benchmark.structured_intervention import (
    compute_paired_event_intervention_metrics,
    make_event_dropout_inputs,
)
from bnn_sim.benchmark.morphology_transfer import MorphologyTransferEvaluator, load_available_morphologies
from bnn_sim.benchmark.branch_adapter import adapt_branch_count, evaluate_with_branch_adaptation
from bnn_sim.benchmark.suite import BenchmarkSuite

__all__ = [
    "compute_regime_stratified_metrics",
    "compute_perturbation_stability",
    "compute_paired_event_intervention_metrics",
    "make_event_dropout_inputs",
    "MorphologyTransferEvaluator",
    "load_available_morphologies",
    "adapt_branch_count",
    "evaluate_with_branch_adaptation",
    "BenchmarkSuite",
]
