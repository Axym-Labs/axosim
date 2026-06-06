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
from bnn_sim.benchmark.state_metrics import compute_state_conditioned_metrics, build_neuron_state_masks
from bnn_sim.benchmark.trace_shape import compute_trace_shape_metrics
from bnn_sim.benchmark.perturbation_stability import compute_perturbation_stability
from bnn_sim.benchmark.structured_intervention import (
    compute_paired_event_intervention_metrics,
    make_event_dropout_inputs,
    make_selective_event_dropout_inputs,
    make_site_silence_inputs,
    make_structured_intervention_inputs,
    make_temporal_jitter_inputs,
)
from bnn_sim.benchmark.morphology_transfer import MorphologyTransferEvaluator, load_available_morphologies
from bnn_sim.benchmark.branch_adapter import adapt_branch_count, evaluate_with_branch_adaptation
from bnn_sim.benchmark.profiles import (
    BenchmarkProfile,
    BenchmarkCostModel,
    LOCAL_CORENEURON_HAY_COST_MODEL,
    build_axis_plan,
    estimate_profile_cost,
    get_benchmark_profile,
    list_benchmark_profiles,
)
from bnn_sim.benchmark.dataset_schema import (
    DIAGNOSTIC_SUITES,
    INTERVENTION_CONDITIONS,
    ORDINARY_SPLITS,
    V1_MORPHOLOGY_COUNT,
    build_diagnostic_suite_catalog,
    build_v1_dataset_manifest_template,
    default_diagnostic_specs,
    default_target_views,
    provisional_v1_morphologies,
    validate_v1_manifest_template,
)
from bnn_sim.benchmark.suite import BenchmarkSuite

__all__ = [
    "compute_regime_stratified_metrics",
    "compute_state_conditioned_metrics",
    "build_neuron_state_masks",
    "compute_trace_shape_metrics",
    "compute_perturbation_stability",
    "compute_paired_event_intervention_metrics",
    "make_event_dropout_inputs",
    "make_selective_event_dropout_inputs",
    "make_site_silence_inputs",
    "make_structured_intervention_inputs",
    "make_temporal_jitter_inputs",
    "MorphologyTransferEvaluator",
    "load_available_morphologies",
    "adapt_branch_count",
    "evaluate_with_branch_adaptation",
    "BenchmarkProfile",
    "BenchmarkCostModel",
    "LOCAL_CORENEURON_HAY_COST_MODEL",
    "build_axis_plan",
    "estimate_profile_cost",
    "get_benchmark_profile",
    "list_benchmark_profiles",
    "DIAGNOSTIC_SUITES",
    "INTERVENTION_CONDITIONS",
    "ORDINARY_SPLITS",
    "V1_MORPHOLOGY_COUNT",
    "build_diagnostic_suite_catalog",
    "build_v1_dataset_manifest_template",
    "default_diagnostic_specs",
    "default_target_views",
    "provisional_v1_morphologies",
    "validate_v1_manifest_template",
    "BenchmarkSuite",
]
