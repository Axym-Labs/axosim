"""NEURO-BENCH: neuron surrogate benchmark utilities.

The package exports the public benchmark helpers lazily so lightweight
dataframe/TCN evaluation can run in environments that intentionally omit
PyTorch.
"""

from __future__ import annotations

from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "compute_regime_stratified_metrics": ("bnn_sim.benchmark.regime_stratified", "compute_regime_stratified_metrics"),
    "compute_state_conditioned_metrics": ("bnn_sim.benchmark.state_metrics", "compute_state_conditioned_metrics"),
    "build_neuron_state_masks": ("bnn_sim.benchmark.state_metrics", "build_neuron_state_masks"),
    "compute_trace_shape_metrics": ("bnn_sim.benchmark.trace_shape", "compute_trace_shape_metrics"),
    "compute_perturbation_stability": ("bnn_sim.benchmark.perturbation_stability", "compute_perturbation_stability"),
    "compute_paired_event_intervention_metrics": (
        "bnn_sim.benchmark.structured_intervention",
        "compute_paired_event_intervention_metrics",
    ),
    "make_event_dropout_inputs": ("bnn_sim.benchmark.structured_intervention", "make_event_dropout_inputs"),
    "make_selective_event_dropout_inputs": (
        "bnn_sim.benchmark.structured_intervention",
        "make_selective_event_dropout_inputs",
    ),
    "make_site_silence_inputs": ("bnn_sim.benchmark.structured_intervention", "make_site_silence_inputs"),
    "make_structured_intervention_inputs": (
        "bnn_sim.benchmark.structured_intervention",
        "make_structured_intervention_inputs",
    ),
    "make_temporal_jitter_inputs": ("bnn_sim.benchmark.structured_intervention", "make_temporal_jitter_inputs"),
    "MorphologyTransferEvaluator": ("bnn_sim.benchmark.morphology_transfer", "MorphologyTransferEvaluator"),
    "load_available_morphologies": ("bnn_sim.benchmark.morphology_transfer", "load_available_morphologies"),
    "adapt_branch_count": ("bnn_sim.benchmark.branch_adapter", "adapt_branch_count"),
    "evaluate_with_branch_adaptation": ("bnn_sim.benchmark.branch_adapter", "evaluate_with_branch_adaptation"),
    "BenchmarkProfile": ("bnn_sim.benchmark.profiles", "BenchmarkProfile"),
    "BenchmarkCostModel": ("bnn_sim.benchmark.profiles", "BenchmarkCostModel"),
    "LOCAL_CORENEURON_HAY_COST_MODEL": ("bnn_sim.benchmark.profiles", "LOCAL_CORENEURON_HAY_COST_MODEL"),
    "build_axis_plan": ("bnn_sim.benchmark.profiles", "build_axis_plan"),
    "estimate_profile_cost": ("bnn_sim.benchmark.profiles", "estimate_profile_cost"),
    "get_benchmark_profile": ("bnn_sim.benchmark.profiles", "get_benchmark_profile"),
    "list_benchmark_profiles": ("bnn_sim.benchmark.profiles", "list_benchmark_profiles"),
    "DIAGNOSTIC_SUITES": ("bnn_sim.benchmark.dataset_schema", "DIAGNOSTIC_SUITES"),
    "INTERVENTION_CONDITIONS": ("bnn_sim.benchmark.dataset_schema", "INTERVENTION_CONDITIONS"),
    "ORDINARY_SPLITS": ("bnn_sim.benchmark.dataset_schema", "ORDINARY_SPLITS"),
    "V1_MORPHOLOGY_COUNT": ("bnn_sim.benchmark.dataset_schema", "V1_MORPHOLOGY_COUNT"),
    "build_diagnostic_suite_catalog": ("bnn_sim.benchmark.dataset_schema", "build_diagnostic_suite_catalog"),
    "build_v1_dataset_manifest_template": ("bnn_sim.benchmark.dataset_schema", "build_v1_dataset_manifest_template"),
    "default_diagnostic_specs": ("bnn_sim.benchmark.dataset_schema", "default_diagnostic_specs"),
    "default_target_views": ("bnn_sim.benchmark.dataset_schema", "default_target_views"),
    "provisional_v1_morphologies": ("bnn_sim.benchmark.dataset_schema", "provisional_v1_morphologies"),
    "validate_v1_manifest_template": ("bnn_sim.benchmark.dataset_schema", "validate_v1_manifest_template"),
    "DiagnosticContext": ("bnn_sim.benchmark.diagnostic_rows", "DiagnosticContext"),
    "evaluate_model_diagnostic_rows": ("bnn_sim.benchmark.diagnostic_rows", "evaluate_model_diagnostic_rows"),
    "morphology_contrast_diagnostic_rows": ("bnn_sim.benchmark.diagnostic_rows", "morphology_contrast_diagnostic_rows"),
    "morphology_routing_qc_rows": ("bnn_sim.benchmark.diagnostic_rows", "morphology_routing_qc_rows"),
    "paired_intervention_diagnostic_rows": ("bnn_sim.benchmark.diagnostic_rows", "paired_intervention_diagnostic_rows"),
    "teacher_qc_diagnostic_rows": ("bnn_sim.benchmark.diagnostic_rows", "teacher_qc_diagnostic_rows"),
    "validate_diagnostic_rows": ("bnn_sim.benchmark.diagnostic_rows", "validate_diagnostic_rows"),
    "write_diagnostic_rows": ("bnn_sim.benchmark.diagnostic_rows", "write_diagnostic_rows"),
    "DatasetSelector": ("bnn_sim.benchmark.selectors", "DatasetSelector"),
    "V1DatasetLayout": ("bnn_sim.benchmark.selectors", "V1DatasetLayout"),
    "load_intervention_pair_npz": ("bnn_sim.benchmark.runner", "load_intervention_pair_npz"),
    "load_selector_batch": ("bnn_sim.benchmark.runner", "load_selector_batch"),
    "run_model_diagnostics_on_selector": ("bnn_sim.benchmark.runner", "run_model_diagnostics_on_selector"),
    "run_paired_intervention_on_npz": ("bnn_sim.benchmark.runner", "run_paired_intervention_on_npz"),
    "run_paired_intervention_on_selector": ("bnn_sim.benchmark.runner", "run_paired_intervention_on_selector"),
    "run_teacher_qc_on_selector": ("bnn_sim.benchmark.runner", "run_teacher_qc_on_selector"),
    "summarize_diagnostic_rows": ("bnn_sim.benchmark.reports", "summarize_diagnostic_rows"),
    "write_diagnostic_summary": ("bnn_sim.benchmark.reports", "write_diagnostic_summary"),
    "run_v1_benchmark_bundle": ("bnn_sim.benchmark.bundle", "run_v1_benchmark_bundle"),
    "infer_pair_metadata": ("bnn_sim.benchmark.bundle", "infer_pair_metadata"),
    "BenchmarkSuite": ("bnn_sim.benchmark.suite", "BenchmarkSuite"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
