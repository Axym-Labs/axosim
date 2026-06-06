"""Default NEURO-BENCH evaluation profiles.

Profiles describe how benchmark axes should be composed before a runner turns
them into concrete simulator/model jobs.  They intentionally avoid hard-coding
paths or checkpoint names so the benchmark can later be shipped as a standalone
package.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from bnn_sim.benchmark.state_metrics import STATE_DEFINITIONS


DEFAULT_METRIC_SUITES = (
    "m1_trace_fidelity",
    "m1b_feature_fidelity",
    "m2_regimes",
    "state_conditioned_fidelity",
    "trace_shape_fidelity",
)

DEFAULT_STATES = tuple(STATE_DEFINITIONS)

DEFAULT_SOURCE_INTERVENTIONS = (
    "event_dropout",
    "exc_dropout",
    "inh_dropout",
    "temporal_jitter",
    "site_silence",
)

DEFAULT_MORPHOLOGY_INTERVENTIONS = (
    "event_dropout",
    "inh_dropout",
)

DEFAULT_INTERVENTION_METRICS = (
    "teacher_response_sanity",
    "soma_delta_rmse",
    "soma_delta_gain",
    "spike_count_delta_mae",
    "spike_count_delta_gain",
    "direction_preservation",
    "representative_trace_ids",
)


@dataclass(frozen=True)
class BenchmarkCostModel:
    """Approximate local cost model for deciding profile feasibility."""

    name: str = "local_coreneuron_hay_cpu_20260606"
    cached_source_teacher_seconds_per_trace: float = 0.0
    source_intervention_teacher_seconds_per_trace: float = 25.0
    morphology_baseline_teacher_seconds_per_trace: float = 14.0
    morphology_intervention_pair_teacher_seconds_per_trace: float = 28.0
    model_eval_seconds_per_trace_per_model: float = 0.15
    default_model_count: int = 3
    notes: tuple[str, ...] = (
        "Anchored to local CoreNEURON/Hay evidence: 16 source event-dropout "
        "intervention traces took about 401 s, and 8 held-out morphology "
        "baseline+intervention traces took about 223 s.",
        "This is a planning estimate, not a replacement for measured runtime "
        "in the final standalone package.",
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LOCAL_CORENEURON_HAY_COST_MODEL = BenchmarkCostModel()


@dataclass(frozen=True)
class BenchmarkProfile:
    """Named benchmark budget and axis-composition policy."""

    name: str
    description: str
    profile_status: str = "provisional"
    design_strategy: str = (
        "Capacity-audited staged design: dense metrics over already-computed "
        "predictions, stratified morphology coverage for the main transfer "
        "axis, and fractional paired interventions for expensive simulator "
        "axes."
    )
    metric_suites: tuple[str, ...] = DEFAULT_METRIC_SUITES
    states: tuple[str, ...] = DEFAULT_STATES
    source_trace_count: int = 64
    morphology_count: int = 15
    morphology_trace_count: int = 16
    source_interventions: tuple[str, ...] = DEFAULT_SOURCE_INTERVENTIONS
    source_intervention_trace_count: int = 16
    morphology_intervention_count: int = 5
    morphology_intervention_trace_count: int = 8
    morphology_interventions: tuple[str, ...] = DEFAULT_MORPHOLOGY_INTERVENTIONS
    intervention_metrics: tuple[str, ...] = DEFAULT_INTERVENTION_METRICS
    compute_full_metrics_on_interventions: bool = False
    model_training_presentation_cap: int = 80_000
    target_wallclock_seconds: int = 4 * 3600
    single_cell_teacher_backend: str = "coreneuron_hay_cpu"
    gpu_validation_policy: str = (
        "Use CPU CoreNEURON/Hay for single-cell teacher generation unless a "
        "local CPU/GPU equivalence and throughput benchmark shows GPU wins at "
        "the requested batch/cell scale."
    )
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable profile."""
        return asdict(self)


PROFILES: dict[str, BenchmarkProfile] = {
    "smoke": BenchmarkProfile(
        name="smoke",
        description="Tiny correctness check for CI and new runner wiring.",
        profile_status="smoke_only",
        source_trace_count=8,
        morphology_count=3,
        morphology_trace_count=4,
        source_interventions=("event_dropout",),
        source_intervention_trace_count=8,
        morphology_intervention_count=1,
        morphology_intervention_trace_count=4,
        morphology_interventions=("event_dropout",),
        notes=(
            "Not publication-grade; intended to catch schema, masking, and "
            "paired-intervention regressions quickly.",
        ),
    ),
    "paper-default": BenchmarkProfile(
        name="paper-default",
        description=(
            "Provisional paper default candidate: dense metric/state "
            "evaluation, wider same-cell-type morphology transfer, and sparse "
            "paired intervention coverage."
        ),
        profile_status="provisional_capacity_audited",
        source_trace_count=64,
        morphology_count=15,
        morphology_trace_count=16,
        source_intervention_trace_count=16,
        morphology_intervention_count=5,
        morphology_intervention_trace_count=8,
        notes=(
            "Metric x state coverage is intentionally dense because it is cheap "
            "once predictions exist and it exposes behavior-specific failures.",
            "Interventions are paired-delta tests by default; full M1/M2 on "
            "every intervention is an extended analysis, not the default.",
            "Morphology-intervention checks are a sparse sanity slice rather "
            "than the full intervention x morphology cartesian product.",
        ),
    ),
    "teacher-screen": BenchmarkProfile(
        name="teacher-screen",
        description=(
            "Teacher-distribution and morphology sanity screen before model "
            "evaluation; useful for ranking candidate morphologies."
        ),
        profile_status="teacher_screen",
        source_trace_count=16,
        morphology_count=30,
        morphology_trace_count=4,
        source_interventions=("event_dropout", "inh_dropout"),
        source_intervention_trace_count=8,
        morphology_intervention_count=5,
        morphology_intervention_trace_count=4,
        morphology_interventions=("event_dropout", "inh_dropout"),
        intervention_metrics=("teacher_response_sanity", "direction_preservation", "representative_trace_ids"),
        notes=(
            "This profile is teacher-first and should not be used as the final "
            "model leaderboard.",
        ),
    ),
}


def list_benchmark_profiles() -> tuple[str, ...]:
    """Return available profile names."""
    return tuple(PROFILES)


def get_benchmark_profile(name: str = "paper-default") -> BenchmarkProfile:
    """Return one named benchmark profile."""
    try:
        return PROFILES[name]
    except KeyError as error:
        available = ", ".join(list_benchmark_profiles())
        raise ValueError(f"unknown benchmark profile {name!r}; available: {available}") from error


def build_axis_plan(profile: BenchmarkProfile | str = "paper-default") -> dict[str, Any]:
    """Summarize the intended non-cartesian benchmark evaluation plan."""
    if isinstance(profile, str):
        profile = get_benchmark_profile(profile)

    metric_state_jobs = len(profile.metric_suites) * len(profile.states)
    full_morphology_intervention_pairs = profile.morphology_count * len(profile.source_interventions)
    selected_morphology_intervention_pairs = profile.morphology_intervention_count * len(profile.morphology_interventions)
    avoided_morphology_intervention_pairs = max(
        0,
        full_morphology_intervention_pairs - selected_morphology_intervention_pairs,
    )

    cost_estimate = estimate_profile_cost(profile)

    return {
        "profile": profile.to_dict(),
        "design_strategy": {
            "profile_status": profile.profile_status,
            "strategy": profile.design_strategy,
            "basis": (
                "Treat profile values as capacity-constrained hypotheses. "
                "Revise them when measured runtime, uncertainty, or failure "
                "coverage shows the budget is too small or poorly allocated."
            ),
        },
        "metric_state_evaluation": {
            "mode": "dense",
            "metric_suites": list(profile.metric_suites),
            "states": list(profile.states),
            "nominal_metric_state_cells": metric_state_jobs,
            "rationale": (
                "Metric x state is not pruned by default: state-conditioned "
                "M1/M1b/M2/trace-shape summaries are cheap after prediction "
                "and are central to interpreting model behavior."
            ),
        },
        "source_distribution": {
            "trace_count": profile.source_trace_count,
            "evaluates": "all metric suites over all states",
        },
        "morphology_transfer": {
            "morphology_count": profile.morphology_count,
            "traces_per_morphology": profile.morphology_trace_count,
            "evaluates": "all metric suites over all states",
            "rationale": (
                "M3 uses the larger morphology budget for default metric "
                "evaluation because morphology transfer is a headline axis."
            ),
        },
        "source_interventions": {
            "interventions": list(profile.source_interventions),
            "traces_per_intervention": profile.source_intervention_trace_count,
            "metrics": list(profile.intervention_metrics),
            "semantics": (
                "Interventions measure paired teacher/model response deltas: "
                "direction, gain, timing/count behavior, and representative "
                "failures. They are not plain M1 repeats by default."
            ),
            "compute_full_metrics_on_interventions": profile.compute_full_metrics_on_interventions,
        },
        "morphology_intervention_sanity": {
            "morphology_count": profile.morphology_intervention_count,
            "interventions": list(profile.morphology_interventions),
            "traces_per_pair": profile.morphology_intervention_trace_count,
            "selected_pairs": selected_morphology_intervention_pairs,
            "full_cartesian_pairs": full_morphology_intervention_pairs,
            "avoided_pairs": avoided_morphology_intervention_pairs,
            "rationale": (
                "A small morphology-intervention slice checks whether the "
                "intervention direction remains meaningful under transfer "
                "without multiplying every intervention by every morphology."
            ),
        },
        "execution_policy": {
            "model_training_presentation_cap": profile.model_training_presentation_cap,
            "single_cell_teacher_backend": profile.single_cell_teacher_backend,
            "gpu_validation_policy": profile.gpu_validation_policy,
        },
        "capacity_estimate": cost_estimate,
    }


def estimate_profile_cost(
    profile: BenchmarkProfile | str = "paper-default",
    *,
    cost_model: BenchmarkCostModel = LOCAL_CORENEURON_HAY_COST_MODEL,
    model_count: int | None = None,
) -> dict[str, Any]:
    """Estimate local runtime and trace counts for a profile.

    The estimate separates expensive teacher-generation work from cheaper model
    prediction/metric work.  It is deliberately transparent rather than
    overfit: users should replace the cost model with measured package-local
    timings when deploying elsewhere.
    """
    if isinstance(profile, str):
        profile = get_benchmark_profile(profile)
    model_count = int(cost_model.default_model_count if model_count is None else model_count)

    source_teacher_traces = profile.source_trace_count
    morphology_baseline_traces = profile.morphology_count * profile.morphology_trace_count
    source_intervention_pairs = len(profile.source_interventions)
    source_intervention_traces = source_intervention_pairs * profile.source_intervention_trace_count
    morphology_intervention_pairs = profile.morphology_intervention_count * len(profile.morphology_interventions)
    morphology_intervention_traces = morphology_intervention_pairs * profile.morphology_intervention_trace_count

    full_cartesian_morphology_intervention_pairs = profile.morphology_count * len(profile.source_interventions)
    full_cartesian_morphology_intervention_traces = (
        full_cartesian_morphology_intervention_pairs * profile.morphology_intervention_trace_count
    )
    avoided_morphology_intervention_traces = max(
        0,
        full_cartesian_morphology_intervention_traces - morphology_intervention_traces,
    )

    teacher_seconds = {
        "cached_source_distribution": (
            source_teacher_traces * cost_model.cached_source_teacher_seconds_per_trace
        ),
        "morphology_baselines": (
            morphology_baseline_traces * cost_model.morphology_baseline_teacher_seconds_per_trace
        ),
        "source_interventions": (
            source_intervention_traces * cost_model.source_intervention_teacher_seconds_per_trace
        ),
        "morphology_intervention_sanity": (
            morphology_intervention_traces * cost_model.morphology_intervention_pair_teacher_seconds_per_trace
        ),
    }
    model_eval_trace_units = (
        profile.source_trace_count
        + morphology_baseline_traces
        + 2 * source_intervention_traces
        + 2 * morphology_intervention_traces
    )
    model_seconds = model_eval_trace_units * model_count * cost_model.model_eval_seconds_per_trace_per_model
    teacher_total_seconds = float(sum(teacher_seconds.values()))
    total_seconds = teacher_total_seconds + float(model_seconds)

    full_cartesian_extra_teacher_seconds = (
        avoided_morphology_intervention_traces
        * cost_model.morphology_intervention_pair_teacher_seconds_per_trace
    )
    return {
        "cost_model": cost_model.to_dict(),
        "model_count": model_count,
        "trace_counts": {
            "source_distribution": source_teacher_traces,
            "morphology_baselines": morphology_baseline_traces,
            "source_intervention_traces": source_intervention_traces,
            "morphology_intervention_traces": morphology_intervention_traces,
            "full_cartesian_morphology_intervention_traces": full_cartesian_morphology_intervention_traces,
            "avoided_morphology_intervention_traces": avoided_morphology_intervention_traces,
        },
        "pair_counts": {
            "source_interventions": source_intervention_pairs,
            "morphology_intervention_selected": morphology_intervention_pairs,
            "morphology_intervention_full_cartesian": full_cartesian_morphology_intervention_pairs,
        },
        "teacher_seconds_by_component": {
            key: round(value, 3) for key, value in teacher_seconds.items()
        },
        "model_eval_seconds": round(float(model_seconds), 3),
        "teacher_total_seconds": round(teacher_total_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "total_hours": round(total_seconds / 3600.0, 3),
        "target_wallclock_seconds": profile.target_wallclock_seconds,
        "target_wallclock_hours": round(profile.target_wallclock_seconds / 3600.0, 3),
        "within_target_wallclock": bool(total_seconds <= profile.target_wallclock_seconds),
        "full_cartesian_extra_teacher_seconds_avoided": round(full_cartesian_extra_teacher_seconds, 3),
        "full_cartesian_extra_teacher_hours_avoided": round(
            full_cartesian_extra_teacher_seconds / 3600.0,
            3,
        ),
        "dominant_cost_component": max(teacher_seconds, key=teacher_seconds.get),
        "recommended_parallel_jobs": max(1, min(8, math.ceil(teacher_total_seconds / 1800.0))),
        "interpretation": (
            "Teacher generation dominates runtime; metric x state expansion is "
            "therefore kept dense after predictions exist, while simulator "
            "axes use staged/fractional coverage."
        ),
    }
