"""Default NEURO-BENCH evaluation profiles.

Profiles describe how benchmark axes should be composed before a runner turns
them into concrete simulator/model jobs.  They intentionally avoid hard-coding
paths or checkpoint names so the benchmark can later be shipped as a standalone
package.
"""

from __future__ import annotations

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
class BenchmarkProfile:
    """Named benchmark budget and axis-composition policy."""

    name: str
    description: str
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
            "Publishable default: dense metric/state evaluation, wider "
            "same-cell-type morphology transfer, and sparse paired intervention "
            "coverage."
        ),
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

    return {
        "profile": profile.to_dict(),
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
            "full_cartesian_pairs_avoided": full_morphology_intervention_pairs,
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
    }
