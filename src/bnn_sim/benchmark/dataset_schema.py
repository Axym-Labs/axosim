"""Dataset and diagnostic schema for the v1 NeuronIO-like benchmark arc."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


V1_MORPHOLOGY_COUNT = 5
V1_TRAIN_TRACES_PER_MORPHOLOGY = 20_000
V1_NON_TRAIN_TRACES_PER_MORPHOLOGY = V1_TRAIN_TRACES_PER_MORPHOLOGY // 8

ORDINARY_SPLITS = ("train", "val", "test", "private-test")
INTERVENTION_CONDITIONS = (
    "event_dropout",
    "exc_dropout",
    "inh_dropout",
    "temporal_jitter",
    "site_silence",
)

DIAGNOSTIC_SUITES = (
    "teacher-qc",
    "trace-core",
    "state-dynamics",
    "spike-behavior",
    "feature-fidelity",
    "morphology-contrast",
    "intervention-response",
    "paper-summary",
    "full",
)

DIAGNOSTIC_ROW_COLUMNS = (
    "diagnostic",
    "suite",
    "model_id",
    "teacher_id",
    "morphology_id",
    "split",
    "condition",
    "intervention",
    "trace_id",
    "state",
    "metric",
    "value",
    "unit",
    "n",
)


@dataclass(frozen=True)
class TargetViewSpec:
    """One stored view of the soma/spike target."""

    name: str
    columns: tuple[str, ...]
    units: str
    description: str
    default_for_training: bool = False
    default_for_historical_comparison: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetCategorySpec:
    """One top-level dataset category such as `train` or `interventions`."""

    name: str
    kind: Literal["ordinary_split", "interventions"]
    morphologies: int
    semantic_units_per_morphology: int
    semantic_unit: Literal["trace", "paired_example"]
    paired: bool = False
    default_diagnostic_selector: bool = False

    @property
    def total_semantic_units(self) -> int:
        return self.morphologies * self.semantic_units_per_morphology

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["total_semantic_units"] = self.total_semantic_units
        return payload


@dataclass(frozen=True)
class MorphologySpec:
    """One provisional v1 morphology entry."""

    specimen_id: int
    role: str
    swc_path: str
    line_name: str
    reconstruction_type: str
    compatibility_score_lower_better: float
    extracted_branches: int
    routed_branches: int
    total_length_ratio_vs_reference: float
    routed_length_ratio_vs_reference: float
    apical_fraction_delta_vs_reference: float
    radial_extent_ratio_vs_reference: float
    routing_shape_match: bool
    caveat: str = "Allen metadata marks this reconstruction dendrite-only."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlotterSpec:
    """A variable-length diagnostic-attached plotting method."""

    name: str
    kind: Literal["atomic", "example"]
    consumes: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiagnosticSpec:
    """An atomic dataframe-first diagnostic."""

    name: str
    suites: tuple[str, ...]
    selectors: tuple[str, ...]
    required_columns: tuple[str, ...] = DIAGNOSTIC_ROW_COLUMNS
    plotters: tuple[PlotterSpec, ...] = ()
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["plotters"] = [plotter.to_dict() for plotter in self.plotters]
        return payload


def build_v1_dataset_manifest_template(
    *,
    morphology_count: int = V1_MORPHOLOGY_COUNT,
    train_traces_per_morphology: int = V1_TRAIN_TRACES_PER_MORPHOLOGY,
) -> dict[str, Any]:
    """Return the v1 public dataset manifest contract as JSON-like data."""
    if morphology_count <= 0:
        raise ValueError("morphology_count must be positive")
    if train_traces_per_morphology <= 0:
        raise ValueError("train_traces_per_morphology must be positive")
    non_train = train_traces_per_morphology // 8
    if non_train <= 0:
        raise ValueError("train_traces_per_morphology must be at least 8")

    categories = [
        DatasetCategorySpec(
            name="train",
            kind="ordinary_split",
            morphologies=morphology_count,
            semantic_units_per_morphology=train_traces_per_morphology,
            semantic_unit="trace",
        ),
        DatasetCategorySpec(
            name="val",
            kind="ordinary_split",
            morphologies=morphology_count,
            semantic_units_per_morphology=non_train,
            semantic_unit="trace",
            default_diagnostic_selector=True,
        ),
        DatasetCategorySpec(
            name="test",
            kind="ordinary_split",
            morphologies=morphology_count,
            semantic_units_per_morphology=non_train,
            semantic_unit="trace",
        ),
        DatasetCategorySpec(
            name="private-test",
            kind="ordinary_split",
            morphologies=morphology_count,
            semantic_units_per_morphology=non_train,
            semantic_unit="trace",
        ),
        DatasetCategorySpec(
            name="interventions",
            kind="interventions",
            morphologies=morphology_count,
            semantic_units_per_morphology=non_train,
            semantic_unit="paired_example",
            paired=True,
        ),
    ]

    return {
        "schema_version": 2,
        "dataset_kind": "deep_neuronio_like_v1",
        "framing": "deep_trainable_diagnostic_rich_lean",
        "broad_morphology_release": False,
        "morphology_count": morphology_count,
        "ordinary_splits": list(ORDINARY_SPLITS),
        "intervention_conditions": list(INTERVENTION_CONDITIONS),
        "categories": [category.to_dict() for category in categories],
        "default_diagnostics": {
            "split": "val",
            "morphology_policy": "all_v1_morphologies",
            "intervention_selector": "interventions/<name>",
        },
        "morphologies": [morphology.to_dict() for morphology in provisional_v1_morphologies()],
        "target_views": [view.to_dict() for view in default_target_views()],
        "trace": {
            "duration_ms": 6000,
            "sample_dt_ms": 1,
            "inputs_shape": "(time, input_channels)",
            "targets_shape": "(time, 2)",
        },
    }


def provisional_v1_morphologies() -> tuple[MorphologySpec, ...]:
    """Return the provisional N=5 morphology set for teacher-QC pilots.

    The final dataset freeze may replace entries that fail teacher sanity, but
    any replacement should preserve the same role: close control, moderate
    shift, line-diversity shift, or stronger sane shift.
    """
    return (
        MorphologySpec(
            specimen_id=488_683_425,
            role="reference_control",
            swc_path="src/bnn_sim/assets/allen_l5_template.swc",
            line_name="Rbp4-Cre_KL100",
            reconstruction_type="dendrite-only",
            compatibility_score_lower_better=0.0,
            extracted_branches=99,
            routed_branches=45,
            total_length_ratio_vs_reference=1.0,
            routed_length_ratio_vs_reference=1.0,
            apical_fraction_delta_vs_reference=0.0,
            radial_extent_ratio_vs_reference=1.0,
            routing_shape_match=True,
            caveat="Bundled template; exact Allen specimen 488683425.",
        ),
        MorphologySpec(
            specimen_id=487_664_663,
            role="closest_heldout_control",
            swc_path="data/morphologies/allen_mouse_l5_spiny/specimen_487664663.swc",
            line_name="Rbp4-Cre_KL100",
            reconstruction_type="dendrite-only",
            compatibility_score_lower_better=0.203,
            extracted_branches=98,
            routed_branches=45,
            total_length_ratio_vs_reference=0.930,
            routed_length_ratio_vs_reference=0.946,
            apical_fraction_delta_vs_reference=0.053,
            radial_extent_ratio_vs_reference=1.022,
            routing_shape_match=True,
        ),
        MorphologySpec(
            specimen_id=479_770_916,
            role="intermediate_line_shift",
            swc_path="data/morphologies/allen_mouse_l5_spiny/specimen_479770916.swc",
            line_name="Rorb-IRES2-Cre",
            reconstruction_type="dendrite-only",
            compatibility_score_lower_better=0.247,
            extracted_branches=110,
            routed_branches=45,
            total_length_ratio_vs_reference=1.116,
            routed_length_ratio_vs_reference=1.037,
            apical_fraction_delta_vs_reference=-0.029,
            radial_extent_ratio_vs_reference=0.930,
            routing_shape_match=True,
        ),
        MorphologySpec(
            specimen_id=589_427_435,
            role="line_diversity_shift",
            swc_path="data/morphologies/allen_mouse_l5_spiny/specimen_589427435.swc",
            line_name="Sim1-Cre_KJ18",
            reconstruction_type="dendrite-only",
            compatibility_score_lower_better=0.284,
            extracted_branches=120,
            routed_branches=45,
            total_length_ratio_vs_reference=1.187,
            routed_length_ratio_vs_reference=1.033,
            apical_fraction_delta_vs_reference=0.054,
            radial_extent_ratio_vs_reference=0.975,
            routing_shape_match=True,
        ),
        MorphologySpec(
            specimen_id=488_679_042,
            role="stronger_sane_length_shift",
            swc_path="data/morphologies/allen_mouse_l5_spiny/specimen_488679042.swc",
            line_name="Rorb-IRES2-Cre",
            reconstruction_type="dendrite-only",
            compatibility_score_lower_better=0.405,
            extracted_branches=132,
            routed_branches=45,
            total_length_ratio_vs_reference=1.325,
            routed_length_ratio_vs_reference=1.095,
            apical_fraction_delta_vs_reference=0.019,
            radial_extent_ratio_vs_reference=0.986,
            routing_shape_match=True,
        ),
    )


def default_target_views() -> tuple[TargetViewSpec, ...]:
    """Return the target views required for v1."""
    return (
        TargetViewSpec(
            name="neuronio_clipped_normalized",
            columns=("spike", "soma"),
            units="spike_indicator, normalized_soma",
            description="Historical NeuronIO target view: clip soma above -55 mV, subtract -67.7 mV, scale by 1/10.",
            default_for_training=True,
            default_for_historical_comparison=True,
        ),
        TargetViewSpec(
            name="raw_soma_mv",
            columns=("spike", "soma_mv"),
            units="spike_indicator, millivolts",
            description="Raw teacher soma voltage for clipping diagnostics and spike-shape/state analysis.",
        ),
    )


def default_diagnostic_specs() -> tuple[DiagnosticSpec, ...]:
    """Return the v1 diagnostic catalog with diagnostic-attached plotters."""
    return (
        DiagnosticSpec(
            name="teacher_distribution_qc",
            suites=("teacher-qc", "paper-summary", "full"),
            selectors=("split=val",),
            plotters=(
                PlotterSpec(
                    name="voltage_spike_distribution",
                    kind="atomic",
                    consumes=("teacher_distribution_qc",),
                    description="Distribution view for soma voltage and spike-rate sanity by morphology.",
                ),
            ),
            description="Teacher voltage/spike distribution sanity against the frozen source regime.",
        ),
        DiagnosticSpec(
            name="trace_core",
            suites=("trace-core", "paper-summary", "full"),
            selectors=("split=val",),
            plotters=(
                PlotterSpec(
                    name="metric_by_morphology",
                    kind="atomic",
                    consumes=("trace_core",),
                    description="Per-morphology trace, spike, count, and timing metric view.",
                ),
            ),
            description="Trace fidelity, spike discrimination, count, and timing basics.",
        ),
        DiagnosticSpec(
            name="state_dynamics",
            suites=("state-dynamics", "paper-summary", "full"),
            selectors=("split=val",),
            plotters=(
                PlotterSpec(
                    name="state_metric_matrix",
                    kind="atomic",
                    consumes=("state_dynamics",),
                    description="Metric-by-state matrix for dense behavior-specific failure inspection.",
                ),
            ),
            description="Dense state/regime-conditioned dynamics diagnostics.",
        ),
        DiagnosticSpec(
            name="spike_behavior",
            suites=("spike-behavior", "paper-summary", "full"),
            selectors=("split=val",),
            plotters=(
                PlotterSpec(
                    name="threshold_calibration_curve",
                    kind="atomic",
                    consumes=("spike_behavior",),
                    description="Spike calibration and threshold sensitivity view.",
                ),
            ),
            description="Spike calibration, count, timing, and threshold behavior.",
        ),
        DiagnosticSpec(
            name="feature_fidelity",
            suites=("feature-fidelity", "full"),
            selectors=("split=val",),
            description="Electrophysiology-style feature summaries.",
        ),
        DiagnosticSpec(
            name="morphology_contrast",
            suites=("morphology-contrast", "paper-summary", "full"),
            selectors=("split=val", "morphologies=all_v1"),
            plotters=(
                PlotterSpec(
                    name="morphology_effect_size",
                    kind="atomic",
                    consumes=("morphology_contrast",),
                    description="Per-morphology effect-size and uncertainty view.",
                ),
            ),
            description="Differences and rough uncertainty across all v1 morphologies.",
        ),
        DiagnosticSpec(
            name="morphology_routing_qc",
            suites=("teacher-qc", "paper-summary", "full"),
            selectors=("metadata/morphologies",),
            plotters=(
                PlotterSpec(
                    name="routing_feature_table",
                    kind="atomic",
                    consumes=("morphology_routing_qc",),
                    description="Morphology and routing feature sanity table for v1 teacher validation.",
                ),
            ),
            description="Morphology/routing sanity rows for the selected v1 morphology set.",
        ),
        DiagnosticSpec(
            name="intervention_response",
            suites=("teacher-qc", "intervention-response", "paper-summary", "full"),
            selectors=("condition=interventions/<name>",),
            plotters=(
                PlotterSpec(
                    name="paired_delta_response",
                    kind="atomic",
                    consumes=("intervention_response",),
                    description="Teacher/model paired-delta direction and gain view.",
                ),
            ),
            description="Paired response deltas under top-level intervention conditions.",
        ),
        DiagnosticSpec(
            name="clipping_distortion",
            suites=("teacher-qc", "trace-core", "paper-summary", "full"),
            selectors=("split=val",),
            plotters=(
                PlotterSpec(
                    name="raw_vs_clipped_voltage",
                    kind="atomic",
                    consumes=("clipping_distortion",),
                    description="Raw soma versus clipped target distortion by state and morphology.",
                ),
                PlotterSpec(
                    name="clipping_worst_examples",
                    kind="example",
                    consumes=("clipping_distortion", "trace_core", "spike_behavior"),
                    description="Worst-case example traces combining raw/clipped voltage, spike markers, and trace error.",
                ),
            ),
            description="Explicit raw-vs-clipped target distortion diagnostic.",
        ),
    )


def build_diagnostic_suite_catalog() -> dict[str, Any]:
    """Return suites, diagnostics, and plotter attachments as JSON-like data."""
    diagnostics = default_diagnostic_specs()
    suite_members: dict[str, list[str]] = {suite: [] for suite in DIAGNOSTIC_SUITES}
    for diagnostic in diagnostics:
        for suite in diagnostic.suites:
            suite_members.setdefault(suite, []).append(diagnostic.name)
    return {
        "suites": suite_members,
        "diagnostics": [diagnostic.to_dict() for diagnostic in diagnostics],
        "plotter_policy": {
            "attachment": "diagnostic.plotters",
            "shape": "variable_length_list",
            "fixed_plotter_attributes": False,
            "suite_plotters": "specific_synthesis_only",
        },
        "row_columns": list(DIAGNOSTIC_ROW_COLUMNS),
    }


def validate_v1_manifest_template(manifest: dict[str, Any]) -> None:
    """Validate the v1 manifest contract without requiring local shard files."""
    required_categories = set(ORDINARY_SPLITS) | {"interventions"}
    categories = {entry["name"]: entry for entry in manifest.get("categories", [])}
    missing = required_categories - set(categories)
    if missing:
        raise ValueError(f"missing required dataset categories: {sorted(missing)}")
    if int(manifest.get("morphology_count", 0)) != V1_MORPHOLOGY_COUNT:
        raise ValueError("v1 manifest must use exactly 5 morphologies")
    morphologies = manifest.get("morphologies", [])
    if len(morphologies) != V1_MORPHOLOGY_COUNT:
        raise ValueError("v1 manifest must declare exactly 5 morphology entries")
    if not all(entry.get("routing_shape_match") for entry in morphologies):
        raise ValueError("all provisional v1 morphologies must match the routing shape")
    for name in required_categories:
        if int(categories[name]["morphologies"]) != V1_MORPHOLOGY_COUNT:
            raise ValueError(f"{name} must include all 5 v1 morphologies")
    interventions = categories["interventions"]
    if interventions["kind"] != "interventions" or not interventions["paired"]:
        raise ValueError("interventions must be a top-level paired category")
    target_views = {view["name"] for view in manifest.get("target_views", [])}
    if {"neuronio_clipped_normalized", "raw_soma_mv"} - target_views:
        raise ValueError("manifest must declare both clipped and raw target views")
