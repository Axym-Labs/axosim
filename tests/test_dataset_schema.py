import pytest

from bnn_sim.benchmark.dataset_schema import (
    DIAGNOSTIC_SUITES,
    INTERVENTION_CONDITIONS,
    V1_MORPHOLOGY_COUNT,
    build_diagnostic_suite_catalog,
    build_v1_dataset_manifest_template,
    default_diagnostic_specs,
    validate_v1_manifest_template,
)


def test_v1_dataset_manifest_encodes_deep_lean_layout():
    manifest = build_v1_dataset_manifest_template()

    validate_v1_manifest_template(manifest)

    categories = {entry["name"]: entry for entry in manifest["categories"]}
    assert manifest["schema_version"] == 2
    assert manifest["broad_morphology_release"] is False
    assert manifest["morphology_count"] == V1_MORPHOLOGY_COUNT
    assert set(categories) == {"train", "val", "test", "private-test", "interventions"}
    assert categories["train"]["semantic_units_per_morphology"] == 20_000
    assert categories["val"]["semantic_units_per_morphology"] == 2_500
    assert categories["interventions"]["semantic_unit"] == "paired_example"
    assert categories["interventions"]["paired"] is True
    assert set(manifest["intervention_conditions"]) == set(INTERVENTION_CONDITIONS)


def test_v1_manifest_requires_raw_and_clipped_target_views():
    manifest = build_v1_dataset_manifest_template()
    manifest["target_views"] = [view for view in manifest["target_views"] if view["name"] != "raw_soma_mv"]

    with pytest.raises(ValueError, match="clipped and raw"):
        validate_v1_manifest_template(manifest)


def test_diagnostic_specs_use_variable_length_plotter_lists():
    diagnostics = default_diagnostic_specs()
    clipping = next(diagnostic for diagnostic in diagnostics if diagnostic.name == "clipping_distortion")
    feature = next(diagnostic for diagnostic in diagnostics if diagnostic.name == "feature_fidelity")

    assert len(clipping.plotters) == 2
    assert len(feature.plotters) == 0
    assert {plotter.kind for plotter in clipping.plotters} == {"atomic", "example"}
    assert all(hasattr(diagnostic, "plotters") for diagnostic in diagnostics)
    assert not any(hasattr(diagnostic, "plot_main") or hasattr(diagnostic, "plot_breakdown") for diagnostic in diagnostics)


def test_diagnostic_suite_catalog_is_overlapping_not_partitioned():
    catalog = build_diagnostic_suite_catalog()

    assert set(DIAGNOSTIC_SUITES).issubset(catalog["suites"])
    assert catalog["plotter_policy"]["shape"] == "variable_length_list"
    assert catalog["plotter_policy"]["fixed_plotter_attributes"] is False
    assert "clipping_distortion" in catalog["suites"]["teacher-qc"]
    assert "clipping_distortion" in catalog["suites"]["paper-summary"]
    assert "intervention_response" in catalog["suites"]["teacher-qc"]
    assert "intervention_response" in catalog["suites"]["intervention-response"]
