from bnn_sim.benchmark.profiles import (
    DEFAULT_STATES,
    build_axis_plan,
    get_benchmark_profile,
    list_benchmark_profiles,
)


def test_paper_default_keeps_metric_state_product_dense():
    profile = get_benchmark_profile("paper-default")
    plan = build_axis_plan(profile)

    assert "paper-default" in list_benchmark_profiles()
    assert tuple(plan["metric_state_evaluation"]["states"]) == DEFAULT_STATES
    assert plan["metric_state_evaluation"]["mode"] == "dense"
    assert plan["morphology_transfer"]["evaluates"] == "all metric suites over all states"


def test_paper_default_prunes_intervention_morphology_cartesian_product():
    plan = build_axis_plan("paper-default")
    morph = plan["morphology_intervention_sanity"]

    assert plan["profile"]["morphology_count"] > 3
    assert morph["morphology_count"] < plan["profile"]["morphology_count"]
    assert morph["selected_pairs"] < morph["full_cartesian_pairs_avoided"]
    assert plan["source_interventions"]["compute_full_metrics_on_interventions"] is False
    assert "soma_delta_gain" in plan["source_interventions"]["metrics"]


def test_smoke_profile_is_not_the_publishable_morphology_budget():
    smoke = get_benchmark_profile("smoke")
    paper = get_benchmark_profile("paper-default")

    assert smoke.morphology_count == 3
    assert paper.morphology_count >= 15
    assert smoke.morphology_intervention_count < paper.morphology_intervention_count
