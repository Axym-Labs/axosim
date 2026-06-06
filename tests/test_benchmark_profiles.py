from bnn_sim.benchmark.profiles import (
    DEFAULT_STATES,
    build_axis_plan,
    estimate_profile_cost,
    get_benchmark_profile,
    list_benchmark_profiles,
)


def test_paper_default_keeps_metric_state_product_dense():
    profile = get_benchmark_profile("paper-default")
    plan = build_axis_plan(profile)

    assert "paper-default" in list_benchmark_profiles()
    assert profile.profile_status == "provisional_capacity_audited"
    assert tuple(plan["metric_state_evaluation"]["states"]) == DEFAULT_STATES
    assert plan["metric_state_evaluation"]["mode"] == "dense"
    assert plan["morphology_transfer"]["evaluates"] == "all metric suites over all states"
    assert plan["design_strategy"]["profile_status"] == "provisional_capacity_audited"


def test_paper_default_prunes_intervention_morphology_cartesian_product():
    plan = build_axis_plan("paper-default")
    morph = plan["morphology_intervention_sanity"]

    assert plan["profile"]["morphology_count"] > 3
    assert morph["morphology_count"] < plan["profile"]["morphology_count"]
    assert morph["selected_pairs"] < morph["full_cartesian_pairs"]
    assert morph["avoided_pairs"] == morph["full_cartesian_pairs"] - morph["selected_pairs"]
    assert plan["source_interventions"]["compute_full_metrics_on_interventions"] is False
    assert "soma_delta_gain" in plan["source_interventions"]["metrics"]
    assert plan["capacity_estimate"]["within_target_wallclock"] is True
    assert plan["capacity_estimate"]["full_cartesian_extra_teacher_hours_avoided"] > 0


def test_smoke_profile_is_not_the_publishable_morphology_budget():
    smoke = get_benchmark_profile("smoke")
    paper = get_benchmark_profile("paper-default")

    assert smoke.morphology_count == 3
    assert paper.morphology_count >= 15
    assert smoke.morphology_intervention_count < paper.morphology_intervention_count


def test_profile_cost_estimate_identifies_teacher_generation_as_dominant():
    cost = estimate_profile_cost("paper-default")

    assert cost["teacher_total_seconds"] > cost["model_eval_seconds"]
    assert cost["dominant_cost_component"] in {
        "morphology_baselines",
        "source_interventions",
        "morphology_intervention_sanity",
    }
    assert cost["trace_counts"]["avoided_morphology_intervention_traces"] > 0
