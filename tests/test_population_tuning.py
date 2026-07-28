from __future__ import annotations

from copy import deepcopy
import math

from axosim.population_tuning import (
    AXOBENCH_POPULATION_CONTRACT,
    role_correct_population_contract,
    score_population_summary,
)


def _matching_summary() -> dict[str, object]:
    contract = AXOBENCH_POPULATION_CONTRACT
    return {
        "firing_rate_hz": contract.firing_rate_hz,
        "tail_firing_rate_hz": contract.firing_rate_hz,
        "excitatory_firing_rate_hz": contract.excitatory_firing_rate_hz,
        "tail_excitatory_firing_rate_hz": (
            contract.excitatory_firing_rate_hz
        ),
        "inhibitory_firing_rate_hz": contract.inhibitory_firing_rate_hz,
        "tail_inhibitory_firing_rate_hz": (
            contract.inhibitory_firing_rate_hz
        ),
        "input_events_per_ms": contract.input_events_per_ms,
        "inhibitory_input_fraction": contract.inhibitory_input_fraction,
        "input_count_cv": contract.input_count_cv,
        "tail_input_count_cv": contract.input_count_cv,
        "input_count_lag_correlation": dict(
            contract.input_count_lag_correlation
        ),
        "tail_input_count_lag_correlation": dict(
            contract.input_count_lag_correlation
        ),
        "morphology_firing_rate_hz": dict(
            contract.morphology_firing_rate_hz
        ),
        "tail_morphology_firing_rate_hz": dict(
            contract.morphology_firing_rate_hz
        ),
    }


def test_exact_contract_has_zero_loss_and_passes_gates() -> None:
    result = score_population_summary(_matching_summary())

    assert result["objective"] == 0.0
    assert result["passes_contract_gate"]
    assert all(result["gates"].values())


def test_role_correct_contract_scales_excitatory_morphology_targets() -> None:
    contract = role_correct_population_contract()
    scale = (
        AXOBENCH_POPULATION_CONTRACT.excitatory_firing_rate_hz
        / AXOBENCH_POPULATION_CONTRACT.firing_rate_hz
    )

    assert contract.firing_rate_hz == (
        AXOBENCH_POPULATION_CONTRACT.firing_rate_hz
    )
    assert contract.inhibitory_firing_rate_hz == (
        AXOBENCH_POPULATION_CONTRACT.inhibitory_firing_rate_hz
    )
    assert contract.morphology_firing_rate_hz == {
        morphology: rate * scale
        for morphology, rate in (
            AXOBENCH_POPULATION_CONTRACT.morphology_firing_rate_hz.items()
        )
    }


def test_score_rejects_correct_mean_with_bursty_unstable_activity() -> None:
    summary = _matching_summary()
    summary["input_count_cv"] = 2.0
    summary["tail_input_count_cv"] = 2.5
    summary["tail_firing_rate_hz"] = 0.2

    result = score_population_summary(summary)

    assert not result["passes_contract_gate"]
    assert not result["gates"]["variability"]
    assert not result["gates"]["stationarity"]
    assert result["objective"] > 1.0


def test_score_rejects_morphology_collapse_hidden_by_global_rate() -> None:
    summary = _matching_summary()
    summary["morphology_firing_rate_hz"] = {
        morphology: AXOBENCH_POPULATION_CONTRACT.firing_rate_hz
        for morphology in AXOBENCH_POPULATION_CONTRACT.morphology_firing_rate_hz
    }
    summary["tail_morphology_firing_rate_hz"] = deepcopy(
        summary["morphology_firing_rate_hz"]
    )

    result = score_population_summary(summary)

    assert not result["passes_contract_gate"]
    assert not result["gates"]["morphology"]
    assert result["components"]["morphology"] > 0.0


def test_score_rejects_wrong_ei_balance_and_input_load() -> None:
    summary = _matching_summary()
    summary["excitatory_firing_rate_hz"] = 4.0
    summary["inhibitory_firing_rate_hz"] = 3.0
    summary["tail_excitatory_firing_rate_hz"] = 4.0
    summary["tail_inhibitory_firing_rate_hz"] = 3.0
    summary["input_events_per_ms"] = 5.0
    summary["inhibitory_input_fraction"] = 0.1

    result = score_population_summary(summary)

    assert not result["passes_contract_gate"]
    assert not result["gates"]["cell_type_rates"]
    assert not result["gates"]["input_load"]


def test_score_returns_finite_penalty_for_silent_population() -> None:
    summary = _matching_summary()
    for key in (
        "firing_rate_hz",
        "tail_firing_rate_hz",
        "excitatory_firing_rate_hz",
        "tail_excitatory_firing_rate_hz",
        "inhibitory_firing_rate_hz",
        "tail_inhibitory_firing_rate_hz",
        "input_count_cv",
        "tail_input_count_cv",
    ):
        summary[key] = 0.0
    summary["input_count_lag_correlation"] = {
        key: float("nan")
        for key in AXOBENCH_POPULATION_CONTRACT.input_count_lag_correlation
    }
    summary["tail_input_count_lag_correlation"] = deepcopy(
        summary["input_count_lag_correlation"]
    )
    summary["morphology_firing_rate_hz"] = {
        key: 0.0
        for key in AXOBENCH_POPULATION_CONTRACT.morphology_firing_rate_hz
    }
    summary["tail_morphology_firing_rate_hz"] = deepcopy(
        summary["morphology_firing_rate_hz"]
    )

    result = score_population_summary(summary)

    assert math.isfinite(result["objective"])
    assert not result["passes_contract_gate"]
