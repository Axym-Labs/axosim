"""Contract scoring for model-generated biological population dynamics."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping


@dataclass(frozen=True)
class PopulationBiologicalContract:
    """Measured targets and practical gates for connected populations."""

    firing_rate_hz: float
    excitatory_firing_rate_hz: float
    inhibitory_firing_rate_hz: float
    input_events_per_ms: float
    inhibitory_input_fraction: float
    input_count_cv: float
    input_count_lag_correlation: Mapping[str, float]
    morphology_firing_rate_hz: Mapping[str, float]


AXOBENCH_POPULATION_CONTRACT = PopulationBiologicalContract(
    firing_rate_hz=3.869090909090909,
    excitatory_firing_rate_hz=2.6272857142857142,
    inhibitory_firing_rate_hz=8.836311688311689,
    input_events_per_ms=13.392972727272728,
    inhibitory_input_fraction=0.456764,
    input_count_cv=0.355,
    input_count_lag_correlation={
        "1": 0.415,
        "4": 0.413,
        "16": 0.402,
        "64": 0.291,
        "256": 0.0679,
    },
    morphology_firing_rate_hz={
        "specimen_479770916": 7.736,
        "specimen_487664663": 2.136,
        "specimen_488679042": 5.373,
        "specimen_488683425": 3.127,
        "specimen_589427435": 0.973,
    },
)

_EPSILON = 1e-9


def role_correct_population_contract(
    contract: PopulationBiologicalContract = AXOBENCH_POPULATION_CONTRACT,
) -> PopulationBiologicalContract:
    """Return morphology targets for excitatory-only morphology groups."""

    scale = contract.excitatory_firing_rate_hz / contract.firing_rate_hz
    return replace(
        contract,
        morphology_firing_rate_hz={
            morphology: rate * scale
            for morphology, rate in contract.morphology_firing_rate_hz.items()
        },
    )


def _log_ratio(observed: float, target: float) -> float:
    observed = float(observed)
    target = float(target)
    if not math.isfinite(observed):
        observed = _EPSILON
    if not math.isfinite(target):
        target = _EPSILON
    return math.log(
        max(observed, _EPSILON) / max(target, _EPSILON)
    )


def _mean_square(values: list[float]) -> float:
    return sum(value * value for value in values) / len(values)


def _within_ratio(observed: float, target: float, ratio: float) -> bool:
    return (
        math.isfinite(observed)
        and target / ratio <= observed <= target * ratio
    )


def _scaled_difference(
    observed: float,
    target: float,
    scale: float,
) -> float:
    if not math.isfinite(observed):
        return 5.0
    return (observed - target) / scale


def _mapping_values(
    observed: Mapping[str, float],
    target: Mapping[str, float],
) -> list[tuple[float, float]]:
    missing = set(target) - set(observed)
    if missing:
        raise ValueError(f"population summary is missing keys: {sorted(missing)}")
    return [(float(observed[key]), float(value)) for key, value in target.items()]


def score_population_summary(
    summary: Mapping[str, object],
    contract: PopulationBiologicalContract = AXOBENCH_POPULATION_CONTRACT,
) -> dict[str, object]:
    """Score a rollout without allowing one statistic to hide another."""

    overall = float(summary["firing_rate_hz"])
    tail = float(summary["tail_firing_rate_hz"])
    excitatory = float(summary["excitatory_firing_rate_hz"])
    inhibitory = float(summary["inhibitory_firing_rate_hz"])
    tail_excitatory = float(summary["tail_excitatory_firing_rate_hz"])
    tail_inhibitory = float(summary["tail_inhibitory_firing_rate_hz"])
    input_events = float(summary["input_events_per_ms"])
    inhibitory_input_fraction = float(summary["inhibitory_input_fraction"])
    count_cv = float(summary["input_count_cv"])
    tail_count_cv = float(summary["tail_input_count_cv"])
    lag_pairs = _mapping_values(
        summary["input_count_lag_correlation"],  # type: ignore[arg-type]
        contract.input_count_lag_correlation,
    )
    tail_lag_pairs = _mapping_values(
        summary["tail_input_count_lag_correlation"],  # type: ignore[arg-type]
        contract.input_count_lag_correlation,
    )
    morphology_pairs = _mapping_values(
        summary["morphology_firing_rate_hz"],  # type: ignore[arg-type]
        contract.morphology_firing_rate_hz,
    )
    tail_morphology_pairs = _mapping_values(
        summary["tail_morphology_firing_rate_hz"],  # type: ignore[arg-type]
        contract.morphology_firing_rate_hz,
    )

    components = {
        "global_rate": _mean_square(
            [
                _log_ratio(overall, contract.firing_rate_hz),
                _log_ratio(tail, contract.firing_rate_hz),
            ]
        ),
        "cell_type_rates": _mean_square(
            [
                _log_ratio(excitatory, contract.excitatory_firing_rate_hz),
                _log_ratio(inhibitory, contract.inhibitory_firing_rate_hz),
                _log_ratio(
                    tail_excitatory,
                    contract.excitatory_firing_rate_hz,
                ),
                _log_ratio(
                    tail_inhibitory,
                    contract.inhibitory_firing_rate_hz,
                ),
            ]
        ),
        "input_load": (
            _log_ratio(input_events, contract.input_events_per_ms) ** 2
            + (
                (inhibitory_input_fraction - contract.inhibitory_input_fraction)
                / 0.1
            )
            ** 2
        )
        / 2.0,
        "variability": _mean_square(
            [
                _log_ratio(count_cv, contract.input_count_cv),
                _log_ratio(tail_count_cv, contract.input_count_cv),
            ]
        ),
        "temporal": _mean_square(
            [
                _scaled_difference(observed, target, 0.2)
                for observed, target in lag_pairs + tail_lag_pairs
            ]
        ),
        "morphology": _mean_square(
            [
                _log_ratio(observed, target)
                for observed, target in morphology_pairs
                + tail_morphology_pairs
            ]
        ),
        "stationarity": _mean_square(
            [
                _log_ratio(tail, overall),
                _log_ratio(tail_excitatory, excitatory),
                _log_ratio(tail_inhibitory, inhibitory),
                _log_ratio(tail_count_cv, count_cv),
            ]
        ),
    }
    weights = {
        "global_rate": 2.0,
        "cell_type_rates": 2.0,
        "input_load": 1.0,
        "variability": 1.0,
        "temporal": 0.5,
        "morphology": 1.5,
        "stationarity": 1.0,
    }
    objective = sum(
        weights[name] * value for name, value in components.items()
    ) / sum(weights.values())

    morphology_gate = all(
        _within_ratio(observed, target, 2.0)
        for observed, target in morphology_pairs + tail_morphology_pairs
    )
    gates = {
        "global_rate": (
            _within_ratio(overall, contract.firing_rate_hz, 1.5)
            and _within_ratio(tail, contract.firing_rate_hz, 1.5)
        ),
        "cell_type_rates": all(
            (
                _within_ratio(excitatory, contract.excitatory_firing_rate_hz, 2.0),
                _within_ratio(inhibitory, contract.inhibitory_firing_rate_hz, 2.0),
                _within_ratio(
                    tail_excitatory,
                    contract.excitatory_firing_rate_hz,
                    2.0,
                ),
                _within_ratio(
                    tail_inhibitory,
                    contract.inhibitory_firing_rate_hz,
                    2.0,
                ),
            )
        ),
        "input_load": (
            _within_ratio(input_events, contract.input_events_per_ms, 1.5)
            and abs(
                inhibitory_input_fraction
                - contract.inhibitory_input_fraction
            )
            <= 0.1
        ),
        "variability": (
            _within_ratio(count_cv, contract.input_count_cv, 2.0)
            and _within_ratio(tail_count_cv, contract.input_count_cv, 2.0)
        ),
        "temporal": components["temporal"] <= 1.0,
        "morphology": morphology_gate,
        "stationarity": (
            _within_ratio(tail, overall, 1.3)
            and _within_ratio(tail_excitatory, excitatory, 1.5)
            and _within_ratio(tail_inhibitory, inhibitory, 1.5)
            and _within_ratio(tail_count_cv, count_cv, 1.5)
        ),
    }
    return {
        "objective": float(objective),
        "components": components,
        "gates": gates,
        "passes_contract_gate": all(gates.values()),
    }
