"""NeuronIO training and evaluation utilities for ELM / Branch-ELM experiments.

Keep the package import lightweight so data utilities can run in environments
that intentionally omit PyTorch.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AxoMamba",
    "AxoMambaConfig",
    "BranchELM",
    "BranchELMConfig",
    "LargePopulationSimulationContract",
    "MILLION_NEURON_REALTIME_CONTRACT",
]


def __getattr__(name: str) -> Any:
    if name in {"AxoMamba", "AxoMambaConfig"}:
        from axosim.axomamba import AxoMamba, AxoMambaConfig

        return {"AxoMamba": AxoMamba, "AxoMambaConfig": AxoMambaConfig}[name]
    if name in {"BranchELM", "BranchELMConfig"}:
        from axosim.model import BranchELM, BranchELMConfig

        return {"BranchELM": BranchELM, "BranchELMConfig": BranchELMConfig}[name]
    if name in {
        "LargePopulationSimulationContract",
        "MILLION_NEURON_REALTIME_CONTRACT",
    }:
        from axosim.simulation_contract import (
            LargePopulationSimulationContract,
            MILLION_NEURON_REALTIME_CONTRACT,
        )

        return {
            "LargePopulationSimulationContract": (
                LargePopulationSimulationContract
            ),
            "MILLION_NEURON_REALTIME_CONTRACT": (
                MILLION_NEURON_REALTIME_CONTRACT
            ),
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
