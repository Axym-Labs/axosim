"""NeuronIO training and evaluation utilities for ELM / Branch-ELM experiments.

Keep the package import lightweight so data utilities can run in environments
that intentionally omit PyTorch.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "MILLION_NEURON_BEST_QUALITY_REALTIME_PROFILE",
    "MILLION_NEURON_MAX_THROUGHPUT_PROFILE",
    "MILLION_NEURON_REALTIME_CONTRACT",
    "AxoMamba",
    "AxoMambaConfig",
    "BranchELM",
    "BranchELMConfig",
    "LargePopulationSimulationContract",
    "MixedQuantizedNeuronBehaviorBank",
    "PopulationInferenceProfile",
    "ProceduralMorphologyConnectomeRouter",
    "QuantizedSynapticEfficacyBank",
    "compact_active_sources",
    "dequantize_synaptic_efficacies",
    "dequantize_mixed_neuron_behavior_parameters",
    "quantize_synaptic_efficacies",
    "quantize_mixed_neuron_behavior_parameters",
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
        "MILLION_NEURON_BEST_QUALITY_REALTIME_PROFILE",
        "MILLION_NEURON_MAX_THROUGHPUT_PROFILE",
        "MILLION_NEURON_REALTIME_CONTRACT",
        "PopulationInferenceProfile",
    }:
        from axosim.simulation_contract import (
            MILLION_NEURON_BEST_QUALITY_REALTIME_PROFILE,
            MILLION_NEURON_MAX_THROUGHPUT_PROFILE,
            MILLION_NEURON_REALTIME_CONTRACT,
            LargePopulationSimulationContract,
            PopulationInferenceProfile,
        )

        return {
            "LargePopulationSimulationContract": (
                LargePopulationSimulationContract
            ),
            "PopulationInferenceProfile": PopulationInferenceProfile,
            "MILLION_NEURON_MAX_THROUGHPUT_PROFILE": (
                MILLION_NEURON_MAX_THROUGHPUT_PROFILE
            ),
            "MILLION_NEURON_BEST_QUALITY_REALTIME_PROFILE": (
                MILLION_NEURON_BEST_QUALITY_REALTIME_PROFILE
            ),
            "MILLION_NEURON_REALTIME_CONTRACT": (
                MILLION_NEURON_REALTIME_CONTRACT
            ),
        }[name]
    if name in {
        "QuantizedSynapticEfficacyBank",
        "dequantize_synaptic_efficacies",
        "quantize_synaptic_efficacies",
    }:
        from axosim.synapse import (
            QuantizedSynapticEfficacyBank,
            dequantize_synaptic_efficacies,
            quantize_synaptic_efficacies,
        )

        return {
            "QuantizedSynapticEfficacyBank": (
                QuantizedSynapticEfficacyBank
            ),
            "dequantize_synaptic_efficacies": (
                dequantize_synaptic_efficacies
            ),
            "quantize_synaptic_efficacies": (
                quantize_synaptic_efficacies
            ),
        }[name]
    if name in {
        "ProceduralMorphologyConnectomeRouter",
        "compact_active_sources",
    }:
        from axosim.connectome import (
            ProceduralMorphologyConnectomeRouter,
            compact_active_sources,
        )

        return {
            "ProceduralMorphologyConnectomeRouter": (
                ProceduralMorphologyConnectomeRouter
            ),
            "compact_active_sources": compact_active_sources,
        }[name]
    if name in {
        "MixedQuantizedNeuronBehaviorBank",
        "dequantize_mixed_neuron_behavior_parameters",
        "quantize_mixed_neuron_behavior_parameters",
    }:
        from axosim.population import (
            MixedQuantizedNeuronBehaviorBank,
            dequantize_mixed_neuron_behavior_parameters,
            quantize_mixed_neuron_behavior_parameters,
        )

        return {
            "MixedQuantizedNeuronBehaviorBank": (
                MixedQuantizedNeuronBehaviorBank
            ),
            "dequantize_mixed_neuron_behavior_parameters": (
                dequantize_mixed_neuron_behavior_parameters
            ),
            "quantize_mixed_neuron_behavior_parameters": (
                quantize_mixed_neuron_behavior_parameters
            ),
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
