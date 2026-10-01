"""NeuronIO training and evaluation utilities for ELM / Branch-ELM experiments.

Keep the package import lightweight so data utilities can run in environments
that intentionally omit PyTorch.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AXOSIM_POPULATION_PROFILE",
    "MILLION_NEURON_REALTIME_CONTRACT",
    "AxoMamba",
    "AxoMambaConfig",
    "AxoSimGRU",
    "AxoSimLite",
    "AxoSimMamba",
    "AxoSimModelProfile",
    "AxoSimPopulation",
    "CudaGraphAdaptationStep",
    "HomeostaticThresholdController",
    "AXOSIM_MODEL_PROFILES",
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
    "create_axosim_profile",
    "ConnectedPopulation",
    "ExplicitConnectome",
    "InputEvents",
    "PopulationFrame",
    "ProceduralConnectome",
    "create_population",
]


def __getattr__(name: str) -> Any:
    if name in {
        "ConnectedPopulation", "ExplicitConnectome", "InputEvents",
        "PopulationFrame", "ProceduralConnectome", "create_population",
    }:
        from axosim.connected_population import (
            ConnectedPopulation, ExplicitConnectome, InputEvents,
            PopulationFrame, ProceduralConnectome, create_population,
        )

        return {
            "ConnectedPopulation": ConnectedPopulation,
            "ExplicitConnectome": ExplicitConnectome,
            "InputEvents": InputEvents,
            "PopulationFrame": PopulationFrame,
            "ProceduralConnectome": ProceduralConnectome,
            "create_population": create_population,
        }[name]
    if name == "CudaGraphAdaptationStep":
        from axosim.adaptation import CudaGraphAdaptationStep

        return CudaGraphAdaptationStep
    if name == "HomeostaticThresholdController":
        from axosim.activity import HomeostaticThresholdController

        return HomeostaticThresholdController
    if name in {
        "AxoSimGRU",
        "AxoSimLite",
        "AxoSimMamba",
        "AxoSimPopulation",
    }:
        from axosim.interfaces import (
            AxoSimGRU,
            AxoSimLite,
            AxoSimMamba,
            AxoSimPopulation,
        )

        return {
            "AxoSimGRU": AxoSimGRU,
            "AxoSimLite": AxoSimLite,
            "AxoSimMamba": AxoSimMamba,
            "AxoSimPopulation": AxoSimPopulation,
        }[name]
    if name in {"AxoMamba", "AxoMambaConfig"}:
        from axosim.axomamba import AxoMamba, AxoMambaConfig

        return {"AxoMamba": AxoMamba, "AxoMambaConfig": AxoMambaConfig}[name]
    if name in {
        "AXOSIM_MODEL_PROFILES",
        "AxoSimModelProfile",
        "create_axosim_profile",
    }:
        from axosim.model_family import (
            AXOSIM_MODEL_PROFILES,
            AxoSimModelProfile,
            create_axosim_profile,
        )

        return {
            "AXOSIM_MODEL_PROFILES": AXOSIM_MODEL_PROFILES,
            "AxoSimModelProfile": AxoSimModelProfile,
            "create_axosim_profile": create_axosim_profile,
        }[name]
    if name in {"BranchELM", "BranchELMConfig"}:
        from axosim.model import BranchELM, BranchELMConfig

        return {"BranchELM": BranchELM, "BranchELMConfig": BranchELMConfig}[name]
    if name in {
        "AXOSIM_POPULATION_PROFILE",
        "LargePopulationSimulationContract",
        "MILLION_NEURON_REALTIME_CONTRACT",
        "PopulationInferenceProfile",
    }:
        from axosim.simulation_contract import (
            AXOSIM_POPULATION_PROFILE,
            MILLION_NEURON_REALTIME_CONTRACT,
            LargePopulationSimulationContract,
            PopulationInferenceProfile,
        )

        return {
            "AXOSIM_POPULATION_PROFILE": AXOSIM_POPULATION_PROFILE,
            "LargePopulationSimulationContract": (
                LargePopulationSimulationContract
            ),
            "PopulationInferenceProfile": PopulationInferenceProfile,
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
