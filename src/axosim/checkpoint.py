from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from axosim.mamba_pytorch import SimpleMambaModel
from axosim.model import BranchELM, BranchELMConfig
from axosim.official import ELM, create_official_elm
from axosim.rnn_variants import BranchTraceRNN, BranchTraceRNNConfig

_RETIRED_AXOMAMBA_CONFIG_KEYS = frozenset(
    {
        "auxiliary_voltage_horizons",
        "branch_impulse_delay_max_steps",
        "branch_impulse_delay_mode",
        "branch_impulse_delay_scale",
        "branch_readout_skip_mode",
        "branch_readout_skip_scale",
        "detach_spike_readout_features",
        "detach_spike_voltage_coupling",
        "hybrid_embedding_dim",
        "hybrid_embedding_mode",
        "hybrid_embedding_scale",
        "local_tcn_gate_mode",
        "local_tcn_gate_scale",
        "nonlinear_bypass_hidden",
        "nonlinear_bypass_mode",
        "nonlinear_bypass_scale",
        "spike_delay_count",
        "spike_delay_max_steps",
        "spike_delay_mode",
        "spike_delay_scale",
        "final_norm_mix",
        "branch_local_mamba_scale",
        "branch_local_model_dim",
        "branch_local_state_dim",
        "branch_memory_taus",
        "branch_memory_kernel",
        "branch_memory_scale",
        "branch_dynamic_gate_rank",
        "branch_dynamic_gate_scale",
        "branch_bilinear_mode",
        "branch_bilinear_temporal_taus",
        "branch_bilinear_temporal_kernel",
        "branch_bilinear_temporal_mode",
        "branch_forcing_reinject_scale",
        "branch_forcing_residual_hidden",
        "branch_forcing_residual_scale",
        "branch_forcing_residual_source",
        "branch_interaction_state_gate_scale",
        "branch_latent_tokens",
        "share_morphology_feature_compute",
        "branch_latent_dim",
        "branch_latent_layers",
        "branch_latent_mode",
        "branch_latent_assignment",
        "branch_latent_temperature",
        "channel_groups",
        "share_group_mamba",
        "hybrid_current_kernel",
        "hybrid_current_mode",
        "input_load_conditioning_scale",
        "liquid_soma_kernel",
        "liquid_soma_mode",
        "liquid_soma_taus",
        "local_tcn_iteration_mode",
        "local_tcn_iterations",
        "local_tcn_coarse_mode",
        "local_tcn_coarse_scale",
        "membrane_branchwise_capacity",
        "membrane_state_dim",
        "membrane_state_kernel",
        "membrane_state_mode",
        "membrane_state_scale",
        "latent_distillation_adapter",
        "event_change_decoder_detach",
        "event_change_decoder_hidden",
        "event_change_decoder_scale",
        "event_change_decoder_source",
        "event_lift_kernel",
        "event_lift_mode",
        "event_lift_scale",
        "morphology_branch_gain_scale",
        "morphology_input_routing",
        "morphology_adapter_mode",
        "morphology_adapter_rank",
        "morphology_adapter_scale",
        "morphology_film_scale",
        "morphology_reinject_scale",
        "morphology_readout_expert_mode",
        "morphology_readout_expert_scale",
        "morphology_route_moment_dim",
        "morphology_route_moment_hidden",
        "morphology_route_moment_scale",
        "morphology_route_moment_source",
        "morphology_route_partition_hidden",
        "morphology_route_partition_scale",
        "morphology_route_partition_source",
        "morphology_route_residual_hidden",
        "morphology_route_residual_scale",
        "morphology_soma_affine_scale",
        "morphology_synapse_graph_degree",
        "morphology_synapse_graph_scale",
        "morphology_synapse_validity_scale",
        "polarity_group_port",
        "predictor_corrector_dim",
        "predictor_corrector_scale",
        "predictor_corrector_source",
        "parameter_padding",
        "recurrent_tail",
        "regime_transition_hidden",
        "regime_transition_mode",
        "regime_transition_scale",
        "soma_final_norm_mix",
        "soma_basis_kernel",
        "soma_basis_mode",
        "soma_basis_scale",
        "soma_basis_taus",
        "support_ode_input",
        "support_ode_mode",
        "support_ode_normalization",
        "support_ode_projection",
        "support_ode_route_weighting",
        "support_ode_scale",
        "soma_calibration_knots",
        "soma_calibration_scale",
        "soma_spike_waveform_detach",
        "soma_spike_waveform_kernel",
        "soma_spike_waveform_scale",
        "soma_tcn_coupling_source",
        "soma_tcn_scale",
        "soma_residual_mlp_activation",
        "soma_residual_mlp_decay",
        "soma_residual_mlp_hidden",
        "soma_residual_mlp_kernel",
        "soma_residual_mlp_mode",
        "detach_spike_waveform",
        "spike_waveform_kernel",
        "spike_waveform_scale",
        "spike_coupling_soma_source",
        "detach_spike_onset",
        "detach_spike_refractory",
        "spike_onset_scale",
        "spike_onset_steps",
        "spike_process_clock",
        "spike_process_decay_init",
        "spike_process_event_threshold",
        "spike_process_mode",
        "spike_process_strength_init",
        "spike_process_temperature",
        "spike_fast_path_hidden",
        "spike_fast_path_scale",
        "spike_fast_path_source",
        "spike_refractory_scale",
        "spike_refractory_steps",
        "spike_temporal_mamba_scale",
        "synapse_trace_kernel",
        "synapse_trace_mixture",
        "synapse_trace_residual",
        "synapse_trace_taus",
        "task_head_mamba_mode",
        "task_head_mamba_scale",
        "task_refiner_mode",
        "voltage_state_decay_init",
        "voltage_state_decoder_hidden",
        "voltage_state_decoder_mode",
        "voltage_state_decoder_scale",
        "voltage_state_kernel",
    }
)
_RETIRED_BLOCK_FORECAST_CONFIG_KEYS = frozenset(
    {
        "context_trajectory_skip",
        "mamba_time_scale",
    }
)
_RETIRED_AXO_TEMPORAL_CONFIG_KEYS = frozenset(
    {
        "patch_mode",
    }
)


def save_checkpoint(
    model: torch.nn.Module, path: str | Path, *, metadata: dict[str, Any] | None = None
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_kind, config = _serialize_model_config(model)
    metadata_payload = {"model_kind": model_kind, **(metadata or {})}
    payload = {
        "model_kind": model_kind,
        "config": config,
        "state_dict": model.state_dict(),
        "metadata": metadata_payload,
    }
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path, *, map_location: str = "cpu"
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model_kind = payload.get(
        "model_kind", payload.get("metadata", {}).get("model_kind", "baseline")
    )
    if model_kind in {"baseline", "branch_elm"}:
        config = BranchELMConfig(**payload["config"])
        model = BranchELM(config)
    elif model_kind == "branch-trace-rnn":
        config = BranchTraceRNNConfig(**payload["config"])
        model = BranchTraceRNN(config)
    elif model_kind == "axomamba":
        from axosim.axomamba import AxoMamba, AxoMambaConfig

        config = AxoMambaConfig(**_active_axomamba_config(payload["config"]))
        model = AxoMamba(config)
    elif model_kind == "axomamba-pytorch":
        from axosim.axomamba import AxoMambaConfig, AxoPyTorchMamba

        config = AxoMambaConfig(**_active_axomamba_config(payload["config"]))
        model = AxoPyTorchMamba(config)
    elif model_kind in {"axo-temporal", "axo-temporal-pytorch"}:
        from axosim.axomamba import AxoMamba, AxoMambaConfig, AxoPyTorchMamba
        from axosim.temporal_core import (
            AxoTemporalCoreConfig,
            AxoTemporalModel,
        )

        values = payload["config"]
        backbone_config = AxoMambaConfig(**_active_axomamba_config(values["backbone"]))
        backbone = (
            AxoPyTorchMamba(backbone_config)
            if model_kind == "axo-temporal-pytorch"
            else AxoMamba(backbone_config)
        )
        model = AxoTemporalModel(
            backbone,
            AxoTemporalCoreConfig(
                **{
                    key: value
                    for key, value in values["temporal_core"].items()
                    if key not in _RETIRED_AXO_TEMPORAL_CONFIG_KEYS
                }
            ),
        )
    elif model_kind in {
        "axo-block-forecast",
        "axo-block-forecast-pytorch",
    }:
        from axosim.axomamba import (
            AxoMamba,
            AxoMambaConfig,
            AxoPyTorchMamba,
        )
        from axosim.block_forecast import (
            BlockForecastConfig,
            CausalBlockForecastModel,
        )

        values = payload["config"]
        backbone_config = AxoMambaConfig(**_active_axomamba_config(values["backbone"]))
        backbone = (
            AxoPyTorchMamba(backbone_config)
            if model_kind == "axo-block-forecast-pytorch"
            else AxoMamba(backbone_config)
        )
        model = CausalBlockForecastModel(
            backbone,
            BlockForecastConfig(
                **{
                    key: value
                    for key, value in values["forecast"].items()
                    if key not in _RETIRED_BLOCK_FORECAST_CONFIG_KEYS
                }
            ),
        )
    elif model_kind == "adaptive-p4-surrogate":
        from axosim.p4_surrogate import (
            AdaptiveP4Surrogate,
            P4SurrogateConfig,
        )

        config = P4SurrogateConfig(**payload["config"])
        route_features = payload["state_dict"]["route_features"]
        model = AdaptiveP4Surrogate(config, route_features)
    elif model_kind == "adaptive-support-p4-surrogate":
        from axosim.support_surrogate import (
            AdaptiveSupportP4Surrogate,
            SupportP4Config,
        )

        config = SupportP4Config(**payload["config"])
        route_features = payload["state_dict"]["route_features"]
        model = AdaptiveSupportP4Surrogate(config, route_features)
    elif model_kind == "mamba-official":
        from axosim.mamba_official import BranchOfficialMamba, BranchOfficialMambaConfig

        config = BranchOfficialMambaConfig(**payload["config"])
        model = BranchOfficialMamba(config)
    elif model_kind == "branch-mamba-pytorch":
        from axosim.mamba_official import BranchOfficialMambaConfig, BranchPyTorchMamba

        config = BranchOfficialMambaConfig(**payload["config"])
        model = BranchPyTorchMamba(config)
    elif model_kind == "mamba-pytorch":
        model = SimpleMambaModel(**payload["config"])
    elif model_kind == "official":
        model = create_official_elm(payload["config"])
    else:
        raise ValueError(f"unknown model_kind in checkpoint: {model_kind}")
    strict = model_kind != "branch-trace-rnn"
    state_dict = dict(payload["state_dict"])
    if (
        "branch_subunit_mask" in model.state_dict()
        and "branch_subunit_mask" not in state_dict
    ):
        state_dict["branch_subunit_mask"] = model.state_dict()["branch_subunit_mask"]
    model.load_state_dict(state_dict, strict=strict)
    return model, dict(payload.get("metadata", {}))


def _active_axomamba_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in _RETIRED_AXOMAMBA_CONFIG_KEYS
    }


def _serialize_model_config(model: torch.nn.Module) -> tuple[str, dict[str, Any]]:
    from axosim.block_forecast import CausalBlockForecastModel
    from axosim.p4_surrogate import AdaptiveP4Surrogate
    from axosim.support_surrogate import AdaptiveSupportP4Surrogate
    from axosim.temporal_core import AxoTemporalModel

    if isinstance(model, AdaptiveP4Surrogate):
        return "adaptive-p4-surrogate", asdict(model.config)
    if isinstance(model, AdaptiveSupportP4Surrogate):
        return "adaptive-support-p4-surrogate", asdict(model.config)
    if isinstance(model, CausalBlockForecastModel):
        pytorch_backbone = model.backbone.__class__.__name__ == "AxoPyTorchMamba"
        return (
            "axo-block-forecast-pytorch" if pytorch_backbone else "axo-block-forecast",
            {
                "backbone": asdict(model.backbone.config),
                "forecast": asdict(model.forecast_config),
            },
        )
    if isinstance(model, AxoTemporalModel):
        pytorch_backbone = model.backbone.__class__.__name__ == "AxoPyTorchMamba"
        return (
            "axo-temporal-pytorch" if pytorch_backbone else "axo-temporal",
            {
                "backbone": asdict(model.backbone.config),
                "temporal_core": asdict(model.temporal_config),
            },
        )
    if isinstance(model, BranchELM):
        return "baseline", asdict(model.config)
    if isinstance(model, BranchTraceRNN):
        return "branch-trace-rnn", {
            "num_input": int(model.num_input),
            "num_output": int(model.num_output),
            "num_branch": int(model.num_branch),
            "num_synapse_per_branch": int(model.num_synapse_per_branch),
            "num_memory": int(model.num_memory),
            "lambda_value": float(model.lambda_value),
            "mlp_num_layers": int(model.mlp_num_layers),
            "mlp_hidden_size": int(model.mlp_hidden_size),
            "mlp_activation": str(model.mlp_activation),
            "memory_tau_min": float(model.memory_tau_min),
            "memory_tau_max": float(model.memory_tau_max),
            "trace_tau_min": float(model.trace_tau_min),
            "trace_tau_max": float(model.trace_tau_max),
            "trace_banks": int(model.trace_banks),
            "conductance_cell": bool(model.conductance_cell),
            "membrane_tau": float(model.membrane_tau),
            "readout_hidden_size": int(model.readout_hidden_size),
            "w_s_value": float(model.w_s_value),
            "delta_t": float(model.delta_t),
            "branch_gain": bool(getattr(model, "branch_gain_enabled", False)),
            "branch_nonlinearity": str(getattr(model, "branch_nonlinearity", "none")),
            "soma_multiplicative_gate": bool(
                getattr(model, "soma_multiplicative_gate_enabled", False)
            ),
            "branch_readout_skip": bool(
                getattr(model, "branch_readout_skip_enabled", False)
            ),
        }
    if model.__class__.__name__ == "AxoMamba" and hasattr(model, "config"):
        return "axomamba", asdict(model.config)
    if model.__class__.__name__ == "AxoPyTorchMamba" and hasattr(model, "config"):
        return "axomamba-pytorch", asdict(model.config)
    if model.__class__.__name__ == "BranchOfficialMamba" and hasattr(model, "config"):
        return "mamba-official", asdict(model.config)
    if model.__class__.__name__ == "BranchPyTorchMamba" and hasattr(model, "config"):
        return "branch-mamba-pytorch", asdict(model.config)
    if isinstance(model, SimpleMambaModel):
        return "mamba-pytorch", dict(model.config)
    if isinstance(model, ELM):
        return "official", {
            "num_input": int(model.num_input),
            "num_output": int(model.num_output),
            "num_memory": int(model.num_memory),
            "lambda_value": float(model.lambda_value),
            "mlp_num_layers": int(model.mlp_num_layers),
            "mlp_hidden_size": int(model.mlp_hidden_size),
            "mlp_activation": str(model.mlp_activation),
            "tau_s_value": float(model.tau_s_value),
            "memory_tau_min": float(model.memory_tau_min),
            "memory_tau_max": float(model.memory_tau_max),
            "learn_memory_tau": bool(model.learn_memory_tau),
            "w_s_value": float(model.w_s_value),
            "num_branch": int(model.num_branch),
            "num_synapse_per_branch": int(model.num_synapse_per_branch),
            "input_to_synapse_routing": model.input_to_synapse_routing,
            "delta_t": float(model.delta_t),
            "readout_hidden_size": int(getattr(model, "readout_hidden_size", 0)),
        }
    raise TypeError(f"unsupported checkpoint model type: {type(model).__name__}")
