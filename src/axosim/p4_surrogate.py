from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from axosim.deployment import (
    GRU_BEHAVIOR_ADAPTATION,
    NeuronBehaviorAdaptation,
)


@dataclass(frozen=True)
class P4SurrogateConfig:
    """Cheap adaptive recurrent model evaluated at exact P4 boundaries."""

    input_dim: int = 1278
    route_feature_dim: int = 9
    state_dim: int = 58
    patch_size: int = 4
    output_dim: int = 2
    morphology_ids: tuple[str, ...] = ()
    behavior_adaptation: bool = True
    feature_nonlinearity: str = "none"
    feature_scale_init: float = 0.1

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.route_feature_dim <= 0:
            raise ValueError("input and route feature dimensions must be positive")
        if self.state_dim != GRU_BEHAVIOR_ADAPTATION.width:
            raise ValueError(
                "state_dim must match the established behavior adaptation width"
            )
        if self.patch_size != 4:
            raise ValueError("the conditional surrogate advances at exact P4")
        if self.output_dim != GRU_BEHAVIOR_ADAPTATION.outputs:
            raise ValueError(
                "output_dim must match the established behavior adaptation outputs"
            )
        if not self.morphology_ids:
            raise ValueError("at least one morphology ID is required")
        if self.feature_nonlinearity not in {"none", "tanh"}:
            raise ValueError("feature_nonlinearity must be 'none' or 'tanh'")
        if self.feature_scale_init <= 0:
            raise ValueError("feature_scale_init must be positive")


def normalize_signed_route_features(features: torch.Tensor) -> torch.Tensor:
    """Make every physical feature signed and normalize its input mass."""

    if features.ndim != 3 or features.shape[-1] < 1:
        raise ValueError(
            "route features must have shape (morphologies, inputs, features)"
        )
    signed = features.clone().float()
    sign = torch.where(
        signed[..., :1] >= 0,
        torch.ones_like(signed[..., :1]),
        -torch.ones_like(signed[..., :1]),
    )
    signed[..., 0:1] = sign
    if signed.shape[-1] > 1:
        signed[..., 1:] = signed[..., 1:] * sign
    denominator = signed.abs().sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(signed.dtype).eps
    )
    return signed / denominator


class AdaptiveP4Surrogate(nn.Module):
    """A low-cost P4 GRU over physical input summaries.

    Each call consumes four ordered native input summaries and predicts the
    following four native outputs. Shared weights describe common neuronal
    dynamics; the established behavior vector modifies the token, transition,
    recurrent state, and readout for every neuron.
    """

    def __init__(
        self,
        config: P4SurrogateConfig,
        route_features: torch.Tensor,
    ) -> None:
        super().__init__()
        expected = (
            len(config.morphology_ids),
            config.input_dim,
            config.route_feature_dim,
        )
        if tuple(route_features.shape) != expected:
            raise ValueError(f"route_features must have shape {expected}")
        self.config = config
        self.num_input = config.input_dim
        self.num_output = config.output_dim
        self.register_buffer(
            "route_features",
            route_features.detach().clone().float(),
        )
        if config.feature_nonlinearity == "tanh":
            self.feature_log_scale: nn.Parameter | None = nn.Parameter(
                torch.full(
                    (config.route_feature_dim,),
                    math.log(config.feature_scale_init),
                )
            )
            self.feature_bias: nn.Parameter | None = nn.Parameter(
                torch.zeros(config.route_feature_dim)
            )
        else:
            self.register_parameter("feature_log_scale", None)
            self.register_parameter("feature_bias", None)
        packed_input = config.patch_size * config.route_feature_dim
        self.input_projection = nn.Linear(packed_input, config.state_dim)
        self.input_norm = nn.LayerNorm(config.state_dim)
        self.gru = nn.GRUCell(config.state_dim, config.state_dim)
        self.phase_embedding = nn.Parameter(
            torch.empty(config.patch_size, config.state_dim)
        )
        self.trajectory_head = nn.Linear(
            config.state_dim,
            config.patch_size * config.output_dim,
        )
        self.adaptation: NeuronBehaviorAdaptation | None = (
            GRU_BEHAVIOR_ADAPTATION
            if config.behavior_adaptation
            else None
        )
        if self.adaptation is None:
            self.register_parameter("behavior_adaptation_bank", None)
        else:
            self.behavior_adaptation_bank = nn.Parameter(
                torch.zeros(
                    len(config.morphology_ids),
                    self.adaptation.parameter_count,
                )
            )
            self._initialize_adaptation_bank()
        self.base_soma_prediction: torch.Tensor | None = None
        self.last_execution_mode = "uninitialized"
        self.reset_parameters()

    @property
    def patch_size(self) -> int:
        return self.config.patch_size

    @property
    def behavior_parameter_count(self) -> int:
        return 0 if self.adaptation is None else self.adaptation.parameter_count

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.input_projection.weight, a=math.sqrt(5))
        nn.init.zeros_(self.input_projection.bias)
        nn.init.normal_(self.phase_embedding, std=0.02)
        nn.init.normal_(self.trajectory_head.weight, std=0.01)
        nn.init.zeros_(self.trajectory_head.bias)

    def _initialize_adaptation_bank(self) -> None:
        if self.adaptation is None or self.behavior_adaptation_bank is None:
            return
        with torch.no_grad():
            for name in ("gru_input_down", "gru_recurrent_down"):
                values = self._adaptation_parameter(
                    self.behavior_adaptation_bank,
                    name,
                )
                nn.init.kaiming_uniform_(values, a=math.sqrt(5))

    def aggregate_inputs(
        self,
        inputs: torch.Tensor,
        *,
        morphology_indices: torch.Tensor,
    ) -> torch.Tensor:
        if (
            inputs.ndim != 3
            or inputs.shape[-1] != self.config.input_dim
            or inputs.shape[0] != morphology_indices.shape[0]
        ):
            raise ValueError(
                "inputs and morphology indices have incompatible shapes"
            )
        selected = self.route_features.index_select(
            0,
            morphology_indices.to(
                device=self.route_features.device,
                dtype=torch.long,
            ),
        )
        summaries = torch.einsum("bti,bif->btf", inputs, selected)
        if self.feature_log_scale is not None:
            assert self.feature_bias is not None
            scale = self.feature_log_scale.exp().to(
                dtype=summaries.dtype,
                device=summaries.device,
            )
            bias = self.feature_bias.to(
                dtype=summaries.dtype,
                device=summaries.device,
            )
            summaries = (
                torch.tanh((summaries + bias) * scale) / scale
            )
        return summaries

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.config.state_dim,
            device=device,
            dtype=dtype,
        )

    def step_p4(
        self,
        feature_patch: torch.Tensor,
        state: torch.Tensor,
        *,
        behavior_parameters: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (
            state.shape[0],
            self.config.patch_size,
            self.config.route_feature_dim,
        )
        if tuple(feature_patch.shape) != expected:
            raise ValueError(f"feature_patch must have shape {expected}")
        if tuple(state.shape) != (
            feature_patch.shape[0],
            self.config.state_dim,
        ):
            raise ValueError("state has incompatible shape")
        behavior_parameters = self._validate_behavior_parameters(
            feature_patch,
            behavior_parameters,
        )
        token = self.input_projection(feature_patch.flatten(start_dim=1))
        if behavior_parameters is not None:
            token = self._adapt_token(token, behavior_parameters)
            next_state = self._adaptive_gru(
                self.input_norm(token),
                state,
                behavior_parameters,
            )
        else:
            next_state = self.gru(self.input_norm(token), state)
        output_state = next_state
        if behavior_parameters is not None:
            recurrent_gain = self._adaptation_parameter(
                behavior_parameters,
                "recurrent_output_gain",
            )
            output_state = output_state * (1.0 + recurrent_gain)
        output = self.trajectory_head(output_state).view(
            feature_patch.shape[0],
            self.config.patch_size,
            self.config.output_dim,
        )
        if behavior_parameters is not None:
            phase_state = (
                output_state[:, None, :]
                + self.phase_embedding[None, :, :].to(output_state.dtype)
            )
            weight = self._adaptation_parameter(
                behavior_parameters,
                "readout_weight",
            )
            bias = self._adaptation_parameter(
                behavior_parameters,
                "readout_bias",
            )
            output = (
                output
                + torch.einsum("bow,bpw->bpo", weight, phase_state)
                + bias[:, None, :]
            )
        return output, next_state

    def feature_state_sequence(
        self,
        summaries: torch.Tensor,
        *,
        morphology_indices: torch.Tensor | None = None,
        behavior_parameters: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance exact P4 summaries and return every recurrent state."""

        if (
            summaries.ndim != 3
            or summaries.shape[-1] != self.config.route_feature_dim
            or summaries.shape[1] <= 0
        ):
            raise ValueError(
                "summaries must have shape (batch, time, route_feature_dim)"
            )
        explicit_behavior = behavior_parameters is not None
        behavior_parameters = self._resolve_behavior_parameters(
            summaries,
            morphology_indices=morphology_indices,
            behavior_parameters=behavior_parameters,
        )
        full_steps = summaries.shape[1] - summaries.shape[1] % self.patch_size
        if full_steps < self.patch_size:
            return summaries.new_empty(
                summaries.shape[0],
                0,
                self.config.state_dim,
            )
        if initial_state is not None:
            expected = (summaries.shape[0], self.config.state_dim)
            if tuple(initial_state.shape) != expected:
                raise ValueError(f"initial_state must have shape {expected}")
            if initial_state.device != summaries.device:
                raise ValueError(
                    "initial_state and summaries must be on the same device"
                )
        if not explicit_behavior and initial_state is None:
            return self._grouped_sequence_states(
                summaries[:, :full_steps],
                morphology_indices=morphology_indices,
                behavior_parameters=behavior_parameters,
            )
        state = (
            initial_state
            if initial_state is not None
            else self.initial_state(
                summaries.shape[0],
                device=summaries.device,
                dtype=summaries.dtype,
            )
        )
        states = []
        for start in range(0, full_steps, self.patch_size):
            _, state = self.step_p4(
                summaries[:, start : start + self.patch_size],
                state,
                behavior_parameters=behavior_parameters,
            )
            states.append(state)
        return torch.stack(states, dim=1)

    def decode_state_sequence(
        self,
        states: torch.Tensor,
        *,
        morphology_indices: torch.Tensor | None = None,
        behavior_parameters: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode recurrent P4 states into distinct native-rate outputs."""

        if (
            states.ndim != 3
            or states.shape[-1] != self.config.state_dim
        ):
            raise ValueError(
                "states must have shape (batch, blocks, state_dim)"
            )
        behavior_parameters = self._resolve_behavior_parameters(
            states,
            morphology_indices=morphology_indices,
            behavior_parameters=behavior_parameters,
        )
        output_states = states
        if behavior_parameters is not None:
            recurrent_gain = self._adaptation_parameter(
                behavior_parameters,
                "recurrent_output_gain",
            )
            output_states = output_states * (
                1.0 + recurrent_gain[:, None, :]
            )
        output = self.trajectory_head(output_states).view(
            states.shape[0],
            states.shape[1],
            self.patch_size,
            self.config.output_dim,
        )
        if behavior_parameters is not None:
            phase_state = (
                output_states[:, :, None, :]
                + self.phase_embedding[None, None, :, :].to(
                    output_states.dtype
                )
            )
            weight = self._adaptation_parameter(
                behavior_parameters,
                "readout_weight",
            )
            bias = self._adaptation_parameter(
                behavior_parameters,
                "readout_bias",
            )
            output = (
                output
                + torch.einsum(
                    "bow,bnpw->bnpo",
                    weight,
                    phase_state,
                )
                + bias[:, None, None, :]
            )
        return output

    def forward_feature_summaries(
        self,
        summaries: torch.Tensor,
        *,
        morphology_indices: torch.Tensor | None = None,
        behavior_parameters: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            summaries.ndim != 3
            or summaries.shape[-1] != self.config.route_feature_dim
            or summaries.shape[1] <= 0
        ):
            raise ValueError(
                "summaries must have shape (batch, time, route_feature_dim)"
            )
        explicit_behavior = behavior_parameters is not None
        behavior_parameters = self._resolve_behavior_parameters(
            summaries,
            morphology_indices=morphology_indices,
            behavior_parameters=behavior_parameters,
        )
        full_steps = summaries.shape[1] - summaries.shape[1] % self.patch_size
        output = summaries.new_zeros(
            summaries.shape[0],
            summaries.shape[1],
            self.config.output_dim,
        )
        if full_steps < self.patch_size:
            self.base_soma_prediction = output[..., 1:2]
            return output
        if not explicit_behavior:
            forecast = self._grouped_sequence_forecast(
                summaries[:, :full_steps],
                morphology_indices=morphology_indices,
                behavior_parameters=behavior_parameters,
            )
            self.last_execution_mode = (
                "grouped_morphology"
                if behavior_parameters is not None
                else "shared_sequence"
            )
            return self._align_forecast(
                output,
                forecast,
            )
        self.last_execution_mode = "explicit_step"
        state = self.initial_state(
            summaries.shape[0],
            device=summaries.device,
            dtype=summaries.dtype,
        )
        predictions = []
        for start in range(0, full_steps, self.patch_size):
            prediction, state = self.step_p4(
                summaries[:, start : start + self.patch_size],
                state,
                behavior_parameters=behavior_parameters,
            )
            predictions.append(prediction)
        forecast = torch.cat(predictions, dim=1)
        return self._align_forecast(output, forecast)

    def _align_forecast(
        self,
        output: torch.Tensor,
        forecast: torch.Tensor,
    ) -> torch.Tensor:
        available = max(output.shape[1] - self.patch_size, 0)
        prefix = output[:, : self.patch_size]
        active = forecast[:, :available]
        suffix = output[:, self.patch_size + available :]
        output = torch.cat((prefix, active, suffix), dim=1)
        self.base_soma_prediction = output[..., 1:2]
        return output

    def _grouped_sequence_forecast(
        self,
        summaries: torch.Tensor,
        *,
        morphology_indices: torch.Tensor | None,
        behavior_parameters: torch.Tensor | None,
    ) -> torch.Tensor:
        states = self._grouped_sequence_states(
            summaries,
            morphology_indices=morphology_indices,
            behavior_parameters=behavior_parameters,
        )
        return self.decode_state_sequence(
            states,
            morphology_indices=morphology_indices,
            behavior_parameters=behavior_parameters,
        ).flatten(start_dim=1, end_dim=2)

    def _grouped_sequence_states(
        self,
        summaries: torch.Tensor,
        *,
        morphology_indices: torch.Tensor | None,
        behavior_parameters: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, steps, features = summaries.shape
        blocks = steps // self.patch_size
        tokens = self.input_projection(
            summaries.reshape(
                batch,
                blocks,
                self.patch_size * features,
            )
        )
        if behavior_parameters is not None:
            tokens = self._adapt_token(
                tokens,
                behavior_parameters[:, None, :],
            )
        tokens = self.input_norm(tokens)
        if behavior_parameters is None:
            states = self._gru_sequence(
                tokens,
                self.gru.weight_ih,
                self.gru.weight_hh,
            )
        else:
            if morphology_indices is None:
                raise ValueError(
                    "grouped adaptive execution requires morphology indices"
                )
            states = torch.zeros_like(tokens)
            for morphology_index in torch.unique(
                morphology_indices,
                sorted=True,
            ):
                selection = torch.nonzero(
                    morphology_indices == morphology_index,
                    as_tuple=False,
                ).flatten()
                row = self.behavior_adaptation_bank[
                    morphology_index : morphology_index + 1
                ]
                input_down = self._adaptation_parameter(
                    row,
                    "gru_input_down",
                ).squeeze(0)
                input_up = self._adaptation_parameter(
                    row,
                    "gru_input_up",
                ).squeeze(0)
                recurrent_down = self._adaptation_parameter(
                    row,
                    "gru_recurrent_down",
                ).squeeze(0)
                recurrent_up = self._adaptation_parameter(
                    row,
                    "gru_recurrent_up",
                ).squeeze(0)
                grouped = self._gru_sequence(
                    tokens.index_select(0, selection),
                    self.gru.weight_ih + input_up @ input_down,
                    self.gru.weight_hh + recurrent_up @ recurrent_down,
                )
                states = states.index_copy(0, selection, grouped)
        return states

    def _gru_sequence(
        self,
        tokens: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
    ) -> torch.Tensor:
        initial = tokens.new_zeros(
            1,
            tokens.shape[0],
            self.config.state_dim,
        )
        output, _ = torch._VF.gru(
            tokens,
            initial,
            [
                weight_ih,
                weight_hh,
                self.gru.bias_ih,
                self.gru.bias_hh,
            ],
            True,
            1,
            0.0,
            self.training,
            False,
            True,
        )
        return output

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        morphology_indices: torch.Tensor | None = None,
        behavior_parameters: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if morphology_indices is None:
            raise ValueError("morphology indices are required")
        summaries = self.aggregate_inputs(
            inputs,
            morphology_indices=morphology_indices,
        )
        return self.forward_feature_summaries(
            summaries,
            morphology_indices=morphology_indices,
            behavior_parameters=behavior_parameters,
        )

    def _resolve_behavior_parameters(
        self,
        reference: torch.Tensor,
        *,
        morphology_indices: torch.Tensor | None,
        behavior_parameters: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.adaptation is None:
            if behavior_parameters is not None:
                raise ValueError("behavior adaptation is not configured")
            return None
        if behavior_parameters is None:
            if morphology_indices is None or self.behavior_adaptation_bank is None:
                raise ValueError(
                    "adaptive surrogate requires morphology indices or "
                    "runtime behavior parameters"
                )
            behavior_parameters = self.behavior_adaptation_bank.index_select(
                0,
                morphology_indices.to(
                    device=self.behavior_adaptation_bank.device,
                    dtype=torch.long,
                ),
            )
        return self._validate_behavior_parameters(
            reference,
            behavior_parameters,
        )

    def _validate_behavior_parameters(
        self,
        reference: torch.Tensor,
        behavior_parameters: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if behavior_parameters is None:
            return None
        if self.adaptation is None:
            raise ValueError("behavior adaptation is not configured")
        expected = (reference.shape[0], self.adaptation.parameter_count)
        if tuple(behavior_parameters.shape) != expected:
            raise ValueError(f"behavior_parameters must have shape {expected}")
        if behavior_parameters.device != reference.device:
            raise ValueError("behavior_parameters must be on the input device")
        return behavior_parameters

    def _adaptation_parameter(
        self,
        behavior_parameters: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if self.adaptation is None:
            raise RuntimeError("behavior adaptation is not configured")
        selection = self.adaptation.tensor_slices[name]
        return behavior_parameters[:, selection].view(
            behavior_parameters.shape[0],
            *self.adaptation.tensor_shapes[name],
        )

    def _adapt_token(
        self,
        token: torch.Tensor,
        behavior_parameters: torch.Tensor,
    ) -> torch.Tensor:
        parameter_batch = behavior_parameters.shape[0]
        flat_parameters = behavior_parameters.reshape(parameter_batch, -1)
        branch_token = self._adaptation_parameter(
            flat_parameters,
            "branch_token_offset",
        )
        latent_gain = self._adaptation_parameter(
            flat_parameters,
            "latent_gain",
        )
        latent_offset = self._adaptation_parameter(
            flat_parameters,
            "latent_offset",
        )
        while branch_token.ndim < token.ndim:
            branch_token = branch_token.unsqueeze(1)
            latent_gain = latent_gain.unsqueeze(1)
            latent_offset = latent_offset.unsqueeze(1)
        return (
            (token + branch_token) * (1.0 + latent_gain)
            + latent_offset
        )

    @staticmethod
    def _low_rank(
        values: torch.Tensor,
        down: torch.Tensor,
        up: torch.Tensor,
    ) -> torch.Tensor:
        compressed = torch.bmm(down, values.unsqueeze(-1))
        return torch.bmm(up, compressed).squeeze(-1)

    def _adaptive_gru(
        self,
        token: torch.Tensor,
        state: torch.Tensor,
        behavior_parameters: torch.Tensor,
    ) -> torch.Tensor:
        input_gates = F.linear(
            token,
            self.gru.weight_ih,
            self.gru.bias_ih,
        )
        recurrent_gates = F.linear(
            state,
            self.gru.weight_hh,
            self.gru.bias_hh,
        )
        input_gates = input_gates + self._low_rank(
            token,
            self._adaptation_parameter(
                behavior_parameters,
                "gru_input_down",
            ),
            self._adaptation_parameter(
                behavior_parameters,
                "gru_input_up",
            ),
        )
        recurrent_gates = recurrent_gates + self._low_rank(
            state,
            self._adaptation_parameter(
                behavior_parameters,
                "gru_recurrent_down",
            ),
            self._adaptation_parameter(
                behavior_parameters,
                "gru_recurrent_up",
            ),
        )
        ir, iz, inn = input_gates.chunk(3, dim=-1)
        hr, hz, hn = recurrent_gates.chunk(3, dim=-1)
        reset = torch.sigmoid(ir + hr)
        update = torch.sigmoid(iz + hz)
        candidate = torch.tanh(inn + reset * hn)
        return (1.0 - update) * candidate + update * state
