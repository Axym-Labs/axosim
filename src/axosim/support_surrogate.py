from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from axosim.deployment import GRU_BEHAVIOR_ADAPTATION


@dataclass(frozen=True)
class SupportP4Config:
    """Trainable low-order support recurrence at exact P4 cadence."""

    input_dim: int = 1278
    route_feature_dim: int = 118
    token_dim: int = 58
    state_dim: int = 16
    patch_size: int = 4
    output_dim: int = 2
    morphology_ids: tuple[str, ...] = ()
    behavior_adaptation: bool = True

    def __post_init__(self) -> None:
        if min(
            self.input_dim,
            self.route_feature_dim,
            self.token_dim,
            self.state_dim,
        ) <= 0:
            raise ValueError("support dimensions must be positive")
        if self.patch_size != 4:
            raise ValueError("the support surrogate advances at exact P4")
        if self.output_dim != 2:
            raise ValueError("the support surrogate predicts two outputs")
        if not self.morphology_ids:
            raise ValueError("at least one morphology ID is required")


class AdaptiveSupportP4Surrogate(nn.Module):
    """Cheap learned support dynamics with compiled neuron adaptation."""

    def __init__(
        self,
        config: SupportP4Config,
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
        self.input_projection = nn.Linear(
            config.patch_size * config.route_feature_dim,
            config.token_dim,
        )
        self.support_input = nn.Linear(
            config.token_dim,
            config.state_dim,
            bias=False,
        )
        self.support_recurrent = nn.Linear(
            config.state_dim,
            config.state_dim,
            bias=False,
        )
        self.state_bias = nn.Parameter(torch.zeros(config.state_dim))
        self.shared_decay_logit = nn.Parameter(
            self._initial_decay_logits(config.state_dim)
        )
        native_outputs = config.patch_size * config.output_dim
        self.output_head = nn.Linear(config.state_dim, native_outputs)
        self.skip_head = nn.Linear(
            config.token_dim,
            native_outputs,
            bias=False,
        )
        self.behavior_parameter_count = (
            GRU_BEHAVIOR_ADAPTATION.parameter_count
            if config.behavior_adaptation
            else 0
        )
        self.cache_width = (
            2 * config.token_dim
            + 3 * config.state_dim
            + 2 * native_outputs
        )
        if config.behavior_adaptation:
            self.behavior_adaptation_bank: nn.Parameter | None = nn.Parameter(
                torch.empty(
                    len(config.morphology_ids),
                    self.behavior_parameter_count,
                )
            )
            self.adaptation_compiler: nn.Linear | None = nn.Linear(
                self.behavior_parameter_count,
                self.cache_width,
                bias=False,
            )
        else:
            self.register_parameter("behavior_adaptation_bank", None)
            self.adaptation_compiler = None
        self.base_soma_prediction: torch.Tensor | None = None
        self.reset_parameters()

    @property
    def patch_size(self) -> int:
        return self.config.patch_size

    @property
    def gate_feature_dim(self) -> int:
        return self.config.token_dim + self.config.state_dim

    @staticmethod
    def _initial_decay_logits(state_dim: int) -> torch.Tensor:
        taus = torch.logspace(
            math.log10(4.0),
            math.log10(128.0),
            state_dim,
        )
        decay = torch.exp(-4.0 / taus).clamp(1e-4, 1.0 - 1e-4)
        return torch.logit(decay)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.input_projection.weight, a=math.sqrt(5))
        nn.init.zeros_(self.input_projection.bias)
        nn.init.orthogonal_(self.support_input.weight)
        nn.init.orthogonal_(self.support_recurrent.weight, gain=0.1)
        nn.init.normal_(self.output_head.weight, std=0.01)
        nn.init.zeros_(self.output_head.bias)
        nn.init.normal_(self.skip_head.weight, std=0.01)
        if self.behavior_adaptation_bank is not None:
            nn.init.normal_(self.behavior_adaptation_bank, std=1e-3)
        if self.adaptation_compiler is not None:
            nn.init.normal_(
                self.adaptation_compiler.weight,
                std=1.0 / math.sqrt(self.behavior_parameter_count),
            )

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
        return torch.einsum("bti,bif->btf", inputs, selected)

    def compile_adaptation(
        self,
        behavior_parameters: torch.Tensor,
    ) -> torch.Tensor:
        if self.adaptation_compiler is None:
            raise ValueError("behavior adaptation is not configured")
        expected = (
            behavior_parameters.shape[0],
            self.behavior_parameter_count,
        )
        if tuple(behavior_parameters.shape) != expected:
            raise ValueError(f"behavior_parameters must have shape {expected}")
        return self.adaptation_compiler(behavior_parameters)

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

    def _cache_slices(
        self,
        cache: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        token = self.config.token_dim
        state = self.config.state_dim
        outputs = self.config.patch_size * self.config.output_dim
        boundaries = (
            0,
            token,
            2 * token,
            2 * token + state,
            2 * token + 2 * state,
            2 * token + 3 * state,
            2 * token + 3 * state + outputs,
            self.cache_width,
        )
        return tuple(
            cache[:, start:stop]
            for start, stop in zip(
                boundaries[:-1],
                boundaries[1:],
                strict=True,
            )
        )

    def step_token(
        self,
        token: torch.Tensor,
        state: torch.Tensor,
        *,
        adaptation_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(token.shape) != (
            state.shape[0],
            self.config.token_dim,
        ):
            raise ValueError("token has incompatible shape")
        if tuple(state.shape) != (
            token.shape[0],
            self.config.state_dim,
        ):
            raise ValueError("state has incompatible shape")
        if adaptation_cache is None:
            adapted_token = token
            decay_delta = state_gain = state_offset = None
            output_gain = output_offset = None
        else:
            if tuple(adaptation_cache.shape) != (
                token.shape[0],
                self.cache_width,
            ):
                raise ValueError("adaptation_cache has incompatible shape")
            (
                token_offset,
                token_gain,
                decay_delta,
                state_gain,
                state_offset,
                output_gain,
                output_offset,
            ) = self._cache_slices(adaptation_cache)
            adapted_token = (
                (token + token_offset) * (1.0 + token_gain)
            )
        proposal = torch.tanh(
            self.support_input(adapted_token)
            + self.support_recurrent(state)
            + self.state_bias
            + (0.0 if state_offset is None else state_offset)
        )
        decay = torch.sigmoid(
            self.shared_decay_logit
            + (0.0 if decay_delta is None else decay_delta)
        )
        if state_gain is not None:
            proposal = proposal * (1.0 + state_gain)
        next_state = decay * state + (1.0 - decay) * proposal
        output = (
            self.output_head(next_state)
            + self.skip_head(adapted_token)
        )
        if output_gain is not None:
            output = output * (1.0 + output_gain) + output_offset
        return output.view(
            token.shape[0],
            self.config.patch_size,
            self.config.output_dim,
        ), next_state

    def step_p4(
        self,
        feature_patch: torch.Tensor,
        state: torch.Tensor,
        *,
        behavior_parameters: torch.Tensor | None = None,
        adaptation_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (
            state.shape[0],
            self.config.patch_size,
            self.config.route_feature_dim,
        )
        if tuple(feature_patch.shape) != expected:
            raise ValueError(f"feature_patch must have shape {expected}")
        if behavior_parameters is not None and adaptation_cache is not None:
            raise ValueError(
                "provide behavior_parameters or adaptation_cache, not both"
            )
        if behavior_parameters is not None:
            adaptation_cache = self.compile_adaptation(
                behavior_parameters
            )
        token = self.input_projection(feature_patch.flatten(start_dim=1))
        return self.step_token(
            token,
            state,
            adaptation_cache=adaptation_cache,
        )

    def _resolve_adaptation_cache(
        self,
        *,
        morphology_indices: torch.Tensor | None = None,
        behavior_parameters: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.behavior_adaptation_bank is None:
            if behavior_parameters is not None:
                raise ValueError("behavior adaptation is not configured")
            return None
        if behavior_parameters is None:
            if morphology_indices is None:
                raise ValueError(
                    "adaptive support surrogate requires morphology "
                    "indices or explicit behavior parameters"
                )
            behavior_parameters = self.behavior_adaptation_bank.index_select(
                0,
                morphology_indices.to(
                    device=self.behavior_adaptation_bank.device,
                    dtype=torch.long,
                ),
            )
        return self.compile_adaptation(behavior_parameters)

    def _rollout_feature_summaries(
        self,
        summaries: torch.Tensor,
        *,
        adaptation_cache: torch.Tensor | None,
        capture_gate_features: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (
            summaries.ndim != 3
            or summaries.shape[-1] != self.config.route_feature_dim
            or summaries.shape[1] <= 0
        ):
            raise ValueError(
                "summaries must have shape (batch, time, route_feature_dim)"
            )
        full_steps = summaries.shape[1] - (
            summaries.shape[1] % self.patch_size
        )
        output = summaries.new_zeros(
            summaries.shape[0],
            summaries.shape[1],
            self.config.output_dim,
        )
        if full_steps < self.patch_size:
            self.base_soma_prediction = output[..., 1:2]
            gate_features = (
                summaries.new_zeros(
                    summaries.shape[0],
                    0,
                    self.gate_feature_dim,
                )
                if capture_gate_features
                else None
            )
            return output, gate_features
        blocks = full_steps // self.patch_size
        tokens = self.input_projection(
            summaries[:, :full_steps].reshape(
                summaries.shape[0],
                blocks,
                self.patch_size * self.config.route_feature_dim,
            )
        )
        state = self.initial_state(
            summaries.shape[0],
            device=summaries.device,
            dtype=summaries.dtype,
        )
        forecasts = []
        states = []
        for block in range(blocks):
            prediction, state = self.step_token(
                tokens[:, block],
                state,
                adaptation_cache=adaptation_cache,
            )
            forecasts.append(prediction)
            if capture_gate_features:
                states.append(state)
        forecast = torch.cat(forecasts, dim=1)
        available = max(output.shape[1] - self.patch_size, 0)
        output = torch.cat(
            (
                output[:, : self.patch_size],
                forecast[:, :available],
                output[:, self.patch_size + available :],
            ),
            dim=1,
        )
        self.base_soma_prediction = output[..., 1:2]
        gate_features = None
        if capture_gate_features:
            generated = torch.cat(
                (tokens, torch.stack(states, dim=1)),
                dim=-1,
            )
            gate_features = summaries.new_zeros(
                summaries.shape[0],
                blocks,
                self.gate_feature_dim,
            )
            gate_features[:, 1:] = generated[:, : max(blocks - 1, 0)]
        return output, gate_features

    def forward_feature_summaries(
        self,
        summaries: torch.Tensor,
        *,
        morphology_indices: torch.Tensor | None = None,
        behavior_parameters: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cache = self._resolve_adaptation_cache(
            morphology_indices=morphology_indices,
            behavior_parameters=behavior_parameters,
        )
        output, _ = self._rollout_feature_summaries(
            summaries,
            adaptation_cache=cache,
            capture_gate_features=False,
        )
        return output

    def forward_with_gate_features(
        self,
        inputs: torch.Tensor,
        *,
        morphology_indices: torch.Tensor | None = None,
        behavior_parameters: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predictions and their existing causal token/state values."""

        if morphology_indices is None:
            raise ValueError("morphology indices are required")
        summaries = self.aggregate_inputs(
            inputs,
            morphology_indices=morphology_indices,
        )
        cache = self._resolve_adaptation_cache(
            morphology_indices=morphology_indices,
            behavior_parameters=behavior_parameters,
        )
        output, gate_features = self._rollout_feature_summaries(
            summaries,
            adaptation_cache=cache,
            capture_gate_features=True,
        )
        assert gate_features is not None
        return output, gate_features

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
