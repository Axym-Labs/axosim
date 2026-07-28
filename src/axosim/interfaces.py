"""Public differentiable interfaces for AxoSim models and populations."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn

from axosim.axomamba import AxoMamba as AxoSimMamba
from axosim.block_forecast import CausalBlockForecastModel as AxoSimGRU
from axosim.support_surrogate import (
    AdaptiveSupportP4Surrogate as AxoSimLite,
)


class AxoSimPopulation(nn.Module):
    """Differentiable AxoSim-Lite population with distinct adaptation banks."""

    def __init__(
        self,
        neuron: AxoSimLite,
        *,
        morphology_indices: torch.Tensor,
        contact_branch_indices: torch.Tensor,
    ) -> None:
        super().__init__()
        if morphology_indices.ndim != 1:
            raise ValueError("morphology_indices must be one-dimensional")
        if (
            contact_branch_indices.ndim != 2
            or contact_branch_indices.shape[0]
            != morphology_indices.shape[0]
        ):
            raise ValueError(
                "contact_branch_indices must have shape "
                "(population, contacts)"
            )
        if (
            contact_branch_indices.numel()
            and (
                contact_branch_indices.min() < 0
                or contact_branch_indices.max() >= neuron.config.input_dim
            )
        ):
            raise ValueError("contact branch index lies outside the model")
        if (
            morphology_indices.numel()
            and (
                morphology_indices.min() < 0
                or morphology_indices.max()
                >= len(neuron.config.morphology_ids)
            )
        ):
            raise ValueError("morphology index lies outside the model")

        self.neuron = neuron
        self.register_buffer(
            "morphology_indices",
            morphology_indices.detach().clone().to(dtype=torch.long),
        )
        self.register_buffer(
            "contact_branch_indices",
            contact_branch_indices.detach()
            .clone()
            .to(dtype=torch.long),
        )
        population = morphology_indices.shape[0]
        morphologies = len(neuron.config.morphology_ids)
        runtime_width = neuron.cache_width
        contacts = contact_branch_indices.shape[1]
        self.behavior_adapter = nn.Parameter(
            torch.zeros(population, runtime_width)
        )
        self.morphology_adapter = nn.Parameter(
            torch.zeros(morphologies, runtime_width)
        )
        self.synaptic_log_efficacy = nn.Parameter(
            torch.zeros(population, contacts)
        )

    @property
    def population_size(self) -> int:
        return self.morphology_indices.numel()

    @property
    def runtime_coefficients_per_neuron(self) -> int:
        return self.behavior_adapter.shape[1]

    @property
    def synaptic_efficacies_per_neuron(self) -> int:
        return self.synaptic_log_efficacy.shape[1]

    def adaptation_parameters(self) -> Iterator[nn.Parameter]:
        yield self.behavior_adapter
        yield self.morphology_adapter
        yield self.synaptic_log_efficacy

    def trainable_adaptation_parameters(self) -> Iterator[nn.Parameter]:
        yield from (
            parameter
            for parameter in self.adaptation_parameters()
            if parameter.requires_grad
        )

    def enable_adaptation_training(
        self,
        *,
        behavior: bool = True,
        morphology: bool = True,
        synaptic: bool = True,
    ) -> None:
        for parameter in self.neuron.parameters():
            parameter.requires_grad_(False)
        self.behavior_adapter.requires_grad_(behavior)
        self.morphology_adapter.requires_grad_(morphology)
        self.synaptic_log_efficacy.requires_grad_(synaptic)

    def enable_full_training(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def synaptic_efficacies(
        self,
        synaptic_log_efficacy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = (
            self.synaptic_log_efficacy
            if synaptic_log_efficacy is None
            else synaptic_log_efficacy
        )
        expected = (
            self.population_size,
            self.synaptic_efficacies_per_neuron,
        )
        if tuple(values.shape) != expected:
            raise ValueError(
                f"synaptic_log_efficacy must have shape {expected}"
            )
        return values.exp()

    def _runtime_coefficients(self) -> torch.Tensor:
        morphology = self.morphology_adapter.index_select(
            0,
            self.morphology_indices,
        )
        return self.behavior_adapter + morphology

    def _branch_inputs(
        self,
        contact_inputs: torch.Tensor,
        *,
        synaptic_log_efficacy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (
            self.population_size,
            contact_inputs.shape[1],
            self.synaptic_efficacies_per_neuron,
        )
        if tuple(contact_inputs.shape) != expected:
            raise ValueError(f"contact_inputs must have shape {expected}")
        weighted = contact_inputs * self.synaptic_efficacies(
            synaptic_log_efficacy
        ).unsqueeze(1)
        branch_inputs = contact_inputs.new_zeros(
            self.population_size,
            contact_inputs.shape[1],
            self.neuron.config.input_dim,
        )
        indices = self.contact_branch_indices.unsqueeze(1).expand(
            -1,
            contact_inputs.shape[1],
            -1,
        )
        return branch_inputs.scatter_add(2, indices, weighted)

    def _contact_route_features(self) -> torch.Tensor:
        return self.neuron.route_features[
            self.morphology_indices.unsqueeze(1),
            self.contact_branch_indices,
        ]

    def _contact_summaries(
        self,
        contact_inputs: torch.Tensor,
        *,
        synaptic_log_efficacy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (
            self.population_size,
            contact_inputs.shape[1],
            self.synaptic_efficacies_per_neuron,
        )
        if tuple(contact_inputs.shape) != expected:
            raise ValueError(f"contact_inputs must have shape {expected}")
        weighted = contact_inputs * self.synaptic_efficacies(
            synaptic_log_efficacy
        ).unsqueeze(1)
        return torch.bmm(
            weighted,
            self._contact_route_features(),
        )

    def forward(
        self,
        contact_inputs: torch.Tensor,
        *,
        synaptic_log_efficacy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        summaries = self._contact_summaries(
            contact_inputs,
            synaptic_log_efficacy=synaptic_log_efficacy,
        )
        return self.neuron.forward_runtime_coefficients(
            summaries,
            runtime_coefficients=self._runtime_coefficients(),
        )

    def forward_batch(
        self,
        contact_inputs: torch.Tensor,
        *,
        synaptic_log_efficacy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run independent examples through one persistent population."""

        if contact_inputs.ndim != 4:
            raise ValueError(
                "contact_inputs must have shape "
                "(batch, population, time, contacts)"
            )
        batch, population, time, contacts = contact_inputs.shape
        expected = (
            batch,
            self.population_size,
            time,
            self.synaptic_efficacies_per_neuron,
        )
        if tuple(contact_inputs.shape) != expected:
            raise ValueError(f"contact_inputs must have shape {expected}")

        efficacy = self.synaptic_efficacies(
            synaptic_log_efficacy
        ).view(1, population, 1, contacts)
        weighted = contact_inputs * efficacy
        route_features = self._contact_route_features()
        summaries = torch.bmm(
            weighted.permute(1, 0, 2, 3).reshape(
                population,
                batch * time,
                contacts,
            ),
            route_features,
        ).reshape(
            population,
            batch,
            time,
            self.neuron.config.route_feature_dim,
        ).permute(1, 0, 2, 3).reshape(
            batch * population,
            time,
            self.neuron.config.route_feature_dim,
        )
        runtime_coefficients = self._runtime_coefficients().view(
            1,
            population,
            -1,
        ).expand(batch, -1, -1).reshape(batch * population, -1)
        output = self.neuron.forward_runtime_coefficients(
            summaries,
            runtime_coefficients=runtime_coefficients,
        )
        return output.reshape(batch, population, time, -1)

    def forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run pre-encoded P4 tokens with full temporal gradients."""

        expected = (
            self.population_size,
            tokens.shape[1],
            self.neuron.config.token_dim,
        )
        if tuple(tokens.shape) != expected:
            raise ValueError(f"tokens must have shape {expected}")
        state = self.neuron.initial_state(
            self.population_size,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        runtime_coefficients = self._runtime_coefficients()
        outputs = []
        for block in range(tokens.shape[1]):
            output, state = self.neuron.step_token(
                tokens[:, block],
                state,
                adaptation_cache=runtime_coefficients,
            )
            outputs.append(output)
        return torch.cat(outputs, dim=1)

    def forward_token_batch(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run batched pre-encoded P4 tokens with full temporal gradients."""

        if tokens.ndim != 4:
            raise ValueError(
                "tokens must have shape "
                "(batch, population, blocks, token_dim)"
            )
        batch, population, blocks, token_dim = tokens.shape
        expected = (
            batch,
            self.population_size,
            blocks,
            self.neuron.config.token_dim,
        )
        if tuple(tokens.shape) != expected:
            raise ValueError(f"tokens must have shape {expected}")
        flat_tokens = tokens.reshape(
            batch * population,
            blocks,
            token_dim,
        )
        state = self.neuron.initial_state(
            batch * population,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        runtime_coefficients = self._runtime_coefficients().view(
            1,
            population,
            -1,
        ).expand(batch, -1, -1).reshape(batch * population, -1)
        outputs = []
        for block in range(blocks):
            output, state = self.neuron.step_token(
                flat_tokens[:, block],
                state,
                adaptation_cache=runtime_coefficients,
            )
            outputs.append(output)
        flat_output = torch.cat(outputs, dim=1)
        return flat_output.reshape(
            batch,
            population,
            flat_output.shape[1],
            flat_output.shape[2],
        )
