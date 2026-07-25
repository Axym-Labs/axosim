from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from axosim.temporal_core import (
    AxoTemporalModel,
    CausalPatchedTemporalCore,
    ResidualGRUTemporalCore,
)


class _GRUPopulationStep(nn.Module):
    """Stateless sparse GRU step suitable for whole-step compilation."""

    def __init__(self, model: AxoTemporalModel) -> None:
        super().__init__()
        model._validate_streaming_configuration()
        core = model._unwrapped_temporal_core()
        if not isinstance(core, ResidualGRUTemporalCore):
            raise TypeError("population runner requires a GRU temporal core")
        if model.backbone.local_tcn_out is not None:
            raise ValueError("population runner does not support local TCNs")
        if model.backbone.branch_subunit_projection is not None:
            raise ValueError(
                "population runner does not support branch subunits"
            )
        self.backbone = model.backbone
        self.pre_norm = core.pre_norm
        self.output_projection = core.output_projection
        self.residual_scale = core.residual_scale
        self.weight_ih = core.gru.weight_ih_l0
        self.weight_hh = core.gru.weight_hh_l0
        self.bias_ih = core.gru.bias_ih_l0
        self.bias_hh = core.gru.bias_hh_l0
        parameter = next(model.parameters())
        self.register_buffer(
            "feature_gains",
            self.backbone._all_morphology_feature_gains(
                dtype=parameter.dtype,
                device=parameter.device,
            ),
        )

    def _encode(
        self,
        event_indices: torch.Tensor,
        event_values: torch.Tensor,
        morphology: torch.Tensor,
    ) -> torch.Tensor:
        branch = self.backbone._sparse_streaming_branch_inputs(
            event_indices,
            event_values,
            morphology_indices=morphology,
            morphology_feature_gains=self.feature_gains,
        )
        hidden = self.backbone.branch_projection(branch)
        hidden = self.backbone._apply_projected_branch_bilinear(
            hidden,
            self.backbone._branch_bilinear_projection(branch),
        )
        if self.backbone.morphology_embedding is not None:
            embedding = self.backbone._morphology_embedding_values(
                morphology,
                dtype=hidden.dtype,
                device=hidden.device,
            )
            assert (
                self.backbone.config.morphology_embedding_scale is not None
            )
            hidden = hidden + (
                self.backbone.config.morphology_embedding_scale
                * embedding.unsqueeze(1)
            )
        return hidden.squeeze(1)

    def _advance_gru(
        self,
        hidden: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cell_input = self.pre_norm(hidden)
        input_gates = F.linear(
            cell_input,
            self.weight_ih,
            self.bias_ih,
        )
        hidden_gates = F.linear(
            recurrent_state,
            self.weight_hh,
            self.bias_hh,
        )
        ir, iz, inn = input_gates.chunk(3, dim=-1)
        hr, hz, hn = hidden_gates.chunk(3, dim=-1)
        reset = torch.sigmoid(ir + hr)
        update = torch.sigmoid(iz + hz)
        candidate = torch.tanh(inn + reset * hn)
        next_state = (
            (1.0 - update) * candidate
            + update * recurrent_state
        )
        correction = (
            self.residual_scale.view(1, -1)
            * self.output_projection(next_state)
        )
        return correction, next_state

    def _decode(self, hidden: torch.Tensor) -> torch.Tensor:
        features = self.backbone.norm(hidden).unsqueeze(1)
        if self.backbone.readout is not None:
            output = self.backbone.readout(features)
            spike = output[..., 0:1]
            soma = output[..., 1:2]
        else:
            assert self.backbone.spike_readout is not None
            assert self.backbone.soma_readout is not None
            spike = self.backbone.spike_readout(features)
            soma = self.backbone._readout_soma(features, None)
        if self.backbone.config.spike_voltage_coupling_scale:
            spike = self.backbone._couple_spike_voltage(spike, soma)
        return torch.cat((spike, soma), dim=-1)


class _NativeGRUPopulationStep(_GRUPopulationStep):
    def forward(
        self,
        event_indices: torch.Tensor,
        event_values: torch.Tensor,
        morphology: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode(event_indices, event_values, morphology)
        correction, next_state = self._advance_gru(
            hidden,
            recurrent_state,
        )
        return self._decode(hidden + correction), next_state


class _PatchedGRUPopulationStep(_GRUPopulationStep):
    def __init__(
        self,
        model: AxoTemporalModel,
        *,
        close_patch: bool,
    ) -> None:
        super().__init__(model)
        self.close_patch = close_patch
        if not isinstance(
            model.temporal_core,
            CausalPatchedTemporalCore,
        ):
            raise TypeError("patched population step requires patching")
        self.patch_size = model.temporal_core.patch_size

    def forward(
        self,
        event_indices: torch.Tensor,
        event_values: torch.Tensor,
        morphology: torch.Tensor,
        recurrent_state: torch.Tensor,
        patch_sum: torch.Tensor,
        patch_correction: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = self._encode(event_indices, event_values, morphology)
        corrected = hidden + patch_correction
        accumulated = patch_sum + hidden
        if self.close_patch:
            next_correction, next_state = self._advance_gru(
                accumulated / self.patch_size,
                recurrent_state,
            )
            next_sum = torch.zeros_like(accumulated)
        else:
            next_state = recurrent_state
            next_sum = accumulated
            next_correction = patch_correction
        return (
            self._decode(corrected),
            next_state,
            next_sum,
            next_correction,
        )


class _GroupedGRUPopulationStep(_GRUPopulationStep):
    GROUP_PARAMETER_NAMES = (
        "group_weight_ih",
        "group_weight_hh",
        "group_bias_ih",
        "group_bias_hh",
        "group_residual_scale",
    )

    def __init__(self, model: AxoTemporalModel, *, groups: int) -> None:
        super().__init__(model)
        if not isinstance(self.output_projection, nn.Identity):
            raise ValueError(
                "grouped runner requires GRU hidden width to match model width"
            )
        self.groups = groups
        self.width = self.weight_hh.shape[1]
        self.group_weight_ih = nn.Parameter(
            self.weight_ih.detach().unsqueeze(0).repeat(groups, 1, 1),
            requires_grad=False,
        )
        self.group_weight_hh = nn.Parameter(
            self.weight_hh.detach().unsqueeze(0).repeat(groups, 1, 1),
            requires_grad=False,
        )
        self.group_bias_ih = nn.Parameter(
            self.bias_ih.detach().unsqueeze(0).repeat(groups, 1),
            requires_grad=False,
        )
        self.group_bias_hh = nn.Parameter(
            self.bias_hh.detach().unsqueeze(0).repeat(groups, 1),
            requires_grad=False,
        )
        self.group_residual_scale = nn.Parameter(
            self.residual_scale.detach().unsqueeze(0).repeat(groups, 1),
            requires_grad=False,
        )
        del self.weight_ih
        del self.weight_hh
        del self.bias_ih
        del self.bias_hh
        del self.residual_scale

    def _encode_grouped(
        self,
        event_indices: torch.Tensor,
        event_values: torch.Tensor,
        morphology: torch.Tensor,
    ) -> torch.Tensor:
        groups, neurons, events = event_indices.shape
        flat = self._encode(
            event_indices.reshape(-1, events),
            event_values.reshape(-1, events),
            morphology.reshape(-1),
        )
        return flat.view(groups, neurons, self.width)

    def _advance_grouped(
        self,
        hidden: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cell_input = self.pre_norm(hidden)
        input_gates = torch.bmm(
            cell_input,
            self.group_weight_ih.transpose(1, 2),
        ) + self.group_bias_ih.unsqueeze(1)
        hidden_gates = torch.bmm(
            recurrent_state,
            self.group_weight_hh.transpose(1, 2),
        ) + self.group_bias_hh.unsqueeze(1)
        ir, iz, inn = input_gates.chunk(3, dim=-1)
        hr, hz, hn = hidden_gates.chunk(3, dim=-1)
        reset = torch.sigmoid(ir + hr)
        update = torch.sigmoid(iz + hz)
        candidate = torch.tanh(inn + reset * hn)
        next_state = (
            (1.0 - update) * candidate
            + update * recurrent_state
        )
        correction = (
            self.group_residual_scale.unsqueeze(1) * next_state
        )
        return correction, next_state

    def _decode_grouped(self, hidden: torch.Tensor) -> torch.Tensor:
        groups, neurons, width = hidden.shape
        output = self._decode(hidden.reshape(-1, width))
        return output.view(groups, neurons, 1, -1)


class _NativeGroupedGRUPopulationStep(_GroupedGRUPopulationStep):
    def forward(
        self,
        event_indices: torch.Tensor,
        event_values: torch.Tensor,
        morphology: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode_grouped(
            event_indices,
            event_values,
            morphology,
        )
        correction, next_state = self._advance_grouped(
            hidden,
            recurrent_state,
        )
        return self._decode_grouped(hidden + correction), next_state


class _PatchedGroupedGRUPopulationStep(_GroupedGRUPopulationStep):
    def __init__(
        self,
        model: AxoTemporalModel,
        *,
        groups: int,
        close_patch: bool,
    ) -> None:
        super().__init__(model, groups=groups)
        if not isinstance(
            model.temporal_core,
            CausalPatchedTemporalCore,
        ):
            raise TypeError("patched population step requires patching")
        self.close_patch = close_patch
        self.patch_size = model.temporal_core.patch_size

    def forward(
        self,
        event_indices: torch.Tensor,
        event_values: torch.Tensor,
        morphology: torch.Tensor,
        recurrent_state: torch.Tensor,
        patch_sum: torch.Tensor,
        patch_correction: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = self._encode_grouped(
            event_indices,
            event_values,
            morphology,
        )
        corrected = hidden + patch_correction
        accumulated = patch_sum + hidden
        if self.close_patch:
            next_correction, next_state = self._advance_grouped(
                accumulated / self.patch_size,
                recurrent_state,
            )
            next_sum = torch.zeros_like(accumulated)
        else:
            next_state = recurrent_state
            next_sum = accumulated
            next_correction = patch_correction
        return (
            self._decode_grouped(corrected),
            next_state,
            next_sum,
            next_correction,
        )


class GRUPopulationRunner:
    """Compiled persistent-state inference for a shared-weight GRU population."""

    def __init__(
        self,
        model: AxoTemporalModel,
        *,
        population: int,
        chunk_size: int,
        compile_step: bool = True,
        compile_mode: str = "reduce-overhead",
    ) -> None:
        if model.training:
            raise ValueError("population runner requires eval mode")
        if population <= 0 or chunk_size <= 0:
            raise ValueError("population and chunk_size must be positive")
        if population % chunk_size:
            raise ValueError("population must be divisible by chunk_size")
        if model.temporal_config.kind != "gru":
            raise TypeError("population runner requires a GRU model")
        parameter = next(model.parameters())
        if compile_step and parameter.device.type != "cuda":
            raise ValueError("compiled population runner requires CUDA")

        self.model = model
        self.population = population
        self.chunk_size = chunk_size
        self.device = parameter.device
        self.dtype = parameter.dtype
        self.state = model.allocate_streaming_state(
            population,
            device=self.device,
            dtype=self.dtype,
        )
        self.output = torch.empty(
            population,
            1,
            model.num_output,
            device=self.device,
            dtype=self.dtype,
        )
        self.patch_size = model.temporal_config.patch_size
        self._phase = 0
        self._compiled = compile_step
        compile_module: Callable[[nn.Module], nn.Module]
        if compile_step:
            compile_module = lambda module: torch.compile(
                module,
                fullgraph=True,
                dynamic=False,
                mode=compile_mode,
            )
        else:
            compile_module = lambda module: module

        if self.patch_size == 1:
            self._native_step = compile_module(
                _NativeGRUPopulationStep(model).eval()
            )
            self._open_step = None
            self._close_step = None
        else:
            self._native_step = None
            self._open_step = compile_module(
                _PatchedGRUPopulationStep(
                    model,
                    close_patch=False,
                ).eval()
            )
            self._close_step = compile_module(
                _PatchedGRUPopulationStep(
                    model,
                    close_patch=True,
                ).eval()
            )

    @property
    def recurrent_state(self) -> torch.Tensor:
        if not isinstance(self.state.core_state, torch.Tensor):
            raise TypeError("GRU state must be a tensor")
        return self.state.core_state

    @property
    def patch_sum(self) -> torch.Tensor | None:
        return self.state.patch_sum

    @property
    def patch_correction(self) -> torch.Tensor | None:
        return self.state.patch_correction

    @property
    def patch_position(self) -> int:
        return self._phase

    def step(
        self,
        event_indices: torch.Tensor,
        event_values: torch.Tensor,
        morphology_indices: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(
            event_indices,
            event_values,
            morphology_indices,
        )
        close_patch = (
            self.patch_size > 1
            and self._phase == self.patch_size - 1
        )
        with torch.inference_mode():
            for start in range(0, self.population, self.chunk_size):
                stop = start + self.chunk_size
                if self._compiled:
                    torch.compiler.cudagraph_mark_step_begin()
                if self.patch_size == 1:
                    assert self._native_step is not None
                    output, next_state = self._native_step(
                        event_indices[start:stop],
                        event_values[start:stop],
                        morphology_indices[start:stop],
                        self.recurrent_state[start:stop],
                    )
                    self.output[start:stop].copy_(output)
                    self.recurrent_state[start:stop].copy_(next_state)
                else:
                    assert self.patch_sum is not None
                    assert self.patch_correction is not None
                    step = (
                        self._close_step
                        if close_patch
                        else self._open_step
                    )
                    assert step is not None
                    values = step(
                        event_indices[start:stop],
                        event_values[start:stop],
                        morphology_indices[start:stop],
                        self.recurrent_state[start:stop],
                        self.patch_sum[start:stop],
                        self.patch_correction[start:stop],
                    )
                    self.output[start:stop].copy_(values[0])
                    self.recurrent_state[start:stop].copy_(values[1])
                    self.patch_sum[start:stop].copy_(values[2])
                    self.patch_correction[start:stop].copy_(values[3])
        self._phase = (self._phase + 1) % self.patch_size
        self.state.patch_position = self._phase
        return self.output

    def _validate_inputs(
        self,
        event_indices: torch.Tensor,
        event_values: torch.Tensor,
        morphology_indices: torch.Tensor,
    ) -> None:
        if (
            event_indices.ndim != 2
            or event_values.shape != event_indices.shape
            or event_indices.shape[0] != self.population
        ):
            raise ValueError(
                "events must have matching (population, events) shapes"
            )
        if morphology_indices.shape != (self.population,):
            raise ValueError(
                "morphology_indices must have shape (population,)"
            )
        for tensor in (
            event_indices,
            event_values,
            morphology_indices,
        ):
            if tensor.device != self.device:
                raise ValueError("population inputs must be on model device")
        if event_indices.is_floating_point() or event_indices.is_complex():
            raise ValueError("event indices must be integers")
        if event_values.dtype != self.dtype:
            raise ValueError("event values must match model dtype")


class GroupedGRUPopulationRunner:
    """Compiled GRU inference with one recurrent parameter set per group."""

    def __init__(
        self,
        model: AxoTemporalModel,
        *,
        population: int,
        groups: int,
        chunk_size: int,
        compile_step: bool = True,
        compile_mode: str = "reduce-overhead",
    ) -> None:
        if model.training:
            raise ValueError("population runner requires eval mode")
        if population <= 0 or groups <= 0 or chunk_size <= 0:
            raise ValueError(
                "population, groups, and chunk_size must be positive"
            )
        if population % groups:
            raise ValueError("population must be divisible by groups")
        if chunk_size % groups:
            raise ValueError("chunk_size must be divisible by groups")
        self.neurons_per_group = population // groups
        self.neurons_per_chunk = chunk_size // groups
        if self.neurons_per_group % self.neurons_per_chunk:
            raise ValueError(
                "neurons per group must be divisible by grouped chunk size"
            )
        if model.temporal_config.kind != "gru":
            raise TypeError("population runner requires a GRU model")
        parameter = next(model.parameters())
        if compile_step and parameter.device.type != "cuda":
            raise ValueError("compiled population runner requires CUDA")

        self.model = model
        self.population = population
        self.groups = groups
        self.chunk_size = chunk_size
        self.device = parameter.device
        self.dtype = parameter.dtype
        self.patch_size = model.temporal_config.patch_size
        self._phase = 0
        self._compiled = compile_step
        width = model._unwrapped_temporal_core().hidden_units
        self.recurrent_state = torch.zeros(
            groups,
            self.neurons_per_group,
            width,
            device=self.device,
            dtype=self.dtype,
        )
        self.patch_sum = None
        self.patch_correction = None
        if self.patch_size > 1:
            self.patch_sum = torch.zeros_like(self.recurrent_state)
            self.patch_correction = torch.zeros_like(
                self.recurrent_state
            )
        self.output = torch.empty(
            groups,
            self.neurons_per_group,
            1,
            model.num_output,
            device=self.device,
            dtype=self.dtype,
        )
        self.group_specific_parameter_count = (
            self._group_parameter_count(model)
        )
        total_parameters = sum(
            value.numel() for value in model.parameters()
        )
        self.shared_parameter_count = (
            total_parameters - self.group_specific_parameter_count
        )

        compile_module: Callable[[nn.Module], nn.Module]
        if compile_step:
            compile_module = lambda module: torch.compile(
                module,
                fullgraph=True,
                dynamic=False,
                mode=compile_mode,
            )
        else:
            compile_module = lambda module: module
        if self.patch_size == 1:
            self._native_step = compile_module(
                _NativeGroupedGRUPopulationStep(
                    model,
                    groups=groups,
                ).eval()
            )
            self._open_step = None
            self._close_step = None
        else:
            self._native_step = None
            open_step = _PatchedGroupedGRUPopulationStep(
                model,
                groups=groups,
                close_patch=False,
            ).eval()
            close_step = _PatchedGroupedGRUPopulationStep(
                model,
                groups=groups,
                close_patch=True,
            ).eval()
            for name in _GroupedGRUPopulationStep.GROUP_PARAMETER_NAMES:
                setattr(close_step, name, getattr(open_step, name))
            self._open_step = compile_module(open_step)
            self._close_step = compile_module(close_step)

    @property
    def resident_parameter_count(self) -> int:
        return (
            self.shared_parameter_count
            + self.groups * self.group_specific_parameter_count
        )

    @property
    def patch_position(self) -> int:
        return self._phase

    def step(
        self,
        event_indices: torch.Tensor,
        event_values: torch.Tensor,
        morphology_indices: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(
            event_indices,
            event_values,
            morphology_indices,
        )
        events = event_indices.shape[1]
        indices = event_indices.view(
            self.groups,
            self.neurons_per_group,
            events,
        )
        values = event_values.view_as(indices)
        morphology = morphology_indices.view(
            self.groups,
            self.neurons_per_group,
        )
        close_patch = (
            self.patch_size > 1
            and self._phase == self.patch_size - 1
        )
        with torch.inference_mode():
            for start in range(
                0,
                self.neurons_per_group,
                self.neurons_per_chunk,
            ):
                stop = start + self.neurons_per_chunk
                if self._compiled:
                    torch.compiler.cudagraph_mark_step_begin()
                if self.patch_size == 1:
                    assert self._native_step is not None
                    output, next_state = self._native_step(
                        indices[:, start:stop],
                        values[:, start:stop],
                        morphology[:, start:stop],
                        self.recurrent_state[:, start:stop],
                    )
                    self.output[:, start:stop].copy_(output)
                    self.recurrent_state[:, start:stop].copy_(
                        next_state
                    )
                else:
                    assert self.patch_sum is not None
                    assert self.patch_correction is not None
                    phase = (
                        self._close_step
                        if close_patch
                        else self._open_step
                    )
                    assert phase is not None
                    result = phase(
                        indices[:, start:stop],
                        values[:, start:stop],
                        morphology[:, start:stop],
                        self.recurrent_state[:, start:stop],
                        self.patch_sum[:, start:stop],
                        self.patch_correction[:, start:stop],
                    )
                    self.output[:, start:stop].copy_(result[0])
                    self.recurrent_state[:, start:stop].copy_(result[1])
                    self.patch_sum[:, start:stop].copy_(result[2])
                    self.patch_correction[:, start:stop].copy_(result[3])
        self._phase = (self._phase + 1) % self.patch_size
        return self.output.view(self.population, 1, -1)

    def load_group_recurrent_parameters(
        self,
        group: int,
        source: AxoTemporalModel,
    ) -> None:
        """Load one group's recurrent weights from a compatible GRU model."""

        if not 0 <= group < self.groups:
            raise IndexError("group index is out of range")
        core = source._unwrapped_temporal_core()
        modules = [
            module
            for module in (
                self._native_step,
                self._open_step,
                self._close_step,
            )
            if module is not None
        ]
        with torch.no_grad():
            for module in modules:
                target = getattr(module, "_orig_mod", module)
                target.group_weight_ih[group].copy_(
                    core.gru.weight_ih_l0
                )
                target.group_weight_hh[group].copy_(
                    core.gru.weight_hh_l0
                )
                target.group_bias_ih[group].copy_(
                    core.gru.bias_ih_l0
                )
                target.group_bias_hh[group].copy_(
                    core.gru.bias_hh_l0
                )
                target.group_residual_scale[group].copy_(
                    core.residual_scale
                )

    @staticmethod
    def _group_parameter_count(model: AxoTemporalModel) -> int:
        core = model._unwrapped_temporal_core()
        return (
            core.gru.weight_ih_l0.numel()
            + core.gru.weight_hh_l0.numel()
            + core.gru.bias_ih_l0.numel()
            + core.gru.bias_hh_l0.numel()
            + core.residual_scale.numel()
        )

    def _validate_inputs(
        self,
        event_indices: torch.Tensor,
        event_values: torch.Tensor,
        morphology_indices: torch.Tensor,
    ) -> None:
        if (
            event_indices.ndim != 2
            or event_values.shape != event_indices.shape
            or event_indices.shape[0] != self.population
        ):
            raise ValueError(
                "events must have matching (population, events) shapes"
            )
        if morphology_indices.shape != (self.population,):
            raise ValueError(
                "morphology_indices must have shape (population,)"
            )
        for tensor in (
            event_indices,
            event_values,
            morphology_indices,
        ):
            if tensor.device != self.device:
                raise ValueError("population inputs must be on model device")
        if event_indices.is_floating_point() or event_indices.is_complex():
            raise ValueError("event indices must be integers")
        if event_values.dtype != self.dtype:
            raise ValueError("event values must match model dtype")
