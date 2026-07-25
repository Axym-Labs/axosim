from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from axosim.temporal_core import ResidualGRUTemporalCore

BlockForecastCore = Literal["gru", "mamba"]


@dataclass(frozen=True)
class BlockForecastConfig:
    """Configuration for causal next-block trajectory prediction."""

    core_kind: BlockForecastCore
    patch_size: int
    max_patch_rank: int = 8
    max_trajectory_rank: int = 8
    keep_local_tcn: bool = False
    gru_hidden_units: int | None = None
    residual_scale_init: float = 0.1
    voltage_anchor_count: int | None = None
    event_template: tuple[float, ...] | None = None
    event_template_center_init: float = 0.9208316802978516
    event_template_temperature: float = 0.25
    event_template_hard: bool = False

    def __post_init__(self) -> None:
        if self.core_kind not in {"gru", "mamba"}:
            raise ValueError(f"unknown block forecast core: {self.core_kind}")
        if self.patch_size <= 1:
            raise ValueError("block forecast patch_size must exceed one")
        if self.max_patch_rank <= 0 or self.max_trajectory_rank <= 0:
            raise ValueError("block forecast ranks must be positive")
        if self.gru_hidden_units is not None and self.gru_hidden_units <= 0:
            raise ValueError("gru_hidden_units must be positive")
        sparse_voltage = self.voltage_anchor_count is not None
        if sparse_voltage != (self.event_template is not None):
            raise ValueError(
                "voltage anchors and event template must be enabled together"
            )
        if self.voltage_anchor_count is not None:
            if not 2 <= self.voltage_anchor_count < self.patch_size:
                raise ValueError("voltage_anchor_count must be in [2, patch_size)")
            if len(self.event_template or ()) < 2:
                raise ValueError("event_template must include bias and weights")
        if self.event_template_temperature <= 0:
            raise ValueError("event_template_temperature must be positive")


class CausalBlockForecastModel(nn.Module):
    """Predict each native-resolution output block from prior input blocks."""

    def __init__(
        self,
        source: nn.Module,
        forecast_config: BlockForecastConfig,
    ) -> None:
        super().__init__()
        self.forecast_config = forecast_config
        self.backbone = deepcopy(source)
        width = int(self.backbone.config.model_dim)
        self.patch_rank = min(
            forecast_config.patch_size,
            forecast_config.max_patch_rank,
        )
        self.trajectory_rank = min(
            forecast_config.patch_size,
            forecast_config.max_trajectory_rank,
        )
        self.patch_basis = nn.Parameter(
            _temporal_basis(
                forecast_config.patch_size,
                self.patch_rank,
            )
        )
        self.patch_input = nn.Linear(
            self.patch_rank * width,
            width,
            bias=False,
        )
        self._initialize_patch_input(width)

        if forecast_config.core_kind == "gru":
            self.backbone.blocks = nn.ModuleList()
            self.temporal_core: nn.Module | None = ResidualGRUTemporalCore(
                width,
                hidden_units=forecast_config.gru_hidden_units,
                residual_scale_init=(forecast_config.residual_scale_init),
            )
        else:
            self.temporal_core = None
        if not forecast_config.keep_local_tcn:
            self.backbone.local_tcn_blocks = nn.ModuleList()
            self.backbone.local_tcn_out = None

        self.trajectory_basis = nn.Parameter(
            _temporal_basis(
                forecast_config.patch_size,
                self.trajectory_rank,
            )
        )
        trajectory_outputs = (
            self.trajectory_rank
            if self.sparse_voltage_output
            else self.num_output * self.trajectory_rank
        )
        self.trajectory_head = nn.Linear(width, trajectory_outputs)
        nn.init.zeros_(self.trajectory_head.weight)
        nn.init.zeros_(self.trajectory_head.bias)
        if self.sparse_voltage_output:
            assert forecast_config.voltage_anchor_count is not None
            assert forecast_config.event_template is not None
            self.voltage_anchor_head: nn.Module | None = nn.Linear(
                width,
                forecast_config.voltage_anchor_count,
            )
            nn.init.zeros_(self.voltage_anchor_head.weight)
            nn.init.zeros_(self.voltage_anchor_head.bias)
            template = torch.tensor(
                forecast_config.event_template,
                dtype=torch.float32,
            )
            self.register_buffer(
                "event_template_bias",
                template[0].clone(),
                persistent=False,
            )
            self.register_buffer(
                "event_template_kernel",
                template[1:].clone(),
                persistent=False,
            )
            self.event_template_center: nn.Parameter | None = nn.Parameter(
                torch.tensor(
                    forecast_config.event_template_center_init,
                    dtype=torch.float32,
                )
            )
            self.register_buffer(
                "voltage_anchor_offsets",
                _anchor_offsets(
                    self.patch_size,
                    forecast_config.voltage_anchor_count,
                ),
                persistent=False,
            )
            self.register_buffer(
                "voltage_interpolation",
                _linear_interpolation_matrix(
                    self.patch_size,
                    self.voltage_anchor_offsets,
                ),
                persistent=False,
            )
        else:
            self.voltage_anchor_head = None
            self.register_buffer("event_template_bias", None, persistent=False)
            self.register_buffer("event_template_kernel", None, persistent=False)
            self.event_template_center = None
            self.register_buffer(
                "voltage_anchor_offsets",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "voltage_interpolation",
                torch.empty(0),
                persistent=False,
            )
        self.base_soma_prediction: torch.Tensor | None = None

    @property
    def config(self):
        return self.backbone.config

    @property
    def num_input(self) -> int:
        return int(self.backbone.num_input)

    @property
    def num_output(self) -> int:
        return int(self.backbone.num_output)

    @property
    def num_branch(self) -> int:
        return int(self.backbone.num_branch)

    @property
    def patch_size(self) -> int:
        return self.forecast_config.patch_size

    @property
    def sparse_voltage_output(self) -> bool:
        return self.forecast_config.voltage_anchor_count is not None

    def _initialize_patch_input(self, width: int) -> None:
        with torch.no_grad():
            self.patch_input.weight.zero_()
            if self.patch_rank == self.patch_size:
                start = (self.patch_rank - 1) * width
                self.patch_input.weight[:, start : start + width].copy_(
                    torch.eye(
                        width,
                        dtype=self.patch_input.weight.dtype,
                    )
                )
            else:
                identity = torch.eye(
                    width,
                    dtype=self.patch_input.weight.dtype,
                )
                for rank_index in range(self.patch_rank):
                    start = rank_index * width
                    self.patch_input.weight[:, start : start + width].copy_(
                        identity * self.patch_basis[-1, rank_index]
                    )

    def _pack_hidden(
        self,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, width = hidden.shape
        if steps % self.patch_size:
            raise ValueError("packed hidden length must divide patch_size")
        patches = hidden.reshape(
            batch,
            steps // self.patch_size,
            self.patch_size,
            width,
        )
        reduced = torch.einsum(
            "bnpw,pr->bnrw",
            patches,
            self.patch_basis,
        )
        return self.patch_input(reduced.flatten(start_dim=-2))

    def _run_block_core(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.backbone._run_temporal_sequence_with_core(
            tokens,
            tokens,
            temporal_core=self.temporal_core,
        )

    def _decode_trajectory(
        self,
        block_state: torch.Tensor,
    ) -> torch.Tensor:
        anchor = self.backbone._decode_sequence(
            block_state,
            None,
        )
        features = self.backbone.norm(block_state)
        coefficients = self.trajectory_head(features).view(
            block_state.shape[0],
            block_state.shape[1],
            self.num_output,
            self.trajectory_rank,
        )
        trajectory = torch.einsum(
            "bnor,pr->bnpo",
            coefficients,
            self.trajectory_basis,
        )
        return anchor.unsqueeze(2) + trajectory

    def _decode_sparse_trajectory(
        self,
        block_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.voltage_anchor_head is None:
            raise RuntimeError("sparse voltage head is not configured")
        anchor = self.backbone._decode_sequence(block_state, None)
        features = self.backbone.norm(block_state)
        spike_coefficients = self.trajectory_head(features)
        spike_trajectory = torch.einsum(
            "bnr,pr->bnp",
            spike_coefficients,
            self.trajectory_basis,
        )
        spike = anchor[..., 0].unsqueeze(2) + spike_trajectory
        voltage_anchors = anchor[..., 1].unsqueeze(2) + self.voltage_anchor_head(
            features
        )
        return spike, voltage_anchors

    def _event_template_voltage(
        self,
        spike_logits: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.event_template_kernel is None
            or self.event_template_bias is None
            or self.event_template_center is None
        ):
            raise RuntimeError("event template is not configured")
        temperature = self.forecast_config.event_template_temperature
        events = torch.sigmoid(
            (spike_logits - self.event_template_center) / temperature
        )
        if self.forecast_config.event_template_hard:
            hard_events = (spike_logits >= self.event_template_center).to(
                events.dtype
            )
            events = (
                hard_events + events - events.detach()
                if self.training
                else hard_events
            )
        history = self.event_template_kernel.numel() - 1
        padded = F.pad(events.unsqueeze(1), (history, 0))
        voltage = F.conv1d(
            padded,
            self.event_template_kernel.view(1, 1, -1),
        ).squeeze(1)
        return voltage + self.event_template_bias

    def _materialize_sparse_voltage(
        self,
        spike_logits: torch.Tensor,
        voltage_anchors: torch.Tensor,
    ) -> torch.Tensor:
        template = self._event_template_voltage(spike_logits)
        forecast_template = template[:, self.patch_size :]
        template_blocks = forecast_template.reshape(
            voltage_anchors.shape[0],
            voltage_anchors.shape[1],
            self.patch_size,
        )
        template_anchors = template_blocks.index_select(
            2,
            self.voltage_anchor_offsets,
        )
        residual_anchors = voltage_anchors - template_anchors
        residual = torch.einsum(
            "bnk,pk->bnp",
            residual_anchors,
            self.voltage_interpolation,
        )
        return template_blocks + residual

    def forward(
        self,
        x: torch.Tensor,
        *,
        morphology_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.num_input or x.shape[1] <= 0:
            raise ValueError(
                f"block forecast input must have shape (batch, time, {self.num_input})"
            )
        batch, steps, _ = x.shape
        full_steps = steps - steps % self.patch_size
        if full_steps < self.patch_size:
            output = x.new_zeros(batch, steps, self.num_output)
            self.base_soma_prediction = output[..., 1:2]
            return output

        active_input = x[:, :full_steps]
        if not active_input.is_contiguous():
            active_input = active_input.contiguous()
        hidden, _, _ = self.backbone._encode_sequence(
            active_input,
            morphology_indices=morphology_indices,
        )
        tokens = self._pack_hidden(hidden)
        block_state = self._run_block_core(tokens)
        available_steps = max(steps - self.patch_size, 0)
        if self.sparse_voltage_output:
            spike_blocks, voltage_anchors = self._decode_sparse_trajectory(block_state)
            spike_forecast = spike_blocks.flatten(start_dim=1)
            cold_spike = spike_forecast.new_zeros(
                batch,
                self.patch_size,
            )
            template_cold_spike = spike_forecast.new_full(
                (batch, self.patch_size),
                -20.0,
            )
            extended_spike = torch.cat(
                (template_cold_spike, spike_forecast),
                dim=1,
            )
            voltage_blocks = self._materialize_sparse_voltage(
                extended_spike,
                voltage_anchors,
            )
            voltage_forecast = voltage_blocks.flatten(start_dim=1)
            cold_voltage = voltage_forecast.new_zeros(
                batch,
                self.patch_size,
            )
            output = torch.stack(
                (
                    torch.cat(
                        (
                            cold_spike,
                            spike_forecast[:, :available_steps],
                        ),
                        dim=1,
                    ),
                    torch.cat(
                        (
                            cold_voltage,
                            voltage_forecast[:, :available_steps],
                        ),
                        dim=1,
                    ),
                ),
                dim=-1,
            )
        else:
            forecast = self._decode_trajectory(block_state).flatten(
                start_dim=1,
                end_dim=2,
            )
            cold_start = forecast.new_zeros(
                batch,
                min(self.patch_size, steps),
                self.num_output,
            )
            output = torch.cat(
                (
                    cold_start,
                    forecast[:, :available_steps],
                ),
                dim=1,
            )
        self.base_soma_prediction = output[..., 1:2]
        return output

    def component_manifest(self) -> dict[str, object]:
        return {
            "temporal_core": self.forecast_config.core_kind,
            "causal_block_forecast": True,
            "patch_size": self.patch_size,
            "patch_rank": self.patch_rank,
            "trajectory_rank": self.trajectory_rank,
            "voltage_anchor_count": (self.forecast_config.voltage_anchor_count),
            "event_template_history_steps": (
                0
                if self.event_template_kernel is None
                else self.event_template_kernel.numel() - 1
            ),
            "event_template_hard": self.forecast_config.event_template_hard,
            "native_input_values_preserved_in_order": True,
            "native_output_values_predicted_distinctly": True,
            "forecast_alignment": "completed_input_block_to_next_output_block",
            "shared_axomamba_front_end": True,
            "shared_axomamba_heads": True,
        }


def _temporal_basis(steps: int, rank: int) -> torch.Tensor:
    if rank == steps:
        return torch.eye(steps)
    positions = torch.arange(steps, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.arange(rank, dtype=torch.float32).unsqueeze(0)
    basis = torch.cos(math.pi * (positions + 0.5) * frequencies / float(steps))
    basis[:, 0] *= math.sqrt(1.0 / steps)
    if rank > 1:
        basis[:, 1:] *= math.sqrt(2.0 / steps)
    return basis


def _anchor_offsets(steps: int, count: int) -> torch.Tensor:
    offsets = torch.round(torch.linspace(0, steps - 1, count)).long()
    if torch.unique(offsets).numel() != count:
        raise ValueError("voltage anchor offsets are not unique")
    return offsets


def _linear_interpolation_matrix(
    steps: int,
    offsets: torch.Tensor,
) -> torch.Tensor:
    positions = torch.arange(steps)
    left = torch.searchsorted(offsets, positions, right=True) - 1
    left = left.clamp(0, offsets.numel() - 2)
    right = left + 1
    fraction = (positions - offsets[left]).float() / (
        offsets[right] - offsets[left]
    ).float()
    matrix = torch.zeros(steps, offsets.numel())
    matrix[positions, left] = 1.0 - fraction
    matrix[positions, right] = fraction
    return matrix
