"""Model-generated activity selection for connected population simulation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class StratifiedPatchPhaseLayout:
    """Deterministic patch phases and a phase-major storage permutation."""

    phase_ids: torch.Tensor
    storage_permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    phase_population_sizes: tuple[int, ...]
    storage_group_offsets: tuple[int, ...]
    storage_group_shape: tuple[int, int]

    @property
    def patch_size(self) -> int:
        return len(self.phase_population_sizes)

    @property
    def population_size(self) -> int:
        return self.phase_ids.numel()

    @property
    def phase_sizes(self) -> tuple[int, ...]:
        return self.phase_population_sizes


@dataclass(frozen=True)
class FixedActivityThresholds:
    """Fixed spike-score thresholds stratified by role and morphology."""

    excitatory: tuple[float, ...]
    inhibitory: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.excitatory:
            raise ValueError("at least one morphology threshold is required")
        if len(self.excitatory) != len(self.inhibitory):
            raise ValueError(
                "excitatory and inhibitory thresholds must have equal length"
            )
        values = self.excitatory + self.inhibitory
        if not all(math.isfinite(value) for value in values):
            raise ValueError("activity thresholds must be finite")

    @property
    def morphology_count(self) -> int:
        return len(self.excitatory)

    def as_tensor(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.tensor(
            (self.excitatory, self.inhibitory),
            device=device,
            dtype=dtype,
        )


@dataclass(frozen=True)
class FixedBlockActivityThresholds:
    """Fixed role/morphology thresholds for each forecast position."""

    forecast_steps: tuple[FixedActivityThresholds, ...]

    def __post_init__(self) -> None:
        if not self.forecast_steps:
            raise ValueError("at least one forecast step is required")
        morphology_counts = {
            step.morphology_count for step in self.forecast_steps
        }
        if len(morphology_counts) != 1:
            raise ValueError(
                "every forecast step must use the same morphologies"
            )

    @property
    def forecast_step_count(self) -> int:
        return len(self.forecast_steps)

    @property
    def morphology_count(self) -> int:
        return self.forecast_steps[0].morphology_count

    def as_tensor(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.stack(
            [
                step.as_tensor(device=device, dtype=dtype)
                for step in self.forecast_steps
            ]
        )


def build_stratified_patch_phase_layout(
    morphology_ids: torch.Tensor,
    inhibitory: torch.Tensor,
    *,
    patch_size: int,
    seed: int,
    tile_ids: torch.Tensor | None = None,
) -> StratifiedPatchPhaseLayout:
    """Balance phases independently inside every morphology and E/I group."""

    if morphology_ids.ndim != 1:
        raise ValueError("morphology_ids must be one-dimensional")
    if inhibitory.shape != morphology_ids.shape:
        raise ValueError("inhibitory must match morphology_ids")
    if morphology_ids.dtype == torch.bool or morphology_ids.is_floating_point():
        raise TypeError("morphology_ids must use an integer dtype")
    if inhibitory.dtype != torch.bool:
        raise TypeError("inhibitory must use bool dtype")
    if morphology_ids.device != inhibitory.device:
        raise ValueError("morphology_ids and inhibitory must share a device")
    if tile_ids is not None:
        if tile_ids.shape != morphology_ids.shape:
            raise ValueError("tile_ids must match morphology_ids")
        if tile_ids.dtype == torch.bool or tile_ids.is_floating_point():
            raise TypeError("tile_ids must use an integer dtype")
        if tile_ids.device != morphology_ids.device:
            raise ValueError("tile_ids and morphology_ids must share a device")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if morphology_ids.numel() and int(morphology_ids.min()) < 0:
        raise ValueError("morphology_ids must be nonnegative")
    if tile_ids is not None and tile_ids.numel() and int(tile_ids.min()) < 0:
        raise ValueError("tile_ids must be nonnegative")

    device = morphology_ids.device
    phase_ids = torch.empty_like(morphology_ids, dtype=torch.long)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    global_phase_counts = [0] * patch_size
    if morphology_ids.numel():
        morphology_span = int(morphology_ids.max()) + 1
        stratum = morphology_ids.long() * 2 + inhibitory.long()
        if tile_ids is not None:
            stratum = (
                tile_ids.long() * morphology_span * 2 + stratum
            )
        stratum_order = torch.argsort(stratum, stable=True)
        _, stratum_counts = torch.unique_consecutive(
            stratum[stratum_order],
            return_counts=True,
        )
        start = 0
        for count_tensor in stratum_counts:
            count = int(count_tensor)
            group = stratum_order[start : start + count]
            start += count
            base_count, remainder = divmod(count, patch_size)
            extra_phases = sorted(
                range(patch_size),
                key=lambda phase: (global_phase_counts[phase], phase),
            )[:remainder]
            phase_counts = [base_count] * patch_size
            for phase in extra_phases:
                phase_counts[phase] += 1
            phases = torch.repeat_interleave(
                torch.arange(
                    patch_size,
                    device=device,
                    dtype=torch.long,
                ),
                torch.tensor(
                    phase_counts,
                    device=device,
                    dtype=torch.long,
                ),
            )
            shuffle = torch.randperm(
                count,
                device=device,
                generator=generator,
            )
            phase_ids[group] = phases[shuffle]
            for phase, phase_count in enumerate(phase_counts):
                global_phase_counts[phase] += phase_count

    if tile_ids is None:
        storage_group = phase_ids
        storage_group_shape = (1, patch_size)
    else:
        tile_count = int(tile_ids.max()) + 1 if tile_ids.numel() else 0
        storage_group = phase_ids * tile_count + tile_ids.long()
        storage_group_shape = (patch_size, tile_count)
    storage_permutation = torch.argsort(storage_group, stable=True)
    inverse_permutation = torch.empty_like(storage_permutation)
    inverse_permutation[storage_permutation] = torch.arange(
        morphology_ids.numel(),
        device=device,
    )
    phase_population_sizes = torch.bincount(
        phase_ids,
        minlength=patch_size,
    ).cpu()
    storage_group_counts = torch.bincount(
        storage_group,
        minlength=storage_group_shape[0] * patch_size,
    ).cpu()
    storage_group_offsets = [0]
    for count in storage_group_counts.tolist():
        storage_group_offsets.append(
            storage_group_offsets[-1] + int(count)
        )
    return StratifiedPatchPhaseLayout(
        phase_ids=phase_ids,
        storage_permutation=storage_permutation,
        inverse_permutation=inverse_permutation,
        phase_population_sizes=tuple(
            int(count) for count in phase_population_sizes.tolist()
        ),
        storage_group_offsets=tuple(storage_group_offsets),
        storage_group_shape=storage_group_shape,
    )


def threshold_model_activity(
    spike_scores: torch.Tensor,
    morphology_ids: torch.Tensor,
    inhibitory: torch.Tensor,
    thresholds: FixedActivityThresholds,
) -> torch.Tensor:
    """Return variable-cardinality events from fixed model-score crossings."""

    if spike_scores.ndim < 1:
        raise ValueError("spike_scores must have a population dimension")
    population = spike_scores.shape[-1]
    if morphology_ids.shape != (population,):
        raise ValueError("morphology_ids must match the population dimension")
    if inhibitory.shape != (population,):
        raise ValueError("inhibitory must match the population dimension")
    if morphology_ids.dtype == torch.bool or morphology_ids.is_floating_point():
        raise TypeError("morphology_ids must use an integer dtype")
    if inhibitory.dtype != torch.bool:
        raise TypeError("inhibitory must use bool dtype")
    if morphology_ids.device != spike_scores.device:
        raise ValueError("morphology_ids and spike_scores must share a device")
    if inhibitory.device != spike_scores.device:
        raise ValueError("inhibitory and spike_scores must share a device")
    if population:
        minimum = int(morphology_ids.min())
        maximum = int(morphology_ids.max())
        if minimum < 0 or maximum >= thresholds.morphology_count:
            raise ValueError("morphology_ids contain an unknown morphology")

    table = thresholds.as_tensor(
        device=spike_scores.device,
        dtype=spike_scores.dtype,
    )
    role_ids = inhibitory.to(dtype=torch.long)
    neuron_thresholds = table[role_ids, morphology_ids.long()]
    return spike_scores >= neuron_thresholds


def threshold_block_model_activity(
    spike_scores: torch.Tensor,
    morphology_ids: torch.Tensor,
    inhibitory: torch.Tensor,
    thresholds: FixedBlockActivityThresholds,
) -> torch.Tensor:
    """Threshold an ordered forecast block without fixing event counts."""

    if spike_scores.ndim < 2:
        raise ValueError(
            "block spike_scores need forecast and population dimensions"
        )
    if spike_scores.shape[0] != thresholds.forecast_step_count:
        raise ValueError(
            "spike_scores forecast dimension does not match thresholds"
        )
    population = spike_scores.shape[-1]
    if morphology_ids.shape != (population,):
        raise ValueError("morphology_ids must match the population dimension")
    if inhibitory.shape != (population,):
        raise ValueError("inhibitory must match the population dimension")
    table = thresholds.as_tensor(
        device=spike_scores.device,
        dtype=spike_scores.dtype,
    )
    role_ids = inhibitory.to(dtype=torch.long)
    neuron_thresholds = table[
        :, role_ids, morphology_ids.to(dtype=torch.long)
    ]
    broadcast_shape = (
        thresholds.forecast_step_count,
        *([1] * (spike_scores.ndim - 2)),
        population,
    )
    return spike_scores >= neuron_thresholds.reshape(broadcast_shape)
