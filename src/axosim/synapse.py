"""Trainable and quantized adaptation for routed synaptic efficacy."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantizedSynapticEfficacyBank:
    """One W8 multiplicative efficacy for every retained incoming contact."""

    values: torch.Tensor
    scale: torch.Tensor
    channels_per_role: int
    recurrent_stride: int
    baseline: float = 1.0

    @property
    def population_size(self) -> int:
        return self.values.shape[0]

    @property
    def values_per_neuron(self) -> int:
        return self.values.shape[1]

    @property
    def storage_bytes(self) -> int:
        return (
            self.values.numel() * self.values.element_size()
            + self.scale.numel() * self.scale.element_size()
        )


def retained_synaptic_efficacies_per_neuron(
    *,
    channels_per_role: int,
    recurrent_stride: int,
) -> int:
    """Return compact E/I contact slots for the declared routed topology."""

    if channels_per_role <= 0:
        raise ValueError("channels_per_role must be positive")
    if recurrent_stride <= 0:
        raise ValueError("recurrent_stride must be positive")
    retained_per_role = (
        channels_per_role + recurrent_stride - 1
    ) // recurrent_stride
    return 2 * retained_per_role


def quantize_synaptic_efficacies(
    efficacies: torch.Tensor,
    *,
    channels_per_role: int,
    recurrent_stride: int,
    baseline: float = 1.0,
) -> QuantizedSynapticEfficacyBank:
    """Quantize positive per-contact multipliers around a shared baseline."""

    expected = retained_synaptic_efficacies_per_neuron(
        channels_per_role=channels_per_role,
        recurrent_stride=recurrent_stride,
    )
    if (
        efficacies.ndim != 2
        or efficacies.shape[1] != expected
        or not efficacies.is_floating_point()
    ):
        raise ValueError(
            "efficacies must be floating point with shape "
            f"(population, {expected})"
        )
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    if bool(torch.any(efficacies <= 0)):
        raise ValueError("synaptic efficacies must remain positive")
    deviation = efficacies - baseline
    maximum = deviation.abs().amax()
    scale = torch.where(
        maximum > 0,
        maximum / 127.0,
        torch.ones_like(maximum),
    )
    values = torch.clamp(
        torch.round(deviation / scale),
        -127,
        127,
    ).to(dtype=torch.int8)
    return QuantizedSynapticEfficacyBank(
        values=values.contiguous(),
        scale=scale.reshape(1).to(dtype=efficacies.dtype),
        channels_per_role=channels_per_role,
        recurrent_stride=recurrent_stride,
        baseline=float(baseline),
    )


def dequantize_synaptic_efficacies(
    bank: QuantizedSynapticEfficacyBank,
) -> torch.Tensor:
    """Materialize a W8 efficacy bank for training checks or export."""

    _validate_synaptic_efficacy_bank(bank)
    return (
        bank.values.to(dtype=bank.scale.dtype) * bank.scale
        + bank.baseline
    )


def _validate_synaptic_efficacy_bank(
    bank: QuantizedSynapticEfficacyBank,
) -> None:
    expected = retained_synaptic_efficacies_per_neuron(
        channels_per_role=bank.channels_per_role,
        recurrent_stride=bank.recurrent_stride,
    )
    if (
        bank.values.ndim != 2
        or bank.values.shape[1] != expected
        or bank.values.dtype != torch.int8
        or not bank.values.is_contiguous()
    ):
        raise ValueError("synaptic efficacy values have an invalid layout")
    if (
        bank.scale.shape != (1,)
        or not bank.scale.is_floating_point()
        or bank.scale.device != bank.values.device
    ):
        raise ValueError(
            "synaptic efficacy scale must be one co-located float"
        )
    if bank.baseline <= 0:
        raise ValueError("synaptic efficacy baseline must be positive")
