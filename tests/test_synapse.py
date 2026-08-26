from __future__ import annotations

import pytest
import torch

from axosim.synapse import (
    dequantize_synaptic_efficacies,
    quantize_synaptic_efficacies,
    retained_synaptic_efficacies_per_neuron,
)


def test_retained_synaptic_efficacy_count_matches_million_contract() -> None:
    assert retained_synaptic_efficacies_per_neuron(
        channels_per_role=724,
        recurrent_stride=4,
    ) == 362


def test_synaptic_efficacy_w8_round_trip() -> None:
    torch.manual_seed(83)
    efficacy = 1.0 + torch.randn(7, 6) * 0.12
    bank = quantize_synaptic_efficacies(
        efficacy,
        channels_per_role=6,
        recurrent_stride=2,
    )
    restored = dequantize_synaptic_efficacies(bank)

    assert bank.values.dtype == torch.int8
    assert bank.values.shape == (7, 6)
    assert bank.storage_bytes == 46
    torch.testing.assert_close(
        restored,
        efficacy,
        atol=float(bank.scale),
        rtol=0,
    )


def test_synaptic_efficacy_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        quantize_synaptic_efficacies(
            torch.tensor([[1.0, 0.0]]),
            channels_per_role=1,
            recurrent_stride=1,
        )
