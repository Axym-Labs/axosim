import math

import torch

from axosim.support_ode import SynapticSupportODE


def _manual_dual_exponential(event: torch.Tensor, rise_ms: float, decay_ms: float) -> torch.Tensor:
    peak_time = rise_ms * decay_ms / (decay_ms - rise_ms) * math.log(decay_ms / rise_ms)
    factor = 1.0 / (-math.exp(-peak_time / rise_ms) + math.exp(-peak_time / decay_ms))
    rise_state = torch.tensor(0.0)
    decay_state = torch.tensor(0.0)
    values = []
    for value in event:
        rise_state = math.exp(-1.0 / rise_ms) * rise_state + value
        decay_state = math.exp(-1.0 / decay_ms) * decay_state + value
        values.append(factor * (decay_state - rise_state))
    return torch.stack(values)


def test_synaptic_support_ode_matches_teacher_dual_exponential_states():
    support = SynapticSupportODE(num_branches=1, mode="synaptic")
    excitatory = torch.zeros(1, 12, 1)
    inhibitory = torch.zeros(1, 12, 1)
    excitatory[0, 0, 0] = 1.0
    inhibitory[0, 0, 0] = 1.0

    output = support(excitatory, inhibitory)[0]

    torch.testing.assert_close(
        output[:, 0],
        _manual_dual_exponential(excitatory[0, :, 0], 0.3, 3.0),
    )
    torch.testing.assert_close(
        output[:, 1],
        _manual_dual_exponential(excitatory[0, :, 0], 2.0, 70.0),
    )
    torch.testing.assert_close(
        output[:, 2],
        2.5 * _manual_dual_exponential(inhibitory[0, :, 0], 0.2, 8.0),
    )


def test_support_ode_controls_preserve_shape_but_change_the_features():
    excitatory = torch.zeros(2, 16, 3)
    inhibitory = torch.zeros(2, 16, 3)
    excitatory[:, 0, :] = 1.0
    inhibitory[:, 4, :] = 1.0
    physical = SynapticSupportODE(num_branches=3, mode="synaptic")
    mismatched = SynapticSupportODE(num_branches=3, mode="generic")
    swapped = SynapticSupportODE(num_branches=3, mode="polarity_swapped")
    instantaneous = SynapticSupportODE(num_branches=3, mode="instantaneous")

    outputs = [module(excitatory, inhibitory) for module in (physical, mismatched, swapped, instantaneous)]

    assert all(output.shape == (2, 16, 9) for output in outputs)
    assert not torch.allclose(outputs[0], outputs[1])
    assert not torch.allclose(outputs[0], outputs[2])
    assert not torch.allclose(outputs[0], outputs[3])
