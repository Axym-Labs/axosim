import torch

from bnn_sim.mamba_official import _EngramMemoryBlock


def test_engram_memory_block_preserves_shape():
    block = _EngramMemoryBlock(
        model_dim=8,
        slots=3,
        engram_dim=4,
        tau=5.0,
        kernel_size=7,
        residual_scale_init=0.1,
    )

    output = block(torch.zeros(2, 11, 8))

    assert output.shape == (2, 11, 8)


def test_engram_memory_block_is_causal():
    torch.manual_seed(123)
    block = _EngramMemoryBlock(
        model_dim=8,
        slots=3,
        engram_dim=4,
        tau=5.0,
        kernel_size=7,
        residual_scale_init=0.1,
    )
    x = torch.randn(2, 11, 8)
    perturbed = x.clone()
    perturbed[:, 6:] = torch.randn_like(perturbed[:, 6:])

    output = block(x)
    perturbed_output = block(perturbed)

    assert torch.allclose(output[:, :6], perturbed_output[:, :6], atol=1e-6)
