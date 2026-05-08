import torch

from bnn_sim.branch_attention import BranchAttention, BranchAttentionConfig


def test_branch_attention_preserves_output_shape():
    model = BranchAttention(
        BranchAttentionConfig(
            num_input=6,
            num_output=2,
            num_branch=3,
            num_synapse_per_branch=2,
            model_dim=8,
            num_layers=1,
            num_heads=2,
            ff_multiplier=2,
        )
    )

    output = model(torch.zeros(2, 7, 6))

    assert output.shape == (2, 7, 2)


def test_branch_attention_is_causal_for_past_outputs():
    torch.manual_seed(123)
    model = BranchAttention(
        BranchAttentionConfig(
            num_input=6,
            num_output=2,
            num_branch=3,
            num_synapse_per_branch=2,
            model_dim=8,
            num_layers=1,
            num_heads=2,
            ff_multiplier=2,
        )
    ).eval()
    x = torch.randn(2, 7, 6)
    changed_future = x.clone()
    changed_future[:, 5:] = torch.randn_like(changed_future[:, 5:])

    with torch.inference_mode():
        base = model(x)
        changed = model(changed_future)

    assert torch.allclose(base[:, :5], changed[:, :5], atol=1e-6)
