from __future__ import annotations

import torch

from axosim.checkpoint import load_checkpoint, save_checkpoint
from axosim.support_surrogate import (
    AdaptiveSupportP4Surrogate,
    SupportP4Config,
)


def _model() -> AdaptiveSupportP4Surrogate:
    config = SupportP4Config(
        input_dim=12,
        route_feature_dim=6,
        token_dim=16,
        state_dim=8,
        morphology_ids=("a", "b"),
    )
    route_features = torch.randn(2, 12, 6)
    return AdaptiveSupportP4Surrogate(config, route_features)


def test_support_surrogate_preserves_native_shape_and_causal_prefix():
    model = _model()
    inputs = torch.randn(3, 20, 12)
    morphology = torch.tensor([0, 1, 0])

    output = model(inputs, morphology_indices=morphology)

    assert output.shape == (3, 20, 2)
    torch.testing.assert_close(output[:, :4], torch.zeros(3, 4, 2))
    assert torch.count_nonzero(output[:, 4:]) > 0


def test_support_adaptation_compiler_reaches_every_behavior_value():
    model = _model()
    summaries = torch.randn(2, 12, 6)
    behavior = torch.randn(
        2, model.behavior_parameter_count, requires_grad=True
    )

    output = model.forward_feature_summaries(
        summaries,
        behavior_parameters=behavior,
    )
    output.square().mean().backward()

    assert behavior.grad is not None
    assert torch.all(behavior.grad != 0)


def test_runtime_coefficient_slices_partition_the_deployed_adapter():
    model = _model()

    slices = model.runtime_coefficient_slices

    assert tuple(slices) == (
        "token_offset",
        "token_gain",
        "decay_delta",
        "state_gain",
        "state_offset",
        "output_gain",
        "output_offset",
    )
    covered = [
        index
        for selection in slices.values()
        for index in range(selection.start, selection.stop)
    ]
    assert covered == list(range(model.cache_width))


def test_support_runtime_cache_matches_direct_behavior_compilation():
    model = _model()
    features = torch.randn(2, 4, 6)
    state = model.initial_state(
        2, device=features.device, dtype=features.dtype
    )
    behavior = torch.randn(2, model.behavior_parameter_count)
    cache = model.compile_adaptation(behavior)

    direct_output, direct_state = model.step_p4(
        features,
        state,
        behavior_parameters=behavior,
    )
    cached_output, cached_state = model.step_p4(
        features,
        state,
        adaptation_cache=cache,
    )

    torch.testing.assert_close(cached_output, direct_output)
    torch.testing.assert_close(cached_state, direct_state)


def test_support_surrogate_checkpoint_round_trip(tmp_path):
    model = _model()
    path = tmp_path / "support.pt"

    save_checkpoint(model, path, metadata={"test": True})
    restored, metadata = load_checkpoint(path)

    assert isinstance(restored, AdaptiveSupportP4Surrogate)
    assert restored.config == model.config
    assert metadata["test"] is True
    for expected, actual in zip(
        model.state_dict().values(),
        restored.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_gate_features_reuse_the_causal_token_and_state():
    model = _model()
    inputs = torch.randn(2, 12, model.config.input_dim)
    morphology = torch.tensor([0, 1])

    expected = model(inputs, morphology_indices=morphology)
    observed, gate_features = model.forward_with_gate_features(
        inputs,
        morphology_indices=morphology,
    )

    torch.testing.assert_close(observed, expected)
    assert gate_features.shape == (2, 3, model.gate_feature_dim)
    assert torch.count_nonzero(gate_features[:, 0]) == 0
    assert torch.count_nonzero(gate_features[:, 1:]) > 0
