from __future__ import annotations

import torch

from axosim.checkpoint import load_checkpoint, save_checkpoint
from axosim.deployment import GRU_BEHAVIOR_ADAPTATION
from axosim.p4_surrogate import (
    AdaptiveP4Surrogate,
    P4SurrogateConfig,
    normalize_signed_route_features,
)


MORPHOLOGIES = ("morph-a", "morph-b")


def _route_features() -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    features = torch.rand(2, 12, 5, generator=generator)
    features[:, :6, 0] = 1.0
    features[:, 6:, 0] = -1.0
    return normalize_signed_route_features(features)


def _model(*, behavior_adaptation: bool = True) -> AdaptiveP4Surrogate:
    return AdaptiveP4Surrogate(
        P4SurrogateConfig(
            input_dim=12,
            route_feature_dim=5,
            state_dim=58,
            patch_size=4,
            morphology_ids=MORPHOLOGIES,
            behavior_adaptation=behavior_adaptation,
        ),
        _route_features(),
    )


def test_route_features_are_signed_and_l1_normalized():
    raw = torch.tensor(
        [
            [
                [1.0, 2.0, 4.0],
                [-1.0, 3.0, 5.0],
            ]
        ]
    )

    normalized = normalize_signed_route_features(raw)

    torch.testing.assert_close(
        normalized.abs().sum(dim=1),
        torch.ones(1, 3),
    )
    assert normalized[0, 1, 1] < 0
    assert normalized[0, 1, 2] < 0


def test_optional_feature_tanh_is_finite_and_bounded():
    features = _route_features()
    model = AdaptiveP4Surrogate(
        P4SurrogateConfig(
            input_dim=12,
            route_feature_dim=5,
            state_dim=58,
            patch_size=4,
            morphology_ids=MORPHOLOGIES,
            behavior_adaptation=False,
            feature_nonlinearity="tanh",
            feature_scale_init=0.5,
        ),
        features,
    )
    inputs = torch.full((2, 8, 12), 1e4)
    summaries = model.aggregate_inputs(
        inputs,
        morphology_indices=torch.tensor([0, 1]),
    )

    assert torch.isfinite(summaries).all()
    assert summaries.abs().amax() <= 2.0


def test_full_sequence_matches_preaggregated_feature_path():
    model = _model()
    inputs = torch.randn(2, 12, 12)
    morphology = torch.tensor([0, 1])
    behavior = torch.randn(
        2,
        GRU_BEHAVIOR_ADAPTATION.parameter_count,
    ) * 0.01

    raw_output = model(
        inputs,
        morphology_indices=morphology,
        behavior_parameters=behavior,
    )
    summaries = model.aggregate_inputs(
        inputs,
        morphology_indices=morphology,
    )
    summary_output = model.forward_feature_summaries(
        summaries,
        behavior_parameters=behavior,
    )

    torch.testing.assert_close(raw_output, summary_output)
    assert raw_output.shape == (2, 12, 2)


def test_p4_outputs_are_causal_and_native_rate():
    model = _model()
    morphology = torch.tensor([0])
    behavior = torch.zeros(
        1,
        GRU_BEHAVIOR_ADAPTATION.parameter_count,
    )
    inputs = torch.randn(1, 12, 12)
    changed = inputs.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:])

    output = model(
        inputs,
        morphology_indices=morphology,
        behavior_parameters=behavior,
    )
    changed_output = model(
        changed,
        morphology_indices=morphology,
        behavior_parameters=behavior,
    )

    torch.testing.assert_close(output[:, :8], changed_output[:, :8])
    torch.testing.assert_close(output[:, :4], torch.zeros_like(output[:, :4]))
    assert not torch.allclose(output[:, 4, :], output[:, 5, :])


def test_streaming_p4_step_matches_sequence_execution():
    model = _model()
    morphology = torch.tensor([0, 1])
    behavior = torch.randn(
        2,
        GRU_BEHAVIOR_ADAPTATION.parameter_count,
    ) * 0.01
    inputs = torch.randn(2, 12, 12)
    summaries = model.aggregate_inputs(
        inputs,
        morphology_indices=morphology,
    )
    sequence = model.forward_feature_summaries(
        summaries,
        behavior_parameters=behavior,
    )

    state = model.initial_state(2, device=inputs.device, dtype=inputs.dtype)
    blocks = [torch.zeros(2, 4, 2)]
    for start in range(0, 8, 4):
        prediction, state = model.step_p4(
            summaries[:, start : start + 4],
            state,
            behavior_parameters=behavior,
        )
        blocks.append(prediction)

    torch.testing.assert_close(sequence, torch.cat(blocks, dim=1))


def test_feature_state_sequence_decodes_to_sequence_forecast():
    model = _model()
    morphology = torch.tensor([0, 1])
    inputs = torch.randn(2, 12, 12)
    summaries = model.aggregate_inputs(
        inputs,
        morphology_indices=morphology,
    )

    states = model.feature_state_sequence(
        summaries,
        morphology_indices=morphology,
    )
    decoded = model.decode_state_sequence(
        states,
        morphology_indices=morphology,
    ).flatten(start_dim=1, end_dim=2)
    output = model.forward_feature_summaries(
        summaries,
        morphology_indices=morphology,
    )

    assert states.shape == (2, 3, 58)
    torch.testing.assert_close(output[:, 4:], decoded[:, :8])


def test_feature_state_sequence_accepts_external_initial_state():
    model = _model()
    morphology = torch.tensor([0, 1])
    behavior = model.behavior_adaptation_bank.index_select(0, morphology)
    summaries = torch.randn(2, 8, model.config.route_feature_dim)
    initial = torch.randn(2, 58)

    states = model.feature_state_sequence(
        summaries,
        behavior_parameters=behavior,
        initial_state=initial,
    )
    _, expected_first = model.step_p4(
        summaries[:, :4],
        initial,
        behavior_parameters=behavior,
    )

    torch.testing.assert_close(states[:, 0], expected_first)


def test_every_adaptation_component_is_trainable_and_behaviorally_active():
    model = _model()
    inputs = torch.randn(2, 12, 12)
    morphology = torch.zeros(2, dtype=torch.long)
    behavior = (
        torch.randn(
            2,
            GRU_BEHAVIOR_ADAPTATION.parameter_count,
        )
        * 0.02
    ).requires_grad_()

    output = model(
        inputs,
        morphology_indices=morphology,
        behavior_parameters=behavior,
    )
    output.square().mean().backward()

    assert behavior.grad is not None
    for name, selection in GRU_BEHAVIOR_ADAPTATION.tensor_slices.items():
        assert behavior.grad[:, selection].abs().sum() > 0, name
    with torch.no_grad():
        centered = output - output.mean(dim=0, keepdim=True)
        assert centered[..., 0].square().mean().sqrt() > 0
        assert centered[..., 1].square().mean().sqrt() > 0


def test_grouped_morphology_path_matches_explicit_adaptation():
    model = _model()
    with torch.no_grad():
        model.behavior_adaptation_bank.normal_(std=0.02)
    inputs = torch.randn(4, 16, 12)
    morphology = torch.tensor([0, 1, 0, 1])
    explicit = model.behavior_adaptation_bank.index_select(0, morphology)

    grouped_output = model(
        inputs,
        morphology_indices=morphology,
    )
    assert model.last_execution_mode == "grouped_morphology"
    explicit_output = model(
        inputs,
        morphology_indices=morphology,
        behavior_parameters=explicit,
    )
    assert model.last_execution_mode == "explicit_step"

    torch.testing.assert_close(grouped_output, explicit_output)


def test_grouped_path_updates_every_adaptation_component():
    model = _model()
    with torch.no_grad():
        model.behavior_adaptation_bank.normal_(std=0.02)
    inputs = torch.randn(4, 16, 12)
    morphology = torch.tensor([0, 1, 0, 1])

    model(inputs, morphology_indices=morphology).square().mean().backward()

    gradient = model.behavior_adaptation_bank.grad
    assert gradient is not None
    for name, selection in GRU_BEHAVIOR_ADAPTATION.tensor_slices.items():
        assert gradient[:, selection].abs().sum() > 0, name


def test_shared_control_rejects_runtime_adaptation():
    model = _model(behavior_adaptation=False)
    inputs = torch.randn(2, 8, 12)
    morphology = torch.tensor([0, 1])
    behavior = torch.zeros(
        2,
        GRU_BEHAVIOR_ADAPTATION.parameter_count,
    )

    try:
        model(
            inputs,
            morphology_indices=morphology,
            behavior_parameters=behavior,
        )
    except ValueError as error:
        assert "not configured" in str(error)
    else:
        raise AssertionError("shared surrogate accepted runtime adaptation")


def test_checkpoint_round_trip(tmp_path):
    model = _model()
    inputs = torch.randn(2, 8, 12)
    morphology = torch.tensor([0, 1])
    expected = model(inputs, morphology_indices=morphology)
    checkpoint = tmp_path / "p4-surrogate.pt"

    save_checkpoint(model, checkpoint, metadata={"iteration": 557})
    loaded, metadata = load_checkpoint(checkpoint)
    actual = loaded(inputs, morphology_indices=morphology)

    assert metadata["iteration"] == 557
    torch.testing.assert_close(actual, expected)
