"""Behavioral contracts for an actually connected, persistent Lite simulator."""

from __future__ import annotations

import pytest
import torch

import axosim
from axosim.support_surrogate import AdaptiveSupportP4Surrogate, SupportP4Config


def public_api(name):
    value = getattr(axosim, name, None)
    assert value is not None, f"axosim must expose {name}"
    return value


def neuron(*, signed_features=False):
    """A real P4 neuron whose spike logit equals its previous patch input sum."""
    config = SupportP4Config(
        input_dim=2,
        route_feature_dim=1,
        token_dim=1,
        state_dim=1,
        morphology_ids=("m0", "m1"),
        behavior_adaptation=False,
    )
    routes = torch.tensor(
        [
            [[1.0], [1.0 if signed_features else -1.0]],
            [[2.0], [2.0 if signed_features else -2.0]],
        ]
    )
    model = AdaptiveSupportP4Surrogate(config, routes)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.input_projection.weight.fill_(1.0)
        model.skip_head.weight[::2].fill_(1.0)
    return model.eval()


def graph(*, roles=(1, 1), efficacies=(0.5, 2.0), delays=(4, 4)):
    return public_api("ExplicitConnectome")(
        sources=[0, 0],
        targets=[1, 1],
        channels=[0, 0],
        delays=delays,
        source_roles=roles,
        efficacies=efficacies,
    )


def stimulus(time_ms):
    if time_ms == 0:
        return public_api("InputEvents")(
            neurons=[0],
            channels=[0],
            values=[1.0],
        )
    return None


def population(connectome=None, **kwargs):
    return public_api("create_population")(
        model=neuron(),
        n=3,
        connectome=graph() if connectome is None else connectome,
        morphology_indices=[0, 0, 1],
        input_encoding="channel",
        channel_roles=[1, -1],
        stream_inputs=stimulus,
        stream_outputs={
            "spikes": "all",
            "input_features": "all",
            "spike_logit": "all",
            "hidden_state": [1],
        },
        spike_threshold=0.5,
        **kwargs,
    )


def test_connected_events_have_exact_delay_and_parallel_contact_efficacies():
    sim = population()
    frames = list(sim.run(duration_ms=16))
    assert [frame.time_ms for frame in frames] == list(range(16))
    assert [frame.valid for frame in frames[:4]] == [False] * 4
    assert frames[4].signals["spikes"].tolist() == [True, False, False]
    assert frames[7].signals["input_features"][1].tolist() == [0.0]
    assert frames[8].signals["input_features"][1].tolist() == [2.5]
    assert frames[11].signals["input_features"][1].tolist() == [2.5]
    assert frames[12].signals["spike_logit"].tolist() == [0.0, 10.0, 0.0]
    assert frames[12].signals["spikes"].tolist() == [False, True, False]
    assert sim.time_ms == 16


def test_heterogeneous_native_delays_match_a_hand_derived_arrival_sequence():
    sim = population(graph(delays=(4, 7)))
    observed = [
        frame.signals["input_features"][1, 0].item()
        for frame in sim.run(duration_ms=20)
    ]
    assert observed == [0.0] * 8 + [0.5, 0.5, 0.5, 2.5, 2.0, 2.0, 2.0] + [0.0] * 5


def test_queued_deliveries_keep_emission_time_efficacy_after_contact_edit():
    sim = population()
    list(sim.run(duration_ms=5))
    sim.set_efficacies([0, 1], [0.0, 0.0])
    arrivals = list(sim.run(duration_ms=5))
    assert arrivals[3].time_ms == 8
    assert arrivals[3].signals["input_features"][1].tolist() == [2.5]
    assert arrivals[4].signals["input_features"][1].tolist() == [0.0]


def test_disconnected_persistent_lite_matches_the_public_supplied_history_model():
    torch.manual_seed(29)
    config = SupportP4Config(
        input_dim=2,
        route_feature_dim=3,
        token_dim=4,
        state_dim=3,
        morphology_ids=("m0", "m1"),
    )
    model = AdaptiveSupportP4Surrogate(config, torch.randn(2, 2, 3)).eval()
    inputs = torch.rand(3, 13, 2)
    ids = torch.tensor([0, 1, 0])
    empty = public_api("ExplicitConnectome")(
        sources=[], targets=[], channels=[], delays=[], source_roles=[], efficacies=[]
    )
    sim = public_api("create_population")(
        model=model,
        n=3,
        connectome=empty,
        morphology_indices=ids,
        input_encoding="channel",
        stream_outputs={"spike_logit": "all", "soma_target": "all"},
    )
    actual = torch.stack(
        [
            torch.stack(
                (frame.signals["spike_logit"], frame.signals["soma_target"]), dim=-1
            )
            for frame in (sim.step(inputs[:, t]) for t in range(13))
        ],
        dim=1,
    )
    with torch.no_grad():
        expected = model(inputs, morphology_indices=ids)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


def test_individual_efficacy_mutation_preserves_parallel_edge_independence():
    sim = population()
    sim.set_efficacies([0], [3.0])
    assert sim.efficacies.tolist() == [3.0, 2.0]
    assert list(sim.run(duration_ms=9))[-1].signals["input_features"][1].tolist() == [
        5.0
    ]


def test_run_continues_and_snapshot_is_owned():
    sim = population()
    first = list(sim.run(duration_ms=6))
    snapshot = sim.observe(neurons=[0, 1], signals=("spike_logit", "hidden_state"))
    frozen = snapshot.signals["spike_logit"].clone()
    second = list(sim.run(duration_ms=10))
    assert second[0].time_ms == 6
    assert torch.equal(snapshot.signals["spike_logit"], frozen)
    assert first[4].signals["spikes"].tolist() == [True, False, False]
    assert sim.observe(neurons=[1], signals=("spike_logit",)).signals[
        "spike_logit"
    ].tolist() == [10.0]


def test_reset_restores_queue_clock_and_initial_efficacies():
    sim = population()
    first = list(sim.run(duration_ms=16))
    sim.set_efficacies([0], [7.0])
    sim.reset()
    assert sim.time_ms == 0
    assert sim.efficacies.tolist() == [0.5, 2.0]
    replay = list(sim.run(duration_ms=16))
    for before, after in zip(first, replay, strict=True):
        assert before.time_ms == after.time_ms
        for key in before.signals:
            assert torch.equal(before.signals[key], after.signals[key])


def test_saved_state_replays_pending_events_and_partial_patch():
    sim = population()
    list(sim.run(duration_ms=7))
    saved = sim.state_dict()
    expected = list(sim.run(duration_ms=10))
    sim.load_state_dict(saved)
    actual = list(sim.run(duration_ms=10))
    assert [frame.time_ms for frame in actual] == list(range(7, 17))
    for before, after in zip(expected, actual, strict=True):
        for key in before.signals:
            assert torch.equal(before.signals[key], after.signals[key])
    assert saved["time_ms"] == 7


@pytest.mark.parametrize(
    "input_encoding,signed_features", [("signed", True), ("channel", False)]
)
def test_inhibition_is_applied_once(input_encoding, signed_features):
    inhibitory = public_api("ExplicitConnectome")(
        sources=[0],
        targets=[1],
        channels=[1],
        delays=[4],
        source_roles=[-1],
        efficacies=[2.0],
    )
    sim = public_api("create_population")(
        model=neuron(signed_features=signed_features),
        n=3,
        connectome=inhibitory,
        morphology_indices=[0, 0, 0],
        input_encoding=input_encoding,
        channel_roles=[1, -1],
        stream_inputs=stimulus,
        stream_outputs={"input_features": [1]},
        spike_threshold=0.5,
    )
    assert list(sim.run(duration_ms=9))[-1].signals["input_features"].tolist() == [
        [-2.0]
    ]


@pytest.mark.parametrize(
    "field,bad,match",
    [
        ("sources", [3], "source"),
        ("targets", [-1], "target"),
        ("channels", [2], "channel"),
        ("delays", [3], "delay"),
        ("source_roles", [0], "role"),
        ("efficacies", [-1.0], "efficac"),
    ],
)
def test_explicit_graph_rejects_invalid_values_without_modifying_them(
    field, bad, match
):
    values = {
        "sources": [0],
        "targets": [1],
        "channels": [0],
        "delays": [4],
        "source_roles": [1],
        "efficacies": [1.0],
    }
    values[field] = bad
    with pytest.raises(ValueError, match=match):
        population(public_api("ExplicitConnectome")(**values))


def test_requires_explicit_model_and_known_voltage_transform():
    with pytest.raises(TypeError, match="model"):
        public_api("create_population")(
            n=3,
            connectome=graph(),
            morphology_indices=[0, 0, 0],
            input_encoding="channel",
        )
    with pytest.raises(ValueError, match="soma_mv"):
        public_api("create_population")(
            model=neuron(),
            n=3,
            connectome=graph(),
            morphology_indices=[0, 0, 0],
            input_encoding="channel",
            stream_outputs={"soma_mv": [0]},
        )


def test_selected_observation_has_ids_native_voltage_and_learned_state():
    sim = public_api("create_population")(
        model=neuron(),
        n=3,
        connectome=graph(),
        morphology_indices=[0, 0, 1],
        input_encoding="channel",
        soma_transform=(10.0, -67.7),
        stream_outputs={"soma_mv": [2, 0], "hidden_state": [1]},
    )
    frame = sim.step()
    assert frame.neuron_ids["soma_mv"].tolist() == [2, 0]
    assert frame.signals["soma_mv"].tolist() == pytest.approx([-67.7, -67.7])
    assert tuple(frame.signals["hidden_state"].shape) == (1, 1)


def test_procedural_topology_matches_its_explicit_materialization():
    spec = public_api("ProceduralConnectome")(
        n=3,
        out_degree=3,
        input_dim=2,
        neuron_roles=[1, -1, 1],
        channel_roles=[1, -1],
        delay_ms=4,
        seed=7,
        efficacy=0.5,
    )
    kwargs = {
        "model": neuron(),
        "n": 3,
        "morphology_indices": [0, 0, 1],
        "input_encoding": "channel",
        "stream_inputs": stimulus,
        "stream_outputs": {"spikes": "all", "input_features": "all"},
        "spike_threshold": 0.5,
    }
    implicit = public_api("create_population")(connectome=spec, **kwargs)
    explicit = public_api("create_population")(
        connectome=spec.materialize(), channel_roles=[1, -1], **kwargs
    )
    for a, b in zip(
        implicit.run(duration_ms=20), explicit.run(duration_ms=20), strict=True
    ):
        assert torch.equal(a.signals["spikes"], b.signals["spikes"])
        assert torch.equal(a.signals["input_features"], b.signals["input_features"])


def test_reference_preserves_independent_autograd_population_contract():
    from axosim import AxoSimPopulation

    model = neuron()
    bank = AxoSimPopulation(
        model,
        morphology_indices=torch.tensor([0, 0]),
        contact_branch_indices=torch.tensor([[0], [0]]),
    )
    inputs = torch.ones(2, 8, 1, requires_grad=True)
    output = bank(inputs)
    output[:, 4:, 0].sum().backward()
    assert inputs.grad is not None
    assert inputs.grad[:, :4, :].abs().sum().item() > 0
    assert torch.equal(bank(inputs.detach()), bank(inputs.detach()))


@pytest.mark.parametrize("threshold", [0.0, -10.0])
def test_causal_warmup_never_emits_even_below_zero_threshold(threshold):
    sim = public_api("create_population")(
        model=neuron(),
        n=3,
        connectome=graph(),
        morphology_indices=[0, 0, 0],
        input_encoding="channel",
        spike_threshold=threshold,
    )
    first = list(sim.run(duration_ms=4))
    assert all(not frame.valid and not frame.signals["spikes"].any() for frame in first)


def test_zero_duration_lazy_iterator_and_chunked_run_parity():
    whole, chunked = population(), population()
    pending = chunked.run(duration_ms=6)
    assert chunked.time_ms == 0
    assert list(chunked.run(duration_ms=0)) == []
    assert chunked.time_ms == 0
    pieces = (
        list(pending)
        + list(chunked.run(duration_ms=3))
        + list(chunked.run(duration_ms=8))
    )
    for a, b in zip(whole.run(duration_ms=17), pieces, strict=True):
        assert a.time_ms == b.time_ms
        for name in a.signals:
            assert torch.equal(a.signals[name], b.signals[name])


def test_sampling_preserves_native_steps_and_caller_frame_mutation_is_isolated():
    sim = population(sample_every_ms=3)
    frames = list(sim.run(duration_ms=11))
    assert [frame.time_ms for frame in frames] == [0, 3, 6, 9]
    frames[-1].signals["spike_logit"].fill_(100.0)
    frames[-1].neuron_ids["spike_logit"].fill_(2)
    now = sim.observe(neurons=[0, 1], signals=("spike_logit",))
    assert now.time_ms == 10
    assert now.signals["spike_logit"].tolist() == [0.0, 0.0]
    assert now.neuron_ids["spike_logit"].tolist() == [0, 1]
    # Consume the full requested duration, not only the iterator's first item.
    assert list(sim.run(duration_ms=2))[0].time_ms == 12  # noqa: RUF015


def test_run_copies_subscribed_signals_only_on_requested_sample_times(monkeypatch):
    sim = population(sample_every_ms=4)
    real_observe, observed = sim.observe, []

    def counted_observation(**kwargs):
        frame = real_observe(**kwargs)
        observed.append(frame.time_ms)
        return frame

    monkeypatch.setattr(sim, "observe", counted_observation)
    frames = list(sim.run(duration_ms=12))
    assert observed == [0, 4, 8]
    assert [frame.time_ms for frame in frames] == [0, 4, 8]
    assert sim.time_ms == 12


def test_half_neuron_does_not_quantize_exact_edge_efficacies():
    precise = public_api("ExplicitConnectome")(
        sources=[0],
        targets=[1],
        channels=[0],
        delays=[4],
        source_roles=[1],
        efficacies=[0.1234567],
    )
    sim = public_api("create_population")(
        model=neuron().half(),
        n=3,
        connectome=precise,
        morphology_indices=[0, 0, 0],
        input_encoding="channel",
        stream_inputs=stimulus,
        spike_threshold=0.5,
        stream_outputs={"input_features": [1]},
    )
    assert sim.efficacies.dtype == torch.float32
    assert torch.equal(sim.efficacies.cpu(), torch.tensor([0.1234567]))
    assert (
        list(sim.run(duration_ms=9))[-1].signals["input_features"].item()
        == torch.tensor(0.1234567).item()
    )


def test_explicit_empty_subscription_produces_no_output_tensors():
    sim = public_api("create_population")(
        model=neuron(),
        n=3,
        connectome=graph(),
        morphology_indices=[0, 0, 0],
        input_encoding="channel",
        stream_outputs={},
    )
    assert sim.step().signals == {}
    assert sim.observe().neuron_ids == {}


def test_public_topology_and_morphology_snapshots_cannot_mutate_runtime_routes():
    sim = population()
    topology = sim.connectome
    topology.targets.fill_(2)
    morphology = sim.morphology_indices
    morphology.fill_(1)
    arrival = list(sim.run(duration_ms=9))[-1]
    assert arrival.signals["input_features"].tolist() == [[0.0], [2.5], [0.0]]


def test_factory_revalidates_graph_tensors_mutated_after_schema_construction():
    bad = graph()
    bad.efficacies.fill_(-1.0)
    with pytest.raises(ValueError, match="efficacies"):
        population(bad)


def test_save_load_device_independent_owned_model_does_not_change_caller_weights(
    tmp_path,
):
    model = neuron()
    sim = public_api("create_population")(
        model=model,
        n=3,
        connectome=graph(),
        morphology_indices=[0, 0, 0],
        input_encoding="channel",
        stream_inputs=stimulus,
        stream_outputs={"spike_logit": "all"},
        spike_threshold=0.5,
    )
    with torch.no_grad():
        model.skip_head.weight.zero_()
    list(sim.run(duration_ms=7))
    path = tmp_path / "population-state.pt"
    torch.save(sim.state_dict(), path)
    saved = torch.load(path, weights_only=True)
    expected = list(sim.run(duration_ms=8))
    sim.load_state_dict(saved)
    replay = list(sim.run(duration_ms=8))
    assert expected[5].signals["spike_logit"].tolist() == [0.0, 10.0, 0.0]
    for a, b in zip(expected, replay, strict=True):
        assert torch.equal(a.signals["spike_logit"], b.signals["spike_logit"])


def test_malformed_state_load_is_atomic_and_foreign_identity_is_rejected():
    sim = population()
    list(sim.run(duration_ms=7))
    before = sim.state_dict()
    malformed = sim.state_dict()
    malformed["tensors"]["state"].fill_(200.0)
    malformed["tensors"]["coefficients"] = torch.zeros(1, 1)
    with pytest.raises(ValueError, match="coefficients"):
        sim.load_state_dict(malformed)
    after = sim.state_dict()
    assert after["time_ms"] == before["time_ms"]
    for name in before["tensors"]:
        assert torch.equal(after["tensors"][name], before["tensors"][name])
    other_graph = public_api("ExplicitConnectome")(
        sources=[0, 0],
        targets=[2, 2],
        channels=[0, 0],
        delays=[4, 4],
        source_roles=[1, 1],
        efficacies=[0.5, 2.0],
    )
    with pytest.raises(ValueError, match="fingerprint"):
        population(other_graph).load_state_dict(before)
    changed_model = neuron()
    with torch.no_grad():
        changed_model.output_head.bias[0] = 0.1
    with pytest.raises(ValueError, match="fingerprint"):
        public_api("create_population")(
            model=changed_model,
            n=3,
            connectome=graph(),
            morphology_indices=[0, 0, 1],
            input_encoding="channel",
            sample_every_ms=1,
        ).load_state_dict(before)


@pytest.mark.parametrize(
    "encoding,channel,value,match",
    [
        ("channel", 0, -1.0, "nonnegative"),
        ("channel", 2, 1.0, "bounds"),
        ("signed", 1, 1.0, "sign"),
    ],
)
def test_invalid_external_encoding_and_ids_raise_without_consuming_time(
    encoding, channel, value, match
):
    sim = public_api("create_population")(
        model=neuron(),
        n=3,
        connectome=graph(),
        morphology_indices=[0, 0, 0],
        input_encoding=encoding,
        channel_roles=[1, -1],
    )
    bad = public_api("InputEvents")(neurons=[0], channels=[channel], values=[value])
    with pytest.raises(ValueError, match=match):
        sim.step(bad)
    assert sim.time_ms == 0
    with pytest.raises(ValueError, match="bounds"):
        sim.step(public_api("InputEvents")(neurons=[3], channels=[0], values=[1.0]))


def test_explicit_graph_checks_source_consistency_and_channel_roles():
    with pytest.raises(ValueError, match="source roles"):
        population(graph(roles=(1, -1)))
    inhibitory_wrong_channel = public_api("ExplicitConnectome")(
        sources=[0],
        targets=[1],
        channels=[0],
        source_roles=[-1],
        delays=[4],
        efficacies=[1.0],
    )
    with pytest.raises(ValueError, match="channel role"):
        population(inhibitory_wrong_channel)


def test_unsupported_model_backend_and_nonreplayable_stream_are_explicit():
    with pytest.raises(TypeError, match="AxoSimLite"):
        public_api("create_population")(
            model=torch.nn.Linear(2, 2),
            n=3,
            connectome=graph(),
            morphology_indices=[0, 0, 0],
            input_encoding="channel",
        )
    with pytest.raises(ValueError, match="backend"):
        population(backend="headline")
    with pytest.raises(ValueError, match="CUDA FP16"):
        population(backend="fused")
    with pytest.raises(TypeError, match="replayable"):
        public_api("create_population")(
            model=neuron(),
            n=3,
            connectome=graph(),
            morphology_indices=[0, 0, 0],
            input_encoding="channel",
            stream_inputs=iter([None]),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires coordinated CUDA/Triton check"
)
def test_fused_lite_matches_reference_for_supported_graph_and_coefficients():
    pytest.importorskip("triton")
    torch.manual_seed(41)
    config = SupportP4Config(
        input_dim=2,
        route_feature_dim=3,
        token_dim=5,
        state_dim=3,
        morphology_ids=("m0",),
        behavior_adaptation=False,
    )
    model = (
        AdaptiveSupportP4Surrogate(config, torch.randn(1, 2, 3)).eval().cuda().half()
    )
    coefficients = (
        torch.randn(3, model.cache_width, device="cuda", dtype=torch.float16) * 0.02
    )
    kwargs = {
        "model": model,
        "n": 3,
        "connectome": graph(),
        "morphology_indices": [0, 0, 0],
        "input_encoding": "signed",
        "runtime_coefficients": coefficients,
        "stream_outputs": {
            "spike_logit": "all",
            "soma_target": "all",
            "hidden_state": "all",
        },
        "spike_threshold": 10.0,
    }
    reference = public_api("create_population")(**kwargs, backend="reference")
    fused = public_api("create_population")(**kwargs, backend="fused")
    for time in range(12):
        inputs = torch.full((3, 2), 0.01 * time, device="cuda", dtype=torch.float16)
        a, b = reference.step(inputs), fused.step(inputs)
        for name in a.signals:
            torch.testing.assert_close(
                a.signals[name], b.signals[name], atol=2e-4, rtol=2e-3
            )
    # Threshold-safe, genuinely coupled oracle: t4 emissions arrive at t8,
    # while one edited parallel edge changes future events independently.
    coupled = {
        "model": neuron().eval().cuda().half(),
        "n": 3,
        "connectome": graph(),
        "morphology_indices": [0, 0, 1],
        "input_encoding": "channel",
        "channel_roles": [1, -1],
        "stream_inputs": stimulus,
        "spike_threshold": 0.5,
        "stream_outputs": {
            "spikes": "all",
            "spike_logit": "all",
            "input_features": "all",
        },
    }
    reference = public_api("create_population")(**coupled, backend="reference")
    fused = public_api("create_population")(**coupled, backend="fused")
    for time in range(16):
        if time == 5:
            reference.set_efficacies([0], [3.0])
            fused.set_efficacies([0], [3.0])
        a, b = reference.step(), fused.step()
        for name in a.signals:
            torch.testing.assert_close(
                a.signals[name], b.signals[name], atol=2e-4, rtol=2e-3
            )
        if time == 8:
            assert b.signals["input_features"][1].item() == 2.5
        if time == 9:
            assert b.signals["input_features"][1].item() == 5.0
        if time == 12:
            assert b.signals["spikes"].tolist() == [False, True, False]
