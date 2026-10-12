"""Connectivity-pattern builders return exact, seeded ExplicitConnectome graphs."""

from __future__ import annotations

import pytest
import torch
from test_connected_population import graph, population

import axosim
from axosim import connectors

ROLES = [1, 1, -1, 1]
CHANNELS = [1, -1]


def edges(connectome):
    return list(zip(connectome.sources.tolist(), connectome.targets.tolist()))


def test_builders_are_lazy_top_level_exports():
    for name in connectors.__all__[1:]:
        assert getattr(axosim, name) is getattr(connectors, name)
    assert axosim.connectors is connectors


def test_all_to_all_and_one_to_one_topologies():
    graph = axosim.all_to_all(3, channel_roles=[1])
    assert edges(graph) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    assert axosim.all_to_all(3, allow_self=True, channel_roles=[1]).edge_count == 9
    subset = axosim.all_to_all(4, pre=[0, 1], post=[2, 3], channel_roles=[1])
    assert edges(subset) == [(0, 2), (0, 3), (1, 2), (1, 3)]
    pairs = axosim.one_to_one(4, pre=[0, 1], post=[3, 2], channel_roles=[1])
    assert edges(pairs) == [(0, 3), (1, 2)]
    assert edges(axosim.one_to_one(2, channel_roles=[1])) == [(0, 0), (1, 1)]
    with pytest.raises(ValueError, match="equal length"):
        axosim.one_to_one(3, pre=[0], post=[1, 2], channel_roles=[1])


def test_fixed_probability_is_seeded_and_supports_pair_kernels():
    kwargs = {"neuron_roles": ROLES, "channel_roles": CHANNELS}
    a = axosim.fixed_probability(40, 0.2, seed=3, channel_roles=CHANNELS)
    b = axosim.fixed_probability(40, 0.2, seed=3, channel_roles=CHANNELS)
    c = axosim.fixed_probability(40, 0.2, seed=4, channel_roles=CHANNELS)
    for name in ("sources", "targets", "channels"):
        assert torch.equal(getattr(a, name), getattr(b, name))
    assert edges(a) != edges(c)
    assert len(set(edges(a))) == a.edge_count
    assert all(source != target for source, target in edges(a))
    assert 0.1 < a.edge_count / (40 * 39) < 0.3
    assert axosim.fixed_probability(4, 0.0, **kwargs).edge_count == 0
    full = axosim.fixed_probability(4, 1.0, **kwargs)
    assert edges(full) == edges(axosim.all_to_all(4, **kwargs))
    ring = axosim.fixed_probability(
        5, lambda s, t: ((s - t).abs() == 1).double(), channel_roles=[1]
    )
    assert edges(ring) == [
        (0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)
    ]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        axosim.fixed_probability(3, 1.5, channel_roles=[1])


def test_fixed_indegree_gives_each_target_exactly_k_distinct_sources():
    graph = axosim.fixed_indegree(10, 3, seed=1, channel_roles=[1])
    again = axosim.fixed_indegree(10, 3, seed=1, channel_roles=[1])
    assert torch.equal(graph.sources, again.sources)
    assert graph.targets.tolist() == [t for t in range(10) for _ in range(3)]
    for target in range(10):
        sources = graph.sources[graph.targets == target].tolist()
        assert sources == sorted(set(sources)) and target not in sources
    assert axosim.fixed_indegree(3, 2, channel_roles=[1]).edge_count == 6
    with pytest.raises(ValueError, match="cannot draw 3"):
        axosim.fixed_indegree(3, 3, channel_roles=[1])
    repeated = axosim.fixed_indegree(
        2, 4, with_replacement=True, allow_self=True, pre=[0], channel_roles=[1]
    )
    assert edges(repeated) == [(0, 0)] * 4 + [(0, 1)] * 4
    subset = axosim.fixed_indegree(6, 2, pre=[0, 1, 2], post=[4, 5], channel_roles=[1])
    assert set(subset.sources.tolist()) <= {0, 1, 2}
    assert subset.targets.tolist() == [4, 4, 5, 5]


def test_from_list_reads_optional_efficacy_and_delay_columns():
    graph = axosim.from_list(3, [(0, 1, 0.5, 4), (0, 1, 2.0, 7)], channel_roles=[1])
    assert edges(graph) == [(0, 1), (0, 1)]
    assert graph.efficacies.tolist() == [0.5, 2.0]
    assert graph.delays.tolist() == [4, 7]
    plain = axosim.from_list(
        3, [[0, 2], [1, 2]], efficacies=[1.0, 3.0], channel_roles=[1]
    )
    assert plain.efficacies.tolist() == [1.0, 3.0]
    assert plain.delays.tolist() == [4, 4]
    assert axosim.from_list(3, [], channel_roles=[1]).edge_count == 0
    with pytest.raises(ValueError, match="both"):
        axosim.from_list(3, [(0, 1, 0.5)], efficacies=1.0, channel_roles=[1])
    with pytest.raises(ValueError, match="integer"):
        axosim.from_list(3, [(0.5, 1)], channel_roles=[1])
    with pytest.raises(ValueError, match="targets"):
        axosim.from_list(3, [(0, 3)], channel_roles=[1])


def test_edge_values_accept_constants_arrays_and_seeded_callables():
    def jitter(sources, targets, generator):
        return 4 + torch.randint(0, 3, (len(sources),), generator=generator)

    def build(seed):
        return axosim.all_to_all(
            4,
            seed=seed,
            channel_roles=[1],
            efficacies=lambda s, t, g: torch.rand(len(s), generator=g),
            delays=jitter,
        )

    a, b = build(0), build(0)
    assert torch.equal(a.efficacies, b.efficacies)
    assert torch.equal(a.delays, b.delays)
    assert not torch.equal(a.efficacies, build(1).efficacies)
    assert set(a.delays.tolist()) <= {4, 5, 6}
    distance = axosim.all_to_all(
        3, channel_roles=[1], delays=lambda s, t, g: 4 + (s - t).abs()
    )
    assert distance.delays.tolist() == [5, 6, 5, 5, 6, 5]
    per_edge = axosim.one_to_one(2, channel_roles=[1], efficacies=[0.25, 0.75])
    assert per_edge.efficacies.tolist() == [0.25, 0.75]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"delays": 3}, ">= 4 native milliseconds"),
        ({"delays": 4.5}, "delays must be integer"),
        ({"efficacies": -1.0}, "nonnegative"),
        ({"efficacies": [1.0, 2.0]}, "one value per edge"),
        ({"neuron_roles": [1, 0, 1]}, "neuron_roles"),
        ({"neuron_roles": [1, 1]}, "one \\+1/-1 role per neuron"),
        ({"channel_roles": None}, "channel_roles is required"),
        ({"channel_roles": [1, 1]}, "inhibitory channel"),
        ({"channels": 1}, "channel role"),
    ],
)
def test_builders_reject_invalid_specifications(kwargs, match):
    options = {"neuron_roles": [1, -1, 1], "channel_roles": CHANNELS, **kwargs}
    with pytest.raises(ValueError, match=match):
        axosim.all_to_all(3, **options)


def test_roles_follow_the_source_and_channels_match_roles():
    graph = axosim.fixed_probability(
        4, 1.0, neuron_roles=ROLES, channel_roles=[1, -1, 1, -1], seed=2
    )
    expected = torch.tensor(ROLES)[graph.sources]
    assert torch.equal(graph.source_roles, expected)
    assert torch.equal(torch.tensor([1, -1, 1, -1])[graph.channels], expected)
    explicit = axosim.all_to_all(
        2, neuron_roles=[1, -1], channel_roles=CHANNELS, channels=[0, 1]
    )
    assert explicit.channels.tolist() == [0, 1]


def test_concatenate_offsets_edge_ids_and_checks_roles():
    first = axosim.one_to_one(3, pre=[0], post=[1], channel_roles=[1])
    second = axosim.from_list(3, [(1, 2, 2.0)], channel_roles=[1])
    joined = axosim.concatenate(first, second)
    assert edges(joined) == [(0, 1), (1, 2)]
    assert joined.efficacies.tolist() == [1.0, 2.0]
    inhibitory = axosim.one_to_one(
        3, pre=[0], post=[2], neuron_roles=[-1, 1, 1], channel_roles=[-1]
    )
    with pytest.raises(ValueError, match="source roles"):
        axosim.concatenate(first, inhibitory)


def test_built_graphs_run_in_a_connected_population():
    hand_built = list(population(graph()).run(duration_ms=16))
    built = axosim.from_list(
        3, [(0, 1, 0.5, 4), (0, 1, 2.0, 4)], channel_roles=CHANNELS
    )
    for a, b in zip(hand_built, population(built).run(duration_ms=16), strict=True):
        for signal in a.signals:
            assert torch.equal(a.signals[signal], b.signals[signal])
    mixed = axosim.fixed_probability(
        3, 0.5, seed=0, neuron_roles=[1, 1, -1], channel_roles=CHANNELS, delays=5
    )
    frames = list(population(mixed).run(duration_ms=12))
    assert population(mixed).connectome.edge_count == mixed.edge_count
    assert [frame.time_ms for frame in frames] == list(range(12))
