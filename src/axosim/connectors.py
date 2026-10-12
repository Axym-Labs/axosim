"""Connectivity-pattern builders that return a validated ``ExplicitConnectome``.

Each builder wires one population of ``n`` neurons (optionally restricted to
``pre`` source and ``post`` target IDs) and returns an ordinary explicit graph
for ``create_population``. Shared keyword arguments:

``neuron_roles``
    +1 (excitatory) or -1 (inhibitory) per neuron, as in ``ProceduralConnectome``;
    each edge takes its source's role. ``None`` makes every neuron excitatory.
``channel_roles``
    The checkpoint's +1/-1 role per input channel. Each edge's destination
    channel is drawn uniformly from the channels with its source's role.
``channels``
    Explicit destination channels instead of that draw; checked against
    ``channel_roles`` when both are given.
``efficacies`` / ``delays``
    A constant, one value per edge, or a callable
    ``fn(sources, targets, generator) -> tensor`` with one value per edge.
    Efficacies are nonnegative magnitudes; delays are integer native
    milliseconds and must be at least 4 (the Lite P4 recurrent minimum).
``seed``
    Seeds one ``torch.Generator`` drawn in a fixed order (topology, channels,
    efficacies, delays), so a seed reproduces the exact graph.

Edge IDs, as used by ``ConnectedPopulation.set_efficacies``, follow the
returned edge order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from axosim.connected_population import ExplicitConnectome, _indices, _roles

__all__ = [
    "MIN_DELAY_MS",
    "all_to_all",
    "concatenate",
    "fixed_indegree",
    "fixed_probability",
    "from_list",
    "one_to_one",
]

MIN_DELAY_MS = 4
_CHUNK = 1 << 22

EdgeValues = (
    float
    | Sequence[float]
    | torch.Tensor
    | Callable[[torch.Tensor, torch.Tensor, torch.Generator], torch.Tensor]
)


def all_to_all(
    n: int,
    *,
    seed: int = 0,
    allow_self: bool = False,
    pre=None,
    post=None,
    **kwargs,
) -> ExplicitConnectome:
    """Connect every ``pre`` neuron to every ``post`` neuron once (source-major)."""
    pre, post = _population(n, pre, post)
    sources = pre.repeat_interleave(len(post))
    targets = post.repeat(len(pre))
    if not allow_self:
        keep = sources != targets
        sources, targets = sources[keep], targets[keep]
    return _build(n, sources, targets, _generator(seed), **kwargs)


def one_to_one(
    n: int, *, seed: int = 0, pre=None, post=None, **kwargs
) -> ExplicitConnectome:
    """Connect ``pre[i]`` to ``post[i]``; the default ``pre == post`` gives autapses."""
    pre, post = _population(n, pre, post)
    if len(pre) != len(post):
        raise ValueError("one_to_one requires pre and post of equal length")
    return _build(n, pre, post, _generator(seed), **kwargs)


def fixed_probability(
    n: int,
    p: float | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    seed: int = 0,
    allow_self: bool = False,
    pre=None,
    post=None,
    **kwargs,
) -> ExplicitConnectome:
    """Keep each ``pre -> post`` pair independently with probability ``p``.

    ``p`` may be an element-wise callable ``p(sources, targets)`` returning one
    probability per candidate pair (evaluated in chunks), e.g. a
    distance-dependent kernel over neuron positions.
    Pairs are drawn once each, so the graph has no parallel edges.
    """
    pre, post = _population(n, pre, post)
    generator = _generator(seed)
    rows = max(1, _CHUNK // max(1, len(post)))
    kept_sources, kept_targets = [], []
    for start in range(0, len(pre), rows):
        sources = pre[start : start + rows].repeat_interleave(len(post))
        targets = post.repeat(len(pre[start : start + rows]))
        probability = (
            torch.as_tensor(p(sources, targets), dtype=torch.float64)
            if callable(p)
            else torch.tensor(float(p), dtype=torch.float64)
        ).expand(len(sources))
        if not ((probability >= 0) & (probability <= 1)).all():
            raise ValueError("connection probabilities must lie in [0, 1]")
        keep = torch.rand(len(sources), generator=generator, dtype=torch.float64)
        keep = keep < probability
        if not allow_self:
            keep &= sources != targets
        kept_sources.append(sources[keep])
        kept_targets.append(targets[keep])
    return _build(
        n, _cat(kept_sources), _cat(kept_targets), generator, **kwargs
    )


def fixed_indegree(
    n: int,
    k: int,
    *,
    seed: int = 0,
    allow_self: bool = False,
    with_replacement: bool = False,
    pre=None,
    post=None,
    **kwargs,
) -> ExplicitConnectome:
    """Give every ``post`` neuron exactly ``k`` sources drawn uniformly from ``pre``.

    Without replacement the ``k`` sources of a target are distinct; with
    replacement repeated draws become parallel edges. Edges are ordered by
    target, then by source.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    pre, post = _population(n, pre, post)
    generator = _generator(seed)
    rows = max(1, _CHUNK // max(1, len(pre)))
    picked_sources, picked_targets = [], []
    for start in range(0, len(post) if k else 0, rows):
        targets = post[start : start + rows]
        weights = torch.ones(len(targets), len(pre), dtype=torch.float64)
        if not allow_self:
            weights[targets[:, None] == pre[None, :]] = 0.0
        available = int((weights > 0).sum(dim=1).min())
        if available < (1 if with_replacement else k):
            raise ValueError(
                f"fixed_indegree cannot draw {k} sources per target from "
                f"{available} eligible sources"
            )
        choice = torch.multinomial(
            weights, k, replacement=with_replacement, generator=generator
        )
        picked_sources.append(pre[choice.sort(dim=1).values].flatten())
        picked_targets.append(targets.repeat_interleave(k))
    return _build(
        n, _cat(picked_sources), _cat(picked_targets), generator, **kwargs
    )


def from_list(
    n: int,
    edges,
    *,
    efficacies: EdgeValues | None = None,
    delays: EdgeValues | None = None,
    seed: int = 0,
    **kwargs,
) -> ExplicitConnectome:
    """Build from rows ``(source, target[, efficacy[, delay]])`` in the given order.

    Efficacy/delay columns and the ``efficacies``/``delays`` arguments are
    mutually exclusive; omitted values default to efficacy 1.0 and 4 ms.
    """
    _check_n(n)
    table = torch.as_tensor(edges, dtype=torch.float64)
    if table.numel() == 0:
        table = table.reshape(0, 2)
    if table.ndim != 2 or table.shape[1] not in (2, 3, 4):
        raise ValueError(
            "edges must be rows of (source, target[, efficacy[, delay]])"
        )
    columns = {2: efficacies, 3: delays}
    for column, name in ((2, "efficacies"), (3, "delays")):
        if table.shape[1] > column:
            if columns[column] is not None:
                raise ValueError(f"{name} given both as an edge column and argument")
            columns[column] = table[:, column]
    endpoints = table[:, :2]
    if not torch.equal(endpoints, endpoints.round()):
        raise ValueError("edge sources and targets must be integer neuron IDs")
    sources, targets = endpoints.long().unbind(dim=1)
    _check_bounds(n, sources, "edge sources")
    _check_bounds(n, targets, "edge targets")
    return _build(
        n,
        sources,
        targets,
        _generator(seed),
        efficacies=1.0 if columns[2] is None else columns[2],
        delays=MIN_DELAY_MS if columns[3] is None else columns[3],
        **kwargs,
    )


def concatenate(*connectomes: ExplicitConnectome) -> ExplicitConnectome:
    """Join graphs edge-wise; part ``i``'s edge IDs follow those of parts ``< i``."""
    if not connectomes:
        raise ValueError("concatenate requires at least one connectome")
    if not all(isinstance(graph, ExplicitConnectome) for graph in connectomes):
        raise TypeError("concatenate accepts ExplicitConnectome values only")
    joined = ExplicitConnectome(
        **{
            name: torch.cat([getattr(graph, name) for graph in connectomes])
            for name in ExplicitConnectome.__dataclass_fields__
        }
    )
    if joined.edge_count:
        size = int(joined.sources.max()) + 1
        role = torch.zeros(size, dtype=torch.long)
        role[joined.sources] = joined.source_roles
        if not torch.equal(role[joined.sources], joined.source_roles):
            raise ValueError(
                "source roles must agree across all outgoing edges of a neuron"
            )
    return joined


def _build(
    n,
    sources,
    targets,
    generator,
    *,
    neuron_roles=None,
    channel_roles=None,
    channels=None,
    efficacies: EdgeValues = 1.0,
    delays: EdgeValues = MIN_DELAY_MS,
) -> ExplicitConnectome:
    roles = (
        torch.ones(n, dtype=torch.long)
        if neuron_roles is None
        else _roles(neuron_roles, "neuron_roles")
    )
    if len(roles) != n:
        raise ValueError("neuron_roles must have one +1/-1 role per neuron")
    source_roles = roles[sources]
    channel_roles = (
        None if channel_roles is None else _roles(channel_roles, "channel_roles")
    )
    if channels is None:
        if channel_roles is None:
            raise ValueError(
                "channel_roles is required to choose destination channels; "
                "pass the checkpoint's channel roles or explicit channels"
            )
        channels = torch.empty_like(sources)
        for role in (-1, 1):
            mask = source_roles == role
            available = (channel_roles == role).nonzero().flatten()
            if not mask.any():
                continue
            if not len(available):
                kind = "excitatory" if role == 1 else "inhibitory"
                raise ValueError(f"{kind} sources need at least one {kind} channel")
            size = (int(mask.sum()),)
            draw = torch.randint(len(available), size, generator=generator)
            channels[mask] = available[draw]
    else:
        channels = _edge_values(channels, sources, targets, generator, "channels")
        channels = _integral(channels, "channels")
        if channel_roles is not None:
            if (channels < 0).any() or (channels >= len(channel_roles)).any():
                raise ValueError(f"channels must be in [0, {len(channel_roles)})")
            if not torch.equal(channel_roles[channels], source_roles):
                raise ValueError(
                    "destination channel role must match the source neuron role"
                )
    efficacies = _edge_values(efficacies, sources, targets, generator, "efficacies")
    delays = _integral(
        _edge_values(delays, sources, targets, generator, "delays"), "delays"
    )
    if len(delays) and int(delays.min()) < MIN_DELAY_MS:
        raise ValueError(
            f"delays must be >= {MIN_DELAY_MS} native milliseconds for Lite P4 "
            f"recurrence; got a minimum of {int(delays.min())} ms"
        )
    return ExplicitConnectome(
        sources=sources,
        targets=targets,
        channels=channels,
        delays=delays,
        source_roles=source_roles,
        efficacies=efficacies,
    )


def _edge_values(spec, sources, targets, generator, name):
    values = torch.as_tensor(
        spec(sources, targets, generator) if callable(spec) else spec
    )
    if values.ndim == 0:
        values = values.expand(len(sources))
    if tuple(values.shape) != (len(sources),):
        raise ValueError(
            f"{name} must be a scalar, one value per edge ({len(sources)}), "
            "or a callable returning one value per edge"
        )
    return values


def _integral(values, name):
    if values.is_floating_point():
        if not torch.isfinite(values).all() or not torch.equal(
            values, values.round()
        ):
            raise ValueError(f"{name} must be integer-valued")
        values = values.long()
    return _indices(values, name)


def _check_n(n):
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")


def _population(n, pre, post):
    _check_n(n)
    ids = []
    for name, value in (("pre", pre), ("post", post)):
        value = torch.arange(n) if value is None else _indices(value, name)
        _check_bounds(n, value, name)
        ids.append(value)
    return ids


def _check_bounds(n, ids, name):
    if (ids < 0).any() or (ids >= n).any():
        raise ValueError(f"{name} must be neuron IDs in [0, {n})")


def _generator(seed):
    if isinstance(seed, bool) or not isinstance(seed, int):
        # Schema validation uses ValueError for invalid arguments.
        raise ValueError("seed must be an integer")  # noqa: TRY004
    return torch.Generator().manual_seed(seed)


def _cat(parts):
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)
