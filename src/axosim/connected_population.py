"""Persistent connected Lite simulation, separate from supplied-history BPTT.

The reference scheduler supports explicit ragged multigraphs and an implicit
deterministic fan-out topology. ``fused`` replaces the Lite neuron step with the
installed Triton kernel; it does not import arbitrary graphs into the historical
specialized million-neuron router or inherit that router's performance claims.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from axosim.support_surrogate import AdaptiveSupportP4Surrogate


def _indices(values, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values)
    if tensor.ndim != 1 or (
        tensor.numel()
        and tensor.dtype
        not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    ):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return tensor.detach().to(device="cpu", dtype=torch.long).clone()


def _values(values, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim != 1 or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return tensor.detach().cpu().clone()


def _roles(values, name: str) -> torch.Tensor:
    tensor = _indices(values, name)
    if not ((tensor == 1) | (tensor == -1)).all():
        raise ValueError(f"{name} entries must be +1 (excitatory) or -1 (inhibitory)")
    return tensor


@dataclass(frozen=True)
class InputEvents:
    """Sparse external inputs in the model's declared input encoding.

    ``neurons``, ``channels`` and ``values`` are equally sized one-dimensional
    arrays. Duplicate neuron/channel entries add. Values already carry the
    native input amplitude/sign; source-role sign is never applied to external
    values a second time. Construction takes owned copies of all arrays.
    """

    neurons: Sequence[int] | torch.Tensor
    channels: Sequence[int] | torch.Tensor
    values: Sequence[float] | torch.Tensor

    def __post_init__(self) -> None:
        for name in ("neurons", "channels"):
            object.__setattr__(self, name, _indices(getattr(self, name), name))
        object.__setattr__(self, "values", _values(self.values, "values"))
        if not (len(self.neurons) == len(self.channels) == len(self.values)):
            raise ValueError(
                "external event neurons, channels and values must have equal length"
            )


@dataclass(frozen=True)
class ExplicitConnectome:
    """An exact directed multigraph with one efficacy per retained edge.

    All fields are one-dimensional arrays of length ``E``. ``sources`` and
    ``targets`` are neuron IDs; ``channels`` index the destination model's input
    channels. ``source_roles`` are +1/-1 and must agree across every outgoing
    edge of a source neuron. ``delays`` are integer native milliseconds.
    Parallel edges and arbitrary incoming/outgoing degrees are preserved.
    Efficacies are finite nonnegative magnitudes; zero disables an edge without
    changing its source role or removing its identity. Bounds and the Lite
    delay minimum are checked by ``create_population`` with the supplied model.
    """

    sources: Sequence[int] | torch.Tensor
    targets: Sequence[int] | torch.Tensor
    channels: Sequence[int] | torch.Tensor
    delays: Sequence[int] | torch.Tensor
    source_roles: Sequence[int] | torch.Tensor
    efficacies: Sequence[float] | torch.Tensor

    def __post_init__(self) -> None:
        for name in ("sources", "targets", "channels", "delays"):
            object.__setattr__(self, name, _indices(getattr(self, name), name))
        object.__setattr__(
            self, "source_roles", _roles(self.source_roles, "source_roles")
        )
        object.__setattr__(self, "efficacies", _values(self.efficacies, "efficacies"))
        lengths = {len(getattr(self, name)) for name in self.__dataclass_fields__}
        if len(lengths) != 1:
            raise ValueError("all explicit connectome fields must have equal length")
        if (self.efficacies < 0).any():
            raise ValueError("efficacies must be nonnegative magnitudes")
        if (self.delays <= 0).any():
            raise ValueError("delays must be positive native milliseconds")

    @property
    def edge_count(self) -> int:
        """Number of retained edges, including parallel edges."""
        return len(self.sources)


@dataclass(frozen=True)
class ProceduralConnectome:
    """Implicit deterministic fan-out graph with an exact efficacy per edge.

    Every source has ``out_degree`` edges. Edge ``source*out_degree+contact``
    targets ``(source + seed + contact*104729) % n``; its destination input
    channel cycles through channels with the source's role. Self and parallel
    edges are allowed and retained. This topology supports any positive ``n``;
    it is not the historical power-of-two, tiled unique-channel connectome.
    Topology is computed for active sources, while efficacies still require
    O(n*out_degree) storage. ``materialize`` creates the equivalent explicit
    graph for inspection/small reference checks, allocating O(E) index arrays.
    """

    n: int
    out_degree: int
    input_dim: int
    neuron_roles: Sequence[int] | torch.Tensor
    channel_roles: Sequence[int] | torch.Tensor
    delay_ms: int = 4
    seed: int = 0
    efficacy: float = 1.0

    def __post_init__(self) -> None:
        for name in ("n", "out_degree", "input_dim", "delay_ms", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                # Schema validation uses ValueError for invalid arguments.
                raise ValueError(f"{name} must be an integer")  # noqa: TRY004
        if min(self.n, self.out_degree, self.input_dim, self.delay_ms) <= 0:
            raise ValueError("n, out_degree, input_dim and delay_ms must be positive")
        if not 0 <= self.seed < 2**31 or self.out_degree >= 2**31:
            raise ValueError(
                "seed and out_degree must fit the procedural integer range"
            )
        object.__setattr__(
            self, "neuron_roles", _roles(self.neuron_roles, "neuron_roles")
        )
        object.__setattr__(
            self, "channel_roles", _roles(self.channel_roles, "channel_roles")
        )
        if (
            len(self.neuron_roles) != self.n
            or len(self.channel_roles) != self.input_dim
        ):
            raise ValueError("neuron_roles/channel_roles must match n/input_dim")
        for role in self.neuron_roles.unique().tolist():
            if not (self.channel_roles == role).any():
                raise ValueError(
                    "every neuron role requires a matching destination channel role"
                )
        if not torch.isfinite(torch.tensor(self.efficacy)) or self.efficacy < 0:
            raise ValueError("efficacy must be a finite nonnegative magnitude")

    @property
    def edge_count(self) -> int:
        """Exact retained edge count; implicit topology does not discard edges."""
        return self.n * self.out_degree

    def materialize(self) -> ExplicitConnectome:
        """Return an equivalent owned explicit graph, allocating all E edges."""
        sources = torch.arange(self.n).repeat_interleave(self.out_degree)
        contacts = torch.arange(self.out_degree).repeat(self.n)
        targets, channels, roles = _procedural_edges(self, sources, contacts)
        return ExplicitConnectome(
            sources=sources,
            targets=targets,
            channels=channels,
            delays=torch.full((self.edge_count,), self.delay_ms),
            source_roles=roles,
            efficacies=torch.full((self.edge_count,), self.efficacy),
        )


def _procedural_edges(spec, sources, contacts):
    roles = spec.neuron_roles.to(sources.device).index_select(0, sources)
    channels = torch.empty_like(sources)
    for role in (-1, 1):
        available = (spec.channel_roles == role).nonzero().flatten().to(sources.device)
        mask = roles == role
        if available.numel():
            channels[mask] = available[contacts[mask] % len(available)]
    return (sources + spec.seed + contacts * 104729) % spec.n, channels, roles


@dataclass(frozen=True)
class PopulationFrame:
    """Owned selected tensors at one native timestamp.

    ``signals`` maps names to tensors; ``neuron_ids`` gives each tensor's row
    identities. Tensors stay on the simulation device until explicitly copied.
    ``valid`` is false during the first four causal-padding samples. Frames do
    not alias runtime buffers and remain unchanged after subsequent steps.
    ``time_ms=-1`` denotes observation before the first native step.
    """

    time_ms: int
    valid: bool
    signals: dict[str, torch.Tensor]
    neuron_ids: dict[str, torch.Tensor]


StreamInput = (
    Callable[[int], InputEvents | torch.Tensor | None]
    | Mapping[int, InputEvents | torch.Tensor]
)


class ConnectedPopulation:
    """A state-owning, closed-loop inference simulator for a supplied Lite model.

    Construct through ``create_population``. One native step is 1 ms; Lite
    state updates once per four-step patch and predicts the next four outputs.
    Discrete thresholding/routing is inference-only. Use ``AxoSimPopulation``
    for full gradients through independently supplied contact histories.
    """

    _SIGNALS = frozenset(
        {
            "spikes",
            "spike_logit",
            "soma_target",
            "soma_mv",
            "hidden_state",
            "input_features",
        }
    )

    def __init__(
        self,
        *,
        model: AdaptiveSupportP4Surrogate,
        n: int,
        connectome: ExplicitConnectome | ProceduralConnectome,
        morphology_indices: Sequence[int] | torch.Tensor,
        input_encoding: Literal["signed", "channel"],
        channel_roles: Sequence[int] | torch.Tensor | None = None,
        stream_inputs: StreamInput | None = None,
        stream_outputs: Mapping[str, Sequence[int] | torch.Tensor | Literal["all"]]
        | None = None,
        backend: Literal["reference", "fused"] = "reference",
        spike_threshold: float | Sequence[float] | torch.Tensor = 0.0,
        sample_every_ms: int = 1,
        soma_transform: tuple[float, float] | None = None,
        runtime_coefficients: torch.Tensor | None = None,
    ) -> None:
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        if not isinstance(model, AdaptiveSupportP4Surrogate):
            raise TypeError(
                "connected simulation currently requires an AxoSimLite model"
            )
        if input_encoding not in ("signed", "channel"):
            raise ValueError("input_encoding must be 'signed' or 'channel'")
        if backend not in ("reference", "fused"):
            raise ValueError("backend must be 'reference' or 'fused'")
        if (
            not isinstance(sample_every_ms, int)
            or isinstance(sample_every_ms, bool)
            or sample_every_ms <= 0
        ):
            raise ValueError("sample_every_ms must be a positive integer")
        if stream_inputs is not None and not (
            callable(stream_inputs) or isinstance(stream_inputs, Mapping)
        ):
            raise TypeError(
                "stream_inputs must be a native-time callback or timestamp mapping; iterators are not replayable"
            )
        self.n = n
        self.backend = backend
        self.input_encoding = input_encoding
        self.sample_every_ms = sample_every_ms
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.device = self.model.route_features.device
        self.dtype = self.model.input_projection.weight.dtype
        if self.dtype not in (
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ):
            raise ValueError("model weights must have floating dtype")
        if backend == "fused":
            if self.device.type != "cuda" or self.dtype != torch.float16:
                raise ValueError(
                    "fused backend requires an explicitly CUDA FP16 Lite model"
                )
            try:
                from axosim._fused_lite import fused_lite_step
            except ImportError as error:
                raise ImportError("fused backend requires Triton") from error
            self._fused_step = fused_lite_step
        self._morphology_indices = _indices(morphology_indices, "morphology_indices")
        if (
            len(self._morphology_indices) != n
            or (self._morphology_indices < 0).any()
            or (self._morphology_indices >= len(model.config.morphology_ids)).any()
        ):
            raise ValueError(
                "morphology_indices must have n valid model morphology IDs"
            )
        self._morphology_indices = self._morphology_indices.to(self.device)
        self._connectome = copy.deepcopy(connectome)
        if isinstance(self._connectome, (ExplicitConnectome, ProceduralConnectome)):
            # Frozen dataclass fields may contain caller-mutated tensor values.
            # Revalidate the owned graph instead of trusting its earlier init.
            self._connectome.__post_init__()
        channel_roles = (
            None if channel_roles is None else _roles(channel_roles, "channel_roles")
        )
        self._validate_connectome(channel_roles)
        if isinstance(self._connectome, ProceduralConnectome):
            channel_roles = self._connectome.channel_roles
        self._channel_roles = (
            None if channel_roles is None else channel_roles.to(self.device)
        )
        self._subscriptions = self._validate_subscriptions(
            {"spikes": "all"} if stream_outputs is None else stream_outputs
        )
        self.soma_transform = soma_transform
        if soma_transform is not None and (
            len(soma_transform) != 2
            or not all(torch.isfinite(torch.tensor(v)) for v in soma_transform)
            or soma_transform[0] <= 0
        ):
            raise ValueError(
                "soma_transform must be (positive finite scale_mv, finite offset_mv)"
            )
        if "soma_mv" in self._subscriptions and soma_transform is None:
            raise ValueError("soma_mv requires an explicit validated soma_transform")
        self.stream_inputs = (
            copy.deepcopy(stream_inputs)
            if isinstance(stream_inputs, Mapping)
            else stream_inputs
        )
        self._thresholds = (
            torch.as_tensor(spike_threshold, device=self.device, dtype=self.dtype)
            .detach()
            .clone()
        )
        if self._thresholds.ndim == 0:
            self._thresholds = self._thresholds.expand(n).clone()
        if (
            tuple(self._thresholds.shape) != (n,)
            or not torch.isfinite(self._thresholds).all()
        ):
            raise ValueError("spike_threshold must be finite scalar or shape (n,)")
        with torch.no_grad():
            if runtime_coefficients is None:
                if self.model.behavior_adaptation_bank is None:
                    self._coefficients = torch.zeros(
                        n, model.cache_width, device=self.device, dtype=self.dtype
                    )
                else:
                    behavior = self.model.behavior_adaptation_bank.index_select(
                        0, self._morphology_indices
                    )
                    self._coefficients = (
                        self.model.compile_adaptation(behavior).detach().clone()
                    )
            else:
                if (
                    tuple(runtime_coefficients.shape) != (n, model.cache_width)
                    or not torch.isfinite(runtime_coefficients).all()
                ):
                    raise ValueError(
                        "runtime_coefficients must be finite shape (n, model.cache_width)"
                    )
                self._coefficients = (
                    runtime_coefficients.detach()
                    .to(device=self.device, dtype=self.dtype)
                    .clone()
                )
        self._coefficients = self._coefficients.contiguous()
        r = model.config.route_feature_dim
        self._state = self.model.initial_state(n, device=self.device, dtype=self.dtype)
        self._patch = torch.zeros(n, 4, r, device=self.device, dtype=self.dtype)
        self._forecast = torch.zeros(n, 4, 2, device=self.device, dtype=self.dtype)
        self._queue = torch.zeros(
            self._max_delay + 1, n, r, device=self.device, dtype=torch.float32
        )
        self._input_features = torch.zeros(
            n, r, device=self.device, dtype=torch.float32
        )
        self._output = torch.zeros(n, 2, device=self.device, dtype=self.dtype)
        self._spikes = torch.zeros(n, device=self.device, dtype=torch.bool)
        self.time_ms = 0
        self._valid = False
        self._fingerprint = self._configuration_fingerprint()
        self._initial_state = self.state_dict()

    def _validate_connectome(self, channel_roles):
        graph = self._connectome
        if (
            channel_roles is not None
            and len(channel_roles) != self.model.config.input_dim
        ):
            raise ValueError("channel_roles must match model input_dim")
        if isinstance(graph, ExplicitConnectome):
            for name, upper in (
                ("sources", self.n),
                ("targets", self.n),
                ("channels", self.model.config.input_dim),
            ):
                indices = getattr(graph, name)
                if (indices < 0).any() or (indices >= upper).any():
                    raise ValueError(f"connectome {name} must be in [0, {upper})")
            source_role = torch.zeros(self.n, dtype=torch.long)
            source_role[graph.sources] = graph.source_roles
            if not torch.equal(source_role[graph.sources], graph.source_roles):
                raise ValueError(
                    "source roles must agree across all outgoing edges of a neuron"
                )
            if channel_roles is not None and not torch.equal(
                channel_roles[graph.channels], graph.source_roles
            ):
                raise ValueError(
                    "destination channel role must match the source neuron role"
                )
            self._max_delay = int(graph.delays.max()) if graph.edge_count else 4
            self._sources = graph.sources.to(self.device)
            self._targets = graph.targets.to(self.device)
            self._channels = graph.channels.to(self.device)
            self._delays = graph.delays.to(self.device)
            self._source_roles = graph.source_roles.to(self.device)
            self._efficacies = graph.efficacies.to(
                self.device, dtype=torch.float32
            ).clone()
            if graph.edge_count and (graph.delays < 4).any():
                raise ValueError(
                    "Lite P4 recurrent delays must be >= 4 native milliseconds; graph is not modified"
                )
        elif isinstance(graph, ProceduralConnectome):
            if graph.n != self.n or graph.input_dim != self.model.config.input_dim:
                raise ValueError("procedural n/input_dim must match population/model")
            if channel_roles is not None and not torch.equal(
                channel_roles, graph.channel_roles
            ):
                raise ValueError("channel_roles disagree with the procedural graph")
            if graph.delay_ms < 4:
                raise ValueError(
                    "Lite P4 recurrent delays must be >= 4 native milliseconds; graph is not modified"
                )
            self._max_delay = graph.delay_ms
            self._efficacies = torch.full(
                (graph.edge_count,),
                graph.efficacy,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            raise TypeError(
                "connectome must be ExplicitConnectome or ProceduralConnectome"
            )
        if not torch.isfinite(self._efficacies).all():
            raise ValueError("efficacies overflow the model's execution dtype")

    def _validate_subscriptions(self, subscriptions):
        result = {}
        for name, selection in subscriptions.items():
            if name not in self._SIGNALS:
                raise ValueError(
                    f"unknown signal {name!r}; choose from {sorted(self._SIGNALS)}"
                )
            ids = (
                torch.arange(self.n)
                if isinstance(selection, str) and selection == "all"
                else _indices(selection, f"{name} neuron selection")
            )
            if (ids < 0).any() or (ids >= self.n).any():
                raise ValueError(f"{name} neuron selection is outside [0, n)")
            result[name] = ids.to(self.device)
        return result

    @property
    def connectome(self) -> ExplicitConnectome | ProceduralConnectome:
        """Owned graph-schema copy; editing it cannot change validated runtime topology."""
        return copy.deepcopy(self._connectome)

    @property
    def morphology_indices(self) -> torch.Tensor:
        """Owned integer (n,) morphology-assignment snapshot."""
        return self._morphology_indices.clone()

    @property
    def efficacies(self) -> torch.Tensor:
        """Owned ``(E,)`` snapshot of exact edge magnitudes; use set_efficacies to edit."""
        return self._efficacies.clone()

    @property
    def thresholds(self) -> torch.Tensor:
        """Owned ``(n,)`` snapshot of native spike-logit thresholds."""
        return self._thresholds.clone()

    def set_efficacies(
        self,
        edge_ids: Sequence[int] | torch.Tensor,
        values: Sequence[float] | torch.Tensor,
    ) -> None:
        """Set selected nonnegative edge magnitudes; queued deliveries are unchanged.

        An emitted event uses the efficacy at emission time. Updating an edge
        therefore affects future emissions, not already scheduled arrivals.
        """
        ids = _indices(edge_ids, "edge_ids").to(self.device)
        values = _values(values, "efficacies").to(
            device=self.device, dtype=torch.float32
        )
        if (
            len(ids) != len(values)
            or (ids < 0).any()
            or (ids >= len(self._efficacies)).any()
        ):
            raise ValueError(
                "edge_ids/efficacies must have equal length and valid edge IDs"
            )
        if (values < 0).any() or not torch.isfinite(values).all():
            raise ValueError(
                "efficacies must be finite nonnegative magnitudes in the model dtype"
            )
        if len(ids.unique()) != len(ids):
            raise ValueError(
                "edge_ids must be unique for an unambiguous efficacy update"
            )
        self._efficacies[ids] = values

    def set_thresholds(self, values: float | Sequence[float] | torch.Tensor) -> None:
        """Replace scalar or per-neuron spike-logit thresholds for future samples."""
        tensor = torch.as_tensor(values, device=self.device, dtype=self.dtype)
        if tensor.ndim == 0:
            tensor = tensor.expand(self.n)
        if tuple(tensor.shape) != (self.n,) or not torch.isfinite(tensor).all():
            raise ValueError("thresholds must be finite scalar or shape (n,)")
        self._thresholds.copy_(tensor)

    def _external_features(self, external):
        if external is None:
            return
        if isinstance(external, InputEvents):
            ids = external.neurons.to(self.device)
            channels = external.channels.to(self.device)
            if (
                (ids < 0).any()
                or (ids >= self.n).any()
                or (channels < 0).any()
                or (channels >= self.model.config.input_dim).any()
            ):
                raise ValueError(
                    "external neuron/channel IDs are outside the model/population bounds"
                )
            routes = self.model.route_features[
                self._morphology_indices[ids], channels
            ].float()
            values = external.values.to(device=self.device, dtype=torch.float32)
            if not torch.isfinite(values).all():
                raise ValueError("external event values overflow the model dtype")
            self._input_features.index_add_(0, ids, routes * values[:, None])
        elif isinstance(external, torch.Tensor):
            if (
                tuple(external.shape) != (self.n, self.model.config.input_dim)
                or not torch.isfinite(external).all()
            ):
                raise ValueError(
                    "dense external inputs must be finite shape (n, model.input_dim)"
                )
            values = external.to(device=self.device, dtype=torch.float32)
            if not torch.isfinite(values).all():
                raise ValueError("dense external values overflow the model dtype")
            selected = self.model.route_features.index_select(
                0, self._morphology_indices
            ).float()
            self._input_features.add_(torch.einsum("nc,ncr->nr", values, selected))
        else:
            raise TypeError(
                "one native external input must be InputEvents, Tensor, or None"
            )

    def _validate_external(self, external):
        """Validate before consuming a delay-queue slot or changing a patch."""
        if external is None:
            return
        if isinstance(external, InputEvents):
            ids, channels = external.neurons, external.channels
            if (
                (ids < 0).any()
                or (ids >= self.n).any()
                or (channels < 0).any()
                or (channels >= self.model.config.input_dim).any()
            ):
                raise ValueError(
                    "external neuron/channel IDs are outside the model/population bounds"
                )
            values = external.values.to(device=self.device, dtype=torch.float32)
            roles = (
                None
                if self._channel_roles is None
                else self._channel_roles[channels.to(self.device)]
            )
        elif isinstance(external, torch.Tensor):
            if tuple(external.shape) != (self.n, self.model.config.input_dim):
                raise ValueError(
                    "dense external inputs must be finite shape (n, model.input_dim)"
                )
            values = external.to(device=self.device, dtype=torch.float32)
            roles = self._channel_roles
        else:
            raise TypeError(
                "one native external input must be InputEvents, Tensor, or None"
            )
        if not torch.isfinite(values).all():
            raise ValueError("external values must be finite in the model dtype")
        if self.input_encoding == "channel" and (values < 0).any():
            raise ValueError(
                "channel-encoded external inputs must be nonnegative; channel identity encodes inhibition"
            )
        if (
            self.input_encoding == "signed"
            and roles is not None
            and (values * roles < 0).any()
        ):
            raise ValueError(
                "signed external input sign must agree with the declared channel role"
            )

    def _route(self):
        graph = self._connectome
        if isinstance(graph, ExplicitConnectome):
            edge_ids = self._spikes[self._sources].nonzero().flatten()
            targets = self._targets[edge_ids]
            channels = self._channels[edge_ids]
            roles = self._source_roles[edge_ids]
            delays = self._delays[edge_ids]
        else:
            active = self._spikes.nonzero().flatten()
            sources = active.repeat_interleave(graph.out_degree)
            contacts = torch.arange(graph.out_degree, device=self.device).repeat(
                len(active)
            )
            edge_ids = sources * graph.out_degree + contacts
            targets, channels, roles = _procedural_edges(graph, sources, contacts)
            delays = torch.full_like(targets, graph.delay_ms)
        if not len(edge_ids):
            return
        routes = self.model.route_features[
            self._morphology_indices[targets], channels
        ].float()
        amplitude = self._efficacies[edge_ids]
        if self.input_encoding == "signed":
            amplitude = amplitude * roles.float()
        slots = (self.time_ms + delays) % len(self._queue)
        flat_indices = slots * self.n + targets
        self._queue.view(-1, self.model.config.route_feature_dim).index_add_(
            0,
            flat_indices,
            routes * amplitude[:, None],
        )

    @torch.inference_mode()
    def _advance(self, inputs: InputEvents | torch.Tensor | None = None) -> None:
        """Advance state without materializing an exported observation."""
        phase = self.time_ms % 4
        external = (
            self.stream_inputs(self.time_ms)
            if callable(self.stream_inputs)
            else (
                self.stream_inputs.get(self.time_ms)
                if self.stream_inputs is not None
                else None
            )
        )
        self._validate_external(external)
        self._validate_external(inputs)
        slot = self.time_ms % len(self._queue)
        self._input_features.copy_(self._queue[slot])
        self._queue[slot].zero_()
        self._external_features(external)
        self._external_features(inputs)
        self._patch[:, phase].copy_(self._input_features)
        self._valid = self.time_ms >= 4
        self._output.copy_(self._forecast[:, phase])
        self._spikes.copy_((self._output[:, 0] > self._thresholds) & self._valid)
        self._route()
        if phase == 3:
            token = self.model.input_projection(self._patch.flatten(1)).contiguous()
            if self.backend == "reference":
                forecast, next_state = self.model.step_token(
                    token, self._state, adaptation_cache=self._coefficients
                )
            else:
                next_state, forecast = self._fused_step(
                    self.model, token, self._state, self._coefficients
                )
            self._state.copy_(next_state)
            self._forecast.copy_(forecast)
        self.time_ms += 1

    def step(self, inputs: InputEvents | torch.Tensor | None = None) -> PopulationFrame:
        """Advance one native millisecond and return an owned subscribed frame.

        Explicit ``inputs`` add to the configured stream at this timestamp.
        Input values are already in the declared native encoding. The frame
        timestamp is the processed sample, while ``time_ms`` becomes the next
        sample's timestamp. Warmup produces no spikes even at thresholds <=0.
        """
        self._advance(inputs)
        return self.observe()

    def run(self, duration_ms: int) -> Iterator[PopulationFrame]:
        """Yield sampled frames while continuing state for exactly duration_ms steps.

        Execution advances when the iterator is consumed. Every native step
        runs; output sampling uses absolute timestamps divisible by
        ``sample_every_ms``. No background queue or unbounded frame history is
        retained; a paused consumer pauses simulation. Partial iteration makes
        only the consumed progress. ``step`` always returns its native frame.
        """
        if (
            not isinstance(duration_ms, int)
            or isinstance(duration_ms, bool)
            or duration_ms < 0
        ):
            raise ValueError("duration_ms must be a nonnegative integer")
        for _ in range(duration_ms):
            self._advance()
            if (self.time_ms - 1) % self.sample_every_ms == 0:
                yield self.observe()

    @torch.inference_mode()
    def observe(
        self,
        *,
        neurons: Sequence[int] | torch.Tensor | None = None,
        signals: Sequence[str] | None = None,
    ) -> PopulationFrame:
        """Copy selected current signals without advancing state or time.

        Without arguments use the output subscriptions. With ``signals``,
        ``neurons`` selects shared row IDs (all neurons if omitted). Tensors
        remain device-resident and owned; callers choose when to transfer them.
        """
        if signals is None:
            if neurons is not None:
                raise ValueError("neurons requires an explicit signals selection")
            selection = self._subscriptions
        else:
            selection = self._validate_subscriptions(
                {name: "all" if neurons is None else neurons for name in signals}
            )
        if "soma_mv" in selection and self.soma_transform is None:
            raise ValueError("soma_mv requires an explicit validated soma_transform")
        values = {
            "spikes": self._spikes,
            "spike_logit": self._output[:, 0],
            "soma_target": self._output[:, 1],
            "hidden_state": self._state,
            "input_features": self._input_features,
        }
        result, identities = {}, {}
        # Export ordinary owned tensors, rather than inference tensors that
        # cannot be edited by a caller outside InferenceMode.
        with torch.inference_mode(False), torch.no_grad():
            for name, ids in selection.items():
                if name == "soma_mv":
                    scale, offset = self.soma_transform
                    tensor = self._output[:, 1].index_select(0, ids) * scale + offset
                else:
                    tensor = values[name].index_select(0, ids)
                result[name] = tensor.clone()
                identities[name] = ids.clone()
        return PopulationFrame(self.time_ms - 1, self._valid, result, identities)

    def _configuration_fingerprint(self):
        digest = hashlib.sha256()
        digest.update(
            repr(
                (
                    self.n,
                    self.backend,
                    self.input_encoding,
                    self.model.config,
                    self.soma_transform,
                    self.sample_every_ms,
                )
            ).encode()
        )
        tensors = dict(self.model.state_dict())
        tensors["morphology_indices"] = self._morphology_indices
        if isinstance(self._connectome, ExplicitConnectome):
            for name in ("sources", "targets", "channels", "delays", "source_roles"):
                tensors["graph_" + name] = getattr(self._connectome, name)
        else:
            graph = self._connectome
            digest.update(
                repr(
                    (
                        graph.n,
                        graph.out_degree,
                        graph.input_dim,
                        graph.delay_ms,
                        graph.seed,
                    )
                ).encode()
            )
            tensors["neuron_roles"] = graph.neuron_roles
            tensors["channel_roles"] = graph.channel_roles
        for name, tensor in sorted(tensors.items()):
            digest.update(name.encode())
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(
                tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            )
        return digest.hexdigest()

    def state_dict(self) -> dict[str, object]:
        """Return owned device tensors sufficient for queue/phase/state replay.

        Save with ``torch.save`` and load using ``torch.load(..., weights_only=True)``.
        Weights/topology are identified by a fingerprint, not copied into this
        runtime snapshot. Recreate the same model/graph/encoding/backend before
        loading. A callback must return the same input for a replayed native
        timestamp; its arbitrary external/random state is not serializable here.
        """
        names = (
            "state",
            "patch",
            "forecast",
            "queue",
            "input_features",
            "output",
            "spikes",
            "efficacies",
            "thresholds",
            "coefficients",
        )
        return {
            "version": 1,
            "fingerprint": self._fingerprint,
            "time_ms": self.time_ms,
            "valid": self._valid,
            "tensors": {name: getattr(self, "_" + name).clone() for name in names},
        }

    @torch.inference_mode()
    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore a compatible owned snapshot; reject mismatch before mutation."""
        if state.get("version") != 1 or state.get("fingerprint") != self._fingerprint:
            raise ValueError(
                "saved state model/topology/encoding/backend fingerprint does not match"
            )
        time = state.get("time_ms")
        if not isinstance(time, int) or isinstance(time, bool) or time < 0:
            raise ValueError("saved time_ms must be a nonnegative integer")
        if not isinstance(state.get("valid"), bool) or state["valid"] != (time > 4):
            raise ValueError(
                "saved validity does not match the processed native timestamp"
            )
        tensors = state.get("tensors")
        names = (
            "state",
            "patch",
            "forecast",
            "queue",
            "input_features",
            "output",
            "spikes",
            "efficacies",
            "thresholds",
            "coefficients",
        )
        if not isinstance(tensors, Mapping) or set(tensors) != set(names):
            raise ValueError("saved state has an incomplete runtime tensor schema")
        checked = {}
        for name in names:
            value, target = tensors[name], getattr(self, "_" + name)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != target.shape
                or value.dtype != target.dtype
            ):
                raise ValueError(f"saved {name} tensor shape/dtype does not match")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"saved {name} must be finite")
            if name == "efficacies" and (value < 0).any():
                raise ValueError("saved efficacies must be nonnegative")
            checked[name] = value.to(self.device)
        for name, value in checked.items():
            getattr(self, "_" + name).copy_(value)
        self.time_ms, self._valid = time, state["valid"]

    def reset(self) -> None:
        """Restore construction-time clock, state, empty queue, thresholds and efficacies."""
        self.load_state_dict(self._initial_state)


def create_population(
    *,
    model: AdaptiveSupportP4Surrogate,
    n: int,
    connectome: ExplicitConnectome | ProceduralConnectome,
    morphology_indices: Sequence[int] | torch.Tensor,
    input_encoding: Literal["signed", "channel"],
    channel_roles: Sequence[int] | torch.Tensor | None = None,
    stream_inputs: StreamInput | None = None,
    stream_outputs: Mapping[str, Sequence[int] | torch.Tensor | Literal["all"]]
    | None = None,
    backend: Literal["reference", "fused"] = "reference",
    spike_threshold: float | Sequence[float] | torch.Tensor = 0.0,
    sample_every_ms: int = 1,
    soma_transform: tuple[float, float] | None = None,
    runtime_coefficients: torch.Tensor | None = None,
) -> ConnectedPopulation:
    """Create an actually connected Lite inference simulation from supplied weights.

    ``input_encoding='signed'`` applies source-role sign to positive recurrent
    spikes. ``'channel'`` sends positive recurrent event counts because inhibitory
    channel identity/route features already encode inhibition. Choose the
    convention used to train the checkpoint; it cannot be inferred reliably
    from arbitrary learned route features. External values are always supplied
    directly in that native encoding. Channel roles, when supplied, validate
    source-role/channel compatibility rather than applying another sign.
    """
    return ConnectedPopulation(
        model=model,
        n=n,
        connectome=connectome,
        morphology_indices=morphology_indices,
        input_encoding=input_encoding,
        channel_roles=channel_roles,
        stream_inputs=stream_inputs,
        stream_outputs=stream_outputs,
        backend=backend,
        spike_threshold=spike_threshold,
        sample_every_ms=sample_every_ms,
        soma_transform=soma_transform,
        runtime_coefficients=runtime_coefficients,
    )
