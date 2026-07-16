from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
import torch.nn.functional as F


def average_repeated_module_state(
    state: Mapping[str, torch.Tensor],
    *,
    prefix_groups: list[list[str]],
) -> dict[str, torch.Tensor]:
    """Initialize tied modules with the mean of their untied source tensors."""
    averaged = dict(state)
    for prefixes in prefix_groups:
        if len(prefixes) < 2:
            raise ValueError("a repeated-module group requires at least two prefixes")
        suffixes = [
            name[len(prefixes[0]) :]
            for name in state
            if name == prefixes[0] or name.startswith(f"{prefixes[0]}.")
        ]
        if not suffixes:
            raise ValueError(f"state contains no tensors below {prefixes[0]!r}")
        for suffix in suffixes:
            names = [f"{prefix}{suffix}" for prefix in prefixes]
            if any(name not in state for name in names):
                raise ValueError(f"repeated modules do not share tensor suffix {suffix!r}")
            values = [state[name] for name in names]
            if len({tuple(value.shape) for value in values}) != 1:
                raise ValueError(f"repeated tensor shapes differ for suffix {suffix!r}")
            if values[0].is_floating_point() or values[0].is_complex():
                mean = torch.stack(values).mean(dim=0)
            else:
                if any(not torch.equal(values[0], value) for value in values[1:]):
                    raise ValueError(f"non-floating repeated tensors differ for {suffix!r}")
                mean = values[0]
            for name in names:
                averaged[name] = mean.clone()
    return averaged


def share_morphology_synapse_gain_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Initialize one shared log-gain bank with the morphology-wise mean."""
    name = "morphology_synapse_log_gain"
    if name not in state:
        raise ValueError(f"state contains no {name!r}")
    gains = state[name]
    if gains.ndim != 3 or gains.shape[0] < 2:
        raise ValueError("morphology synapse gains must contain multiple banks")
    shared = dict(state)
    shared[name] = gains.mean(dim=0, keepdim=True)
    return shared


@torch.no_grad()
def initialize_shared_tcn_adapters_from_state(
    model: nn.Module,
    untied_state: Mapping[str, torch.Tensor],
) -> list[str]:
    """Factor each untied TCN pointwise residual around the shared base map."""
    initialized = []
    for index, block in enumerate(getattr(model, "local_tcn_blocks", ())):
        adapter_in = getattr(block, "pointwise_adapter_in", None)
        adapter_out = getattr(block, "pointwise_adapter_out", None)
        if adapter_in is None and adapter_out is None:
            continue
        if not isinstance(adapter_in, nn.Conv1d) or not isinstance(adapter_out, nn.Conv1d):
            raise ValueError("local TCN pointwise adapter modules are incomplete")
        prefix = f"local_tcn_blocks.{index}.pointwise"
        weight_name = f"{prefix}.weight"
        bias_name = f"{prefix}.bias"
        if weight_name not in untied_state or bias_name not in untied_state:
            raise ValueError(f"untied state is missing {prefix!r}")
        source_weight = untied_state[weight_name].flatten(1).float()
        base_weight = block.pointwise.weight.detach().flatten(1).float()
        residual = source_weight - base_weight
        u, singular, vh = torch.linalg.svd(residual, full_matrices=False)
        rank = adapter_in.out_channels
        scale = singular[:rank].clamp_min(0.0).sqrt()
        adapter_out.weight.copy_(
            (u[:, :rank] * scale).to(
                device=adapter_out.weight.device,
                dtype=adapter_out.weight.dtype,
            ).unsqueeze(-1)
        )
        adapter_in.weight.copy_(
            (scale.unsqueeze(1) * vh[:rank]).to(
                device=adapter_in.weight.device,
                dtype=adapter_in.weight.dtype,
            ).unsqueeze(-1)
        )
        assert adapter_out.bias is not None and block.pointwise.bias is not None
        adapter_out.bias.copy_(
            (
                untied_state[bias_name].float()
                - block.pointwise.bias.detach().float()
            ).to(device=adapter_out.bias.device, dtype=adapter_out.bias.dtype)
        )
        initialized.append(prefix)
    if not initialized:
        raise ValueError("model contains no local TCN pointwise adapters")
    return initialized


def initialize_shared_tcn_adapters_from_pretrained_state(
    model: nn.Module,
    pretrained_state: Mapping[str, torch.Tensor],
) -> list[str]:
    """Preserve compatible adapters, or derive them from untied dense maps."""
    model_state = model.state_dict()
    adapter_names = sorted(
        name for name in model_state if "pointwise_adapter" in name
    )
    if not adapter_names:
        return []
    if all(
        name in pretrained_state
        and pretrained_state[name].shape == model_state[name].shape
        for name in adapter_names
    ):
        return []

    dense_names = [
        f"local_tcn_blocks.{index}.pointwise.{suffix}"
        for index, block in enumerate(getattr(model, "local_tcn_blocks", ()))
        if getattr(block, "pointwise_adapter_in", None) is not None
        for suffix in ("weight", "bias")
    ]
    if dense_names and all(name in pretrained_state for name in dense_names):
        return initialize_shared_tcn_adapters_from_state(model, pretrained_state)
    missing = sorted(
        name
        for name in (*adapter_names, *dense_names)
        if name not in pretrained_state
    )
    raise ValueError(
        "pretrained state has neither compatible TCN adapters nor complete "
        f"untied pointwise maps; missing {missing}"
    )


class FactorizedLinear(nn.Module):
    """Linear map represented by a trainable truncated matrix factorization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        bias: bool,
        source_weight_name: str,
    ) -> None:
        super().__init__()
        if not 0 < rank <= min(in_features, out_features):
            raise ValueError("factorization rank must fit the linear dimensions")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.source_weight_name = source_weight_name
        self.left = nn.Parameter(torch.empty(out_features, rank))
        self.right = nn.Parameter(torch.empty(rank, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.left, a=5**0.5)
        nn.init.kaiming_uniform_(self.right, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @property
    def weight(self) -> torch.Tensor:
        return self.left @ self.right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int,
        source_weight_name: str,
    ) -> FactorizedLinear:
        factorized = cls(
            linear.in_features,
            linear.out_features,
            rank,
            bias=linear.bias is not None,
            source_weight_name=source_weight_name,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        factorized.set_from_weight(linear.weight.detach())
        if linear.bias is not None:
            assert factorized.bias is not None
            factorized.bias.data.copy_(linear.bias.detach())
        return factorized

    @torch.no_grad()
    def set_from_weight(self, weight: torch.Tensor) -> None:
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError("dense source weight has the wrong shape")
        source = weight.to(device=self.left.device, dtype=torch.float32)
        u, singular, vh = torch.linalg.svd(source, full_matrices=False)
        scale = singular[: self.rank].clamp_min(0.0).sqrt()
        self.left.copy_((u[:, : self.rank] * scale).to(dtype=self.left.dtype))
        self.right.copy_((scale.unsqueeze(1) * vh[: self.rank]).to(dtype=self.right.dtype))


class FactorizedPointwiseConv1d(nn.Module):
    """A 1x1 convolution represented by two trainable width factors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int,
        *,
        bias: bool,
        source_weight_name: str,
    ) -> None:
        super().__init__()
        if not 0 < rank <= min(in_channels, out_channels):
            raise ValueError("factorization rank must fit the convolution channels")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.rank = int(rank)
        self.source_weight_name = source_weight_name
        self.right = nn.Conv1d(in_channels, rank, kernel_size=1, bias=False)
        self.left = nn.Conv1d(rank, out_channels, kernel_size=1, bias=bias)

    @property
    def weight(self) -> torch.Tensor:
        matrix = self.left.weight.flatten(1) @ self.right.weight.flatten(1)
        return matrix.unsqueeze(-1)

    @property
    def bias(self) -> torch.Tensor | None:
        return self.left.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.left(self.right(x))

    @classmethod
    def from_conv(
        cls,
        convolution: nn.Conv1d,
        *,
        rank: int,
        source_weight_name: str,
    ) -> FactorizedPointwiseConv1d:
        if convolution.kernel_size != (1,) or convolution.groups != 1:
            raise ValueError("pointwise factorization requires an ungrouped 1x1 convolution")
        factorized = cls(
            convolution.in_channels,
            convolution.out_channels,
            rank,
            bias=convolution.bias is not None,
            source_weight_name=source_weight_name,
        ).to(device=convolution.weight.device, dtype=convolution.weight.dtype)
        factorized.set_from_weight(convolution.weight.detach())
        if convolution.bias is not None:
            assert factorized.bias is not None
            factorized.bias.data.copy_(convolution.bias.detach())
        return factorized

    @torch.no_grad()
    def set_from_weight(self, weight: torch.Tensor) -> None:
        expected = (self.out_channels, self.in_channels, 1)
        if tuple(weight.shape) != expected:
            raise ValueError("dense pointwise source weight has the wrong shape")
        source = weight.flatten(1).to(
            device=self.left.weight.device,
            dtype=torch.float32,
        )
        u, singular, vh = torch.linalg.svd(source, full_matrices=False)
        scale = singular[: self.rank].clamp_min(0.0).sqrt()
        self.left.weight.copy_(
            (u[:, : self.rank] * scale)
            .to(dtype=self.left.weight.dtype)
            .unsqueeze(-1)
        )
        self.right.weight.copy_(
            (scale.unsqueeze(1) * vh[: self.rank])
            .to(dtype=self.right.weight.dtype)
            .unsqueeze(-1)
        )


def initialize_factorized_linears_from_state(
    model: nn.Module,
    dense_state: Mapping[str, torch.Tensor],
    *,
    compatible_only: bool = False,
) -> list[str]:
    """Initialize every factorized map from its named dense checkpoint tensor."""
    initialized = []
    for module in model.modules():
        if not isinstance(module, (FactorizedLinear, FactorizedPointwiseConv1d)):
            continue
        if module.source_weight_name not in dense_state:
            if compatible_only:
                continue
            raise ValueError(
                f"dense checkpoint has no {module.source_weight_name!r} for factorization"
            )
        source = dense_state[module.source_weight_name]
        expected_shape = (
            (module.out_features, module.in_features)
            if isinstance(module, FactorizedLinear)
            else (module.out_channels, module.in_channels, 1)
        )
        if tuple(source.shape) != expected_shape:
            if compatible_only:
                continue
            raise ValueError(
                f"dense checkpoint tensor {module.source_weight_name!r} has the wrong shape"
            )
        module.set_from_weight(source)
        initialized.append(module.source_weight_name)
    return sorted(initialized)


def balanced_proxy_compress_mamba_state(
    dense_state: Mapping[str, torch.Tensor],
    *,
    source_state_dim: int,
    target_state_dim: int,
) -> tuple[dict[str, torch.Tensor], dict[str, list[int]]]:
    """Retain Mamba states with high diagonal-system controllability/observability.

    Mamba's input-dependent B and C vectors are produced by rows of ``x_proj``.
    Their row norms proxy input/output strength; ``exp(-A_log)`` favors states
    with longer continuous-time memory. The product is the diagonal analogue of
    a balanced-truncation score, not an exact Hankel singular value.
    """
    if source_state_dim <= 0 or not 0 < target_state_dim < source_state_dim:
        raise ValueError("target_state_dim must be between zero and source_state_dim")
    compressed = dict(dense_state)
    selections: dict[str, list[int]] = {}
    suffix = ".A_log"
    for a_name, a_log in dense_state.items():
        if not a_name.endswith(suffix):
            continue
        prefix = a_name[: -len(suffix)]
        x_name = f"{prefix}.x_proj.weight"
        if x_name not in dense_state:
            raise ValueError(f"Mamba state tensor {a_name!r} has no matching x_proj")
        x_weight = dense_state[x_name]
        if a_log.ndim != 2 or a_log.shape[1] != source_state_dim:
            raise ValueError(f"{a_name!r} does not match source_state_dim")
        dt_rank = x_weight.shape[0] - 2 * source_state_dim
        if x_weight.ndim != 2 or dt_rank <= 0:
            raise ValueError(f"{x_name!r} has an invalid Mamba projection shape")
        b_rows = x_weight[dt_rank : dt_rank + source_state_dim].float()
        c_rows = x_weight[dt_rank + source_state_dim :].float()
        timescale = torch.exp(-a_log.float()).mean(dim=0)
        score = b_rows.norm(dim=1) * c_rows.norm(dim=1) * timescale
        selected = torch.topk(score, k=target_state_dim, largest=True).indices.sort().values
        selections[prefix] = [int(index) for index in selected.tolist()]
        compressed[a_name] = a_log.index_select(1, selected.to(a_log.device))
        row_indices = torch.cat(
            [
                torch.arange(dt_rank, device=x_weight.device),
                selected.to(x_weight.device) + dt_rank,
                selected.to(x_weight.device) + dt_rank + source_state_dim,
            ]
        )
        compressed[x_name] = x_weight.index_select(0, row_indices)
    if not selections:
        raise ValueError("dense_state contains no Mamba state tensors")
    return compressed, selections
