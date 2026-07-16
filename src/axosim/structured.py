from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
import torch.nn.functional as F


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


def initialize_factorized_linears_from_state(
    model: nn.Module,
    dense_state: Mapping[str, torch.Tensor],
    *,
    compatible_only: bool = False,
) -> list[str]:
    """Initialize every factorized map from its named dense checkpoint tensor."""
    initialized = []
    for module in model.modules():
        if not isinstance(module, FactorizedLinear):
            continue
        if module.source_weight_name not in dense_state:
            if compatible_only:
                continue
            raise ValueError(
                f"dense checkpoint has no {module.source_weight_name!r} for factorization"
            )
        source = dense_state[module.source_weight_name]
        if tuple(source.shape) != (module.out_features, module.in_features):
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
