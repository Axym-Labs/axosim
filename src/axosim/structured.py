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
) -> list[str]:
    """Initialize every factorized map from its named dense checkpoint tensor."""
    initialized = []
    for module in model.modules():
        if not isinstance(module, FactorizedLinear):
            continue
        if module.source_weight_name not in dense_state:
            raise ValueError(
                f"dense checkpoint has no {module.source_weight_name!r} for factorization"
            )
        module.set_from_weight(dense_state[module.source_weight_name])
        initialized.append(module.source_weight_name)
    return sorted(initialized)
