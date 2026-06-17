"""NeuronIO training and evaluation utilities for ELM / Branch-ELM experiments.

Keep the package import lightweight so data/benchmark utilities can run in
environments that intentionally omit PyTorch, e.g. TensorFlow-only TCN checks.
"""

from __future__ import annotations

from typing import Any

__all__ = ["BranchELM", "BranchELMConfig"]


def __getattr__(name: str) -> Any:
    if name in {"BranchELM", "BranchELMConfig"}:
        from bnn_sim.model import BranchELM, BranchELMConfig

        return {"BranchELM": BranchELM, "BranchELMConfig": BranchELMConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
