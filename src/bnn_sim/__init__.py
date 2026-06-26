"""Compatibility package for the former ``bnn_sim`` import path.

New code should import ``axosim``. This shim keeps older scripts importable
while the project transition settles.
"""

from __future__ import annotations

from importlib import import_module

_axosim = import_module("axosim")
__path__ = _axosim.__path__
__all__ = getattr(_axosim, "__all__", ())


def __getattr__(name: str):
    return getattr(_axosim, name)
