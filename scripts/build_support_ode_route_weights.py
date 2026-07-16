#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "docs/axomamba-autoresearch/artifacts/routes/v1-i140-feature-gate.npz"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "docs/axomamba-autoresearch/artifacts/routes/v1-support-ode-route-weights.npz"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build exact source-to-target multiplicity weights for support ODEs."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with np.load(args.source, allow_pickle=False) as artifact:
        morphology_ids = np.asarray(artifact["morphology_ids"])
        real = _decode_multiplicity(artifact["features_full"][..., 5])
        shuffled = _decode_multiplicity(
            artifact["features_occupancy_shuffled"][..., 5]
        )
    summary = {
        "source": str(args.source.resolve()),
        "feature": "target multiplicity",
        "values": [0, 1, 2],
        "shape": list(real.shape),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        morphology_ids=morphology_ids,
        multiplicity=real,
        multiplicity_shuffled=shuffled,
        metadata_json=np.asarray(json.dumps(summary, sort_keys=True)),
    )
    print(json.dumps({**summary, "output": str(args.output.resolve())}, indent=2))
    return 0


def _decode_multiplicity(normalized: np.ndarray) -> np.ndarray:
    values = np.expm1(np.asarray(normalized, dtype=np.float64) * np.log1p(2.0))
    decoded = np.rint(values)
    if not np.allclose(values, decoded, atol=1e-5) or not np.isin(decoded, [0, 1, 2]).all():
        raise ValueError("route feature does not encode exact 0/1/2 multiplicity")
    return decoded.astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
