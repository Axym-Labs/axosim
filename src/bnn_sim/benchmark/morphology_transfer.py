"""M3: Morphology Transfer Evaluation.

Tests generalization to unseen dendritic morphologies from Allen Cell Types Database.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from bnn_sim.benchmark.swc_utils import (
    create_routing_from_branches,
    extract_branches,
    load_swc,
    select_branches,
    summarize_morphology,
)


class MorphologyTransferEvaluator:
    """Evaluator for morphology transfer (M3).

    Downloads and preprocesses Allen Cell Types morphologies,
    then evaluates models trained on one morphology against others.
    """

    def __init__(self, cache_dir: str | Path = "data/morphologies"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._morphology_cache: dict[str, dict[str, Any]] = {}

    def load_morphology(self, swc_path: str | Path, n_branches: int = 45) -> dict[str, Any]:
        """Load and preprocess a morphology.

        Args:
            swc_path: Path to SWC file.
            n_branches: Number of branches to select.

        Returns:
            Dictionary with morphology info and routing matrix.
        """
        swc_path = Path(swc_path)
        cache_key = f"{swc_path.stem}_{n_branches}"

        if cache_key in self._morphology_cache:
            return self._morphology_cache[cache_key]

        swc_data = load_swc(swc_path)
        all_branches = extract_branches(swc_data, target_types=[3, 4])
        branches = select_branches(swc_data, n_branches=n_branches)
        routing = create_routing_from_branches(swc_data, branches)
        morphology_features = summarize_morphology(swc_data, all_branches if all_branches else branches)
        routing_features = summarize_morphology(swc_data, branches)

        result = {
            "swc_path": str(swc_path),
            "n_nodes": len(swc_data),
            "n_branches": len(branches),
            "branches": branches,
            "routing_matrix": routing,
            "routing_shape": routing.shape,
            "features": morphology_features,
            "morphology_features": morphology_features,
            "routing_features": routing_features,
        }

        self._morphology_cache[cache_key] = result
        return result

    def evaluate_transfer(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        train_morphology: dict[str, Any],
        test_morphologies: list[dict[str, Any]],
        test_inputs: list[np.ndarray],
        test_targets: list[np.ndarray],
    ) -> dict[str, Any]:
        """Evaluate model transfer to unseen morphologies.

        For surrogate models, "transfer" means the model was trained with one
        branch routing matrix and is tested with another. Since the model
        architecture depends on the routing, true zero-shot transfer requires
        the model to handle variable branch counts or use a branch-agnostic
        architecture.

        For now, this metric documents the routing mismatch and measures
        performance degradation.

        Args:
            model_fn: Model prediction function.
            train_morphology: Training morphology info.
            test_morphologies: List of test morphology info.
            test_inputs: List of test input arrays (one per morphology).
            test_targets: List of test target arrays.

        Returns:
            Dictionary with transfer metrics.
        """
        results = {
            "train_morphology": train_morphology["swc_path"],
            "train_n_branches": train_morphology["n_branches"],
            "test_results": [],
        }

        for test_morph, inputs, targets in zip(test_morphologies, test_inputs, test_targets):
            # Check routing compatibility
            train_routing = train_morphology["routing_matrix"]
            test_routing = test_morph["routing_matrix"]

            routing_compatible = (
                train_routing.shape == test_routing.shape and
                np.allclose(train_routing, test_routing)
            )
            route_mismatch = _route_mismatch_summary(train_morphology, test_morph)

            # Evaluate model
            preds = model_fn(inputs)

            # Compute metrics
            rmse_mv = np.sqrt(np.mean((preds[:, :, 1] - targets[:, :, 1]) ** 2)) / 0.1

            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(targets[:, :, 0].flatten(), preds[:, :, 0].flatten())
            except ValueError:
                auc = 0.5  # All same class

            results["test_results"].append({
                "test_morphology": test_morph["swc_path"],
                "test_n_branches": test_morph["n_branches"],
                "routing_compatible": bool(routing_compatible),
                "routing_shape_match": train_routing.shape == test_routing.shape,
                "morphology_features": test_morph["morphology_features"],
                "routing_features": test_morph["routing_features"],
                "route_mismatch": route_mismatch,
                "rmse_mv": float(rmse_mv),
                "auc": float(auc),
                "n_samples": inputs.shape[0],
            })

        return results

    def compare_morphology_set(
        self,
        train_morphology: dict[str, Any],
        test_morphologies: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compare held-out morphology features without model predictions."""
        return {
            "status": "morphology_feature_comparison",
            "train_morphology": train_morphology["swc_path"],
            "train_features": train_morphology["morphology_features"],
            "train_routing_features": train_morphology["routing_features"],
            "test_morphologies": [
                {
                    "swc_path": morph["swc_path"],
                    "features": morph["morphology_features"],
                    "routing_features": morph["routing_features"],
                    "route_mismatch": _route_mismatch_summary(train_morphology, morph),
                }
                for morph in test_morphologies
            ],
            "what_failure_means": (
                "Large feature/routing mismatch means a later transfer score "
                "mixes model generalization with morphology distribution shift; "
                "same-cell-type transfer should control these axes where possible."
            ),
        }


def _route_mismatch_summary(train_morphology: dict[str, Any], test_morphology: dict[str, Any]) -> dict[str, Any]:
    train_routing = train_morphology["routing_matrix"]
    test_routing = test_morphology["routing_matrix"]
    shape_match = train_routing.shape == test_routing.shape
    train_features = train_morphology["morphology_features"]
    test_features = test_morphology["morphology_features"]
    train_routing_features = train_morphology["routing_features"]
    test_routing_features = test_morphology["routing_features"]
    summary = {
        "routing_shape_match": bool(shape_match),
        "train_routing_shape": tuple(int(x) for x in train_routing.shape),
        "test_routing_shape": tuple(int(x) for x in test_routing.shape),
        "morphology_branch_count_delta": int(test_features["n_branches"] - train_features["n_branches"]),
        "routing_branch_count_delta": int(
            test_routing_features["n_branches"] - train_routing_features["n_branches"]
        ),
        "total_length_ratio": _safe_ratio(
            test_features["total_dendritic_length_um"],
            train_features["total_dendritic_length_um"],
        ),
        "routed_length_ratio": _safe_ratio(
            test_routing_features["total_dendritic_length_um"],
            train_routing_features["total_dendritic_length_um"],
        ),
        "apical_fraction_delta": _safe_delta(
            test_features["apical_length_fraction"],
            train_features["apical_length_fraction"],
        ),
        "max_radial_extent_ratio": _safe_ratio(
            test_features["max_radial_extent_um"],
            train_features["max_radial_extent_um"],
        ),
    }
    if shape_match:
        summary["routing_l1_mean"] = float(np.mean(np.abs(test_routing - train_routing)))
    else:
        summary["routing_l1_mean"] = None
    return summary


def _safe_ratio(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference == 0:
        return None
    return float(value / reference)


def _safe_delta(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    return float(value - reference)


def load_available_morphologies(cache_dir: str = "data/morphologies") -> list[Path]:
    """Find all SWC files in cache directory."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []
    return sorted(cache_dir.glob("*.swc"))


if __name__ == "__main__":
    # Test morphology loading
    swc_files = load_available_morphologies()
    print(f"Found {len(swc_files)} morphologies")

    evaluator = MorphologyTransferEvaluator()

    for swc_path in swc_files:
        morph = evaluator.load_morphology(swc_path, n_branches=45)
        print(f"\n{swc_path.name}:")
        print(f"  Nodes: {morph['n_nodes']}")
        print(f"  Branches: {morph['n_branches']}")
        print(f"  Routing shape: {morph['routing_shape']}")
