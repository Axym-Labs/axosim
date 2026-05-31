"""M3: Morphology Transfer Evaluation.

Tests generalization to unseen dendritic morphologies from Allen Cell Types Database.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from bnn_sim.benchmark.swc_utils import load_swc, select_branches, create_routing_from_branches


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
        branches = select_branches(swc_data, n_branches=n_branches)
        routing = create_routing_from_branches(swc_data, branches)

        result = {
            "swc_path": str(swc_path),
            "n_nodes": len(swc_data),
            "n_branches": len(branches),
            "branches": branches,
            "routing_matrix": routing,
            "routing_shape": routing.shape,
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
                "rmse_mv": float(rmse_mv),
                "auc": float(auc),
                "n_samples": inputs.shape[0],
            })

        return results


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
