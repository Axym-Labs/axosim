#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AXOBENCH_ROOT = PROJECT_ROOT.parent / "axobench"
sys.path.insert(0, str(AXOBENCH_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from axobench.generation.run_coreneuron_event_dropout_pair import (  # noqa: E402
    _build_source_site_route,
    _normalized_source_indices,
    _segment_route,
)
from axobench.generation.run_coreneuron_hay_probe import load_probe_mechanisms  # noqa: E402
from axobench.generation.run_hay_neuron_driven_probe import collect_segments  # noqa: E402
DATASET_ROOT = PROJECT_ROOT / "data" / "v1-neuronio-like-gpu-20260611"
MORPHOLOGY_ROOT = PROJECT_ROOT / "data" / "morphologies" / "allen_mouse_l5_spiny"
GENERATION_ROOT = AXOBENCH_ROOT / "artifacts" / "generation"
MECHANISM_ROOT = GENERATION_ROOT / "hay_swc_template"
SOURCE_ROOT = GENERATION_ROOT / "source_models" / "neuron_as_deep_net" / "L5PC_NEURON_simulation"
TEMPLATE_HOC = MECHANISM_ROOT / "L5PCtemplate_swc.hoc"
OUTPUT_ROOT = PROJECT_ROOT / "docs" / "axomamba-autoresearch" / "artifacts" / "routes"
FEATURE_NAMES = (
    "sign",
    "is_apical",
    "is_basal",
    "source_arclength",
    "target_used",
    "target_multiplicity",
    "target_arclength",
    "target_length",
    "arclength_mismatch",
)


def main() -> int:
    morphology_ids = _morphology_ids()
    h = _prepare_neuron()
    source_route = _build_source_site_route(h, SOURCE_ROOT)
    source_sites = int(source_route["types"].shape[0])
    if source_sites * 2 != 1278:
        raise ValueError(f"expected 639 source sites per sign, got {source_sites}")

    feature_rows = []
    routes: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for morphology_id in morphology_ids:
        cell = h.L5PCtemplate(str(MORPHOLOGY_ROOT / f"{morphology_id}.swc"))
        try:
            _, segment_types, segment_lengths = collect_segments(cell)
            target_route = _segment_route(segment_types, segment_lengths)
            source_indices = _normalized_source_indices(segment_types, segment_lengths, source_route)
        finally:
            del cell
        routes[morphology_id] = source_indices
        summaries[morphology_id] = _route_summary(
            source_indices, source_sites, segment_types
        )
        feature_rows.append(
            _source_features(
                source_route,
                target_route,
                source_indices,
                segment_lengths,
            )
        )

    metadata_audit = _audit_metadata(summaries)
    features = np.stack(feature_rows).astype(np.float32, copy=False)
    shuffled_features = _shuffled_features(features, seed=20260758)
    constant_features = features.copy()
    constant_features[..., 3:] = 0.0
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    artifact_path = OUTPUT_ROOT / "v1-normalized-route-features.npz"
    arrays: dict[str, np.ndarray] = {
        "features": features,
        "features_shuffled": shuffled_features,
        "features_constant": constant_features,
        "morphology_ids": np.asarray(morphology_ids),
        "feature_names": np.asarray(FEATURE_NAMES),
        "source_types": np.asarray(source_route["types"]),
        "source_positions": np.asarray(source_route["positions"], dtype=np.float32),
    }
    for morphology_id, source_indices in routes.items():
        arrays[f"route_{morphology_id}"] = source_indices.astype(np.int64, copy=False)
    np.savez_compressed(artifact_path, **arrays)
    feature_ablation_path = OUTPUT_ROOT / "v1-coordinate-feature-ablation.npz"
    np.savez_compressed(
        feature_ablation_path,
        morphology_ids=np.asarray(morphology_ids),
        features=features,
        features_shuffled=shuffled_features,
        features_occupancy=_feature_subset(features, keep=(0, 1, 2, 4, 5, 7)),
        features_geometry=_feature_subset(features, keep=(0, 1, 2, 3, 6, 8)),
        features_validity=_feature_subset(features, keep=(0, 1, 2, 4)),
    )
    report = {
        "artifact": str(artifact_path.resolve()),
        "feature_ablation_artifact": str(feature_ablation_path.resolve()),
        "dataset_root": str(DATASET_ROOT.resolve()),
        "feature_names": list(FEATURE_NAMES),
        "feature_shape": list(features.shape),
        "morphology_ids": morphology_ids,
        "source_count_by_type": source_route["count_by_type"],
        "source_sites_per_sign": source_sites,
        "summaries": summaries,
        "metadata_audit": metadata_audit,
        "status": "pass",
    }
    report_path = OUTPUT_ROOT / "v1-normalized-route-verification.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _morphology_ids() -> list[str]:
    values = json.loads((PROJECT_ROOT / "configs" / "axomamba.json").read_text(encoding="utf-8"))
    return [str(value) for value in values["morphology_ids"]]


def _prepare_neuron():
    from neuron import h

    os.chdir(MECHANISM_ROOT)
    load_probe_mechanisms(h, MECHANISM_ROOT)
    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    h.load_file(str(SOURCE_ROOT / "L5PCbiophys5b.hoc"))
    h.load_file(str(TEMPLATE_HOC))
    return h


def _route_summary(
    source_indices: np.ndarray,
    source_sites: int,
    segment_types: np.ndarray,
) -> dict[str, Any]:
    unique = int(np.unique(source_indices).size)
    return {
        "morphology_segments": int(source_indices.size),
        "unique_source_sites_per_sign": unique,
        "duplicated_source_sites_per_sign": int(source_indices.size - unique),
        "unused_source_sites_per_sign": int(source_sites - unique),
        "target_count_by_type": {
            str(label): int(np.count_nonzero(segment_types == label))
            for label in np.unique(segment_types)
        },
    }


def _source_features(
    source_route: dict[str, Any],
    target_route: dict[str, Any],
    source_indices: np.ndarray,
    target_lengths: np.ndarray,
) -> np.ndarray:
    source_sites = int(source_route["types"].shape[0])
    counts = np.bincount(source_indices, minlength=source_sites).astype(np.float64)
    target_positions = np.asarray(target_route["positions"], dtype=np.float64)
    target_lengths = np.asarray(target_lengths, dtype=np.float64)
    position_sum = np.bincount(source_indices, weights=target_positions, minlength=source_sites)
    length_sum = np.bincount(source_indices, weights=target_lengths, minlength=source_sites)
    used = counts > 0
    mean_position = np.asarray(source_route["positions"], dtype=np.float64).copy()
    mean_length = np.zeros(source_sites, dtype=np.float64)
    mean_position[used] = position_sum[used] / counts[used]
    mean_length[used] = length_sum[used] / counts[used]
    max_count = max(float(counts.max()), 1.0)
    max_length = max(float(mean_length.max()), 1e-12)
    source_types = np.asarray(source_route["types"], dtype=str)
    source_positions = np.asarray(source_route["positions"], dtype=np.float64)
    unsigned = np.stack(
        [
            (source_types == "apical").astype(np.float64),
            (source_types == "basal").astype(np.float64),
            source_positions,
            used.astype(np.float64),
            np.log1p(counts) / math.log1p(max_count),
            mean_position,
            np.log1p(mean_length) / math.log1p(max_length),
            np.abs(mean_position - source_positions),
        ],
        axis=-1,
    )
    excitatory = np.concatenate([np.ones((source_sites, 1)), unsigned], axis=-1)
    inhibitory = np.concatenate([-np.ones((source_sites, 1)), unsigned], axis=-1)
    return np.concatenate([excitatory, inhibitory], axis=0)


def _feature_subset(features: np.ndarray, *, keep: tuple[int, ...]) -> np.ndarray:
    result = np.zeros_like(features)
    result[..., list(keep)] = features[..., list(keep)]
    return result


def _audit_metadata(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    by_morphology = {key: 0 for key in summaries}
    keys = (
        "morphology_segments",
        "unique_source_sites_per_sign",
        "duplicated_source_sites_per_sign",
        "unused_source_sites_per_sign",
    )
    for path in DATASET_ROOT.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        morphology_id = payload.get("morphology_id")
        routing = payload.get("input_routing")
        if morphology_id not in summaries or not isinstance(routing, dict) or routing.get("status") != "recorded":
            continue
        expected = summaries[morphology_id]
        for key in keys:
            recorded = routing[key]
            values = [recorded[field] for field in ("min", "mean", "max")]
            if not all(float(value) == float(expected[key]) for value in values):
                raise ValueError(f"routing mismatch for {morphology_id} {key} in {path}")
        checked += 1
        by_morphology[morphology_id] += 1
    if not checked or any(count == 0 for count in by_morphology.values()):
        raise ValueError(f"incomplete routing metadata audit: {by_morphology}")
    return {"checked_files": checked, "files_by_morphology": by_morphology}


def _shuffled_features(features: np.ndarray, *, seed: int) -> np.ndarray:
    result = np.array(features, copy=True)
    rng = np.random.default_rng(seed)
    for morphology_index in range(result.shape[0]):
        for sign in (-1.0, 1.0):
            for type_column in (1, 2):
                indices = np.flatnonzero(
                    (result[morphology_index, :, 0] == sign)
                    & (result[morphology_index, :, type_column] == 1.0)
                )
                permutation = rng.permutation(indices)
                result[morphology_index, indices, 3:] = result[
                    morphology_index,
                    permutation,
                    3:,
                ]
    return result


if __name__ == "__main__":
    raise SystemExit(main())
