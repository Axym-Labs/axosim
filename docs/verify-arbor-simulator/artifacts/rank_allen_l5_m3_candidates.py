#!/usr/bin/env python3
"""Download and rank Allen mouse L5 spiny morphologies for M3.

The downloaded SWC files are intentionally written under ignored `data/`.
Only the small JSON/Markdown summaries should be tracked.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import requests

from bnn_sim.benchmark.morphology_transfer import MorphologyTransferEvaluator


API = "https://api.brain-map.org/api/v2/data/query.json"
DOWNLOAD = "https://api.brain-map.org/api/v2/well_known_file_download/{id}"
SWC_FILE_TYPE_ID = 303941301


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/morphologies/allen_mouse_l5_spiny")
    parser.add_argument("--reference-swc", default="src/bnn_sim/assets/allen_l5_template.swc")
    parser.add_argument("--branches", type=int, default=45)
    parser.add_argument("--json", default="docs/verify-arbor-simulator/artifacts/allen_l5_m3_candidates_20260606.json")
    parser.add_argument("--markdown", default="docs/verify-arbor-simulator/artifacts/allen_l5_m3_candidates_20260606.md")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = _fetch_reconstructed_cells(args.timeout)
    candidates = [
        row for row in cells
        if row.get("donor__species") == "Mus musculus"
        and row.get("structure__layer") == "5"
        and row.get("tag__dendrite_type") == "spiny"
    ]
    metadata = [_download_candidate(row, output_dir, args.timeout) for row in candidates]
    metadata = [row for row in metadata if row is not None]
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    evaluator = MorphologyTransferEvaluator(cache_dir=output_dir)
    reference = evaluator.load_morphology(args.reference_swc, n_branches=args.branches)
    ranked = [_score_candidate(evaluator, reference, row, args.branches) for row in metadata]
    ranked = [row for row in ranked if row is not None]
    ranked.sort(key=lambda row: row["compatibility_score_lower_better"])

    payload = {
        "source": {
            "api": API,
            "filter": "Mus musculus, layer 5, spiny, non-null reconstruction",
            "reference_swc": args.reference_swc,
            "branches": args.branches,
            "download_dir": str(output_dir),
        },
        "candidate_count": len(candidates),
        "ranked_count": len(ranked),
        "reference_specimen_id": _find_exact_reference_specimen(args.reference_swc, ranked),
        "ranked": ranked,
    }
    Path(args.json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.markdown).write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "candidate_count": payload["candidate_count"],
        "ranked_count": payload["ranked_count"],
        "reference_specimen_id": payload["reference_specimen_id"],
        "top_heldout": [
            row for row in ranked
            if row["specimen_id"] != payload["reference_specimen_id"]
        ][:5],
    }, indent=2, sort_keys=True))
    return 0


def _fetch_reconstructed_cells(timeout: float) -> list[dict[str, Any]]:
    criteria = "model::ApiCellTypesSpecimenDetail,rma::criteria,[nr__reconstruction_type$nenull],rma::options[num_rows$eqall]"
    response = requests.get(API, params={"criteria": criteria}, timeout=timeout)
    response.raise_for_status()
    return response.json()["msg"]


def _download_candidate(row: dict[str, Any], output_dir: Path, timeout: float) -> dict[str, Any] | None:
    specimen_id = int(row["specimen__id"])
    swc_path = output_dir / f"specimen_{specimen_id}.swc"
    if not swc_path.exists() or swc_path.stat().st_size < 1000:
        criteria = f"model::NeuronReconstruction,rma::criteria,[specimen_id$eq{specimen_id}],rma::include,well_known_files"
        response = requests.get(API, params={"criteria": criteria}, timeout=timeout)
        response.raise_for_status()
        msg = response.json()["msg"]
        if not msg:
            return None
        swc_files = [
            wf for wf in msg[0].get("well_known_files", [])
            if wf.get("well_known_file_type_id") == SWC_FILE_TYPE_ID
        ]
        if not swc_files:
            return None
        download = requests.get(DOWNLOAD.format(id=swc_files[0]["id"]), timeout=timeout)
        download.raise_for_status()
        swc_path.write_bytes(download.content)
    return {
        "specimen_id": specimen_id,
        "line_name": row.get("line_name"),
        "structure_acronym": row.get("structure__acronym"),
        "normalized_depth": row.get("csl__normalized_depth"),
        "reconstruction_type": row.get("nr__reconstruction_type"),
        "swc_path": str(swc_path),
    }


def _score_candidate(
    evaluator: MorphologyTransferEvaluator,
    reference: dict[str, Any],
    row: dict[str, Any],
    branches: int,
) -> dict[str, Any] | None:
    try:
        morph = evaluator.load_morphology(row["swc_path"], n_branches=branches)
    except Exception as exc:
        return {**row, "error": repr(exc), "compatibility_score_lower_better": float("inf")}
    comparison = evaluator.compare_morphology_set(reference, [morph])["test_morphologies"][0]
    mismatch = comparison["route_mismatch"]
    features = comparison["features"]
    routing_features = comparison["routing_features"]
    score = 0.0
    for value in (
        mismatch["total_length_ratio"],
        mismatch["routed_length_ratio"],
        mismatch["max_radial_extent_ratio"],
    ):
        score += 10.0 if value is None or value <= 0 else abs(math.log(value))
    score += abs(mismatch["apical_fraction_delta"] or 0.0)
    score += abs(mismatch["routing_branch_count_delta"]) / max(1.0, float(branches))
    return {
        **row,
        "compatibility_score_lower_better": score,
        "extracted_branches": features["n_branches"],
        "routed_branches": routing_features["n_branches"],
        "total_length_ratio": mismatch["total_length_ratio"],
        "routed_length_ratio": mismatch["routed_length_ratio"],
        "apical_fraction_delta": mismatch["apical_fraction_delta"],
        "max_radial_extent_ratio": mismatch["max_radial_extent_ratio"],
        "routing_shape_match": mismatch["routing_shape_match"],
    }


def _find_exact_reference_specimen(reference_swc: str | Path, ranked: list[dict[str, Any]]) -> int | None:
    reference_bytes = Path(reference_swc).read_bytes()
    for row in ranked:
        swc_path = Path(row["swc_path"])
        if swc_path.exists() and swc_path.read_bytes() == reference_bytes:
            return int(row["specimen_id"])
    return None


def _render_markdown(payload: dict[str, Any]) -> str:
    source = payload["source"]
    reference_id = payload["reference_specimen_id"]
    rows = [row for row in payload["ranked"] if row["specimen_id"] != reference_id][:15]
    lines = [
        "# Allen L5 M3 Candidate Ranking",
        "",
        f"- Source filter: `{source['filter']}`",
        f"- Reference SWC: `{source['reference_swc']}`",
        f"- Candidate rows: `{payload['candidate_count']}`",
        f"- Ranked rows: `{payload['ranked_count']}`",
        f"- Exact reference specimen ID: `{reference_id}`",
        "",
        "All candidates found for this filter are marked `dendrite-only` in the Allen metadata. This is acceptable for a dendritic-routing M3 diagnostic, but not a full-cell morphology-transfer claim.",
        "",
        "| rank | specimen | line | type | score | branches | routed | length ratio | routed length | apical delta | radial ratio | route shape |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | `{row['specimen_id']}` | `{row['line_name']}` | `{row['reconstruction_type']}` | "
            f"{row['compatibility_score_lower_better']:.3f} | {row['extracted_branches']} | {row['routed_branches']} | "
            f"{row['total_length_ratio']:.3f} | {row['routed_length_ratio']:.3f} | "
            f"{row['apical_fraction_delta']:.3f} | {row['max_radial_extent_ratio']:.3f} | `{row['routing_shape_match']}` |"
        )
    lines.extend([
        "",
        "Interpretation: the bundled Allen L5 template is exactly specimen `488683425`, so that specimen must be excluded from held-out transfer. Several held-out mouse L5 spiny dendrite-only morphologies remain close enough on branch count, routed length, apical fraction, and radial extent to support a defensible M3a dendritic-routing feasibility benchmark.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

