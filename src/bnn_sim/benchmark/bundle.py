"""Promoted v1 benchmark bundle runner."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from bnn_sim.benchmark.dataset_schema import INTERVENTION_CONDITIONS
from bnn_sim.benchmark.diagnostic_rows import DiagnosticContext, morphology_contrast_diagnostic_rows, write_diagnostic_rows
from bnn_sim.benchmark.reports import write_diagnostic_summary
from bnn_sim.benchmark.runner import (
    run_model_diagnostics_on_selector,
    run_paired_intervention_on_npz,
    run_paired_intervention_on_selector,
    run_teacher_qc_on_selector,
)
from bnn_sim.benchmark.selectors import V1DatasetLayout


DEFAULT_MODEL_SUITES = (
    "trace-core",
    "state-dynamics",
    "spike-behavior",
    "feature-fidelity",
    "paper-summary",
)
DEFAULT_INTERVENTION_SUITES = ("teacher-qc", "intervention-response", "paper-summary")


def run_v1_benchmark_bundle(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    base_context: DiagnosticContext | None = None,
    model_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    model_id: str | None = None,
    intervention_pair_paths: Iterable[str | Path] = (),
    max_val_samples: int | None = None,
    max_intervention_samples: int | None = None,
    cache_shards: int = 1,
) -> dict[str, Any]:
    """Run the promoted v1 validation plus intervention diagnostic bundle."""
    layout = V1DatasetLayout(dataset_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = base_context or DiagnosticContext()
    artifacts: list[dict[str, Any]] = []
    intervention_row_sets: list[dict[str, Any]] = []

    teacher_context = _replace_context(
        base,
        suites=("teacher-qc",),
        model_id="teacher",
        split="val",
        condition="val",
        intervention=None,
    )
    teacher_rows = run_teacher_qc_on_selector(
        layout.default_validation(),
        context=teacher_context,
        max_samples=max_val_samples,
        output=output / "teacher_qc_val.jsonl",
        cache_shards=cache_shards,
    )
    _write_summary_artifact(teacher_rows, output / "teacher_qc_val.md", artifacts)

    if model_fn is not None:
        resolved_model_id = model_id or base.model_id
        model_context = _replace_context(
            base,
            suites=DEFAULT_MODEL_SUITES,
            model_id=resolved_model_id,
            split="val",
            condition="val",
            intervention=None,
        )
        model_rows = run_model_diagnostics_on_selector(
            model_fn,
            layout.default_validation(),
            context=model_context,
            max_samples=max_val_samples,
            output=output / f"model_val_{_slug(resolved_model_id)}.jsonl",
            cache_shards=cache_shards,
        )
        _write_summary_artifact(model_rows, output / f"model_val_{_slug(resolved_model_id)}.md", artifacts)

    pair_paths = [Path(path) for path in intervention_pair_paths]
    if pair_paths:
        for pair_path in pair_paths:
            intervention, morphology_id = infer_pair_metadata(pair_path, default_morphology=base.morphology_id)
            intervention_model_id = model_id if model_fn is not None else "teacher"
            context = _replace_context(
                base,
                suites=DEFAULT_INTERVENTION_SUITES,
                model_id=intervention_model_id,
                morphology_id=morphology_id,
                split="",
                condition=f"interventions/{intervention}",
                intervention=intervention,
            )
            filename = f"intervention_{_slug(morphology_id)}_{_slug(intervention)}"
            rows = run_paired_intervention_on_npz(
                pair_path,
                model_fn=model_fn,
                context=context,
                max_samples=max_intervention_samples,
                output=output / f"{filename}.jsonl",
            )
            intervention_row_sets.extend(rows)
            _write_summary_artifact(rows, output / f"{filename}.md", artifacts)
    else:
        for intervention in layout.available_interventions():
            selector = layout.intervention(intervention)
            intervention_model_id = model_id if model_fn is not None else "teacher"
            context = _replace_context(
                base,
                suites=DEFAULT_INTERVENTION_SUITES,
                model_id=intervention_model_id,
                split="",
                condition=f"interventions/{intervention}",
                intervention=intervention,
            )
            filename = f"intervention_{_slug(base.morphology_id)}_{_slug(intervention)}"
            rows = run_paired_intervention_on_selector(
                selector,
                model_fn=model_fn,
                context=context,
                max_samples=max_intervention_samples,
                output=output / f"{filename}.jsonl",
            )
            intervention_row_sets.extend(rows)
            _write_summary_artifact(rows, output / f"{filename}.md", artifacts)

    contrast_rows = morphology_contrast_diagnostic_rows(
        intervention_row_sets,
        context=_replace_context(base, suites=("morphology-contrast", "paper-summary"), morphology_id="all_v1"),
    )
    if contrast_rows:
        contrast_path = output / "morphology_contrast_interventions.jsonl"
        write_diagnostic_rows(contrast_rows, contrast_path)
        _write_summary_artifact(contrast_rows, output / "morphology_contrast_interventions.md", artifacts)

    manifest = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output),
        "model_id": model_id,
        "max_val_samples": max_val_samples,
        "max_intervention_samples": max_intervention_samples,
        "artifacts": artifacts,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def infer_pair_metadata(path: str | Path, *, default_morphology: str = "unknown") -> tuple[str, str]:
    """Infer intervention and morphology labels from a pilot pair-file path."""
    text = str(path)
    intervention = next((name for name in INTERVENTION_CONDITIONS if name in text), "event_dropout")
    match = re.search(r"specimen_(\d+)", text)
    morphology_id = f"specimen_{match.group(1)}" if match else default_morphology
    return intervention, morphology_id


def _write_summary_artifact(rows: list[dict[str, Any]], path: Path, artifacts: list[dict[str, Any]]) -> None:
    write_diagnostic_summary(rows, path)
    artifacts.append(
        {
            "rows": len(rows),
            "rows_path": str(path.with_suffix(".jsonl")),
            "summary_path": str(path),
        }
    )


def _replace_context(context: DiagnosticContext, **updates: Any) -> DiagnosticContext:
    data = {**context.__dict__, **updates}
    return DiagnosticContext(**data)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"
