"""Promoted v1 benchmark bundle runner."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import json
from pathlib import Path
import re
from typing import Any, Literal

import numpy as np

from bnn_sim.benchmark.dataset_schema import INTERVENTION_CONDITIONS
from bnn_sim.benchmark.diagnostic_rows import (
    DiagnosticContext,
    morphology_contrast_diagnostic_rows,
    morphology_routing_qc_rows,
    write_diagnostic_rows,
)
from bnn_sim.benchmark.reports import write_diagnostic_summary
from bnn_sim.benchmark.runner import (
    resolve_intervention_pair_paths,
    run_model_diagnostics_on_selector,
    run_paired_intervention_on_npz,
    run_paired_intervention_on_npzs,
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
SuiteProfile = Literal["teacher-qc", "paper-summary", "full"]


def run_v1_benchmark_bundle(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    base_context: DiagnosticContext | None = None,
    model_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    model_id: str | None = None,
    intervention_pair_paths: Iterable[str | Path] = (),
    suite: SuiteProfile = "paper-summary",
    max_val_samples: int | None = None,
    max_intervention_samples: int | None = None,
    cache_shards: int = 1,
    write_summaries: bool = True,
) -> dict[str, Any]:
    """Run the promoted v1 validation plus intervention diagnostic bundle."""
    layout = V1DatasetLayout(dataset_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = base_context or DiagnosticContext()
    suite_config = _suite_config(suite)
    artifacts: list[dict[str, Any]] = []
    intervention_row_sets: list[dict[str, Any]] = []

    teacher_context = _replace_context(
        base,
        suites=suite_config["teacher"],
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
    _record_artifact(teacher_rows, output / "teacher_qc_val.jsonl", artifacts, write_summaries=write_summaries)

    routing_rows = morphology_routing_qc_rows(
        context=_replace_context(base, suites=suite_config["routing"], model_id="teacher")
    )
    write_diagnostic_rows(routing_rows, output / "morphology_routing_qc.jsonl")
    _record_artifact(
        routing_rows,
        output / "morphology_routing_qc.jsonl",
        artifacts,
        write_summaries=write_summaries,
    )

    if model_fn is not None and suite_config["model"] is not None:
        resolved_model_id = model_id or base.model_id
        model_context = _replace_context(
            base,
            suites=suite_config["model"],
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
        _record_artifact(
            model_rows,
            output / f"model_val_{_slug(resolved_model_id)}.jsonl",
            artifacts,
            write_summaries=write_summaries,
        )

    pair_paths = [Path(path) for path in intervention_pair_paths]
    if pair_paths:
        for intervention, morphology_id, grouped_paths in _group_pair_paths_by_morphology(
            pair_paths,
            default_morphology=base.morphology_id,
        ):
            rows_path, rows = _run_grouped_intervention(
                grouped_paths,
                output=output,
                base=base,
                suites=suite_config["intervention"],
                model_fn=model_fn,
                model_id=model_id,
                intervention=intervention,
                morphology_id=morphology_id,
                max_samples=max_intervention_samples,
            )
            intervention_row_sets.extend(rows)
            _record_artifact(rows, rows_path, artifacts, write_summaries=write_summaries)
    else:
        for intervention in layout.available_interventions():
            selector = layout.intervention(intervention)
            selector_pair_paths = resolve_intervention_pair_paths(selector)
            if len(selector_pair_paths) == 1:
                intervention_model_id = model_id if model_fn is not None else "teacher"
                context = _replace_context(
                    base,
                    suites=suite_config["intervention"],
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
                _record_artifact(rows, output / f"{filename}.jsonl", artifacts, write_summaries=write_summaries)
                continue
            for inferred_intervention, morphology_id, grouped_paths in _group_pair_paths_by_morphology(
                selector_pair_paths,
                default_morphology=base.morphology_id,
            ):
                rows_path, rows = _run_grouped_intervention(
                    grouped_paths,
                    output=output,
                    base=base,
                    suites=suite_config["intervention"],
                    model_fn=model_fn,
                    model_id=model_id,
                    intervention=inferred_intervention,
                    morphology_id=morphology_id,
                    max_samples=max_intervention_samples,
                )
                intervention_row_sets.extend(rows)
                _record_artifact(rows, rows_path, artifacts, write_summaries=write_summaries)

    contrast_rows = morphology_contrast_diagnostic_rows(
        intervention_row_sets,
        context=_replace_context(base, suites=suite_config["morphology_contrast"], morphology_id="all_v1"),
    )
    if contrast_rows:
        contrast_path = output / "morphology_contrast_interventions.jsonl"
        write_diagnostic_rows(contrast_rows, contrast_path)
        _record_artifact(
            contrast_rows,
            contrast_path,
            artifacts,
            write_summaries=write_summaries,
        )

    manifest = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output),
        "model_id": model_id,
        "suite": suite,
        "suite_config": {key: list(value) if value is not None else None for key, value in suite_config.items()},
        "max_val_samples": max_val_samples,
        "max_intervention_samples": max_intervention_samples,
        "write_summaries": write_summaries,
        "artifacts": artifacts,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _suite_config(suite: SuiteProfile) -> dict[str, tuple[str, ...] | None]:
    if suite == "teacher-qc":
        return {
            "teacher": ("teacher-qc",),
            "routing": ("teacher-qc",),
            "model": None,
            "intervention": ("teacher-qc",),
            "morphology_contrast": ("morphology-contrast",),
        }
    if suite == "paper-summary":
        return {
            "teacher": ("teacher-qc", "paper-summary"),
            "routing": ("teacher-qc", "paper-summary"),
            "model": DEFAULT_MODEL_SUITES,
            "intervention": DEFAULT_INTERVENTION_SUITES,
            "morphology_contrast": ("morphology-contrast", "paper-summary"),
        }
    if suite == "full":
        return {
            "teacher": ("teacher-qc", "paper-summary", "full"),
            "routing": ("teacher-qc", "paper-summary", "full"),
            "model": (*DEFAULT_MODEL_SUITES, "full"),
            "intervention": ("teacher-qc", "intervention-response", "paper-summary", "full"),
            "morphology_contrast": ("morphology-contrast", "paper-summary", "full"),
        }
    raise ValueError(f"unknown v1 benchmark suite profile: {suite}")


def infer_pair_metadata(path: str | Path, *, default_morphology: str = "unknown") -> tuple[str, str]:
    """Infer intervention and morphology labels from a pilot pair-file path."""
    path = Path(path)
    parts = path.parts
    intervention = None
    if "interventions" in parts:
        index = parts.index("interventions")
        if index + 1 < len(parts):
            intervention = parts[index + 1]
    text = str(path)
    if intervention is None:
        intervention = next((name for name in INTERVENTION_CONDITIONS if name in text), "event_dropout")
    match = re.search(r"specimen_(\d+)", text)
    morphology_id = f"specimen_{match.group(1)}" if match else default_morphology
    return intervention, morphology_id


def _group_pair_paths_by_morphology(
    pair_paths: Iterable[Path],
    *,
    default_morphology: str,
) -> list[tuple[str, str, list[Path]]]:
    grouped: dict[tuple[str, str], list[Path]] = {}
    for pair_path in pair_paths:
        intervention, morphology_id = infer_pair_metadata(pair_path, default_morphology=default_morphology)
        grouped.setdefault((intervention, morphology_id), []).append(pair_path)
    return [
        (intervention, morphology_id, sorted(paths))
        for (intervention, morphology_id), paths in sorted(grouped.items())
    ]


def _run_grouped_intervention(
    pair_paths: list[Path],
    *,
    output: Path,
    base: DiagnosticContext,
    suites: tuple[str, ...],
    model_fn: Callable[[np.ndarray], np.ndarray] | None,
    model_id: str | None,
    intervention: str,
    morphology_id: str,
    max_samples: int | None,
) -> tuple[Path, list[dict[str, Any]]]:
    intervention_model_id = model_id if model_fn is not None else "teacher"
    context = _replace_context(
        base,
        suites=suites,
        model_id=intervention_model_id,
        morphology_id=morphology_id,
        split="",
        condition=f"interventions/{intervention}",
        intervention=intervention,
    )
    rows_path = output / f"intervention_{_slug(morphology_id)}_{_slug(intervention)}.jsonl"
    if len(pair_paths) == 1:
        rows = run_paired_intervention_on_npz(
            pair_paths[0],
            model_fn=model_fn,
            context=context,
            max_samples=max_samples,
            output=rows_path,
        )
    else:
        rows = run_paired_intervention_on_npzs(
            pair_paths,
            model_fn=model_fn,
            context=context,
            max_samples=max_samples,
            output=rows_path,
        )
    return rows_path, rows


def _record_artifact(
    rows: list[dict[str, Any]],
    rows_path: Path,
    artifacts: list[dict[str, Any]],
    *,
    write_summaries: bool,
) -> None:
    artifact = {
        "rows": len(rows),
        "rows_path": str(rows_path),
    }
    if write_summaries:
        summary_path = rows_path.with_suffix(".md")
        write_diagnostic_summary(rows, summary_path)
        artifact["summary_path"] = str(summary_path)
    artifacts.append(artifact)


def _replace_context(context: DiagnosticContext, **updates: Any) -> DiagnosticContext:
    data = {**context.__dict__, **updates}
    return DiagnosticContext(**data)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"
