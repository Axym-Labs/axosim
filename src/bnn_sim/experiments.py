from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from typing import Any

DEFAULT_REGISTRY_BASENAME = "experiment-registry.jsonl"

_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "train": {
        "paper-random-fulltrace": {
            "description": "Recommended default: official model with random full-trace windows and corrected full-trace validation.",
            "options": {
                "model_kind": "official",
                "model_config": "configs/branch_elm_30_official.json",
                "window_size": 500,
                "ignore_start": 500,
                "window_sampling": "random-full-trace",
                "full_trace_length": 6000,
                "burn_in": 150,
                "batch_size": 8,
                "learning_rate": 5e-4,
                "lr_schedule": "cosine",
                "lr_schedule_steps": 342000,
                "optimizer": "adam",
                "spike_loss_weight": 0.5,
                "soma_loss_weight": 0.5,
                "prefetch_batches": 1,
                "val_metric_ignore_start": 500,
                "val_soma_affine_calibration": True,
            },
        },
        "official-fulltrace-reference": {
            "description": "Reference path: official model with official-style file-level full-trace sampling.",
            "options": {
                "model_kind": "official",
                "model_config": "configs/branch_elm_30_official.json",
                "window_size": 500,
                "ignore_start": 500,
                "window_sampling": "official-full-trace",
                "full_trace_length": 6000,
                "file_load_fraction": 0.3,
                "burn_in": 150,
                "batch_size": 8,
                "learning_rate": 5e-4,
                "lr_schedule": "cosine",
                "lr_schedule_steps": 342000,
                "optimizer": "adam",
                "spike_loss_weight": 0.5,
                "soma_loss_weight": 0.5,
                "prefetch_batches": 1,
                "val_metric_ignore_start": 500,
                "val_soma_affine_calibration": True,
            },
        },
        "mamba-lite-probe": {
            "description": "Mamba-inspired experimental path with the standard random full-trace training setup.",
            "options": {
                "model_kind": "mamba-lite",
                "model_config": "configs/branch_mamba_lite_45x64.json",
                "window_size": 500,
                "ignore_start": 500,
                "window_sampling": "random-full-trace",
                "full_trace_length": 6000,
                "burn_in": 150,
                "batch_size": 8,
                "learning_rate": 3e-4,
                "lr_schedule": "cosine",
                "lr_schedule_steps": 342000,
                "optimizer": "adamw",
                "weight_decay": 1e-4,
                "spike_loss_weight": 0.5,
                "soma_loss_weight": 0.5,
                "prefetch_batches": 1,
                "val_metric_ignore_start": 500,
                "val_soma_affine_calibration": True,
            },
        },
        "mamba-lite-integrative-probe": {
            "description": "Mamba-inspired temporal core with explicit damping and a lightweight long-timescale path.",
            "options": {
                "model_kind": "mamba-lite",
                "model_config": "configs/branch_mamba_lite_integrative_45x64.json",
                "window_size": 500,
                "ignore_start": 500,
                "window_sampling": "random-full-trace",
                "full_trace_length": 6000,
                "burn_in": 150,
                "batch_size": 8,
                "learning_rate": 3e-4,
                "lr_schedule": "cosine",
                "lr_schedule_steps": 342000,
                "optimizer": "adamw",
                "weight_decay": 1e-4,
                "spike_loss_weight": 0.5,
                "soma_loss_weight": 0.5,
                "prefetch_batches": 1,
                "val_metric_ignore_start": 500,
                "val_soma_affine_calibration": True,
            },
        },
        "mamba-lite-branchtrace-probe": {
            "description": "Mamba-inspired temporal core with a Branch-ELM-style branch-level synaptic trace before temporal mixing.",
            "options": {
                "model_kind": "mamba-lite",
                "model_config": "configs/branch_mamba_lite_branchtrace_45x64.json",
                "window_size": 500,
                "ignore_start": 500,
                "window_sampling": "random-full-trace",
                "full_trace_length": 6000,
                "burn_in": 150,
                "batch_size": 8,
                "learning_rate": 3e-4,
                "lr_schedule": "cosine",
                "lr_schedule_steps": 342000,
                "optimizer": "adamw",
                "weight_decay": 1e-4,
                "spike_loss_weight": 0.5,
                "soma_loss_weight": 0.5,
                "prefetch_batches": 1,
                "val_metric_ignore_start": 500,
                "val_soma_affine_calibration": True,
            },
        },
        "mamba-lite-statetrace-probe": {
            "description": "Mamba-inspired temporal core with gated feature-level leaky state traces inside each temporal block.",
            "options": {
                "model_kind": "mamba-lite",
                "model_config": "configs/branch_mamba_lite_statetrace_45x64.json",
                "window_size": 500,
                "ignore_start": 500,
                "window_sampling": "random-full-trace",
                "full_trace_length": 6000,
                "burn_in": 150,
                "batch_size": 8,
                "learning_rate": 3e-4,
                "lr_schedule": "cosine",
                "lr_schedule_steps": 342000,
                "optimizer": "adamw",
                "weight_decay": 1e-4,
                "spike_loss_weight": 0.5,
                "soma_loss_weight": 0.5,
                "prefetch_batches": 1,
                "val_metric_ignore_start": 500,
                "val_soma_affine_calibration": True,
            },
        },
        "mamba-selective-probe": {
            "description": "Dependency-free Mamba-v1 style selective SSM probe with input-dependent dt/B/C and diagonal state scan.",
            "options": {
                "model_kind": "mamba-selective",
                "model_config": "configs/branch_mamba_selective_45x64.json",
                "window_size": 500,
                "ignore_start": 500,
                "window_sampling": "random-full-trace",
                "full_trace_length": 6000,
                "burn_in": 150,
                "batch_size": 8,
                "learning_rate": 3e-4,
                "lr_schedule": "cosine",
                "lr_schedule_steps": 342000,
                "optimizer": "adamw",
                "weight_decay": 1e-4,
                "spike_loss_weight": 0.5,
                "soma_loss_weight": 0.5,
                "prefetch_batches": 1,
                "val_metric_ignore_start": 500,
                "val_soma_affine_calibration": True,
            },
        },
        "mamba-official-probe": {
            "description": "Official mamba-ssm backend probe with fused selective scan in an explicitly configured environment.",
            "options": {
                "model_kind": "mamba-official",
                "model_config": "configs/branch_mamba_official_45x64.json",
                "window_size": 500,
                "ignore_start": 500,
                "window_sampling": "random-full-trace",
                "full_trace_length": 6000,
                "burn_in": 150,
                "batch_size": 8,
                "learning_rate": 3e-4,
                "lr_schedule": "cosine",
                "lr_schedule_steps": 342000,
                "optimizer": "adamw",
                "weight_decay": 1e-4,
                "spike_loss_weight": 0.5,
                "soma_loss_weight": 0.5,
                "prefetch_batches": 1,
                "val_metric_ignore_start": 500,
                "val_soma_affine_calibration": True,
            },
        },
        "legacy-baseline-smoke": {
            "description": "Small local BranchELM smoke path for toy data and quick tests.",
            "options": {
                "model_kind": "baseline",
                "window_sampling": "deterministic",
                "lr_schedule": "constant",
            },
        },
    },
    "evaluate": {
        "fulltrace-paper-metric": {
            "description": "Recommended default: official model with corrected full-trace evaluation semantics.",
            "options": {
                "model_kind": "official",
                "model_config": "configs/branch_elm_30_official.json",
                "metric_ignore_start": 500,
                "soma_affine_calibration": True,
            },
        },
        "legacy-baseline-smoke": {
            "description": "Small local BranchELM smoke path for toy data and quick tests.",
            "options": {
                "model_kind": "baseline",
            },
        },
    },
}


def list_presets() -> dict[str, dict[str, str]]:
    return {
        command: {name: spec["description"] for name, spec in command_presets.items()}
        for command, command_presets in _PRESETS.items()
    }


def apply_preset(command: str, preset_name: str | None, args: Any, explicit_dests: set[str]) -> None:
    if not preset_name:
        return
    command_presets = _PRESETS.get(command, {})
    if preset_name not in command_presets:
        available = ", ".join(sorted(command_presets)) or "none"
        raise ValueError(f"unknown preset {preset_name!r} for {command}; available: {available}")
    for dest, value in command_presets[preset_name]["options"].items():
        if dest not in explicit_dests and hasattr(args, dest):
            setattr(args, dest, value)


def resolve_registry_path(
    command: str,
    *,
    registry: str | None,
    output_path: str | Path,
) -> Path | None:
    if registry == "off":
        return None
    if registry is not None:
        return Path(registry)
    output = Path(output_path)
    return output.parent / DEFAULT_REGISTRY_BASENAME


def append_registry_entry(path: str | Path, entry: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def build_registry_entry(
    *,
    command: str,
    preset: str | None,
    args: Any,
    metrics_path: str | Path,
    metrics: dict[str, Any],
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "command": command,
        "preset": preset,
        "data": str(getattr(args, "data", "")),
        "val_data": str(getattr(args, "val_data", "")) if getattr(args, "val_data", None) else None,
        "metrics_path": str(metrics_path),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "model_kind": getattr(args, "model_kind", None),
        "window_sampling": getattr(args, "window_sampling", None),
        "best_validation_soma_rmse": metrics.get("best_validation_soma_rmse", metrics.get("soma_rmse")),
        "validation_spike_auc": metrics.get("spike_auc"),
        "train_presentations": metrics.get("train_presentations"),
        "total_seconds": metrics.get("total_seconds"),
    }


def _git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
