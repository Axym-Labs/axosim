#!/usr/bin/env python3
"""Robustness/OOD/linear-stability probes for official NeuronIO baselines.

The probe deliberately separates teacher-grounded tests from model-only stress
tests:

* OOD/regime rows use original NeuronIO teacher pickles.
* Adversarial event edits and local linearity rows perturb the input without
  re-running the teacher, so they are stability diagnostics, not correctness
  claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from bnn_sim.data import ShardedNeuronIODataset
from bnn_sim.metrics import binary_auc
from bnn_sim.neuronio_raw import (
    DEFAULT_Y_SOMA_THRESHOLD,
    DEFAULT_Y_TRAIN_SOMA_BIAS,
    DEFAULT_Y_TRAIN_SOMA_SCALE,
    create_neuronio_input_type,
    normalize_soma,
    parse_neuronio_pickle,
)
from bnn_sim.official import (
    create_official_elm,
    load_official_elm_config,
    load_official_state_dict,
)


MODEL_DIRS = {
    "branch_elm_best": "best_elm_neuron",
    "branch_elm_m30": "num_memory_30",
    "branch_elm_m100": "num_memory_100",
    "elm_m1": "num_memory_1",
}

REGIME_DIRS = {
    "canonical_test": "Data_test",
    "ood_rates": "Data_test_OOD",
    "combined_ergodic": "Data_test_combined_ergodic",
    "subthreshold": "Data_test_subthreshold",
}


@dataclass(frozen=True)
class RawRegimeBatch:
    regime: str
    inputs: np.ndarray
    targets_clipped: np.ndarray
    raw_soma_mv: np.ndarray
    sample_ids: list[str]
    source_files: list[str]


@dataclass(frozen=True)
class WindowSample:
    group: str
    trace_id: str
    start: int
    inputs: np.ndarray
    targets: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/nmda-test"))
    parser.add_argument("--val-root", type=Path, default=Path("data/full-trace-shards/nmda-val"))
    parser.add_argument("--model-root", type=Path, default=Path(".external/elmneuron/models"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/ta1/artifacts/analysis_sources/existing_models/robustness_ood_stability_20260619"),
    )
    parser.add_argument("--models", default="branch_elm_best,branch_elm_m30,branch_elm_m100")
    parser.add_argument("--stress-models", default="branch_elm_best,branch_elm_m30,branch_elm_m100")
    parser.add_argument("--max-ood-samples", type=int, default=64)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ignore-start", type=int, default=500)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--spike-windows", type=int, default=3)
    parser.add_argument("--quiet-windows", type=int, default=3)
    parser.add_argument("--event-budgets", default="8,32,128")
    parser.add_argument("--linearity-eps", default="0.0001,0.001,0.01,0.05,0.1")
    parser.add_argument("--skip-ood", action="store_true", help="Skip original NeuronIO regime/OOD evaluation.")
    parser.add_argument("--skip-stress", action="store_true", help="Skip adversarial-event and local-linearity model-only stress probes.")
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_ids = _parse_list(args.models)
    stress_model_ids = _parse_list(args.stress_models)
    budgets = [int(x) for x in _parse_list(args.event_budgets)]
    eps_values = [float(x) for x in _parse_list(args.linearity_eps)]
    rng = np.random.default_rng(args.seed)

    hypotheses = _hypotheses()
    rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    models = {model_id: _load_official_model(args.model_root, model_id, args.device) for model_id in sorted(set(model_ids + stress_model_ids))}
    calibration_inputs, calibration_targets, calibration_ids = _load_sharded_batch(args.val_root, args.calibration_samples)
    thresholds: dict[str, float] = {}
    for model_id in model_ids:
        pred = _predict_batched(models[model_id], calibration_inputs, args.batch_size, args.device)
        threshold = _prevalence_threshold(
            pred[:, args.ignore_start :, 0],
            int((calibration_targets[:, args.ignore_start :, 0] > 0.5).sum()),
        )
        thresholds[model_id] = threshold
        calibration_rows.append(
            {
                "kind": "calibration",
                "model_id": model_id,
                "split": "nmda_val",
                "n_traces": int(calibration_inputs.shape[0]),
                "threshold": threshold,
                "target_spikes": int((calibration_targets[:, args.ignore_start :, 0] > 0.5).sum()),
                "predicted_spikes_at_threshold": int((pred[:, args.ignore_start :, 0] >= threshold).sum()),
            }
        )
        del pred

    if not args.skip_ood:
        for regime, dirname in REGIME_DIRS.items():
            batch = _load_raw_regime(
                args.raw_root / dirname,
                regime=regime,
                max_samples=args.max_ood_samples,
            )
            for model_id in model_ids:
                pred = _predict_batched(models[model_id], batch.inputs, args.batch_size, args.device)
                rows.append(
                    _ood_row(
                        model_id=model_id,
                        batch=batch,
                        pred=pred,
                        threshold=thresholds[model_id],
                        ignore_start=args.ignore_start,
                    )
                )
                del pred
            del batch

    if not args.skip_stress and stress_model_ids:
        windows = _select_windows(
            calibration_inputs,
            calibration_targets,
            calibration_ids,
            window_size=args.window_size,
            ignore_start=args.ignore_start,
            spike_windows=args.spike_windows,
            quiet_windows=args.quiet_windows,
            rng=rng,
        )
        for model_id in stress_model_ids:
            model = models[model_id]
            for window in windows:
                rows.extend(
                    _adversarial_rows(
                        model=model,
                        model_id=model_id,
                        window=window,
                        budgets=budgets,
                        device=args.device,
                        rng=rng,
                    )
                )
                rows.extend(
                    _linearity_rows(
                        model=model,
                        model_id=model_id,
                        window=window,
                        eps_values=eps_values,
                        device=args.device,
                        rng=rng,
                    )
            )

    payload = {
        "metadata": {
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": time.time() - started,
            "raw_root": str(args.raw_root),
            "val_root": str(args.val_root),
            "model_root": str(args.model_root),
            "models": model_ids,
            "stress_models": stress_model_ids,
            "max_ood_samples": args.max_ood_samples,
            "calibration_samples": args.calibration_samples,
            "ignore_start": args.ignore_start,
            "window_size": args.window_size,
            "event_budgets": budgets,
            "linearity_eps": eps_values,
            "seed": args.seed,
            "device": args.device,
            "skip_ood": args.skip_ood,
            "skip_stress": args.skip_stress,
        },
        "hypotheses": hypotheses,
        "calibration_rows": calibration_rows,
        "rows": rows,
        "interpretation": _interpret(rows),
    }
    _write_json(args.output_dir / "summary.json", payload)
    _write_jsonl(args.output_dir / "rows.jsonl", rows)
    _write_csv(args.output_dir / "rows.csv", rows)
    _write_markdown(args.output_dir / "report.md", payload)
    print(json.dumps(payload["interpretation"], indent=2, sort_keys=True))


def _hypotheses() -> list[dict[str, str]]:
    return [
        {
            "id": "H1_ood_operating_point",
            "claim": (
                "Official ELM/Branch-ELM spike discrimination can remain high on NeuronIO regimes while "
                "a validation-calibrated operating threshold transfers poorly, revealing a gap hidden by AUC."
            ),
            "test": "Evaluate original NeuronIO Data_test, OOD, combined-ergodic, and subthreshold pickles with a threshold calibrated on nmda-val.",
            "falsifier": "Validation-calibrated spike-count and event-timing metrics remain close to in-domain performance across regimes.",
        },
        {
            "id": "H2_adversarial_event_brittleness",
            "claim": (
                "A small number of gradient-targeted event insertions/deletions can move predictions much more than random edits "
                "with the same edit budget."
            ),
            "test": "Compare model output deltas from first-order targeted discrete event edits against random controls.",
            "falsifier": "Targeted/random effect ratios stay near 1 for spike-rich and quiet windows across edit budgets.",
        },
        {
            "id": "H3_local_linear_instability",
            "claim": (
                "Local affine approximations are less stable near spikes than in quiet windows, either through larger finite-difference "
                "linearity error or more hidden-activation sign crossings. For the downloaded SiLU checkpoints this is not a literal "
                "piecewise-linear region count."
            ),
            "test": "Measure output linearity error and hidden preactivation sign flip fraction under random and gradient directions for multiple epsilons.",
            "falsifier": "Spike-rich and quiet windows have similar linearity error and activation sign flip rates.",
        },
    ]


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_official_model(model_root: Path, model_id: str, device: str) -> torch.nn.Module:
    if device != "cpu":
        warnings.warn("The local default PyTorch build may not support RTX 5090 CUDA; CPU is the validated path for this probe.")
    dirname = MODEL_DIRS[model_id]
    root = model_root / dirname
    model = create_official_elm(load_official_elm_config(root / "model_config.json"))
    model.load_state_dict(load_official_state_dict(root / "neuronio_best_model_state.pt", map_location="cpu"))
    return model.to(device).eval()


def _load_sharded_batch(root: Path, n_samples: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    dataset = ShardedNeuronIODataset(root, cache_shards=1)
    n = min(n_samples, len(dataset))
    inputs, targets = dataset.get_batch(range(n))
    sample_ids = dataset.get_sample_ids(range(n))
    return inputs.astype(np.float32, copy=False), targets.astype(np.float32, copy=False), sample_ids


def _load_raw_regime(root: Path, *, regime: str, max_samples: int) -> RawRegimeBatch:
    files = sorted(path for path in root.glob("*.p") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no raw NeuronIO pickles found in {root}")
    inputs_chunks: list[np.ndarray] = []
    targets_chunks: list[np.ndarray] = []
    raw_soma_chunks: list[np.ndarray] = []
    sample_ids: list[str] = []
    source_files: list[str] = []
    remaining = max_samples
    for path in files:
        if remaining <= 0:
            break
        raw = parse_neuronio_pickle(path)
        take = min(remaining, raw.inputs.shape[0])
        sign = create_neuronio_input_type(raw.inputs.shape[-1])
        signed_inputs = (raw.inputs[:take] * sign).astype(np.float32, copy=False)
        clipped_soma = normalize_soma(raw.soma[:take], DEFAULT_Y_SOMA_THRESHOLD, DEFAULT_Y_TRAIN_SOMA_BIAS, DEFAULT_Y_TRAIN_SOMA_SCALE)
        targets = np.stack([raw.spikes[:take], clipped_soma], axis=-1).astype(np.float32, copy=False)
        inputs_chunks.append(signed_inputs)
        targets_chunks.append(targets)
        raw_soma_chunks.append(raw.soma[:take].astype(np.float32, copy=False))
        sample_ids.extend([f"{path.stem}:{idx}" for idx in range(take)])
        source_files.append(path.name)
        remaining -= take
    return RawRegimeBatch(
        regime=regime,
        inputs=np.concatenate(inputs_chunks, axis=0),
        targets_clipped=np.concatenate(targets_chunks, axis=0),
        raw_soma_mv=np.concatenate(raw_soma_chunks, axis=0),
        sample_ids=sample_ids,
        source_files=source_files,
    )


def _predict_batched(model: torch.nn.Module, inputs: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], batch_size):
            x = torch.from_numpy(inputs[start : start + batch_size].astype(np.float32, copy=False)).to(device)
            outputs.append(model(x).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _ood_row(
    *,
    model_id: str,
    batch: RawRegimeBatch,
    pred: np.ndarray,
    threshold: float,
    ignore_start: int,
) -> dict[str, Any]:
    pred_m = pred[:, ignore_start:, :]
    tgt_m = batch.targets_clipped[:, ignore_start:, :]
    raw_soma = batch.raw_soma_mv[:, ignore_start:]
    labels = tgt_m[..., 0] > 0.5
    scores = pred_m[..., 0]
    pred_soma_mv = pred_m[..., 1] / DEFAULT_Y_TRAIN_SOMA_SCALE + DEFAULT_Y_TRAIN_SOMA_BIAS
    clipped_soma_mv = tgt_m[..., 1] / DEFAULT_Y_TRAIN_SOMA_SCALE + DEFAULT_Y_TRAIN_SOMA_BIAS
    residual_clip = pred_soma_mv - clipped_soma_mv
    residual_raw = pred_soma_mv - raw_soma
    near = _near_spike_mask(labels, before=5, after=10)
    zero_pred = scores >= 0.0
    val_pred = scores >= threshold
    regime_threshold = _prevalence_threshold(scores, int(labels.sum()))
    regime_pred = scores >= regime_threshold
    return {
        "kind": "teacher_grounded_ood",
        "model_id": model_id,
        "regime": batch.regime,
        "n_traces": int(pred.shape[0]),
        "source_files": batch.source_files,
        "target_spikes": int(labels.sum()),
        "event_fraction": float(np.count_nonzero(batch.inputs) / batch.inputs.size),
        "clipped_rmse_mv": _rmse(residual_clip),
        "raw_rmse_mv": _rmse(residual_raw),
        "near_spike_fraction": float(near.mean()),
        "near_spike_clipped_rmse_mv": _rmse(residual_clip[near]) if near.any() else None,
        "near_spike_raw_rmse_mv": _rmse(residual_raw[near]) if near.any() else None,
        "spike_auc": float(binary_auc(scores, labels.astype(np.float32))),
        "zero_threshold_predicted_spikes": int(zero_pred.sum()),
        "zero_threshold_event_f1_2ms": _event_f1(zero_pred, labels, tolerance=2)["f1"],
        "val_threshold": float(threshold),
        "val_threshold_predicted_spikes": int(val_pred.sum()),
        "val_threshold_count_mae_per_trace": _count_mae(val_pred, labels),
        "val_threshold_event_f1_2ms": _event_f1(val_pred, labels, tolerance=2)["f1"],
        "regime_prevalence_threshold": float(regime_threshold),
        "regime_prevalence_predicted_spikes": int(regime_pred.sum()),
        "regime_prevalence_event_f1_2ms": _event_f1(regime_pred, labels, tolerance=2)["f1"],
        "claim_scope": "teacher-grounded original NeuronIO regime evaluation",
    }


def _select_windows(
    inputs: np.ndarray,
    targets: np.ndarray,
    sample_ids: list[str],
    *,
    window_size: int,
    ignore_start: int,
    spike_windows: int,
    quiet_windows: int,
    rng: np.random.Generator,
) -> list[WindowSample]:
    windows: list[WindowSample] = []
    spike_candidates: list[tuple[int, int]] = []
    quiet_candidates: list[int] = []
    for trace_idx in range(targets.shape[0]):
        spike_idx = np.flatnonzero(targets[trace_idx, ignore_start:, 0] > 0.5) + ignore_start
        if spike_idx.size:
            spike_candidates.append((trace_idx, int(spike_idx[spike_idx.size // 2])))
        else:
            quiet_candidates.append(trace_idx)
    rng.shuffle(spike_candidates)
    rng.shuffle(quiet_candidates)
    for trace_idx, spike_time in spike_candidates[:spike_windows]:
        start = _bounded_window_start(spike_time - window_size // 2, targets.shape[1], window_size, ignore_start)
        windows.append(
            WindowSample(
                group="spike_rich",
                trace_id=sample_ids[trace_idx],
                start=start,
                inputs=inputs[trace_idx : trace_idx + 1, start : start + window_size],
                targets=targets[trace_idx : trace_idx + 1, start : start + window_size],
            )
        )
    if len(quiet_candidates) < quiet_windows:
        no_spike = {idx for idx in quiet_candidates}
        fallback = [idx for idx in range(targets.shape[0]) if idx not in no_spike]
        quiet_candidates.extend(fallback)
    for trace_idx in quiet_candidates[:quiet_windows]:
        start = _bounded_window_start(ignore_start, targets.shape[1], window_size, ignore_start)
        windows.append(
            WindowSample(
                group="quiet_or_low_spike",
                trace_id=sample_ids[trace_idx],
                start=start,
                inputs=inputs[trace_idx : trace_idx + 1, start : start + window_size],
                targets=targets[trace_idx : trace_idx + 1, start : start + window_size],
            )
        )
    return windows


def _bounded_window_start(start: int, trace_len: int, window_size: int, ignore_start: int) -> int:
    low = min(ignore_start, max(0, trace_len - window_size))
    high = max(low, trace_len - window_size)
    return int(np.clip(start, low, high))


def _adversarial_rows(
    *,
    model: torch.nn.Module,
    model_id: str,
    window: WindowSample,
    budgets: list[int],
    device: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    x = window.inputs.astype(np.float32, copy=True)
    base = _predict_tensor(model, x, device)
    grad = _input_gradient(model, x, device)
    objective_name = "mean_spike_logit_plus_0p05_soma_after_128ms"
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        adv, adv_meta = _make_targeted_event_edit(x, grad, budget)
        random_edit, random_meta = _make_random_event_edit(x, budget, adv_meta["n_insertions"], adv_meta["n_deletions"], rng)
        for perturbation, meta, kind in [
            (adv, adv_meta, "gradient_targeted"),
            (random_edit, random_meta, "random_control"),
        ]:
            pert = _predict_tensor(model, perturbation, device)
            delta = pert - base
            rows.append(
                {
                    "kind": "model_only_adversarial_event",
                    "model_id": model_id,
                    "window_group": window.group,
                    "trace_id": window.trace_id,
                    "window_start": int(window.start),
                    "perturbation": kind,
                    "budget": int(budget),
                    "n_insertions": int(meta["n_insertions"]),
                    "n_deletions": int(meta["n_deletions"]),
                    "changed_events": int(meta["changed_events"]),
                    "baseline_events": int(np.count_nonzero(x)),
                    "objective": objective_name,
                    "soma_delta_rmse_mv": _rmse(delta[..., 1] / DEFAULT_Y_TRAIN_SOMA_SCALE),
                    "soma_delta_max_abs_mv": float(np.max(np.abs(delta[..., 1] / DEFAULT_Y_TRAIN_SOMA_SCALE))),
                    "spike_logit_delta_mean": float(np.mean(delta[..., 0])),
                    "spike_logit_delta_max_abs": float(np.max(np.abs(delta[..., 0]))),
                    "spike_probability_mass_delta": float(np.sum(_sigmoid(pert[..., 0]) - _sigmoid(base[..., 0]))),
                    "claim_scope": "model-only input stability; teacher was not resimulated",
                }
            )
    return rows


def _input_gradient(model: torch.nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    xt = torch.from_numpy(x.astype(np.float32, copy=False)).to(device)
    xt.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    y = model(xt)
    objective = y[:, 128:, 0].mean() + 0.05 * y[:, 128:, 1].mean()
    objective.backward()
    return xt.grad.detach().cpu().numpy().astype(np.float32, copy=False)


def _make_targeted_event_edit(x: np.ndarray, grad: np.ndarray, budget: int) -> tuple[np.ndarray, dict[str, int]]:
    edited = x.copy()
    sign_map = _signed_channel_template(x.shape[-1]).reshape(1, 1, -1)
    existing = x != 0
    empty = x == 0
    delete_gain = np.where(existing, -grad * x, -np.inf)
    insert_gain = np.where(empty, grad * sign_map, -np.inf)
    delete_flat = delete_gain.reshape(-1)
    insert_flat = insert_gain.reshape(-1)
    candidates: list[tuple[float, str, int]] = []
    for flat in _topk_indices(delete_flat, budget):
        if np.isfinite(delete_flat[flat]):
            candidates.append((float(delete_flat[flat]), "delete", int(flat)))
    for flat in _topk_indices(insert_flat, budget):
        if np.isfinite(insert_flat[flat]):
            candidates.append((float(insert_flat[flat]), "insert", int(flat)))
    candidates.sort(reverse=True, key=lambda row: row[0])
    n_insertions = 0
    n_deletions = 0
    for _, action, flat in candidates[:budget]:
        idx = np.unravel_index(flat, x.shape)
        if action == "delete":
            edited[idx] = 0.0
            n_deletions += 1
        else:
            edited[idx] = float(sign_map[(0, 0, idx[-1])])
            n_insertions += 1
    changed = int(np.count_nonzero(edited != x))
    return edited, {"n_insertions": n_insertions, "n_deletions": n_deletions, "changed_events": changed}


def _make_random_event_edit(
    x: np.ndarray,
    budget: int,
    n_insertions: int,
    n_deletions: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    del budget
    edited = x.copy()
    sign_map = _signed_channel_template(x.shape[-1])
    existing = np.flatnonzero(x.reshape(-1) != 0)
    empty = np.flatnonzero(x.reshape(-1) == 0)
    delete_choices = rng.choice(existing, size=min(n_deletions, existing.size), replace=False) if n_deletions else []
    insert_choices = rng.choice(empty, size=min(n_insertions, empty.size), replace=False) if n_insertions else []
    flat = edited.reshape(-1)
    flat[np.asarray(delete_choices, dtype=np.int64)] = 0.0
    for choice in np.asarray(insert_choices, dtype=np.int64):
        channel = int(np.unravel_index(int(choice), x.shape)[-1])
        flat[int(choice)] = float(sign_map[channel])
    changed = int(np.count_nonzero(edited != x))
    return edited, {"n_insertions": int(len(insert_choices)), "n_deletions": int(len(delete_choices)), "changed_events": changed}


def _linearity_rows(
    *,
    model: torch.nn.Module,
    model_id: str,
    window: WindowSample,
    eps_values: list[float],
    device: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    x = window.inputs.astype(np.float32, copy=False)
    base = _predict_tensor(model, x, device)
    base_pattern = _relu_pattern(model, x, device)
    grad = _input_gradient(model, x, device)
    directions = {
        "random_gaussian": _unit_rms(rng.standard_normal(x.shape).astype(np.float32)),
        "gradient_sign": _unit_rms(np.sign(grad).astype(np.float32)),
    }
    rows: list[dict[str, Any]] = []
    eps0 = min(eps_values)
    for direction_name, direction in directions.items():
        ref = _predict_tensor(model, x + eps0 * direction, device)
        tangent = (ref - base) / eps0
        for eps in eps_values:
            perturbed_x = x + eps * direction
            pert = _predict_tensor(model, perturbed_x, device)
            actual = pert - base
            linear = tangent * eps
            pattern = _relu_pattern(model, perturbed_x, device)
            rows.append(
                {
                    "kind": "model_only_local_linearity",
                    "model_id": model_id,
                    "window_group": window.group,
                    "trace_id": window.trace_id,
                    "window_start": int(window.start),
                    "direction": direction_name,
                    "epsilon": float(eps),
                    "soma_linearity_relative_error": _relative_error(actual[..., 1], linear[..., 1]),
                    "spike_linearity_relative_error": _relative_error(actual[..., 0], linear[..., 0]),
                    "soma_delta_rmse_mv": _rmse(actual[..., 1] / DEFAULT_Y_TRAIN_SOMA_SCALE),
                    "spike_logit_delta_rms": _rmse(actual[..., 0]),
                    "activation_type": str(getattr(model, "mlp_activation", "")),
                    "activation_sign_flip_fraction": float(np.mean(pattern != base_pattern)) if pattern.size else None,
                    "literal_relu_region": bool(str(getattr(model, "mlp_activation", "")) == "relu"),
                    "claim_scope": "model-only local linearity; inputs are continuous relaxations of event tensors",
                }
            )
    return rows


def _predict_tensor(model: torch.nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    with torch.inference_mode():
        xt = torch.from_numpy(x.astype(np.float32, copy=False)).to(device)
        return model(xt).detach().cpu().numpy().astype(np.float32, copy=False)


def _relu_pattern(model: torch.nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    """Return hidden preactivation sign patterns across timesteps.

    For ReLU networks this is the literal piecewise-linear activation region.
    The downloaded ELM/Branch-ELM checkpoints use SiLU, so in the main probe it
    is a high-curvature sign-crossing proxy rather than a true linear region.
    """
    with torch.inference_mode():
        xt = torch.from_numpy(x.astype(np.float32, copy=False)).to(device)
        routed = model.route_input_to_synapses(xt)
        batch_size = routed.shape[0]
        w_s = model.w_s
        kappa_s, kappa_m = model.kappa_s, model.kappa_m
        s_prev = torch.zeros(batch_size, len(kappa_s), device=device)
        m_prev = torch.zeros(batch_size, len(kappa_m), device=device)
        patterns: list[torch.Tensor] = []
        for t in range(routed.shape[1]):
            x_t = routed[:, t]
            s_t = kappa_s * s_prev + w_s * x_t
            syn_input = s_t.view(batch_size, model.num_branch, -1).sum(dim=-1)
            mlp_in = torch.cat([syn_input, kappa_m * m_prev], dim=-1)
            mlp_out, pattern = _mlp_forward_with_relu_pattern(model.mlp.network, mlp_in)
            delta_m_t = torch.tanh(mlp_out * 2 / 3) * 1.7159
            m_t = kappa_m * m_prev + model.lambda_value * (1 - kappa_m) * delta_m_t
            if pattern is not None:
                patterns.append(pattern.detach().cpu())
            s_prev, m_prev = s_t, m_t
        if not patterns:
            return np.zeros((0,), dtype=bool)
        return torch.cat(patterns, dim=0).numpy().astype(bool, copy=False).reshape(-1)


def _mlp_forward_with_relu_pattern(network: nn.Sequential, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    patterns: list[torch.Tensor] = []
    out = x
    last_linear: torch.Tensor | None = None
    for layer in network:
        if isinstance(layer, nn.Linear):
            out = layer(out)
            last_linear = out
            continue
        if isinstance(layer, (nn.ReLU, nn.SiLU, nn.GELU, nn.Softplus)) and last_linear is not None:
            patterns.append(last_linear > 0)
        out = layer(out)
    return out, torch.cat(patterns, dim=-1) if patterns else None


def _topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or values.size == 0:
        return np.asarray([], dtype=np.int64)
    k = min(k, values.size)
    idx = np.argpartition(values, values.size - k)[values.size - k :]
    return idx[np.argsort(values[idx])[::-1]]


def _signed_channel_template(n_channels: int) -> np.ndarray:
    half = n_channels // 2
    if half * 2 != n_channels:
        raise ValueError(f"expected even input channels, got {n_channels}")
    return np.concatenate([np.ones(half), -np.ones(half)]).astype(np.float32)


def _unit_rms(direction: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(direction.astype(np.float64) ** 2)))
    if rms == 0.0:
        return np.zeros_like(direction, dtype=np.float32)
    return (direction / rms).astype(np.float32, copy=False)


def _near_spike_mask(labels: np.ndarray, *, before: int, after: int) -> np.ndarray:
    result = np.zeros_like(labels, dtype=bool)
    for row in range(labels.shape[0]):
        for idx in np.flatnonzero(labels[row]):
            start = max(0, idx - before)
            stop = min(labels.shape[1], idx + after + 1)
            result[row, start:stop] = True
    return result


def _event_f1(pred: np.ndarray, target: np.ndarray, *, tolerance: int) -> dict[str, float]:
    tp = 0
    total_pred = 0
    total_target = 0
    for row in range(pred.shape[0]):
        pred_idx = list(map(int, np.flatnonzero(pred[row])))
        target_idx = list(map(int, np.flatnonzero(target[row])))
        total_pred += len(pred_idx)
        total_target += len(target_idx)
        used: set[int] = set()
        for t in target_idx:
            candidates = [p for p in pred_idx if p not in used and abs(p - t) <= tolerance]
            if not candidates:
                continue
            p = min(candidates, key=lambda value: abs(value - t))
            used.add(p)
            tp += 1
    precision = tp / total_pred if total_pred else 0.0
    recall = tp / total_target if total_target else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _prevalence_threshold(scores: np.ndarray, positive_count: int) -> float:
    flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    if positive_count <= 0:
        return float("inf")
    if positive_count >= flat.size:
        return float("-inf")
    return float(np.partition(flat, flat.size - positive_count)[flat.size - positive_count])


def _count_mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred.sum(axis=1) - target.sum(axis=1))))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _rmse(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(arr**2)))


def _relative_error(actual: np.ndarray, linear: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    linear = np.asarray(linear, dtype=np.float64)
    denom = float(np.linalg.norm(actual.reshape(-1)))
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm((actual - linear).reshape(-1)) / denom)


def _interpret(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ood = [row for row in rows if row["kind"] == "teacher_grounded_ood"]
    adversarial = [row for row in rows if row["kind"] == "model_only_adversarial_event"]
    linearity = [row for row in rows if row["kind"] == "model_only_local_linearity"]
    adv_ratios = _adversarial_ratios(adversarial)
    return {
        "teacher_grounded_ood_rows": len(ood),
        "model_only_adversarial_rows": len(adversarial),
        "model_only_linearity_rows": len(linearity),
        "worst_val_threshold_f1": _min_row(ood, "val_threshold_event_f1_2ms"),
        "worst_raw_near_spike_rmse_mv": _max_row(ood, "near_spike_raw_rmse_mv"),
        "largest_adversarial_random_ratio": _max_row(adv_ratios, "soma_delta_ratio"),
        "largest_linearity_error": _max_row(linearity, "soma_linearity_relative_error"),
        "scope_warning": (
            "OOD rows are teacher-grounded; adversarial and linearity rows are model-only because the perturbed inputs were not re-simulated."
        ),
    }


def _adversarial_ratios(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["model_id"], row["window_group"], row["trace_id"], int(row["budget"]))
        grouped.setdefault(key, {})[row["perturbation"]] = row
    ratios: list[dict[str, Any]] = []
    for (model_id, group, trace_id, budget), pair in grouped.items():
        adv = pair.get("gradient_targeted")
        rnd = pair.get("random_control")
        if not adv or not rnd:
            continue
        rnd_soma = float(rnd["soma_delta_rmse_mv"])
        rnd_spike = abs(float(rnd["spike_probability_mass_delta"]))
        ratios.append(
            {
                "model_id": model_id,
                "window_group": group,
                "trace_id": trace_id,
                "budget": budget,
                "soma_delta_ratio": float(adv["soma_delta_rmse_mv"]) / rnd_soma if rnd_soma > 0 else math.inf,
                "spike_mass_ratio": abs(float(adv["spike_probability_mass_delta"])) / rnd_spike if rnd_spike > 0 else math.inf,
            }
        )
    return ratios


def _max_row(rows: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get(field) is not None and not (isinstance(row.get(field), float) and math.isnan(row[field]))]
    if not valid:
        return None
    row = max(valid, key=lambda item: float(item[field]))
    return {key: row[key] for key in row if key in {"model_id", "regime", "window_group", "trace_id", "budget", "direction", "epsilon", field}}


def _min_row(rows: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get(field) is not None and not (isinstance(row.get(field), float) and math.isnan(row[field]))]
    if not valid:
        return None
    row = min(valid, key=lambda item: float(item[field]))
    return {key: row[key] for key in row if key in {"model_id", "regime", "window_group", "trace_id", "budget", "direction", "epsilon", field}}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, default=_json_default) + "\n" for row in rows), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    ood = [row for row in rows if row["kind"] == "teacher_grounded_ood"]
    adv_ratios = _adversarial_ratios([row for row in rows if row["kind"] == "model_only_adversarial_event"])
    linearity = [row for row in rows if row["kind"] == "model_only_local_linearity"]
    lines = [
        "# Robustness, OOD, and Linear-Stability Probe",
        "",
        "## Scope",
        "",
        "OOD rows use original NeuronIO teacher pickles and are teacher-grounded. "
        "Adversarial event edits and local linearity rows perturb model inputs without re-running the teacher, "
        "so they diagnose model stability rather than simulator correctness. The downloaded official checkpoints use SiLU activations; "
        "therefore the stability probe reports local affine-error and hidden preactivation sign crossings, not literal ReLU linear regions.",
        "",
        "## Hypotheses",
        "",
    ]
    for hyp in payload["hypotheses"]:
        lines.extend([f"### {hyp['id']}", "", f"Claim: {hyp['claim']}", "", f"Test: {hyp['test']}", "", f"Falsifier: {hyp['falsifier']}", ""])
    lines.extend(["## Calibration", ""])
    for row in payload["calibration_rows"]:
        lines.append(
            f"- `{row['model_id']}` val prevalence threshold `{row['threshold']:.6g}` "
            f"for `{row['target_spikes']}` target spikes over `{row['n_traces']}` traces."
        )
    lines.extend(["", "## Teacher-Grounded OOD", ""])
    lines.append("| model | regime | n | spikes | clipped RMSE mV | raw RMSE mV | near-spike raw RMSE mV | AUC | val-threshold spikes | val-threshold F1@2ms | prevalence F1@2ms |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in ood:
        lines.append(
            f"| {row['model_id']} | {row['regime']} | {row['n_traces']} | {row['target_spikes']} | "
            f"{_fmt(row['clipped_rmse_mv'])} | {_fmt(row['raw_rmse_mv'])} | {_fmt(row['near_spike_raw_rmse_mv'])} | "
            f"{_fmt(row['spike_auc'])} | {row['val_threshold_predicted_spikes']} | "
            f"{_fmt(row['val_threshold_event_f1_2ms'])} | {_fmt(row['regime_prevalence_event_f1_2ms'])} |"
        )
    lines.extend(["", "## Adversarial Event Stability", ""])
    grouped_adv: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in adv_ratios:
        grouped_adv.setdefault((row["model_id"], row["window_group"], int(row["budget"])), []).append(row)
    lines.append("| model | group | budget | windows | mean soma ratio | max soma ratio | mean spike-mass ratio | max spike-mass ratio |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key, group_rows in sorted(grouped_adv.items()):
        model_id, window_group, budget = key
        lines.append(
            f"| {model_id} | {window_group} | {budget} | {len(group_rows)} | "
            f"{_fmt(_mean(row['soma_delta_ratio'] for row in group_rows))} | "
            f"{_fmt(max(row['soma_delta_ratio'] for row in group_rows))} | "
            f"{_fmt(_mean(row['spike_mass_ratio'] for row in group_rows))} | "
            f"{_fmt(max(row['spike_mass_ratio'] for row in group_rows))} |"
        )
    lines.extend(["", "## Local Linearity", ""])
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = {}
    for row in linearity:
        grouped.setdefault((row["model_id"], row["window_group"], row["direction"], float(row["epsilon"])), []).append(row)
    lines.append("| model | group | direction | epsilon | mean soma linearity error | mean activation sign flip fraction | mean soma delta mV |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for key, group_rows in sorted(grouped.items()):
        model_id, window_group, direction, eps = key
        lines.append(
            f"| {model_id} | {window_group} | {direction} | {eps:g} | "
            f"{_fmt(_mean(row['soma_linearity_relative_error'] for row in group_rows))} | "
            f"{_fmt(_mean(row['activation_sign_flip_fraction'] for row in group_rows))} | "
            f"{_fmt(_mean(row['soma_delta_rmse_mv'] for row in group_rows))} |"
        )
    lines.extend(["", "## Machine Readout", "", "```json", json.dumps(payload["interpretation"], indent=2, sort_keys=True, default=_json_default), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.4g}"


def _mean(values: Any) -> float | None:
    arr = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
