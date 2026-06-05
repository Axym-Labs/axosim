#!/usr/bin/env python3
"""Compare NeuronIO and Arbor teacher protocol/input statistics.

This is an evidence-gathering diagnostic. It compares observable split-level
input and output statistics, and records source-level protocol differences from
the NeuronIO/Beniaguev generator and the current Arbor generator.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle
import warnings
from typing import Any

import numpy as np

from bnn_sim.official.constants import DEFAULT_Y_TRAIN_SOMA_BIAS, DEFAULT_Y_TRAIN_SOMA_SCALE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmda", default="data/full-trace-shards/nmda-val")
    parser.add_argument("--arbor", default="data/arbor-calibrated-nmda-proxy-scale256_64-clipped-20260605/val")
    parser.add_argument("--morphology", default="data/raw/nmda-test/Morphology/morphology_dict.pickle")
    parser.add_argument("--output-json", default="docs/verify-arbor-simulator/artifacts/teacher_protocol_comparison_20260605.json")
    parser.add_argument("--output-md", default="docs/verify-arbor-simulator/artifacts/teacher_protocol_comparison_20260605.md")
    parser.add_argument("--max-samples", type=int, default=256, help="diagnostic sample cap; 0 means all samples")
    parser.add_argument("--soma-scale", type=float, default=DEFAULT_Y_TRAIN_SOMA_SCALE)
    parser.add_argument("--soma-bias-mv", type=float, default=DEFAULT_Y_TRAIN_SOMA_BIAS)
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nmda_regions = load_nmda_channel_regions(Path(args.morphology))
    arbor_root = Path(args.arbor)
    arbor_regions = load_arbor_channel_regions(arbor_root)
    arbor_manifest = load_arbor_manifest_summary(arbor_root)
    max_samples = None if args.max_samples <= 0 else args.max_samples

    report = {
        "metadata": {
            "nmda": args.nmda,
            "arbor": args.arbor,
            "morphology": args.morphology,
            "max_samples": max_samples,
            "soma_scale": args.soma_scale,
            "soma_bias_mv": args.soma_bias_mv,
            "arbor_manifest": arbor_manifest,
        },
        "source_protocol_diff": source_protocol_diff(arbor_manifest.get("input_protocol", "unknown")),
        "split_statistics": {
            "nmda": analyze_split(
                Path(args.nmda),
                channel_regions=nmda_regions,
                max_samples=max_samples,
                soma_scale=args.soma_scale,
                soma_bias_mv=args.soma_bias_mv,
            ),
            "arbor": analyze_split(
                arbor_root,
                channel_regions=arbor_regions,
                max_samples=max_samples,
                soma_scale=args.soma_scale,
                soma_bias_mv=args.soma_bias_mv,
            ),
        },
        "interpretation": interpretation(),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(compact(report), indent=2, sort_keys=True))
    return 0


def load_nmda_channel_regions(morphology_path: Path) -> dict[str, np.ndarray]:
    with morphology_path.open("rb") as handle:
        morphology = pickle.load(handle, encoding="latin1")
    segment_types = np.asarray(morphology["all_segments_type"]).astype(str)
    if segment_types.shape[0] * 2 != 1278:
        raise ValueError(f"expected 639 NeuronIO segments, got {segment_types.shape[0]}")
    region = np.concatenate([segment_types, segment_types])
    sign = np.concatenate([np.ones(segment_types.shape[0], dtype=np.int8), -np.ones(segment_types.shape[0], dtype=np.int8)])
    return {"region": region, "sign": sign}


def load_arbor_channel_regions(split_root: Path) -> dict[str, np.ndarray]:
    manifest = json.loads((split_root / "manifest.json").read_text(encoding="utf-8"))
    simulator = manifest.get("simulator", {})
    branch_regions = np.asarray(simulator.get("branch_regions", []), dtype=str)
    input_dim = int(simulator.get("config", {}).get("input_dim", 1278))
    if branch_regions.size == 0:
        raise ValueError("Arbor manifest is missing branch_regions")
    routing_info = simulator.get("routing", {})
    if isinstance(routing_info, dict) and "channel_branch_routing" in routing_info:
        routing = np.asarray(routing_info["channel_branch_routing"], dtype=np.int64)
    else:
        routing = interlocking_branch_routing(input_dim, int(branch_regions.shape[0]))
    region = branch_regions[routing]
    sign = np.where(np.arange(input_dim) < input_dim // 2, 1, -1).astype(np.int8)
    return {"region": region, "sign": sign}


def load_arbor_manifest_summary(split_root: Path) -> dict[str, Any]:
    manifest_path = split_root / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    simulator = manifest.get("simulator", {})
    config = simulator.get("config", {})
    return {
        "dataset_kind": manifest.get("dataset_kind"),
        "input_protocol": config.get("input_protocol", "arbor_poisson"),
        "synapse_routing": config.get("synapse_routing", "branch"),
        "target_columns": simulator.get("target_columns"),
        "soma_target_units": simulator.get("soma_target_units"),
    }


def interlocking_branch_routing(input_dim: int, num_branch: int) -> np.ndarray:
    even = np.arange(0, input_dim, 2, dtype=np.int64)
    odd = np.arange(1, input_dim, 2, dtype=np.int64)
    interlocked = np.concatenate([even, odd])
    routing = np.empty(input_dim, dtype=np.int64)
    routing[interlocked] = np.arange(input_dim, dtype=np.int64) % num_branch
    return routing


def analyze_split(
    root: Path,
    *,
    channel_regions: dict[str, np.ndarray],
    max_samples: int | None,
    soma_scale: float,
    soma_bias_mv: float,
) -> dict[str, Any]:
    shard_paths = load_shard_paths(root)
    accum = Accumulator(channel_regions)
    seen = 0
    for shard_path in shard_paths:
        with np.load(shard_path) as data:
            inputs = data["inputs"]
            targets = data["targets"]
            take = inputs.shape[0]
            if max_samples is not None:
                take = min(take, max_samples - seen)
            if take <= 0:
                break
            soma_mv = targets[:take, :, 1].astype(np.float64) / soma_scale + soma_bias_mv
            accum.add(inputs[:take], targets[:take, :, 0] > 0.5, soma_mv)
            seen += take
    return accum.finish()


def load_shard_paths(root: Path) -> list[Path]:
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = []
        for shard in manifest.get("shards", []):
            if "file" in shard:
                paths.append(root / shard["file"])
        if paths:
            return paths
    return sorted(root.glob("*.npz"))


class Accumulator:
    def __init__(self, channel_regions: dict[str, np.ndarray]) -> None:
        self.region = np.asarray(channel_regions["region"]).astype(str)
        self.sign = np.asarray(channel_regions["sign"]).astype(np.int8)
        self.channels = self.region.shape[0]
        self.total_samples = 0
        self.total_time = 0
        self.total_input_slots = 0
        self.total_events = 0
        self.total_exc = 0
        self.total_inh = 0
        self.total_spikes = 0
        self.soma_values: list[np.ndarray] = []
        self.soma_dvdt_values: list[np.ndarray] = []
        self.event_count_values: list[np.ndarray] = []
        self.exc_count_values: list[np.ndarray] = []
        self.inh_count_values: list[np.ndarray] = []
        self.window_100ms: dict[str, list[np.ndarray]] = defaultdict(list)
        self.trace_event_rate_per_ms: list[float] = []
        self.trace_spike_rate_per_1000: list[float] = []
        self.channel_event_counts = np.zeros(self.channels, dtype=np.float64)
        self.channel_region_counts: dict[str, float] = defaultdict(float)
        self.autocorr_by_lag: dict[int, list[float]] = {lag: [] for lag in (1, 5, 25, 50, 100, 200)}
        self.spike_triggered: list[np.ndarray] = []

    def add(self, inputs: np.ndarray, spikes: np.ndarray, soma_mv: np.ndarray) -> None:
        abs_inputs = np.abs(inputs.astype(np.int8, copy=False))
        event_counts = abs_inputs.sum(axis=2).astype(np.float64)
        exc_mask = inputs > 0
        inh_mask = inputs < 0
        exc_counts = exc_mask.sum(axis=2).astype(np.float64)
        inh_counts = inh_mask.sum(axis=2).astype(np.float64)

        n, t, c = inputs.shape
        if c != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {c}")
        self.total_samples += int(n)
        self.total_time += int(n * t)
        self.total_input_slots += int(n * t * c)
        self.total_events += int(event_counts.sum())
        self.total_exc += int(exc_counts.sum())
        self.total_inh += int(inh_counts.sum())
        self.total_spikes += int(spikes.sum())
        self.soma_values.append(soma_mv.astype(np.float32, copy=False).ravel())
        dvdt = np.diff(soma_mv, axis=1, prepend=soma_mv[:, :1])
        self.soma_dvdt_values.append(dvdt.astype(np.float32, copy=False).ravel())
        self.event_count_values.append(event_counts.ravel().astype(np.float32, copy=False))
        self.exc_count_values.append(exc_counts.ravel().astype(np.float32, copy=False))
        self.inh_count_values.append(inh_counts.ravel().astype(np.float32, copy=False))
        self.channel_event_counts += abs_inputs.sum(axis=(0, 1))

        for label in ("basal", "apical", "dend", "apic"):
            region_mask = self.region == label
            if region_mask.any():
                self.channel_region_counts[label] += float(abs_inputs[:, :, region_mask].sum())
        for label, mask in {
            "exc": exc_mask,
            "inh": inh_mask,
            "basal_exc": exc_mask[:, :, self.region == "basal"],
            "basal_inh": inh_mask[:, :, self.region == "basal"],
            "apical_exc": exc_mask[:, :, self.region == "apical"],
            "apical_inh": inh_mask[:, :, self.region == "apical"],
            "dend_exc": exc_mask[:, :, self.region == "dend"],
            "dend_inh": inh_mask[:, :, self.region == "dend"],
            "apic_exc": exc_mask[:, :, self.region == "apic"],
            "apic_inh": inh_mask[:, :, self.region == "apic"],
        }.items():
            if mask.shape[-1] == 0:
                continue
            self.window_100ms[label].append(window_sum(mask, 100).astype(np.float32, copy=False).ravel())
        self.window_100ms["total"].append(window_sum(abs_inputs, 100).astype(np.float32, copy=False).ravel())
        self.window_100ms["ei_diff"].append((window_sum(exc_mask, 100) - window_sum(inh_mask, 100)).astype(np.float32, copy=False).ravel())

        self.trace_event_rate_per_ms.extend((event_counts.sum(axis=1) / t).tolist())
        self.trace_spike_rate_per_1000.extend((1000.0 * spikes.sum(axis=1) / t).tolist())
        for row in event_counts:
            centered = row - row.mean()
            denom = float(np.dot(centered, centered))
            if denom <= 0:
                continue
            for lag in self.autocorr_by_lag:
                if lag >= row.shape[0]:
                    continue
                value = float(np.dot(centered[:-lag], centered[lag:]) / denom)
                self.autocorr_by_lag[lag].append(value)
        collect_spike_triggered(soma_mv, spikes, self.spike_triggered)

    def finish(self) -> dict[str, Any]:
        soma = np.concatenate(self.soma_values)
        dvdt = np.concatenate(self.soma_dvdt_values)
        event_counts = np.concatenate(self.event_count_values)
        exc_counts = np.concatenate(self.exc_count_values)
        inh_counts = np.concatenate(self.inh_count_values)
        window_stats = {
            name: summarize_array(np.concatenate(values))
            for name, values in sorted(self.window_100ms.items())
            if values
        }
        channel_rates = self.channel_event_counts / max(self.total_samples * (self.total_time // max(self.total_samples, 1)), 1)
        nonzero_channel_rates = channel_rates[channel_rates > 0]
        return {
            "samples": self.total_samples,
            "time_steps_per_sample": int(self.total_time / max(self.total_samples, 1)),
            "event_fraction": float(self.total_events / max(self.total_input_slots, 1)),
            "exc_fraction_of_events": float(self.total_exc / max(self.total_events, 1)),
            "inh_fraction_of_events": float(self.total_inh / max(self.total_events, 1)),
            "spike_rate_per_1000_steps": float(1000.0 * self.total_spikes / max(self.total_time, 1)),
            "trace_event_rate_per_ms": summarize_array(np.asarray(self.trace_event_rate_per_ms)),
            "trace_spike_rate_per_1000_steps": summarize_array(np.asarray(self.trace_spike_rate_per_1000)),
            "event_count_per_ms": summarize_array(event_counts),
            "exc_count_per_ms": summarize_array(exc_counts),
            "inh_count_per_ms": summarize_array(inh_counts),
            "event_count_fano_per_ms": fano(event_counts),
            "event_count_autocorr": {
                str(lag): float(np.mean(values)) if values else None
                for lag, values in self.autocorr_by_lag.items()
            },
            "window_100ms_counts": window_stats,
            "channel_event_rate_per_ms": {
                "mean": float(channel_rates.mean()),
                "std": float(channel_rates.std()),
                "cv_nonzero": float(nonzero_channel_rates.std() / max(nonzero_channel_rates.mean(), 1e-12)),
                "p01_p50_p99": [float(x) for x in np.quantile(channel_rates, [0.01, 0.5, 0.99])],
                "gini": gini(channel_rates),
            },
            "region_event_counts": dict(sorted(self.channel_region_counts.items())),
            "soma_mv": summarize_array(soma),
            "abs_dvdt_mv_per_step": summarize_array(np.abs(dvdt)),
            "spike_triggered_waveform": spike_triggered_summary(self.spike_triggered),
        }


def window_sum(mask: np.ndarray, size: int) -> np.ndarray:
    n, t = mask.shape[:2]
    usable = (t // size) * size
    if usable == 0:
        return np.zeros((n, 0), dtype=np.float32)
    reshaped = mask[:, :usable].reshape(n, usable // size, size, -1)
    return reshaped.sum(axis=(2, 3))


def summarize_array(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "p01": float("nan"), "p50": float("nan"), "p95": float("nan"), "p99": float("nan")}
    qs = np.quantile(values, [0.01, 0.5, 0.95, 0.99])
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p01": float(qs[0]),
        "p50": float(qs[1]),
        "p95": float(qs[2]),
        "p99": float(qs[3]),
    }


def fano(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values.var() / max(values.mean(), 1e-12))


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.all(values == 0):
        return 0.0
    sorted_values = np.sort(values)
    n = sorted_values.size
    return float((2 * np.arange(1, n + 1) @ sorted_values) / (n * sorted_values.sum()) - (n + 1) / n)


def collect_spike_triggered(soma_mv: np.ndarray, spikes: np.ndarray, out: list[np.ndarray], before: int = 20, after: int = 80) -> None:
    for row in range(soma_mv.shape[0]):
        for index in np.flatnonzero(spikes[row]):
            if index - before < 0 or index + after >= soma_mv.shape[1]:
                continue
            out.append(soma_mv[row, index - before : index + after + 1].astype(np.float32, copy=False))


def spike_triggered_summary(snippets: list[np.ndarray]) -> dict[str, Any]:
    if not snippets:
        return {"count": 0, "mean_peak_mv": None, "mean_pre_spike_mv": None, "mean_post_20_mv": None}
    arr = np.stack(snippets)
    mean = arr.mean(axis=0)
    return {
        "count": int(arr.shape[0]),
        "mean_peak_mv": float(mean.max()),
        "mean_pre_spike_mv": float(mean[19]),
        "mean_post_20_mv": float(mean[40]),
    }


def source_protocol_diff(arbor_input_protocol: str) -> list[dict[str, str]]:
    if arbor_input_protocol == "neuronio_smoothed":
        arbor_input = "NeuronIO-like smoothed basal/apical E/I rate-field approximation on the selected Arbor branches, with branch-length weighting."
    else:
        arbor_input = "Per-ms Poisson event count over all channels with branch weights, E/I balance, apic fraction, and occasional same-branch synchrony."
    return [
        {
            "axis": "cell template",
            "neuronio": "Hay et al. L5PC NEURON model, morphology `cell1.asc`, active dendrites, Ih shift 0, SK_E2 scale 1.",
            "arbor": "Allen L5 template SWC/fit from Arbor single-cell example.",
            "evidence": "NeuronIO `simulate_L5PC_and_create_dataset.py`; Arbor `src/bnn_sim/arbor_sim.py::_load_allen_template`.",
        },
        {
            "axis": "synapse mechanisms",
            "neuronio": "ProbAMPANMDA2 for NMDA condition, ProbUDFsyn2 for GABA_A; generator uses Use=1, Dep=0, Fac=0, so the MOD files' release/STP machinery is present but effectively disabled for this dataset.",
            "arbor": "Separate deterministic Exp2Syn AMPA, sourced NMDA_CA1_pyr_SC mechanism, and Exp2Syn inhibitory synapse.",
            "evidence": "NeuronIO `DefineSynapse_NMDA/GABA_A`; Arbor `_build_cell` placements.",
        },
        {
            "axis": "input process",
            "neuronio": "Coarse basal/apical E/I rates per 100 ms, 25-450 ms rate-hold intervals, 25-600 ms Gaussian smoothing, exponential instantaneous probabilities, segment-length weighting.",
            "arbor": arbor_input,
            "evidence": "NeuronIO `generate_input_spike_trains_for_simulation`; Arbor `_sample_sparse_inputs`.",
        },
        {
            "axis": "spike labeling",
            "neuronio": "Detect high-resolution local maxima above -25 mV; low-res labels use `(outputSpikeTimes - 0.5).astype(int)`.",
            "arbor": "Detect threshold crossings of raw soma voltage at a configurable threshold, currently -32 mV in calibrated fixed params.",
            "evidence": "NeuronIO spike detection block; Arbor `_threshold_spikes`.",
        },
        {
            "axis": "voltage observation",
            "neuronio": "CVode trace interpolated to 1 ms low-res soma voltage, stored as float16, then historical training/eval clips above -55 mV.",
            "arbor": "Sampled every 1 ms from a fixed-step Arbor integration trace; raw soma stored, clipped target view derived later for comparability.",
            "evidence": "NeuronIO recording/interpolation block; Arbor `_simulate_arbor_batch` and clipped target derivation.",
        },
    ]


def interpretation() -> list[str]:
    return [
        "The current Arbor teacher is not a direct reimplementation of the NeuronIO/Beniaguev simulator; it is a calibrated proxy with different morphology, channel fit, synapse mechanisms, input process, and spike-label semantics.",
        "Clipped voltage comparability is valid for old-model RMSE reporting, but it does not by itself make Arbor dynamically equivalent to NeuronIO.",
        "If the goal is to reduce the Arbor RMSE gap without touching the model, the most defensible path is to port NeuronIO-like protocol pieces into Arbor one at a time: input process/routing first, then spike-label semantics, then closer synapse mechanisms if feasible.",
        "If those pieces cannot be faithfully ported, the benchmark should report Arbor as a new teacher distribution and treat the RMSE gap as expected teacher/protocol shift.",
    ]


def compact(report: dict[str, Any]) -> dict[str, Any]:
    out = {"metadata": report["metadata"], "interpretation": report["interpretation"]}
    out["split_statistics"] = {}
    for name, stats in report["split_statistics"].items():
        out["split_statistics"][name] = {
            "samples": stats["samples"],
            "event_fraction": stats["event_fraction"],
            "spike_rate_per_1000_steps": stats["spike_rate_per_1000_steps"],
            "event_count_per_ms": stats["event_count_per_ms"],
            "window_100ms_counts": {
                key: stats["window_100ms_counts"][key]
                for key in ("total", "exc", "inh", "ei_diff")
                if key in stats["window_100ms_counts"]
            },
            "event_count_autocorr": stats["event_count_autocorr"],
            "soma_mv": stats["soma_mv"],
            "abs_dvdt_mv_per_step": stats["abs_dvdt_mv_per_step"],
            "spike_triggered_waveform": stats["spike_triggered_waveform"],
        }
    return out


def render_markdown(report: dict[str, Any]) -> str:
    nmda = report["split_statistics"]["nmda"]
    arbor = report["split_statistics"]["arbor"]
    arbor_protocol = report["metadata"].get("arbor_manifest", {}).get("input_protocol", "unknown")
    arbor_label = f"Arbor `{arbor_protocol}` target view"
    lines = [
        "# Teacher Protocol Comparison",
        "",
        "## Source Protocol Diff",
        "",
        "| axis | NeuronIO/Beniaguev | current Arbor |",
        "|---|---|---|",
    ]
    for row in report["source_protocol_diff"]:
        lines.append(f"| {row['axis']} | {row['neuronio']} | {row['arbor']} |")
    lines.extend([
        "",
        "## Split Statistics",
        "",
        f"Diagnostic sample cap: `{report['metadata']['max_samples']}` (`null` means all samples).",
        "",
        f"| metric | NeuronIO/NMDA | {arbor_label} |",
        "|---|---:|---:|",
        f"| samples | {nmda['samples']} | {arbor['samples']} |",
        f"| event fraction | {nmda['event_fraction']:.6f} | {arbor['event_fraction']:.6f} |",
        f"| spike rate / 1000 steps | {nmda['spike_rate_per_1000_steps']:.4f} | {arbor['spike_rate_per_1000_steps']:.4f} |",
        f"| event count / ms p50/p95/p99 | {qfmt(nmda['event_count_per_ms'])} | {qfmt(arbor['event_count_per_ms'])} |",
        f"| 100ms total events p50/p95/p99 | {qfmt(nmda['window_100ms_counts']['total'])} | {qfmt(arbor['window_100ms_counts']['total'])} |",
        f"| 100ms E-I diff p50/p95/p99 | {qfmt(nmda['window_100ms_counts']['ei_diff'])} | {qfmt(arbor['window_100ms_counts']['ei_diff'])} |",
        f"| event-count Fano / ms | {nmda['event_count_fano_per_ms']:.3f} | {arbor['event_count_fano_per_ms']:.3f} |",
        f"| event autocorr lag 100ms | {fmt_nullable(nmda['event_count_autocorr']['100'])} | {fmt_nullable(arbor['event_count_autocorr']['100'])} |",
        f"| channel-rate Gini | {nmda['channel_event_rate_per_ms']['gini']:.3f} | {arbor['channel_event_rate_per_ms']['gini']:.3f} |",
        f"| soma mean/std mV | {nmda['soma_mv']['mean']:.3f}/{nmda['soma_mv']['std']:.3f} | {arbor['soma_mv']['mean']:.3f}/{arbor['soma_mv']['std']:.3f} |",
        f"| |dV/dt| p95/p99 mV/step | {nmda['abs_dvdt_mv_per_step']['p95']:.3f}/{nmda['abs_dvdt_mv_per_step']['p99']:.3f} | {arbor['abs_dvdt_mv_per_step']['p95']:.3f}/{arbor['abs_dvdt_mv_per_step']['p99']:.3f} |",
        f"| spike-triggered mean peak mV | {fmt_nullable(nmda['spike_triggered_waveform']['mean_peak_mv'])} | {fmt_nullable(arbor['spike_triggered_waveform']['mean_peak_mv'])} |",
        "",
        "## Region Counts",
        "",
        "| view | region event counts |",
        "|---|---|",
        f"| NeuronIO/NMDA | `{json.dumps(nmda['region_event_counts'], sort_keys=True)}` |",
        f"| current Arbor | `{json.dumps(arbor['region_event_counts'], sort_keys=True)}` |",
        "",
        "## Interpretation",
        "",
    ])
    lines.extend(f"- {item}" for item in report["interpretation"])
    lines.append("")
    return "\n".join(lines)


def qfmt(stats: dict[str, float]) -> str:
    return f"{stats['p50']:.2f}/{stats['p95']:.2f}/{stats['p99']:.2f}"


def fmt_nullable(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
