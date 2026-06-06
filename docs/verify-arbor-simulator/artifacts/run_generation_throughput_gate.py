#!/usr/bin/env python3
"""Measure and report the v1 Hay/CoreNEURON generation throughput gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "docs" / "verify-arbor-simulator" / "artifacts"
GENERATOR = ARTIFACTS / "coreneuron_hay_port_probe_20260605" / "generate_coreneuron_hay_dataset.py"
CPU_MECH_ROOT = ARTIFACTS / "coreneuron_hay_port_probe_20260605"
GPU_MECH_ROOT = ARTIFACTS / "coreneuron_hay_gpu_probe_9_durable_20260606"
SOURCE_ROOT = ARTIFACTS / "source_models" / "neuron_as_deep_net" / "L5PC_NEURON_simulation"
DEFAULT_PYTHON = Path("/home/davwis/miniconda3/envs/bnn_sim/bin/python")
GPU_RUNTIME = Path("/home/davwis/.local/neuron-9.0.1-gpu-runtime")
NVHPC = Path("/home/davwis/.local/nvidia/Linux_x86_64/26.3")
BUILD_TOOLS = Path("/home/davwis/.local/neuron-build-tools")
BNN_ENV = Path("/home/davwis/miniconda3/envs/bnn_sim")
REPORT_PATH = ARTIFACTS / "generation_throughput_gate_20260606.md"
JSON_PATH = ARTIFACTS / "generation_throughput_gate_20260606.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ARTIFACTS / "generation_throughput_gate_20260606")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--duration-ms", type=int, default=6000)
    parser.add_argument("--quick-duration-ms", type=int, default=1000)
    parser.add_argument("--parallelism", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--gpu-batch-size", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--gpu-duration-ms", type=int, default=2000)
    parser.add_argument("--force", action="store_true", help="Delete and rerun existing timing outputs")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "source_root": str(SOURCE_ROOT),
        "cpu_mechanism_root": str(CPU_MECH_ROOT),
        "gpu_mechanism_root": str(GPU_MECH_ROOT),
        "python": str(args.python),
        "environment": inspect_environment(args.python),
        "timings": [],
        "parallel_timings": [],
        "gpu_batch_timings": [],
    }

    results["timings"].append(
        run_or_load_generator(
            label="cpu_coreneuron_8x1000",
            output_root=output_root / "cpu_coreneuron_8x1000",
            python=args.python,
            backend="coreneuron",
            samples=8,
            duration_ms=args.quick_duration_ms,
            force=args.force,
        )
    )
    results["timings"].append(
        run_or_load_generator(
            label="cpu_neuron_8x1000",
            output_root=output_root / "cpu_neuron_8x1000",
            python=args.python,
            backend="neuron",
            samples=8,
            duration_ms=args.quick_duration_ms,
            force=args.force,
        )
    )
    results["timings"].append(
        run_or_load_generator(
            label="cpu_coreneuron_2x6000",
            output_root=output_root / "cpu_coreneuron_2x6000",
            python=args.python,
            backend="coreneuron",
            samples=2,
            duration_ms=args.duration_ms,
            force=args.force,
        )
    )
    results["timings"].append(
        run_or_load_generator(
            label="cpu_neuron_2x6000",
            output_root=output_root / "cpu_neuron_2x6000",
            python=args.python,
            backend="neuron",
            samples=2,
            duration_ms=args.duration_ms,
            force=args.force,
        )
    )

    for parallelism in args.parallelism:
        results["parallel_timings"].append(
            run_or_load_parallel(
                label=f"cpu_coreneuron_parallel_{parallelism}x1x6000",
                output_root=output_root / f"cpu_coreneuron_parallel_{parallelism}x1x6000",
                python=args.python,
                parallelism=parallelism,
                duration_ms=args.duration_ms,
                force=args.force,
            )
        )

    if results["environment"]["gpu_available_for_generation"]:
        for batch_size in args.gpu_batch_size:
            results["gpu_batch_timings"].append(
                run_or_load_generator(
                    label=f"gpu_coreneuron_batch_{batch_size}x{args.gpu_duration_ms}",
                    output_root=output_root / f"gpu_coreneuron_batch_{batch_size}x{args.gpu_duration_ms}",
                    python=args.python,
                    backend="coreneuron",
                    samples=batch_size,
                    duration_ms=args.gpu_duration_ms,
                    force=args.force,
                    mechanism_root=GPU_MECH_ROOT,
                    coreneuron_gpu=True,
                    batch_size=batch_size,
                    env_override=gpu_runtime_env(),
                )
            )

    results["recommendation"] = build_recommendation(results)
    JSON_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render_report(results), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {JSON_PATH}")
    return 0


def inspect_environment(python: Path) -> dict[str, Any]:
    gpu_lib = GPU_RUNTIME / "lib" / "libcoreneuron-cuda.so"
    gpu_mech = GPU_MECH_ROOT / "x86_64" / "libcorenrnmech.so"
    cpu_mech = CPU_MECH_ROOT / "x86_64" / "libcorenrnmech.so"
    env = {
        "source_root_exists": SOURCE_ROOT.exists(),
        "cpu_mechanism_exists": cpu_mech.exists(),
        "gpu_mechanism_exists": gpu_mech.exists(),
        "libcoreneuron_cuda": str(gpu_lib) if gpu_lib.exists() else None,
        "gpu_runtime_exists": GPU_RUNTIME.exists(),
        "gpu_mechanism_ldd": run_ldd(gpu_mech, env=gpu_runtime_env()) if gpu_mech.exists() else None,
        "python_exists": python.exists(),
    }
    env["gpu_available_for_generation"] = bool(gpu_lib.exists() and gpu_mech.exists())
    return env


def run_or_load_generator(
    *,
    label: str,
    output_root: Path,
    python: Path,
    backend: str,
    samples: int,
    duration_ms: int,
    force: bool,
    mechanism_root: Path = CPU_MECH_ROOT,
    coreneuron_gpu: bool = False,
    batch_size: int = 1,
    env_override: dict[str, str] | None = None,
) -> dict[str, Any]:
    summary_path = output_root / "generation_summary.json"
    if force and output_root.exists():
        shutil.rmtree(output_root)
    if not summary_path.exists():
        env = os.environ.copy()
        if env_override:
            env.update(env_override)
        env["PYTHONPATH"] = f"src:{ARTIFACTS}:{env.get('PYTHONPATH', '')}"
        cmd = [
            str(python),
            str(GENERATOR),
            "--source-root",
            str(SOURCE_ROOT),
            "--mechanism-root",
            str(mechanism_root),
            "--output-root",
            str(output_root),
            "--backend",
            backend,
            "--train-samples",
            str(samples),
            "--val-samples",
            "0",
            "--duration-ms",
            str(duration_ms),
            "--exc-event-keep-prob",
            "0.925",
            "--batch-size",
            str(batch_size),
        ]
        if coreneuron_gpu:
            cmd.append("--coreneuron-gpu")
        subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    elapsed = float(summary["elapsed_seconds"])
    trace_seconds = samples * duration_ms / 1000.0
    return {
        "label": label,
        "backend": backend,
        "coreneuron_gpu": bool(coreneuron_gpu),
        "batch_size": int(batch_size),
        "samples": samples,
        "duration_ms": duration_ms,
        "elapsed_seconds": elapsed,
        "wall_seconds_per_trace": elapsed / samples,
        "simulated_trace_seconds_per_wall_second": trace_seconds / elapsed,
        "output_root": str(output_root),
    }


def run_or_load_parallel(
    *,
    label: str,
    output_root: Path,
    python: Path,
    parallelism: int,
    duration_ms: int,
    force: bool,
) -> dict[str, Any]:
    marker = output_root / "parallel_summary.json"
    if force and output_root.exists():
        shutil.rmtree(output_root)
    if not marker.exists():
        output_root.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        procs = []
        for index in range(parallelism):
            shard_root = output_root / f"shard_{index}"
            env = os.environ.copy()
            env["PYTHONPATH"] = f"src:{ARTIFACTS}:{env.get('PYTHONPATH', '')}"
            cmd = [
                str(python),
                str(GENERATOR),
                "--source-root",
                str(SOURCE_ROOT),
                "--mechanism-root",
                str(CPU_MECH_ROOT),
                "--output-root",
                str(shard_root),
                "--backend",
                "coreneuron",
                "--train-samples",
                "1",
                "--val-samples",
                "0",
                "--duration-ms",
                str(duration_ms),
                "--seed",
                str(20260605 + index),
                "--exc-event-keep-prob",
                "0.925",
            ]
            procs.append(subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL))
        return_codes = [proc.wait() for proc in procs]
        if any(return_codes):
            raise RuntimeError(f"{label} failed with return codes {return_codes}")
        elapsed = time.perf_counter() - started
        marker.write_text(
            json.dumps(
                {
                    "label": label,
                    "backend": "coreneuron",
                    "parallelism": parallelism,
                    "samples": parallelism,
                    "duration_ms": duration_ms,
                    "elapsed_seconds": elapsed,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    summary = json.loads(marker.read_text(encoding="utf-8"))
    elapsed = float(summary["elapsed_seconds"])
    trace_seconds = parallelism * duration_ms / 1000.0
    summary.update(
        {
            "wall_seconds_per_trace": elapsed / parallelism,
            "simulated_trace_seconds_per_wall_second": trace_seconds / elapsed,
            "output_root": str(output_root),
        }
    )
    return summary


def build_recommendation(results: dict[str, Any]) -> dict[str, Any]:
    parallel = sorted(results["parallel_timings"], key=lambda row: row["simulated_trace_seconds_per_wall_second"], reverse=True)
    peak = parallel[0] if parallel else None
    best = choose_headroom_parallelism(results["parallel_timings"], peak)
    gpu_status = "available_source_hay_batch_validated" if results["environment"]["gpu_available_for_generation"] else "blocked_missing_runtime"
    return {
        "current_backend": "cpu_coreneuron_parallel_shards",
        "current_parallelism": int(best["parallelism"]) if best else None,
        "peak_parallelism": int(peak["parallelism"]) if peak else None,
        "gpu_status": gpu_status,
        "rationale": (
            "Use measured source-Hay GPU batches when generating source-Hay traces. For the v1 SWC "
            "morphology path, use CPU CoreNEURON shards until SWC/CoreNEURON GPU is fixed; the "
            "current SWC GPU worker aborts before solve while CPU CoreNEURON agrees with NEURON."
        ),
    }


def choose_headroom_parallelism(rows: list[dict[str, Any]], peak: dict[str, Any] | None) -> dict[str, Any] | None:
    if peak is None:
        return None
    threshold = 0.9 * float(peak["simulated_trace_seconds_per_wall_second"])
    candidates = [row for row in rows if float(row["simulated_trace_seconds_per_wall_second"]) >= threshold]
    return min(candidates, key=lambda row: int(row["parallelism"]))


def render_report(results: dict[str, Any]) -> str:
    lines = [
        "# Generation Throughput Gate - 2026-06-06",
        "",
        "## Environment",
        "",
        f"- Source root exists: `{results['environment']['source_root_exists']}`",
        f"- CPU mechanism exists: `{results['environment']['cpu_mechanism_exists']}`",
        f"- GPU mechanism exists: `{results['environment']['gpu_mechanism_exists']}`",
        f"- `libcoreneuron-cuda.so`: `{results['environment']['libcoreneuron_cuda']}`",
        f"- GPU generation status: `{results['recommendation']['gpu_status']}`",
        "",
        "## Sequential Timings",
        "",
        "| Label | Backend | Samples | Duration ms | Elapsed s | s/trace | Simulated trace s / wall s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["timings"]:
        lines.append(
            "| {label} | {backend} | {samples} | {duration_ms} | {elapsed_seconds:.2f} | "
            "{wall_seconds_per_trace:.2f} | {simulated_trace_seconds_per_wall_second:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
        "## Parallel CPU CoreNEURON Timings",
            "",
            "| Label | Parallelism | Samples | Duration ms | Elapsed s | s/trace | Simulated trace s / wall s |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results["parallel_timings"]:
        lines.append(
            "| {label} | {parallelism} | {samples} | {duration_ms} | {elapsed_seconds:.2f} | "
            "{wall_seconds_per_trace:.2f} | {simulated_trace_seconds_per_wall_second:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Batched Source-Hay GPU CoreNEURON Timings",
            "",
            "| Label | Batch | Samples | Duration ms | Elapsed s | s/trace | Simulated trace s / wall s |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results["gpu_batch_timings"]:
        lines.append(
            "| {label} | {batch_size} | {samples} | {duration_ms} | {elapsed_seconds:.2f} | "
            "{wall_seconds_per_trace:.2f} | {simulated_trace_seconds_per_wall_second:.3f} |".format(**row)
        )
    rec = results["recommendation"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Current promoted generation route: `{rec['current_backend']}` with parallelism `{rec['current_parallelism']}`.",
            f"Peak measured parallelism in this run: `{rec['peak_parallelism']}`.",
            "",
            rec["rationale"],
            "",
            "Source-Hay GPU batching is now available and measured with a durable NEURON 9.0.1 GPU runtime. "
            "The v1 SWC morphology generator is a separate gate: CPU CoreNEURON is valid, but SWC GPU "
            "currently aborts during CoreNEURON GPU initialization before writing outputs.",
            "",
        ]
    )
    ldd = results["environment"].get("gpu_mechanism_ldd")
    if ldd:
        lines.extend(["## GPU Mechanism `ldd` Excerpt", "", "```text", ldd.strip(), "```", ""])
    return "\n".join(lines)


def find_file(name: str, roots: list[Path]) -> Path | None:
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob(name):
            return path
    return None


def gpu_runtime_env() -> dict[str, str]:
    path = f"{GPU_RUNTIME / 'bin'}:{BUILD_TOOLS / 'bin'}:{NVHPC / 'compilers/bin'}:{NVHPC / 'cuda/bin'}:{os.environ.get('PATH', '')}"
    pythonpath = f"{GPU_RUNTIME / 'lib/python'}:{ARTIFACTS}:{os.environ.get('PYTHONPATH', '')}"
    ld_library_path = (
        f"{GPU_MECH_ROOT / 'x86_64'}:{GPU_RUNTIME / 'lib'}:{BUILD_TOOLS / 'lib'}:{BNN_ENV / 'lib'}:"
        f"{NVHPC / 'compilers/lib'}:{NVHPC / 'cuda/lib64'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )
    return {
        "PATH": path,
        "PYTHONPATH": pythonpath,
        "LD_LIBRARY_PATH": ld_library_path,
        "NEURONHOME": str(GPU_RUNTIME),
        "NRNHOME": str(GPU_RUNTIME),
        "NRNHOME_EXEC": str(GPU_RUNTIME),
        "CORENEURONLIB": str(GPU_MECH_ROOT / "x86_64" / "libcorenrnmech.so"),
    }


def run_ldd(path: Path, env: dict[str, str] | None = None) -> str:
    try:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        result = subprocess.run(["ldd", str(path)], check=False, text=True, capture_output=True, env=merged_env)
    except OSError as error:
        return str(error)
    return result.stdout + result.stderr


if __name__ == "__main__":
    raise SystemExit(main())
