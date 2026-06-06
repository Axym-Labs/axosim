#!/usr/bin/env bash
set -euo pipefail

# Reproduce the NEURON 9.0.1 CPU-vs-GPU CoreNEURON/Hay validation on this
# machine after the local NEURON 9/NVHPC build has been prepared.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ART="${ART:-$ROOT/docs/verify-arbor-simulator/artifacts/coreneuron_hay_gpu_probe_9_durable_20260606}"
PROBE="$ROOT/docs/verify-arbor-simulator/artifacts/coreneuron_hay_port_probe_20260605/run_coreneuron_hay_probe.py"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT/docs/verify-arbor-simulator/artifacts/source_models/neuron_as_deep_net/L5PC_NEURON_simulation}"
PYTHON="${PYTHON:-/home/davwis/miniconda3/envs/bnn_sim/bin/python}"
NRN_BUILD="${NRN_BUILD:-/home/davwis/.local/neuron-9.0.1-gpu-runtime}"
NVHPC="${NVHPC:-/home/davwis/.local/nvidia/Linux_x86_64/26.3}"
BUILD_TOOLS="${BUILD_TOOLS:-/home/davwis/.local/neuron-build-tools}"
BNN_ENV="${BNN_ENV:-/home/davwis/miniconda3/envs/bnn_sim}"

if [[ ! -d "$NRN_BUILD/lib/python/neuron" ]]; then
  echo "Missing NEURON 9 build at $NRN_BUILD" >&2
  exit 2
fi

if [[ ! -f "$ART/x86_64/libcorenrnmech.so" ]]; then
  echo "Missing GPU mechanism library at $ART/x86_64/libcorenrnmech.so" >&2
  exit 2
fi

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "Missing source Hay root at $SOURCE_ROOT" >&2
  exit 2
fi

export PATH="$NRN_BUILD/bin:$NVHPC/compilers/bin:$NVHPC/cuda/bin:$PATH"
export PYTHONPATH="$NRN_BUILD/lib/python:$ROOT/docs/verify-arbor-simulator/artifacts:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ART/x86_64:$NRN_BUILD/lib:$BUILD_TOOLS/lib:$BNN_ENV/lib:$NVHPC/compilers/lib:$NVHPC/cuda/lib64:${LD_LIBRARY_PATH:-}"
export NEURONHOME="$NRN_BUILD"
export NRNHOME="$NRN_BUILD"
export NRNHOME_EXEC="$NRN_BUILD"
export CORENEURONLIB="$ART/x86_64/libcorenrnmech.so"

cd "$ART"
"$PYTHON" "$PROBE" --source-root "$SOURCE_ROOT" --mechanism-root "$ART" --backend neuron --samples 8 --duration-ms 500 --output neuron9_cpu_8x500ms_validation.json
"$PYTHON" "$PROBE" --source-root "$SOURCE_ROOT" --mechanism-root "$ART" --backend coreneuron --coreneuron-gpu --samples 8 --duration-ms 500 --output neuron9_gpu_8x500ms_validation.json

"$PYTHON" - <<'PY'
import json
from pathlib import Path

p = Path.cwd()
cpu = json.loads((p / "neuron9_cpu_8x500ms_validation.json").read_text())
gpu = json.loads((p / "neuron9_gpu_8x500ms_validation.json").read_text())
keys = [
    k
    for k in cpu["rows"][0]
    if isinstance(cpu["rows"][0][k], (int, float))
    and k not in {"sample_index", "sample_seed", "recorded_points"}
]
print(f"rows: cpu={len(cpu['rows'])} gpu={len(gpu['rows'])}")
print(f"elapsed_seconds: cpu={cpu['elapsed_seconds']:.3f} gpu={gpu['elapsed_seconds']:.3f}")
for key in sorted(keys):
    max_delta = max(abs(float(a[key]) - float(b[key])) for a, b in zip(cpu["rows"], gpu["rows"]))
    print(f"{key}: max_abs_delta={max_delta:.12g}")
PY
