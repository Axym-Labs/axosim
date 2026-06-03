#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-src}"

if [[ ! -f runs/verify_t3_low_input/arbor_allen_hetero_00000.npz ]]; then
  "$PYTHON_BIN" -m bnn_sim.cli make-arbor-hetero-data --config configs/sim/verify_low_vs_high_input.json
fi

if [[ ! -f runs/verify_t3_high_input/arbor_allen_hetero_00000.npz ]]; then
  "$PYTHON_BIN" -m bnn_sim.cli make-arbor-hetero-data --config configs/sim/verify_high_vs_low_input.json
fi

"$PYTHON_BIN" docs/verify-arbor-simulator/artifacts/run_neuro_bench_mvp.py "$@"
