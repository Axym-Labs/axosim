#!/usr/bin/env bash
set -euo pipefail

# Generate a direct Hay/NEURON source-teacher dataset and evaluate the trained
# NMDA checkpoints with the correct runtime environments.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

NEURON_PY="${NEURON_PY:-/home/davwis/miniconda3/envs/bnn_sim/bin/python}"
BRANCH_EVAL_PY="${BRANCH_EVAL_PY:-/home/davwis/miniconda3/envs/bnn_sim/bin/python}"
MAMBA_EVAL_PY="${MAMBA_EVAL_PY:-.venv-mamba/bin/python}"

TRAIN_SAMPLES="${TRAIN_SAMPLES:-32}"
VAL_SAMPLES="${VAL_SAMPLES:-32}"
DURATION_MS="${DURATION_MS:-6000}"
SEED="${SEED:-20260605}"
DATA_ROOT="${DATA_ROOT:-data/hay-neuron-source-benchmark-${TRAIN_SAMPLES}_${VAL_SAMPLES}-20260605}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-docs/verify-arbor-simulator/artifacts}"

"$NEURON_PY" "$ARTIFACT_ROOT/generate_hay_neuron_dataset.py" \
  --output-root "$DATA_ROOT" \
  --train-samples "$TRAIN_SAMPLES" \
  --val-samples "$VAL_SAMPLES" \
  --duration-ms "$DURATION_MS" \
  --seed "$SEED" \
  --use-cvode

PYTHONPATH=src "$BRANCH_EVAL_PY" "$ARTIFACT_ROOT/evaluate_nmda_checkpoints_neuro_bench.py" \
  --report-title "Direct Hay Source NEURO-BENCH Evaluation" \
  --val-data "$DATA_ROOT/val" \
  --output-dir "$ARTIFACT_ROOT/hay_source_neuro_bench_eval_branch_${TRAIN_SAMPLES}_${VAL_SAMPLES}_20260605" \
  --max-samples "$VAL_SAMPLES" \
  --eval-batch-size 2 \
  --plot-count 3 \
  --checkpoint branch_elm_official80k="$ARTIFACT_ROOT/nmda_branch_elm_reproduction_20260604/branch_elm_official_fulltrace_80k_best.pt" \
  --checkpoint branch_elm_random800k="$ARTIFACT_ROOT/nmda_branch_elm_paper_random_800k_20260604/branch_elm_paper_random_800k_best.pt"

PYTHONPATH=src "$MAMBA_EVAL_PY" "$ARTIFACT_ROOT/evaluate_nmda_checkpoints_neuro_bench.py" \
  --report-title "Direct Hay Source NEURO-BENCH Evaluation" \
  --val-data "$DATA_ROOT/val" \
  --output-dir "$ARTIFACT_ROOT/hay_source_neuro_bench_eval_mamba_${TRAIN_SAMPLES}_${VAL_SAMPLES}_20260605" \
  --max-samples "$VAL_SAMPLES" \
  --eval-batch-size 2 \
  --plot-count 3 \
  --checkpoint mamba_compact80k="$ARTIFACT_ROOT/nmda_mamba_compact_80k_20260604/mamba_compact_80k_best.pt" \
  --checkpoint mamba_compact480k_partial="$ARTIFACT_ROOT/nmda_mamba_compact_long_b16_2p4m_20260604/mamba_compact_long_b16_2p4m_best.pt"
