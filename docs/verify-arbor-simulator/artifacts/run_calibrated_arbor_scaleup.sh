#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PY_ARBOR="${PY_ARBOR:-/home/davwis/miniconda3/envs/bnn_sim/bin/python}"
PY_TRAIN="${PY_TRAIN:-.venv-mamba/bin/python}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-256}"
VAL_SAMPLES="${VAL_SAMPLES:-64}"
EPOCHS="${EPOCHS:-20}"
EPOCH_SAMPLES="${EPOCH_SAMPLES:-4096}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
SIM_BATCH_SIZE="${SIM_BATCH_SIZE:-8}"
ARTEFACT_DATE="${ARTEFACT_DATE:-$(date +%Y%m%d)}"

DATA_RAW="${DATA_RAW:-data/arbor-calibrated-nmda-proxy-scale${TRAIN_SAMPLES}_${VAL_SAMPLES}-${ARTEFACT_DATE}}"
DATA_CLIPPED="${DATA_CLIPPED:-data/arbor-calibrated-nmda-proxy-scale${TRAIN_SAMPLES}_${VAL_SAMPLES}-clipped-${ARTEFACT_DATE}}"
RUN_DIR="${RUN_DIR:-docs/verify-arbor-simulator/artifacts/arbor_scale${TRAIN_SAMPLES}_${VAL_SAMPLES}_branch_elm_clipped_${EPOCH_SAMPLES}x${EPOCHS}_${ARTEFACT_DATE}}"
BENCH_DIR="${BENCH_DIR:-docs/verify-arbor-simulator/artifacts/arbor_scale${TRAIN_SAMPLES}_${VAL_SAMPLES}_branch_elm_clipped_neuro_bench_${ARTEFACT_DATE}}"

mkdir -p "$RUN_DIR" "$BENCH_DIR"

echo "[1/4] Generating calibrated raw Arbor data at $DATA_RAW"
PYTHONPATH=src "$PY_ARBOR" docs/verify-arbor-simulator/artifacts/generate_calibrated_arbor_pilot.py \
  --output "$DATA_RAW" \
  --train-samples "$TRAIN_SAMPLES" \
  --val-samples "$VAL_SAMPLES" \
  --shards-per-split 1 \
  --time-steps 6000 \
  --base-event-rate-hz 10 \
  --burst-probability 0 \
  --integration-dt-ms 0.05 \
  --sim-batch-size "$SIM_BATCH_SIZE" \
  --arbor-gpu-id 0

echo "[2/4] Deriving NeuronIO-compatible clipped target view at $DATA_CLIPPED"
PYTHONPATH=src "$PY_TRAIN" docs/verify-arbor-simulator/artifacts/derive_clipped_arbor_targets.py \
  --source "$DATA_RAW" \
  --output "$DATA_CLIPPED" \
  --clean

echo "[3/4] Training Branch-ELM on clipped Arbor data"
PYTHONPATH=src "$PY_TRAIN" -m bnn_sim.cli train \
  --preset paper-random-fulltrace \
  --data "$DATA_CLIPPED/train" \
  --val-data "$DATA_CLIPPED/val" \
  --checkpoint "$RUN_DIR/branch_elm_clipped.pt" \
  --best-checkpoint "$RUN_DIR/branch_elm_clipped_best.pt" \
  --metrics "$RUN_DIR/branch_elm_clipped_metrics.json" \
  --epochs "$EPOCHS" \
  --epoch-samples "$EPOCH_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate 5e-4 \
  --device cuda \
  --val-batch-size "$VAL_BATCH_SIZE" \
  --no-val-soma-affine-calibration \
  --train-update-log-interval 200 \
  --train-update-log "$RUN_DIR/branch_elm_clipped_metrics-updates.jsonl"

echo "[4/4] Evaluating Branch-ELM on NEURO-BENCH with SVG plots"
PYTHONPATH=src "$PY_TRAIN" docs/verify-arbor-simulator/artifacts/evaluate_nmda_checkpoints_neuro_bench.py \
  --val-data "$DATA_CLIPPED/val" \
  --output-dir "$BENCH_DIR" \
  --max-samples "$VAL_SAMPLES" \
  --ignore-start 500 \
  --eval-batch-size 4 \
  --device cuda \
  --checkpoint branch_elm_arbor_scale_clipped="$RUN_DIR/branch_elm_clipped_best.pt"

echo "done"
echo "raw_data=$DATA_RAW"
echo "clipped_data=$DATA_CLIPPED"
echo "run_dir=$RUN_DIR"
echo "bench_dir=$BENCH_DIR"
