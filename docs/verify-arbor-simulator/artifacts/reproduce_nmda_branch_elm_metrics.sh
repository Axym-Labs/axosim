#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON_BIN="${PYTHON_BIN:-.venv-mamba/bin/python}"
export PYTHONPATH="${PYTHONPATH:-src}"

RAW_TRAIN="${RAW_TRAIN:-data/raw/nmda-train}"
RAW_TEST="${RAW_TEST:-data/raw/nmda-test/Data_test}"
DATA_ROOT="${DATA_ROOT:-data/full-trace-shards}"
TRAIN_DATA="${TRAIN_DATA:-$DATA_ROOT/nmda-train}"
VAL_DATA="${VAL_DATA:-$DATA_ROOT/nmda-val}"
VAL_LINK_DIR="${VAL_LINK_DIR:-$DATA_ROOT/raw-nmda-val10-links}"
OUT_DIR="${OUT_DIR:-docs/verify-arbor-simulator/artifacts/nmda_branch_elm_reproduction_20260604}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUT_DIR"

VAL_SHARDS_EXPECTED="${VAL_SHARDS_EXPECTED:-5}"
TRAIN_SHARDS_EXPECTED="${TRAIN_SHARDS_EXPECTED:-150}"

converted_shards() {
  local path="$1"
  find "$path" -maxdepth 1 -type f -name 'neuronio_*.npz' 2>/dev/null | wc -l
}

if [[ "$(converted_shards "$VAL_DATA")" -ne "$VAL_SHARDS_EXPECTED" ]]; then
  rm -rf "$VAL_DATA"
  mkdir -p "$VAL_LINK_DIR"
  rm -f "$VAL_LINK_DIR"/*
  while IFS= read -r raw_file; do
    ln -s "$(realpath --relative-to="$VAL_LINK_DIR" "$raw_file")" "$VAL_LINK_DIR/$(basename "$raw_file")"
  done < <(find "$RAW_TEST" -maxdepth 1 -type f | sort | head -10)
  "$PYTHON_BIN" -m bnn_sim.cli convert-neuronio \
    --input "$VAL_LINK_DIR" \
    --output "$VAL_DATA" \
    --shard-size 256
fi

if [[ "$(converted_shards "$TRAIN_DATA")" -ne "$TRAIN_SHARDS_EXPECTED" ]]; then
  rm -rf "$TRAIN_DATA"
  "$PYTHON_BIN" -m bnn_sim.cli convert-neuronio \
    --input "$RAW_TRAIN" \
    --output "$TRAIN_DATA" \
    --shard-size 256
fi

"$PYTHON_BIN" -m bnn_sim.cli train \
  --preset official-fulltrace-reference \
  --data "$TRAIN_DATA" \
  --val-data "$VAL_DATA" \
  --epochs 4 \
  --epoch-samples 20000 \
  --batch-size 8 \
  --checkpoint "$OUT_DIR/branch_elm_official_fulltrace_80k.pt" \
  --best-checkpoint "$OUT_DIR/branch_elm_official_fulltrace_80k_best.pt" \
  --metrics "$OUT_DIR/branch_elm_official_fulltrace_80k_metrics.json" \
  --train-update-log-interval "${UPDATE_LOG_INTERVAL:-500}" \
  --device "$DEVICE"

"$PYTHON_BIN" -m bnn_sim.cli train \
  --preset paper-random-fulltrace \
  --data "$TRAIN_DATA" \
  --val-data "$VAL_DATA" \
  --epochs "${LONG_EPOCHS:-10}" \
  --epoch-samples "${LONG_EPOCH_SAMPLES:-80000}" \
  --shard-reuse-batches 58 \
  --checkpoint "$OUT_DIR/branch_elm_paper_random_fulltrace.pt" \
  --best-checkpoint "$OUT_DIR/branch_elm_paper_random_fulltrace_best.pt" \
  --metrics "$OUT_DIR/branch_elm_paper_random_fulltrace_metrics.json" \
  --train-update-log-interval "${UPDATE_LOG_INTERVAL:-500}" \
  --device "$DEVICE"
