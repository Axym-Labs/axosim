#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-src}"

DATA_ROOT="${DATA_ROOT:-data/arbor-allen-nmda-source-medium-20260510}"
TRAIN_DATA="${TRAIN_DATA:-$DATA_ROOT/train}"
VAL_DATA="${VAL_DATA:-$DATA_ROOT/val}"
OUT_DIR="${OUT_DIR:-docs/verify-arbor-simulator/artifacts/long_model_runs}"
DEVICE="${DEVICE:-auto}"

RUN_RNN="${RUN_RNN:-1}"
RUN_MAMBA="${RUN_MAMBA:-1}"
RNN_EPOCHS="${RNN_EPOCHS:-1}"
RNN_EPOCH_SAMPLES="${RNN_EPOCH_SAMPLES:-8000}"
RNN_BATCH_SIZE="${RNN_BATCH_SIZE:-16}"
RNN_LEARNING_RATE="${RNN_LEARNING_RATE:-5e-4}"
RNN_PREFIX="${RNN_PREFIX:-rnn_branchtanh_fixed_${RNN_EPOCH_SAMPLES}}"
RNN_UPDATE_LOG_INTERVAL="${RNN_UPDATE_LOG_INTERVAL:-0}"
MAMBA_EPOCHS="${MAMBA_EPOCHS:-1}"
MAMBA_EPOCH_SAMPLES="${MAMBA_EPOCH_SAMPLES:-40000}"
MAMBA_BATCH_SIZE="${MAMBA_BATCH_SIZE:-16}"
MAMBA_LEARNING_RATE="${MAMBA_LEARNING_RATE:-3e-4}"
MAMBA_PREFIX="${MAMBA_PREFIX:-mamba_branchtanh_somagate_${MAMBA_EPOCH_SAMPLES}}"

mkdir -p "$OUT_DIR"

if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$("$PYTHON_BIN" - <<'PY'
import torch

device = "cpu"
if torch.cuda.is_available():
    try:
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
        if arch in torch.cuda.get_arch_list():
            device = "cuda"
    except Exception:
        device = "cpu"
print(device)
PY
)"
fi
echo "Using DEVICE=$DEVICE"

EXTRA_ARGS=()

if [[ "$RUN_RNN" == "1" ]]; then
  "$PYTHON_BIN" -m bnn_sim.cli train \
    --data "$TRAIN_DATA" \
    --val-data "$VAL_DATA" \
    --model-kind branch-trace-rnn \
    --model-config configs/branch_trace_rnn_k3_readouthidden64_branchtanh_fixed.json \
    --checkpoint "$OUT_DIR/${RNN_PREFIX}.pt" \
    --best-checkpoint "$OUT_DIR/${RNN_PREFIX}_best.pt" \
    --metrics "$OUT_DIR/${RNN_PREFIX}_metrics.json" \
    --epochs "$RNN_EPOCHS" \
    --epoch-samples "$RNN_EPOCH_SAMPLES" \
    --batch-size "$RNN_BATCH_SIZE" \
    --window-size 500 \
    --ignore-start 500 \
    --window-sampling random-full-trace \
    --full-trace-length 6000 \
    --shard-reuse-batches 64 \
    --burn-in 150 \
    --learning-rate "$RNN_LEARNING_RATE" \
    --optimizer adam \
    --spike-loss-weight 0.5 \
    --soma-loss-weight 0.5 \
    --val-metric-ignore-start 500 \
    --val-soma-affine-calibration \
    --train-update-log-interval "$RNN_UPDATE_LOG_INTERVAL" \
    --device "$DEVICE"
  EXTRA_ARGS+=(--extra-checkpoint "${RNN_PREFIX}=$OUT_DIR/${RNN_PREFIX}_best.pt")
fi

if [[ "$RUN_MAMBA" == "1" ]]; then
  "$PYTHON_BIN" -m bnn_sim.cli train \
    --data "$TRAIN_DATA" \
    --val-data "$VAL_DATA" \
    --model-kind mamba-official \
    --model-config configs/branch_mamba_official_branchgain_integrative_pre_branchtanh_somagate_45x64.json \
    --checkpoint "$OUT_DIR/${MAMBA_PREFIX}.pt" \
    --best-checkpoint "$OUT_DIR/${MAMBA_PREFIX}_best.pt" \
    --metrics "$OUT_DIR/${MAMBA_PREFIX}_metrics.json" \
    --epochs "$MAMBA_EPOCHS" \
    --epoch-samples "$MAMBA_EPOCH_SAMPLES" \
    --batch-size "$MAMBA_BATCH_SIZE" \
    --window-size 500 \
    --ignore-start 500 \
    --window-sampling random-full-trace \
    --full-trace-length 6000 \
    --shard-reuse-batches 64 \
    --burn-in 150 \
    --learning-rate "$MAMBA_LEARNING_RATE" \
    --optimizer adamw \
    --weight-decay 1e-4 \
    --spike-loss-weight 0.5 \
    --soma-loss-weight 0.5 \
    --val-metric-ignore-start 500 \
    --val-soma-affine-calibration \
    --device "$DEVICE"
  EXTRA_ARGS+=(--extra-checkpoint "${MAMBA_PREFIX}=$OUT_DIR/${MAMBA_PREFIX}_best.pt")
fi

"$PYTHON_BIN" docs/verify-arbor-simulator/artifacts/run_neuro_bench_mvp.py \
  --device "$DEVICE" \
  --output-json "$OUT_DIR/neuro_bench_long_model_results.json" \
  --output-md "$OUT_DIR/neuro_bench_long_model_report.md" \
  "${EXTRA_ARGS[@]}"
