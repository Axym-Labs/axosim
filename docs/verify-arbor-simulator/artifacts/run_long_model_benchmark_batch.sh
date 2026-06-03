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
    --checkpoint "$OUT_DIR/rnn_branchtanh_fixed_8k.pt" \
    --best-checkpoint "$OUT_DIR/rnn_branchtanh_fixed_8k_best.pt" \
    --metrics "$OUT_DIR/rnn_branchtanh_fixed_8k_metrics.json" \
    --epochs 1 \
    --epoch-samples 8000 \
    --batch-size 16 \
    --window-size 500 \
    --ignore-start 500 \
    --window-sampling random-full-trace \
    --full-trace-length 6000 \
    --shard-reuse-batches 64 \
    --burn-in 150 \
    --learning-rate 5e-4 \
    --optimizer adam \
    --spike-loss-weight 0.5 \
    --soma-loss-weight 0.5 \
    --val-metric-ignore-start 500 \
    --val-soma-affine-calibration \
    --device "$DEVICE"
  EXTRA_ARGS+=(--extra-checkpoint "rnn_branchtanh_fixed_8k=$OUT_DIR/rnn_branchtanh_fixed_8k_best.pt")
fi

if [[ "$RUN_MAMBA" == "1" ]]; then
  "$PYTHON_BIN" -m bnn_sim.cli train \
    --data "$TRAIN_DATA" \
    --val-data "$VAL_DATA" \
    --model-kind mamba-official \
    --model-config configs/branch_mamba_official_branchgain_integrative_pre_branchtanh_somagate_45x64.json \
    --checkpoint "$OUT_DIR/mamba_branchtanh_somagate_40k.pt" \
    --best-checkpoint "$OUT_DIR/mamba_branchtanh_somagate_40k_best.pt" \
    --metrics "$OUT_DIR/mamba_branchtanh_somagate_40k_metrics.json" \
    --epochs 1 \
    --epoch-samples 40000 \
    --batch-size 16 \
    --window-size 500 \
    --ignore-start 500 \
    --window-sampling random-full-trace \
    --full-trace-length 6000 \
    --shard-reuse-batches 64 \
    --burn-in 150 \
    --learning-rate 3e-4 \
    --optimizer adamw \
    --weight-decay 1e-4 \
    --spike-loss-weight 0.5 \
    --soma-loss-weight 0.5 \
    --val-metric-ignore-start 500 \
    --val-soma-affine-calibration \
    --device "$DEVICE"
  EXTRA_ARGS+=(--extra-checkpoint "mamba_branchtanh_somagate_40k=$OUT_DIR/mamba_branchtanh_somagate_40k_best.pt")
fi

"$PYTHON_BIN" docs/verify-arbor-simulator/artifacts/run_neuro_bench_mvp.py \
  --device "$DEVICE" \
  --output-json "$OUT_DIR/neuro_bench_long_model_results.json" \
  --output-md "$OUT_DIR/neuro_bench_long_model_report.md" \
  "${EXTRA_ARGS[@]}"
