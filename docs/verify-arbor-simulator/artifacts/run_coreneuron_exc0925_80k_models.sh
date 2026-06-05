#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PYTHONPATH="${PYTHONPATH:-src}"

TRAIN_DATA="${TRAIN_DATA:-data/hay-coreneuron-source-train32-exc0925-20260605/train}"
VAL_DATA="${VAL_DATA:-data/hay-coreneuron-source-val16-exc0925-20260605/val}"
OUT_DIR="${OUT_DIR:-docs/verify-arbor-simulator/artifacts/hay_coreneuron_exc0925_80k_models_20260605}"
PYTHON_BIN="${PYTHON_BIN:-.venv-mamba/bin/python}"
DEVICE="${DEVICE:-auto}"
RUN_MAMBA="${RUN_MAMBA:-1}"
RUN_RNN="${RUN_RNN:-1}"
PRESENTATIONS="${PRESENTATIONS:-80000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCH_SAMPLES="${EPOCH_SAMPLES:-20000}"
EPOCHS="${EPOCHS:-4}"

if (( EPOCHS * EPOCH_SAMPLES > 80000 )); then
  echo "Refusing >80k presentations: EPOCHS*EPOCH_SAMPLES=$((EPOCHS * EPOCH_SAMPLES))" >&2
  exit 2
fi
if (( PRESENTATIONS > 80000 )); then
  echo "Refusing PRESENTATIONS=$PRESENTATIONS > 80000" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$("$PYTHON_BIN" - <<'PY'
import torch

device = "cpu"
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    if f"sm_{major}{minor}" in torch.cuda.get_arch_list():
        device = "cuda"
print(device)
PY
)"
fi
echo "Using DEVICE=$DEVICE"
if [[ "$DEVICE" != "cuda" ]]; then
  echo "Refusing CPU training for this benchmark run." >&2
  exit 3
fi

EXTRA_CHECKPOINTS=()

if [[ "$RUN_MAMBA" == "1" ]]; then
  "$PYTHON_BIN" -m bnn_sim.cli train \
    --data "$TRAIN_DATA" \
    --val-data "$VAL_DATA" \
    --model-kind mamba-official \
    --model-config configs/branch_mamba_official_branchgain_integrative_pre_branchtanh_somagate_45x64.json \
    --checkpoint "$OUT_DIR/mamba_exc0925_80k.pt" \
    --best-checkpoint "$OUT_DIR/mamba_exc0925_80k_best.pt" \
    --metrics "$OUT_DIR/mamba_exc0925_80k_metrics.json" \
    --epochs "$EPOCHS" \
    --epoch-samples "$EPOCH_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
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
  EXTRA_CHECKPOINTS+=(--checkpoint "mamba_exc0925_80k=$OUT_DIR/mamba_exc0925_80k_best.pt")
fi

if [[ "$RUN_RNN" == "1" ]]; then
  "$PYTHON_BIN" -m bnn_sim.cli train \
    --data "$TRAIN_DATA" \
    --val-data "$VAL_DATA" \
    --model-kind branch-trace-rnn \
    --model-config configs/branch_trace_rnn_k3_readouthidden64_branchtanh_fixed.json \
    --checkpoint "$OUT_DIR/rnn_exc0925_80k.pt" \
    --best-checkpoint "$OUT_DIR/rnn_exc0925_80k_best.pt" \
    --metrics "$OUT_DIR/rnn_exc0925_80k_metrics.json" \
    --epochs "$EPOCHS" \
    --epoch-samples "$EPOCH_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
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
  EXTRA_CHECKPOINTS+=(--checkpoint "rnn_exc0925_80k=$OUT_DIR/rnn_exc0925_80k_best.pt")
fi

if ((${#EXTRA_CHECKPOINTS[@]})); then
  "$PYTHON_BIN" docs/verify-arbor-simulator/artifacts/evaluate_nmda_checkpoints_neuro_bench.py \
    --val-data "$VAL_DATA" \
    --output-dir "$OUT_DIR/neuro_bench_eval" \
    --max-samples 16 \
    --eval-batch-size 4 \
    --report-title "CoreNEURON/Hay Exc0925 80k Model Evaluation" \
    "${EXTRA_CHECKPOINTS[@]}"
fi
