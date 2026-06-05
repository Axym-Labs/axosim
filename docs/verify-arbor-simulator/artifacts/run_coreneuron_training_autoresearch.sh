#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PYTHONPATH="${PYTHONPATH:-src}"

TRAIN_DATA="${TRAIN_DATA:-data/hay-coreneuron-source-train32-exc0925-20260605/train}"
VAL_DATA="${VAL_DATA:-data/hay-coreneuron-source-val16-exc0925-20260605/val}"
OUT_DIR="${OUT_DIR:-docs/verify-arbor-simulator/artifacts/coreneuron_training_autoresearch_20260605}"
REPORT_PATH="${REPORT_PATH:-docs/verify-arbor-simulator/artifacts/coreneuron_training_autoresearch_20260605.md}"
PYTHON_BIN="${PYTHON_BIN:-.venv-mamba/bin/python}"
CANDIDATES="${CANDIDATES:-baseline_repro longctx1000 soma075 slope005 sparse005 official_sampling}"
EPOCHS="${EPOCHS:-4}"
EPOCH_SAMPLES="${EPOCH_SAMPLES:-20000}"
MAX_PRESENTATIONS="${MAX_PRESENTATIONS:-80000}"
PREFETCH_BATCHES="${PREFETCH_BATCHES:-2}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
GPU_IDS="${GPU_IDS:-}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

if (( EPOCHS * EPOCH_SAMPLES > MAX_PRESENTATIONS )); then
  echo "Refusing >${MAX_PRESENTATIONS} presentations: EPOCHS*EPOCH_SAMPLES=$((EPOCHS * EPOCH_SAMPLES))" >&2
  exit 2
fi

if [[ "$DEVICE" != "cuda" ]]; then
  echo "Refusing non-CUDA training for this autoresearch loop: DEVICE=$DEVICE" >&2
  exit 3
fi

if [[ ! -d "$TRAIN_DATA" ]]; then
  echo "Missing TRAIN_DATA=$TRAIN_DATA" >&2
  exit 4
fi
if [[ ! -d "$VAL_DATA" ]]; then
  echo "Missing VAL_DATA=$VAL_DATA" >&2
  exit 4
fi

if [[ -z "$GPU_IDS" ]]; then
  GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
fi
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if ((${#GPU_ARRAY[@]} == 0)) || [[ -z "${GPU_ARRAY[0]}" ]]; then
  echo "No GPU ids available. Set GPU_IDS explicitly if nvidia-smi is unavailable." >&2
  exit 5
fi
if [[ -z "$MAX_PARALLEL" ]]; then
  MAX_PARALLEL="${#GPU_ARRAY[@]}"
fi
if (( MAX_PARALLEL < 1 )); then
  echo "MAX_PARALLEL must be >=1" >&2
  exit 5
fi

mkdir -p "$OUT_DIR/candidates" .codex/experiments

"$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
major, minor = torch.cuda.get_device_capability()
arch = f"sm_{major}{minor}"
if arch not in torch.cuda.get_arch_list():
    raise SystemExit(f"CUDA arch {arch} is not supported by this torch build")
print(f"CUDA OK: {torch.cuda.get_device_name()} ({arch})")
PY

common_train_args=(
  -m bnn_sim.cli train
  --data "$TRAIN_DATA"
  --val-data "$VAL_DATA"
  --model-kind mamba-official
  --model-config configs/branch_mamba_official_branchgain_integrative_pre_branchtanh_somagate_45x64.json
  --epochs "$EPOCHS"
  --epoch-samples "$EPOCH_SAMPLES"
  --ignore-start 500
  --full-trace-length 6000
  --learning-rate 3e-4
  --optimizer adamw
  --weight-decay 1e-4
  --val-metric-ignore-start 500
  --val-soma-affine-calibration
  --prefetch-batches "$PREFETCH_BATCHES"
  --pin-memory
  --seed "$SEED"
  --device "$DEVICE"
)

candidate_args() {
  local name="$1"
  case "$name" in
    baseline_repro)
      printf '%s\n' "--window-size" "500" "--burn-in" "150" "--batch-size" "16" "--window-sampling" "random-full-trace" "--shard-reuse-batches" "64" "--spike-loss-weight" "0.5" "--soma-loss-weight" "0.5"
      ;;
    longctx1000)
      printf '%s\n' "--window-size" "1000" "--burn-in" "300" "--batch-size" "8" "--window-sampling" "random-full-trace" "--shard-reuse-batches" "64" "--spike-loss-weight" "0.5" "--soma-loss-weight" "0.5"
      ;;
    soma075)
      printf '%s\n' "--window-size" "500" "--burn-in" "150" "--batch-size" "16" "--window-sampling" "random-full-trace" "--shard-reuse-batches" "64" "--spike-loss-weight" "0.25" "--soma-loss-weight" "0.75"
      ;;
    slope005)
      printf '%s\n' "--window-size" "500" "--burn-in" "150" "--batch-size" "16" "--window-sampling" "random-full-trace" "--shard-reuse-batches" "64" "--spike-loss-weight" "0.5" "--soma-loss-weight" "0.5" "--soma-slope-loss-weight" "0.05"
      ;;
    sparse005)
      printf '%s\n' "--window-size" "500" "--burn-in" "150" "--batch-size" "16" "--window-sampling" "random-full-trace" "--shard-reuse-batches" "64" "--spike-loss-weight" "0.5" "--soma-loss-weight" "0.5" "--sparse-soma-loss-weight" "0.05"
      ;;
    official_sampling)
      printf '%s\n' "--window-size" "500" "--burn-in" "150" "--batch-size" "16" "--window-sampling" "official-full-trace" "--file-load-fraction" "0.3" "--spike-loss-weight" "0.5" "--soma-loss-weight" "0.5"
      ;;
    *)
      echo "Unknown candidate: $name" >&2
      return 2
      ;;
  esac
}

candidate_note() {
  local name="$1"
  case "$name" in
    baseline_repro) echo "Control rerun of accepted 80k Mamba protocol." ;;
    longctx1000) echo "Longer context/burn-in to test slow-state conditioning." ;;
    soma075) echo "Voltage-heavy objective for the headline RMSE metric." ;;
    slope005) echo "Small voltage-slope auxiliary loss for dynamics fidelity." ;;
    sparse005) echo "Sparse soma emphasis on rare voltage/high-dVdt/event regimes." ;;
    official_sampling) echo "Official-style full-trace file/time sampling policy." ;;
  esac
}

run_candidate() {
  local gpu_id="$1"
  local name="$2"
  local candidate_dir="$OUT_DIR/candidates/$name"
  local checkpoint="$candidate_dir/${name}.pt"
  local best_checkpoint="$candidate_dir/${name}_best.pt"
  local metrics="$candidate_dir/${name}_metrics.json"
  local train_log="$candidate_dir/train.log"
  local eval_log="$candidate_dir/eval.log"
  local status="$candidate_dir/status.txt"
  local eval_dir="$candidate_dir/neuro_bench_eval"
  local args=()

  mkdir -p "$candidate_dir"
  mapfile -t args < <(candidate_args "$name")
  {
    echo "candidate=$name"
    echo "gpu_id=$gpu_id"
    echo "note=$(candidate_note "$name")"
    echo "presentations=$((EPOCHS * EPOCH_SAMPLES))"
    echo "args=${args[*]}"
    echo "started=$(date -Is)"
  } > "$status"

  if [[ ! -f "$best_checkpoint" || ! -f "$metrics" ]]; then
    echo "[$name] training on GPU $gpu_id"
    if ! CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "${common_train_args[@]}" \
      --checkpoint "$checkpoint" \
      --best-checkpoint "$best_checkpoint" \
      --metrics "$metrics" \
      "${args[@]}" > "$train_log" 2>&1; then
      echo "failed_train=$(date -Is)" >> "$status"
      return 1
    fi
    echo "trained=$(date -Is)" >> "$status"
  else
    echo "[$name] training artifacts already exist; skipping train"
    echo "train_skipped=$(date -Is)" >> "$status"
  fi

  if [[ ! -f "$eval_dir/results.json" ]]; then
    echo "[$name] evaluating"
    if ! CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" docs/verify-arbor-simulator/artifacts/evaluate_nmda_checkpoints_neuro_bench.py \
      --val-data "$VAL_DATA" \
      --output-dir "$eval_dir" \
      --checkpoint "$name=$best_checkpoint" \
      --max-samples 16 \
      --eval-batch-size 4 \
      --device "$DEVICE" \
      --report-title "CoreNEURON/Hay Training Autoresearch: $name" > "$eval_log" 2>&1; then
      echo "failed_eval=$(date -Is)" >> "$status"
      return 1
    fi
    echo "evaluated=$(date -Is)" >> "$status"
  else
    echo "[$name] evaluation artifacts already exist; skipping eval"
    echo "eval_skipped=$(date -Is)" >> "$status"
  fi
  echo "completed=$(date -Is)" >> "$status"
}

pids=()
active=0
failures=0
gpu_cursor=0
read -r -a CANDIDATE_ARRAY <<< "$CANDIDATES"
for candidate in "${CANDIDATE_ARRAY[@]}"; do
  gpu_id="${GPU_ARRAY[$((gpu_cursor % ${#GPU_ARRAY[@]}))]}"
  gpu_cursor=$((gpu_cursor + 1))
  run_candidate "$gpu_id" "$candidate" &
  pids+=("$!")
  active=$((active + 1))
  if (( active >= MAX_PARALLEL )); then
    if ! wait -n; then
      failures=$((failures + 1))
    fi
    active=$((active - 1))
  fi
done
while (( active > 0 )); do
  if ! wait -n; then
    failures=$((failures + 1))
  fi
  active=$((active - 1))
done

"$PYTHON_BIN" - "$OUT_DIR" "$REPORT_PATH" "$CANDIDATES" "$TRAIN_DATA" "$VAL_DATA" "$EPOCHS" "$EPOCH_SAMPLES" <<'PY'
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

out_dir = Path(sys.argv[1])
report_path = Path(sys.argv[2])
candidates = sys.argv[3].split()
train_data = sys.argv[4]
val_data = sys.argv[5]
epochs = int(sys.argv[6])
epoch_samples = int(sys.argv[7])

def get(d, path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def regime_rmse(m2, label):
    rows = get(m2, ["regime_error"], []) or get(m2, ["spike_proximity"], [])
    for row in rows:
        if row.get("regime_label") == label:
            return row.get("rmse_mv") or row.get("rmse")
    return None

def fmt(value, digits=4):
    if value is None:
        return "NA"
    try:
        if math.isnan(float(value)):
            return "NA"
    except Exception:
        return str(value)
    return f"{float(value):.{digits}f}"

rows = []
for name in candidates:
    eval_path = out_dir / "candidates" / name / "neuro_bench_eval" / "results.json"
    status_path = out_dir / "candidates" / name / "status.txt"
    status = status_path.read_text(encoding="utf-8") if status_path.exists() else ""
    if not eval_path.exists():
        rows.append({"name": name, "status": "missing_eval" if "failed" not in status else "failed", "status_text": status})
        continue
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    model = data["models"][name]
    bench = model["benchmark"]
    m1 = bench.get("m1_trace_fidelity") or bench.get("m1_open_loop") or {}
    m1_open = bench.get("m1_open_loop_fidelity") or {}
    m2 = bench["m2_regimes"]
    rows.append(
        {
            "name": name,
            "status": "ok",
            "rmse": m1.get("soma_rmse_mv", m1_open.get("rmse_mv")),
            "auc": m1.get("spike_auc", m1_open.get("auc")),
            "near": regime_rmse(m2, "near_spike"),
            "count_mae0": m1.get("spike_count_mae_at_threshold_0", m1.get("spike_count_mae_per_trace")),
            "timing_f1": get(m2, ["spike_timing_prevalence_matched", "f1_at_2steps"]),
            "worst": get(model, ["trace_error_summary", "max_rmse_mv"]),
            "flags": ",".join(model.get("sanity", {}).get("flags", [])) or "none",
            "plots": bool(model.get("plots")),
            "report": str(eval_path.parent / "report.md"),
        }
    )

ok_rows = [row for row in rows if row["status"] == "ok"]
baseline = next((row for row in ok_rows if row["name"] == "baseline_repro"), None)
best = min(ok_rows, key=lambda row: row["rmse"]) if ok_rows else None

lines = [
    "# CoreNEURON/Hay Training Autoresearch",
    "",
    f"Created: `{time.strftime('%Y-%m-%d %H:%M:%S %Z')}`.",
    "",
    "## Setup",
    "",
    f"- Train data: `{train_data}`.",
    f"- Validation data: `{val_data}`.",
    f"- Presentation cap: `{epochs * epoch_samples}` presentations per candidate.",
    "- Simulator/teacher fixed: accepted CoreNEURON/Hay `exc_event_keep_prob=0.925`, `inh_event_keep_prob=1.0`.",
    "- Interpretation guard: these are training/data-policy interventions, not simulator-fidelity improvements.",
    "",
    "## Results",
    "",
    "| candidate | intervention | RMSE mV | AUC | near-spike RMSE | count MAE @0 | timing F1 matched | worst trace RMSE | sanity flags | plots |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
]
notes = {
    "baseline_repro": "control rerun",
    "longctx1000": "1000-step context, 300 burn-in",
    "soma075": "0.25 spike / 0.75 soma loss",
    "slope005": "soma-slope loss 0.05",
    "sparse005": "sparse soma loss 0.05",
    "official_sampling": "official-style window sampling",
}
for row in rows:
    if row["status"] != "ok":
        lines.append(f"| `{row['name']}` | {notes.get(row['name'], row['status'])} | NA | NA | NA | NA | NA | NA | `{row['status']}` | no |")
        continue
    lines.append(
        f"| `{row['name']}` | {notes.get(row['name'], '')} | `{fmt(row['rmse'])}` | `{fmt(row['auc'])}` | "
        f"`{fmt(row['near'])}` | `{fmt(row['count_mae0'])}` | `{fmt(row['timing_f1'])}` | "
        f"`{fmt(row['worst'])}` | `{row['flags']}` | `{row['plots']}` |"
    )

lines.extend(["", "## Decision", ""])
if best is None:
    lines.append("No completed candidate produced NEURO-BENCH metrics.")
elif baseline is None:
    lines.append(f"Best completed RMSE was `{best['name']}` at `{fmt(best['rmse'])} mV`, but no same-loop baseline completed, so treat ranking as provisional.")
else:
    delta = best["rmse"] - baseline["rmse"]
    lines.append(
        f"Best completed RMSE was `{best['name']}` at `{fmt(best['rmse'])} mV`, "
        f"versus same-loop baseline `{fmt(baseline['rmse'])} mV` "
        f"(`{fmt(delta)}` mV delta)."
    )
    if best["name"] != "baseline_repro" and delta < -0.02:
        lines.append("This is worth a follow-up validation rerun on a fresh seed/split before accepting as a paper default.")
    else:
        lines.append("No non-baseline candidate cleared a meaningful RMSE-improvement threshold in this loop.")

lines.extend(
    [
        "",
        "## Artifacts",
        "",
        f"- Candidate directories: `{out_dir}/candidates/<candidate>/`.",
        "- Each completed candidate has `neuro_bench_eval/results.json`, `report.md`, and diagnostic SVG trace plots.",
    ]
)
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

results_tsv = Path(".codex/experiments/results.tsv")
if results_tsv.exists():
    with results_tsv.open("a", encoding="utf-8") as handle:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        for row in rows:
            if row["status"] != "ok":
                handle.write(f"{stamp}\tcoreneuron_training_autoresearch\t{row['name']}\tstatus\tNA\trecord\t{row['status']}\n")
                continue
            decision = "candidate-best" if best and row["name"] == best["name"] else "record"
            handle.write(
                f"{stamp}\tcoreneuron_training_autoresearch\t{row['name']}\tm1_rmse_mv\t{row['rmse']:.10f}\t{decision}\t"
                f"auc={row['auc']:.6f}; near_spike_rmse={fmt(row['near'])}; timing_f1={fmt(row['timing_f1'])}; flags={row['flags']}\n"
            )
PY

echo "Wrote $REPORT_PATH"
if (( failures > 0 )); then
  echo "$failures candidate job(s) failed; see $OUT_DIR/candidates/*/*.log" >&2
  exit 10
fi
