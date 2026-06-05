#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PYTHONPATH="${PYTHONPATH:-src}"

OUT_DIR="${OUT_DIR:-docs/verify-arbor-simulator/artifacts/coreneuron_gmax_autoresearch_20260605}"
GEN_PY="${GEN_PY:-/home/davwis/miniconda3/envs/bnn_sim/bin/python}"
TRAIN_PY="${TRAIN_PY:-.venv-mamba/bin/python}"
GENERATOR="docs/verify-arbor-simulator/artifacts/coreneuron_hay_port_probe_20260605/generate_coreneuron_hay_dataset.py"
EVAL_SCRIPT="docs/verify-arbor-simulator/artifacts/evaluate_nmda_checkpoints_neuro_bench.py"
MECH_ROOT="${MECH_ROOT:-docs/verify-arbor-simulator/artifacts/coreneuron_hay_port_probe_20260605}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-32}"
VAL_SAMPLES="${VAL_SAMPLES:-16}"
DURATION_MS="${DURATION_MS:-6000}"
EPOCHS="${EPOCHS:-4}"
EPOCH_SAMPLES="${EPOCH_SAMPLES:-20000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RUN_TRAIN="${RUN_TRAIN:-1}"
SELECTED="${SELECTED:-}"

if (( EPOCHS * EPOCH_SAMPLES > 80000 )); then
  echo "Refusing >80k presentations: EPOCHS*EPOCH_SAMPLES=$((EPOCHS * EPOCH_SAMPLES))" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

# name exc_keep inh_keep exc_gmax_scale inh_gmax_scale
CANDIDATES=(
  "keep0925 0.925 1.0 1.0 1.0"
  "gexc090 1.0 1.0 0.90 1.0"
  "gexc0925 1.0 1.0 0.925 1.0"
  "gexc095 1.0 1.0 0.95 1.0"
  "ginh105 1.0 1.0 1.0 1.05"
  "ginh110 1.0 1.0 1.0 1.10"
  "gexc095_ginh105 1.0 1.0 0.95 1.05"
)

for row in "${CANDIDATES[@]}"; do
  read -r name exc_keep inh_keep exc_gmax inh_gmax <<<"$row"
  cand_dir="$OUT_DIR/candidates/$name"
  if [[ ! -f "$cand_dir/val/generation_report.json" ]]; then
    "$GEN_PY" "$GENERATOR" \
      --mechanism-root "$MECH_ROOT" \
      --output-root "$cand_dir" \
      --backend coreneuron \
      --train-samples 0 \
      --val-samples "$VAL_SAMPLES" \
      --duration-ms "$DURATION_MS" \
      --exc-event-keep-prob "$exc_keep" \
      --inh-event-keep-prob "$inh_keep" \
      --exc-gmax-scale "$exc_gmax" \
      --inh-gmax-scale "$inh_gmax"
  fi
done

"$GEN_PY" - "$OUT_DIR" "${CANDIDATES[@]}" <<'PY'
import json
import math
import sys
from pathlib import Path

out = Path(sys.argv[1])
candidates = []
for raw in sys.argv[2:]:
    name, exc_keep, inh_keep, exc_gmax, inh_gmax = raw.split()
    report = json.loads((out / "candidates" / name / "val" / "generation_report.json").read_text())
    agg = report["aggregate"]
    stat = {
        "name": name,
        "exc_event_keep_prob": float(exc_keep),
        "inh_event_keep_prob": float(inh_keep),
        "exc_gmax_scale": float(exc_gmax),
        "inh_gmax_scale": float(inh_gmax),
        "event_fraction": agg["event_fraction"]["mean"],
        "exc_fraction": agg["exc_fraction"]["mean"],
        "inh_fraction": agg["inh_fraction"]["mean"],
        "spike_rate_hz": agg["spike_rate_hz"]["mean"],
        "clipped_soma_mean_mv": agg["clipped_soma_mean_mv"]["mean"],
        "clipped_soma_std_mv": agg["clipped_soma_std_mv"]["mean"],
        "clipped_abs_dvdt_p95": agg["clipped_abs_dvdt_p95"]["mean"],
        "clipped_abs_dvdt_p99": agg["clipped_abs_dvdt_p99"]["mean"],
    }
    target = {
        "spike_rate_hz": 0.9896,
        "clipped_soma_mean_mv": -67.595,
        "clipped_soma_std_mv": 3.236,
        "clipped_abs_dvdt_p95": 0.438,
        "clipped_abs_dvdt_p99": 0.687,
    }
    stat["teacher_objective"] = (
        abs(math.log((stat["spike_rate_hz"] + 0.05) / (target["spike_rate_hz"] + 0.05)))
        + abs(stat["clipped_soma_mean_mv"] - target["clipped_soma_mean_mv"]) / 2.0
        + abs(stat["clipped_soma_std_mv"] - target["clipped_soma_std_mv"]) / 1.0
        + abs(stat["clipped_abs_dvdt_p95"] - target["clipped_abs_dvdt_p95"]) / 0.2
        + abs(stat["clipped_abs_dvdt_p99"] - target["clipped_abs_dvdt_p99"]) / 0.3
    )
    candidates.append(stat)

candidates.sort(key=lambda x: x["teacher_objective"])
payload = {
    "target": target,
    "selection_rule": "rank by spike-rate log error plus normalized clipped soma mean/std and clipped dV/dt p95/p99 errors; model metrics are not used for selection",
    "candidates": candidates,
    "selected": candidates[0],
}
(out / "teacher_candidate_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

lines = [
    "# CoreNEURON/Hay Gmax Autoresearch",
    "",
    "Selection uses teacher statistics only; model RMSE is not part of the teacher-setting objective.",
    "",
    "| rank | candidate | objective | exc keep | exc gmax | inh gmax | event frac | spike Hz | mean/std mV | dV/dt p95/p99 |",
    "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for i, row in enumerate(candidates, start=1):
    lines.append(
        "| {rank} | `{name}` | `{objective:.4f}` | `{exc_keep:.3g}` | `{exc_gmax:.3g}` | `{inh_gmax:.3g}` | `{event:.6f}` | `{spike:.4f}` | `{mean:.4f}/{std:.4f}` | `{dv95:.4f}/{dv99:.4f}` |".format(
            rank=i,
            name=row["name"],
            objective=row["teacher_objective"],
            exc_keep=row["exc_event_keep_prob"],
            exc_gmax=row["exc_gmax_scale"],
            inh_gmax=row["inh_gmax_scale"],
            event=row["event_fraction"],
            spike=row["spike_rate_hz"],
            mean=row["clipped_soma_mean_mv"],
            std=row["clipped_soma_std_mv"],
            dv95=row["clipped_abs_dvdt_p95"],
            dv99=row["clipped_abs_dvdt_p99"],
        )
    )
lines += [
    "",
    f"Selected candidate: `{candidates[0]['name']}`.",
]
(out / "teacher_candidate_summary.md").write_text("\n".join(lines) + "\n")
PY

if [[ -z "$SELECTED" ]]; then
  SELECTED="$("$GEN_PY" - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1]) / "teacher_candidate_summary.json"
print(json.loads(p.read_text())["selected"]["name"])
PY
)"
fi

if [[ "$RUN_TRAIN" != "1" ]]; then
  echo "Selected=$SELECTED; RUN_TRAIN=$RUN_TRAIN, stopping after teacher sweep."
  exit 0
fi

selected_row=""
for row in "${CANDIDATES[@]}"; do
  read -r name _ <<<"$row"
  if [[ "$name" == "$SELECTED" ]]; then
    selected_row="$row"
    break
  fi
done
if [[ -z "$selected_row" ]]; then
  echo "Unknown selected candidate: $SELECTED" >&2
  exit 2
fi
read -r name exc_keep inh_keep exc_gmax inh_gmax <<<"$selected_row"

dataset="$OUT_DIR/datasets/$name"
if [[ ! -f "$dataset/val/generation_report.json" ]]; then
  "$GEN_PY" "$GENERATOR" \
    --mechanism-root "$MECH_ROOT" \
    --output-root "$dataset" \
    --backend coreneuron \
    --train-samples "$TRAIN_SAMPLES" \
    --val-samples "$VAL_SAMPLES" \
    --duration-ms "$DURATION_MS" \
    --exc-event-keep-prob "$exc_keep" \
    --inh-event-keep-prob "$inh_keep" \
    --exc-gmax-scale "$exc_gmax" \
    --inh-gmax-scale "$inh_gmax"
fi

device="$("$TRAIN_PY" - <<'PY'
import torch
device = "cpu"
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    if f"sm_{major}{minor}" in torch.cuda.get_arch_list():
        device = "cuda"
print(device)
PY
)"
if [[ "$device" != "cuda" ]]; then
  echo "Refusing CPU training for this benchmark run." >&2
  exit 3
fi

model_dir="$OUT_DIR/models/$name"
mkdir -p "$model_dir"
"$TRAIN_PY" -m bnn_sim.cli train \
  --data "$dataset/train" \
  --val-data "$dataset/val" \
  --model-kind mamba-official \
  --model-config configs/branch_mamba_official_branchgain_integrative_pre_branchtanh_somagate_45x64.json \
  --checkpoint "$model_dir/mamba_${name}_80k.pt" \
  --best-checkpoint "$model_dir/mamba_${name}_80k_best.pt" \
  --metrics "$model_dir/mamba_${name}_80k_metrics.json" \
  --epochs "$EPOCHS" \
  --epoch-samples "$EPOCH_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --window-size 500 \
  --ignore-start 500 \
  --window-sampling random-full-trace \
  --full-trace-length "$DURATION_MS" \
  --shard-reuse-batches 64 \
  --burn-in 150 \
  --learning-rate 3e-4 \
  --optimizer adamw \
  --weight-decay 1e-4 \
  --spike-loss-weight 0.5 \
  --soma-loss-weight 0.5 \
  --val-metric-ignore-start 500 \
  --val-soma-affine-calibration \
  --device "$device"

"$TRAIN_PY" "$EVAL_SCRIPT" \
  --val-data "$dataset/val" \
  --output-dir "$model_dir/neuro_bench_eval" \
  --max-samples "$VAL_SAMPLES" \
  --eval-batch-size 4 \
  --report-title "CoreNEURON/Hay ${name} 80k Mamba Evaluation" \
  --checkpoint "mamba_${name}_80k=$model_dir/mamba_${name}_80k_best.pt"
