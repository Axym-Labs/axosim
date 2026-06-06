#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SAMPLES="${SAMPLES:-8}"
SEED="${SEED:-20260606}"
BACKEND="${BACKEND:-neuron}"
SOURCE_ROOT="${SOURCE_ROOT:-docs/verify-arbor-simulator/artifacts/source_models/neuron_as_deep_net/L5PC_NEURON_simulation}"
MECHANISM_ROOT="${MECHANISM_ROOT:-docs/verify-arbor-simulator/artifacts/hay_swc_template}"
TEMPLATE_HOC="${TEMPLATE_HOC:-$ROOT/docs/verify-arbor-simulator/artifacts/hay_swc_template/L5PCtemplate_swc.hoc}"
SOURCE_DATA="${SOURCE_DATA:-data/hay-coreneuron-source-val16-exc0925-20260605/val}"
OUTPUT_ROOT="${OUTPUT_ROOT:-docs/verify-arbor-simulator/artifacts/m3_normroute_suite_20260606}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [[ $# -gt 0 ]]; then
  CANDIDATES=("$@")
else
  CANDIDATES=(487664663 515249852 479770916)
fi
INTERVENTIONS=(event_dropout inh_dropout)

mkdir -p "$OUTPUT_ROOT"

for specimen in "${CANDIDATES[@]}"; do
  morphology="$ROOT/data/morphologies/allen_mouse_l5_spiny/specimen_${specimen}.swc"
  if [[ ! -f "$morphology" ]]; then
    echo "missing morphology: $morphology" >&2
    exit 1
  fi
  for intervention in "${INTERVENTIONS[@]}"; do
    out_dir="$OUTPUT_ROOT/specimen_${specimen}_${intervention}_simbase${SAMPLES}"
    if [[ "$SKIP_EXISTING" == "1" && -f "$out_dir/intervention_report.json" ]]; then
      echo "skip existing $out_dir"
      continue
    fi
    PYTHONPATH="src:docs/verify-arbor-simulator/artifacts:docs/verify-arbor-simulator/artifacts/coreneuron_hay_port_probe_20260605" \
      "$PYTHON_BIN" docs/verify-arbor-simulator/artifacts/run_coreneuron_event_dropout_pair.py \
        --backend "$BACKEND" \
        --source-root "$SOURCE_ROOT" \
        --mechanism-root "$MECHANISM_ROOT" \
        --template-hoc "$TEMPLATE_HOC" \
        --morphology "$morphology" \
        --source-data "$SOURCE_DATA" \
        --output-dir "$out_dir" \
        --samples "$SAMPLES" \
        --intervention "$intervention" \
        --dropout-probability 0.35 \
        --seed "$SEED" \
        --v-init-mv -76.0 \
        --input-routing normalized_by_type \
        --simulate-baseline
  done
done

SUMMARY_ROOT="$OUTPUT_ROOT" PYTHONPATH=src "$PYTHON_BIN" - <<'PY'
from __future__ import annotations
import json
import os
from pathlib import Path

root = Path(os.environ["SUMMARY_ROOT"])
rows = []
for report_path in sorted(root.glob("specimen_*_*_simbase*/intervention_report.json")):
    report = json.loads(report_path.read_text())
    meta = report["metadata"]
    response = report["teacher_intervention"]["teacher_response"]
    rows.append({
        "specimen": Path(meta["morphology"]).stem.replace("specimen_", ""),
        "intervention": meta["intervention"],
        "n_traces": response["n_traces"],
        "delta_soma_mean_mv": response["delta_soma_mean_mv"],
        "delta_spike_rate_per_1000_steps": response["delta_spike_rate_per_1000_steps"],
        "spike_count_delta_mean": response["delta_spike_count_trace_mean"]["mean"],
        "spike_count_delta_ci95_low": response["delta_spike_count_trace_mean"]["ci95_low"],
        "spike_count_delta_ci95_high": response["delta_spike_count_trace_mean"]["ci95_high"],
        "teacher_direction": response["teacher_direction"],
        "response_detected": response["response_detected"],
        "routing_policy": meta["input_routing"]["policy"],
        "morphology_segments": meta["input_routing"]["morphology_segments"]["mean"],
    })
summary = {
    "kind": "m3_normalized_routing_teacher_suite",
    "rows": rows,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# M3 Normalized-Routing Teacher Suite",
    "",
    "| specimen | intervention | traces | segments | delta soma mV | delta spike rate / 1000 | spike-count delta | spike-count interval | direction |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
]
def fmt(value, digits=4):
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"

for row in rows:
    lines.append(
        f"| `{row['specimen']}` | `{row['intervention']}` | `{row['n_traces']}` | "
        f"`{fmt(row['morphology_segments'], 0)}` | `{fmt(row['delta_soma_mean_mv'])}` | "
        f"`{fmt(row['delta_spike_rate_per_1000_steps'])}` | `{fmt(row['spike_count_delta_mean'])}` | "
        f"`[{fmt(row['spike_count_delta_ci95_low'])}, {fmt(row['spike_count_delta_ci95_high'])}]` | "
        f"`{row['teacher_direction']}` |"
    )
(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
