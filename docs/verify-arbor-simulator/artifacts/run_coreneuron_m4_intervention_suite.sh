#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

NEURON_PY="${NEURON_PY:-/home/davwis/miniconda3/envs/bnn_sim/bin/python}"
EVAL_PY="${EVAL_PY:-.venv-mamba/bin/python}"
SAMPLES="${SAMPLES:-16}"
DATE_TAG="${DATE_TAG:-20260606}"
SOURCE_DATA="${SOURCE_DATA:-data/hay-coreneuron-source-val16-exc0925-20260605/val}"
SOURCE_REPO="${SOURCE_REPO:-docs/verify-arbor-simulator/artifacts/source_models/neuron_as_deep_net}"
SOURCE_ROOT="${SOURCE_ROOT:-$SOURCE_REPO/L5PC_NEURON_simulation}"

if [[ ! -f "$SOURCE_ROOT/L5PCtemplate_2.hoc" ]]; then
  mkdir -p "$(dirname "$SOURCE_REPO")"
  git clone --depth 1 https://github.com/SelfishGene/neuron_as_deep_net.git "$SOURCE_REPO"
fi

declare -a INTERVENTIONS=(
  "event_dropout --dropout-probability 0.35"
  "exc_dropout --dropout-probability 0.35"
  "inh_dropout --dropout-probability 0.35"
  "temporal_jitter --jitter-steps 2"
  "site_silence --site-start-fraction 0.25 --site-fraction 0.25"
)

for spec in "${INTERVENTIONS[@]}"; do
  # shellcheck disable=SC2206
  parts=($spec)
  intervention="${parts[0]}"
  out_dir="docs/verify-arbor-simulator/artifacts/coreneuron_m4_${intervention}_${SAMPLES}_${DATE_TAG}"
  eval_dir="docs/verify-arbor-simulator/artifacts/coreneuron_neuro_bench_m4_${intervention}_${SAMPLES}_${DATE_TAG}"
  pair_path="$out_dir/intervention_pair.npz"

  if [[ ! -f "$pair_path" ]]; then
    PYTHONPATH=src "$NEURON_PY" docs/verify-arbor-simulator/artifacts/run_coreneuron_event_dropout_pair.py \
      --source-data "$SOURCE_DATA" \
      --source-root "$SOURCE_ROOT" \
      --output-dir "$out_dir" \
      --samples "$SAMPLES" \
      --backend coreneuron \
      --intervention "${parts[@]}"
  fi

  PYTHONPATH=src "$EVAL_PY" docs/verify-arbor-simulator/artifacts/run_coreneuron_neuro_bench_mvp.py \
    --m4-event-dropout-pair "$pair_path" \
    --output-json "$eval_dir/results.json" \
    --output-md "$eval_dir/report.md" \
    --checkpoint soma075_calibrated_80k=docs/verify-arbor-simulator/artifacts/coreneuron_soma075_calibrated_20260605/soma075_calibrated_80k_best.pt \
    --checkpoint mamba_exc0925_80k=docs/verify-arbor-simulator/artifacts/hay_coreneuron_exc0925_mamba80k_20260605/mamba_exc0925_80k_best.pt \
    --checkpoint rnn_exc0925_80k=docs/verify-arbor-simulator/artifacts/hay_coreneuron_exc0925_rnn80k_20260605/rnn_exc0925_80k_best.pt
done
