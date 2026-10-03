#!/usr/bin/env bash
# EXP-DSA05-008 round 1 (design §3, §4): 12 formal training runs (R0, F1-F5 x seeds
# 101, 202), the R0 reproducibility check against EXP-006 phase 1 (§4.4) before
# any evaluation, the screening evaluation of every checkpoint, the per-run
# selection, the validation of the selected checkpoints and the round-1 rules.
# Needs a clean committed tree (every evaluation and training run is --formal).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/exp008-round1
PHASE1=outputs/exp006-phase1
CONDITIONS="r0 f1 f2 f3 f4 f5"
SEEDS="101 202"
PY=.venv/bin/python
mkdir -p "$OUT/screen"

pids=()
for c in $CONDITIONS; do
  for seed in $SEEDS; do
    $PY -m dsa_period.experiments.train_capacity_mappo --config "configs/exp008-round1-$c.toml" \
      --variant fixed --seed "$seed" --formal --output "$OUT/$c-seed$seed" \
      > "$OUT/$c-seed$seed.log" 2>&1 &
    pids+=($!)
  done
done
echo "${pids[@]}" > "$OUT/train.pids"
training() { for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }

r0_check() {  # §4.4: every common checkpoint and every logged row must be identical
  for seed in $SEEDS; do
    $PY scripts/compare_capacity_runs.py checkpoints \
      --reference "$PHASE1/fixed-seed$seed" --candidate "$OUT/r0-seed$seed" || return 1
    $PY scripts/compare_capacity_runs.py log \
      --reference "$PHASE1/fixed-seed$seed" --candidate "$OUT/r0-seed$seed" || return 1
  done
}

nth_checkpoint() { ls "$OUT/$1/checkpoints/"*.pt 2>/dev/null | sort | sed -n "$2p"; }

for k in 1 2 3 4 5; do
  for c in $CONDITIONS; do
    for seed in $SEEDS; do
      until [ -n "$(nth_checkpoint "$c-seed$seed" "$k")" ]; do sleep 30; done
    done
  done
  sleep 20  # let torch.save finish
  if ! r0_check; then
    echo "$(date) R0 differs from EXP-006 phase 1 (design §4.4): stopping round 1"
    kill "${pids[@]}" 2>/dev/null || true
    exit 1
  fi
  specs=""
  for c in $CONDITIONS; do
    for seed in $SEEDS; do specs="$specs marl:$(nth_checkpoint "$c-seed$seed" "$k")"; done
  done
  workers=4; training || workers=15
  echo "$(date) screening batch $k ($workers workers)"
  # shellcheck disable=SC2086
  $PY -m dsa_period.experiments.evaluate_capacity --config configs/scenario-v2.toml \
    --split dev --screen exp008-screen256 --policies $specs --workers "$workers" --formal \
    --output "$OUT/screen/batch-$k"
done
wait
r0_check || { echo "$(date) R0 differs after training (design §4.4)"; exit 1; }

$PY -m dsa_period.experiments.select_exp008 select --screen "$OUT"/screen/batch-* \
  --output "$OUT/selection.json"
selected=$($PY -c "import json; print(' '.join(r['selected'] for r in json.load(open('$OUT/selection.json'))['runs'].values()))")
# shellcheck disable=SC2086
$PY -m dsa_period.experiments.evaluate_capacity --config configs/scenario-v2.toml \
  --split validation --grid 2 4 6 8 10 12 16 --policies $selected --workers 15 --formal \
  --output "$OUT/validation"
# §4.4: the R0 selected checkpoint must score exactly as in EXP-006 phase 1 (common N).
for k in 1 2 3 4 5; do
  verdict=$($PY scripts/compare_capacity_runs.py validation \
    --reference "docs/results/exp006-phase1_20261001/validation-batch-$k.json" \
    --candidate "$OUT/validation/summary.json" \
    --runs fixed-seed101=r0-seed101 fixed-seed202=r0-seed202 || true)
  echo "phase-1 validation batch $k: $verdict"
  case "$verdict" in
    DIFFERENT*) echo "$(date) the evaluation path changed (design §4.4): stopping"; exit 1 ;;
  esac
done
$PY -m dsa_period.experiments.select_exp008 round1 --selection "$OUT/selection.json" \
  --validation "$OUT/validation" --output "$OUT/round1.json"
echo "$(date) round 1 done"
