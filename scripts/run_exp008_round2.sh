#!/usr/bin/env bash
# EXP-DSA05-008 round 2 (design §5): on B1 = R0, 6 formal training runs (A-2400,
# C-600, C-2400 x seeds 101, 202); A-600 reuses the two R0 runs of round 1. Every
# checkpoint is screened as it appears, each run selects on the screening set,
# the selected checkpoints are validated once, and the §5.3 rule chooses B2.
# Needs a clean committed tree (every evaluation and training run is --formal).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/exp008-round2
ROUND1=outputs/exp008-round1
SEEDS="101 202"
RUNS="a2400:fixed c600:local c2400:local"
PY=.venv/bin/python
test -f "$ROUND1/selection.json" && test -f "$ROUND1/validation/summary.json"
mkdir -p "$OUT/screen"

pids=()
names=()
for item in $RUNS; do
  c=${item%%:*}
  variant=${item##*:}
  for seed in $SEEDS; do
    $PY -m dsa_period.experiments.train_capacity_mappo --config "configs/exp008-round2-$c.toml" \
      --variant "$variant" --seed "$seed" --formal --output "$OUT/$c-seed$seed" \
      > "$OUT/$c-seed$seed.log" 2>&1 &
    pids+=($!)
    names+=("$c-seed$seed")
  done
done
echo "${pids[@]}" > "$OUT/train.pids"
training() { for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
nth_checkpoint() { ls "$OUT/$1/checkpoints/"*.pt 2>/dev/null | sort | sed -n "$2p"; }

for k in 1 2 3 4 5; do
  for run in "${names[@]}"; do
    until [ -n "$(nth_checkpoint "$run" "$k")" ]; do
      training || { echo "$(date) $run stopped before checkpoint $k"; exit 1; }
      sleep 30
    done
  done
  sleep 20  # let torch.save finish
  specs=""
  for run in "${names[@]}"; do specs="$specs marl:$(nth_checkpoint "$run" "$k")"; done
  workers=9; training || workers=15
  echo "$(date) screening batch $k ($workers workers)"
  # shellcheck disable=SC2086
  $PY -m dsa_period.experiments.evaluate_capacity --config configs/scenario-v2.toml \
    --split dev --screen exp008-screen256 --policies $specs --workers "$workers" --formal \
    --output "$OUT/screen/batch-$k"
done
wait

$PY -m dsa_period.experiments.select_exp008 select --screen "$OUT"/screen/batch-* \
  --output "$OUT/selection.json"
selected=$($PY -c "import json; print(' '.join(r['selected'] for r in json.load(open('$OUT/selection.json'))['runs'].values()))")
# shellcheck disable=SC2086
$PY -m dsa_period.experiments.evaluate_capacity --config configs/scenario-v2.toml \
  --split validation --grid 2 4 6 8 10 12 16 --policies $selected --workers 15 --formal \
  --output "$OUT/validation"
$PY -m dsa_period.experiments.select_exp008 round2 \
  --selection "$ROUND1/selection.json" "$OUT/selection.json" \
  --validation "$ROUND1/validation" "$OUT/validation" --a600 R0 --output "$OUT/round2.json"
echo "$(date) round 2 done"
