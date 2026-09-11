#!/bin/bash
# Fold-level A/B/A2 for the mask-after-move flag on qb2 card 1, one model at a time.
# Card 1 is the one card in this grant that may host a hash-equality verdict: pc card 0
# miscomputes matmuls at a low location-keyed rate and its fold shas are unreadable in both
# directions. A2 is still the A/A floor, measured in the same session on the same card.
set -u
WT=/home/ttuser/.coworker/wt/b2x-trimul-e6-crossmodel-parity
cd "$WT" || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT="$WT/perf/b2x_crossmodel/out_qb2"
mkdir -p "$OUT" "$WT/foldout"
COMMON="--single_sequence --seed 0 --recycling_steps 1 --sampling_steps 20 --diffusion_samples 1 --output_format cif"
for model in "$@"; do
  for arm in A:0 B:1 A2:0; do
    tag=${arm%%:*}; flag=${arm##*:}
    d="$WT/foldout/${model}_${tag}"
    rep="$OUT/fold_${model}_${tag}.json"
    sdir="$OUT/stats_${model}_${tag}"
    [ -s "$rep" ] && { echo "skip $model $tag (already have $rep)"; continue; }
    rm -rf "$d" "$sdir"; mkdir -p "$sdir"
    echo "=== $model arm $tag flag=$flag start $(date -u +%H:%M:%S)"
    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
    TT_BIO_LEASE_HOLDER=worker:b2x-trimul-e6-crossmodel-parity \
    TT_BIO_REBLOCK_STATS_DIR="$sdir" \
    PYTHONPATH="$WT/perf/b2x_crossmodel/hook:$WT" timeout 2400 "$PY" \
      perf/b2x_crossmodel/fold_arm.py --flag "$flag" --tag "$tag" --report "$rep" \
      --results "$d/${model}_results_615/results.json" -- \
      predict examples/615.yaml --model "$model" $COMMON --out_dir "$d" \
      2>&1 | grep -E "^ARM |Done:|Error|Traceback|failed" | tail -5
    echo "--- $model $tag end $(date -u +%H:%M:%S)  child counters:"
    grep -h "gated_moves\|mask_after_move" "$sdir"/*.json 2>/dev/null | sort | uniq -c
  done
done
echo "FOLD_SWEEP_DONE $(date -u +%H:%M:%S)"
