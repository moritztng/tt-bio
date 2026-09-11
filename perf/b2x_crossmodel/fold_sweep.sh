#!/bin/bash
# Fold-level A/B/A2 for the mask-after-move flag, one model at a time on one card.
# A2 is the A/A floor: pc card 0 miscomputes at a low rate, so a fold-level sha comparison is
# only readable against a floor measured in the same session.
set -u
WT=/home/moritz/.coworker/wt/b2x-trimul-e6-crossmodel-parity
cd "$WT" || exit 1
OUT="$WT/perf/b2x_crossmodel/out"
mkdir -p "$OUT" "$WT/foldout"
COMMON="--single_sequence --seed 0 --recycling_steps 1 --sampling_steps 20 --diffusion_samples 1 --output_format cif"
for model in "$@"; do
  for arm in A:0 B:1 A2:0; do
    tag=${arm%%:*}; flag=${arm##*:}
    d="$WT/foldout/${model}_${tag}"
    rep="$OUT/fold_${model}_${tag}.json"
    [ -s "$rep" ] && { echo "skip $model $tag (already have $rep)"; continue; }
    rm -rf "$d"
    echo "=== $model arm $tag flag=$flag $(date -u +%H:%M:%S)"
    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
    TT_BIO_LEASE_HOLDER=worker:b2x-trimul-e6-crossmodel-parity \
    PYTHONPATH="$WT" timeout 2400 /home/moritz/tt-bio/env/bin/python3 \
      perf/b2x_crossmodel/fold_arm.py --flag "$flag" --tag "$tag" --report "$rep" \
      --results "$d/${model}_results_615/results.json" -- \
      predict examples/615.yaml --model "$model" $COMMON --out_dir "$d" \
      2>&1 | grep -E "^ARM |✓|Done:|Error|Traceback|failed" | tail -5
  done
done
echo "FOLD_SWEEP_DONE $(date -u +%H:%M:%S)"
