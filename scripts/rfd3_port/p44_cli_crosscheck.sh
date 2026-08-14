#!/bin/sh
# Cross-check the p44 harness against the shipped CLI at the row where the published table and a
# real run disagree most (419 atoms, batch 1). A CLI wall clock includes weight load and
# featurize, so one run cannot be compared against a per-design number; two runs can. Marginal
# s/design = (wall(N) - wall(1)) / (N - 1) cancels every one-time cost, the same segmentation
# rule the A/B harnesses use for ms/step.
set -e
WT=/home/ttuser/.coworker/wt/rfd3-host-half-defaults-on
cd "$WT"
PY=/home/ttuser/tt-bio-dev/env/bin
OUT=perf/p44/cli_crosscheck
mkdir -p "$OUT"

run() {  # $1 = num_designs
  rm -rf "$OUT"/d$1
  t0=$(date +%s.%N)
  env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:rfd3-host-half-defaults-on \
    PYTHONPATH="$WT" "$PY"/tt-bio design scripts/rfd3_port/p44_spec419.json --model rfd3 \
    --from_pdb --out_dir "$OUT"/d$1 --num_designs "$1" --batch_size 1 --num_timesteps 200 \
    --seed 42 > "$OUT"/n$1.log 2>&1
  t1=$(date +%s.%N)
  echo "$1 $(echo "$t1 - $t0" | bc)" >> "$OUT"/wall.txt
}

: > "$OUT"/wall.txt
run 1
run 3
awk '{w[$1]=$2} END {m=(w[3]-w[1])/2; printf "wall(1)=%.2fs wall(3)=%.2fs marginal=%.2f s/design => %.4f designs/sec\n", w[1], w[3], m, 1/m}' "$OUT"/wall.txt | tee "$OUT"/result.txt
