#!/usr/bin/env bash
# The full-track bfp8 (TT_BIO_PAIR_B8) fold A/B, 512 aa first because it is the deciding size.
# Each size's arms interleave block by block and the A/A floor comes free off the base blocks.
# Clock is pinned by a separate FORCE_AICLK holder and sampled inside every fold by the harness.
set -u
WT=/home/ttuser/.coworker/wt/bfp8-revive-accuracy-killed
cd "$WT" || exit 1
P=/home/ttuser/tt-bio-dev/env/bin/python3
E="env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:bfp8-revive-accuracy-killed"

for size in 512 298; do
  echo "=== $size aa START $(date -uIs)"
  /home/ttuser/.coworker/scripts/benchlock.sh bfp8-revive-accuracy-killed -- \
    $E $P "$WT/perf/bfp8_revive/pairb8_fold_ab.py" \
      --sizes "$size" --blocks 4 --folds 3 --card 2 \
      --flag TT_BIO_PAIR_B8 \
      --out "$WT/perf/bfp8_revive/pairb8_ab_${size}.json" \
      --cifdir "$WT/perf/bfp8_revive/cifs" 2>&1 \
    | grep -avE "DEBUG|Config\{|UMD \||leaked|nanobind|^ - |^See http|^ --- "
  echo "=== $size aa rc=${PIPESTATUS[0]} $(date -uIs)"
done
echo "=== CHAIN DONE $(date -uIs)"
